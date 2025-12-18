import SimpleITK as sitk
import numpy as np
import logging
from scipy import ndimage
from totalsegmentator.python_api import totalsegmentator
import utils.io as io
from typing import Optional
import configparser as cpars
import pydicom
from typing import Any
import os
import fnmatch
import yaml
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def rtk_to_HU(img: sitk.Image) -> sitk.Image:
    """
    Converts RTK reconstructed image to Hounsfield Units (HU)
    CBCT_HU =  CBCT_μ * 2^16 - 1024
    """
    img = img*(2**16)-1024
    return img

def fix_array_order(img: sitk.Image, order = (1, 2, 0), flip=()) -> sitk.Image:
    """
    This function ensures the RTK reconstructed image is in a similar configuration as the .SCAN file loaded with xdrt
    """
    img_np = sitk.GetArrayFromImage(img)
    img_np = np.transpose(img_np, order)
    img_np = np.flip(img_np, axis=flip)
    
    new_img = sitk.GetImageFromArray(img_np)
    
    old_origin = img.GetOrigin()
    new_origin = (old_origin[order[2]], old_origin[order[1]], old_origin[order[0]])
    
    old_spacing = img.GetSpacing()
    new_spacing = (old_spacing[order[2]], old_spacing[order[1]], old_spacing[order[0]])
    
    new_img.SetOrigin(new_origin)
    new_img.SetSpacing(new_spacing)
    
    return new_img

def allign_images_ini(image: sitk.Image, reconstruction_ini=None) -> sitk.Image:
    """
    alligns Elekta CBCT with corresponding planning CT using parameters written in a .INI file in the Reconstructin folder
    """
    
    correction = reconstruction_ini['ALIGNMENT']['onlinetoreftransformcorrection']
    values = correction.split(' ')
    values = [float(v) for v in values if v != '']
    
    R = np.array([[values[0], values[4], values[8]],
                [values[1], values[5], values[9]],
                [values[2], values[6], values[10]]], dtype=float)
    t = np.array(values[12:15], dtype=float)

    origin = np.array(image.GetOrigin())
    o_new = origin + t*10  # assuming t is in cm, convert to mm
    image.SetOrigin(o_new.tolist())

    return image

def _load_tags(file_path):
    """Reads the tags configuration file and returns a list of keywords."""
    if not os.path.exists(file_path):
        print(f"Error: Configuration file '{file_path}' not found.")
        return []
    
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def _to_builtin(v):
    if v is None:
        return None

    if isinstance(v, (list, tuple)):
        return [_to_builtin(x) for x in v]

    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        return [_to_builtin(x) for x in v]
    
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
        return float(v)
    
    if isinstance(v, pydicom.valuerep.IS):
        return int(v)

    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            return v.hex()

    if isinstance(v, str):
        s = v.strip()
        if s:
            try:
                i = int(s)
                return i
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return v
        return None  # empty string -> null (optional)

    return v

def extract_dicom_metadata(directory_path, tags_file):
    """
    Extracts specified DICOM metadata tags from the first DICOM file in the given directory.
    """
    target_tags = _load_tags(tags_file)
    extracted_tags = {} 

    if not target_tags:
        return {}

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory_path)

    # find series with most files
    max_files = 0
    longest_series = None
    
    for sid in series_ids:
        filenames = reader.GetGDCMSeriesFileNames(directory_path, sid)
        if len(filenames) > max_files:
            max_files = len(filenames)
            longest_series = sid
    
    fn = reader.GetGDCMSeriesFileNames(directory_path, longest_series)[0]
    # fn = sorted(fnmatch.filter(os.listdir(directory_path), '*.dcm'))[0]
    file_path = os.path.join(directory_path, fn)
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        for tag in target_tags:
            elem = ds.get(tag)
            extracted_tags[tag] = _to_builtin(elem) if elem is not None else None
                    
    except Exception as e:
        print(f"Error reading DICOM file '{file_path}': {e}")
    
    return extracted_tags

def extract_elekta_metadata(tags_yaml: str, reconstruction_ini: dict, projections: sitk.Image, cbct_clinical: sitk.Image, geometry: Any) -> dict:
    """
    Extracts Elekta CBCT metadata from reconstruction INI files and other related files.
    """
    with open(tags_yaml, 'r') as f:
        tags = yaml.safe_load(f)

    metadata_dict = {}

    # extract tags from ini files
    for tag in tags:
        for section in reconstruction_ini:
            if tags[tag]['Elekta'] in reconstruction_ini[section]:
                metadata_dict[tag] = _to_builtin(reconstruction_ini[section][tags[tag]['Elekta']])
                break

    # extract tags from projections.mha files
    metadata_dict['Frames'] = int(projections.GetSize()[2])
    metadata_dict['ImagerResX'] = _to_builtin(projections.GetSpacing()[0])
    metadata_dict['ImagerResY'] = _to_builtin(projections.GetSpacing()[1])
    metadata_dict['ImagerSizeX'] = _to_builtin(projections.GetSize()[0])
    metadata_dict['ImagerSizeY'] = _to_builtin(projections.GetSize()[1])

    # extract tags from cbct_clinical.mha
    metadata_dict['ReconstructionSpacingX'] = _to_builtin(cbct_clinical.GetSpacing()[0])
    metadata_dict['ReconstructionSpacingY'] = _to_builtin(cbct_clinical.GetSpacing()[1])
    metadata_dict['ReconstructionSpacingZ'] = _to_builtin(cbct_clinical.GetSpacing()[2])
    metadata_dict['ReconstructionSizeX'] = _to_builtin(cbct_clinical.GetSize()[0])
    metadata_dict['ReconstructionSizeY'] = _to_builtin(cbct_clinical.GetSize()[1])
    metadata_dict['ReconstructionSizeZ'] = _to_builtin(cbct_clinical.GetSize()[2])
    
    # extract tags from geometry 
    # check if it's a half or full arc
    angles = geometry.GetGantryAngles()
    angles = [angle * 180.0 / 3.141592653589793 for angle in angles]
    if angles[0] + angles[-1] < 300:
        metadata_dict['Trajectory'] = 'half'
    else:
        metadata_dict['Trajectory'] = 'full'
    metadata_dict['StartAngle'] = angles[0]
    metadata_dict['StopAngle'] = angles[-1]
    # check if detector is offset
    if abs(geometry.GetProjectionOffsetsX()[0]) >= 5 or abs(geometry.GetProjectionOffsetsY()[0]) >= 5:
        metadata_dict['Fan'] = 'Half'
    else:
        metadata_dict['Fan'] = 'Full'
    metadata_dict['DetectorOffsetX'] = geometry.GetProjectionOffsetsX()[0]
    metadata_dict['DetectorOffsetY'] = geometry.GetProjectionOffsetsY()[0]

    # others
    metadata_dict['ScatterGrid'] = 'None'
    metadata_dict['Manufacturer'] = 'Elekta'
    
    return metadata_dict

def extract_varian_metadata(tags_yaml:str, scan_xml: str, projections: sitk.Image, cbct_clinical: sitk.Image, geometry: Any) -> dict:
    # load yaml file with vendor tags
    with open(tags_yaml, 'r') as f:
        tags_dict = yaml.safe_load(f)

    # read scan xml file and extract available metadata
    tree = ET.parse(scan_xml)   # load file
    root = tree.getroot() 
    ns = {
        "lib": "http://baden.varian.com/cr.xsd"
    }

    metadata_dict = {}

    for tag in tags_dict:
        if tags_dict[tag]['Varian'] is not None:
            for book in root.findall("lib:Acquisitions", ns):
                if book.find(f"lib:{tags_dict[tag]['Varian']}", ns) is not None:
                    value = book.find(f"lib:{tags_dict[tag]['Varian']}", ns).text
                else:
                    value = None
            metadata_dict[tag] = _to_builtin(value)

    # extract tags from projections.mha files
    metadata_dict['Frames'] = int(projections.GetSize()[2])
    metadata_dict['ImagerResX'] = _to_builtin(projections.GetSpacing()[0])
    metadata_dict['ImagerResY'] = _to_builtin(projections.GetSpacing()[1])
    metadata_dict['ImagerSizeX'] = _to_builtin(projections.GetSize()[0])
    metadata_dict['ImagerSizeY'] = _to_builtin(projections.GetSize()[1])

    # extract tags from cbct_clinical.mha
    metadata_dict['ReconstructionSpacingX'] = _to_builtin(cbct_clinical.GetSpacing()[0])
    metadata_dict['ReconstructionSpacingY'] = _to_builtin(cbct_clinical.GetSpacing()[1])
    metadata_dict['ReconstructionSpacingZ'] = _to_builtin(cbct_clinical.GetSpacing()[2])
    metadata_dict['ReconstructionSizeX'] = _to_builtin(cbct_clinical.GetSize()[0])
    metadata_dict['ReconstructionSizeY'] = _to_builtin(cbct_clinical.GetSize()[1])
    metadata_dict['ReconstructionSizeZ'] = _to_builtin(cbct_clinical.GetSize()[2])

    # extract tags from geometry 
    angles = geometry.GetGantryAngles()
    angles = [angle * 180.0 / 3.141592653589793 for angle in angles]
    metadata_dict['StartAngle'] = angles[0]
    metadata_dict['StopAngle'] = angles[-1]
    metadata_dict['DetectorOffsetX'] = geometry.GetProjectionOffsetsX()[0]
    metadata_dict['DetectorOffsetY'] = geometry.GetProjectionOffsetsY()[0]
    
    #etc.
    metadata_dict['Manufacturer'] = 'Varian'
        
    return metadata_dict



# def clip_image(image: sitk.Image, lower_bound: float, upper_bound: float) -> sitk.Image:
#     """Clips an image using SimpleITK."""
#     logger.info(f'Clipping image between {lower_bound} and {upper_bound}')
#     image = sitk.Clamp(image, lowerBound=lower_bound, upperBound=upper_bound)
#     return image

# def mask_image(image: Optional[sitk.Image], mask: Optional[sitk.Image], mask_value=-1024) -> sitk.Image:
#     """
#     Masks the input image using the provided mask image.
#     """
#     mask = sitk.Cast(mask, sitk.sitkUInt8)
#     masked_image = sitk.Mask(image, mask, outsideValue=mask_value)
#     return masked_image

# def resample_reference(image: sitk.Image, ref_image: sitk.Image, default_value=0, interpolator=sitk.sitkLinear) -> sitk.Image:
#     """
#     Resamples the given image to the grid of a reference image.
#     """
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(ref_image)
#     resampler.SetDefaultPixelValue(float(default_value))
#     resampler.SetInterpolator(interpolator)
#     resampled_image = resampler.Execute(image)
#     logger.info(f'Image resampled to reference grid.')
#     return resampled_image

# def resample_image(image: sitk.Image, new_spacing=[1.0, 1.0, 1.0]) -> sitk.Image:
#     """
#     Resamples the given image to a new spacing.
#     """
#     original_spacing = image.GetSpacing()
#     original_size = image.GetSize()

#     new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]

#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(new_spacing)
#     resampler.SetSize(new_size)
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetTransform(sitk.Transform())
#     resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    
#     # Use linear interpolation for intensity images, nearest for masks
#     if image.GetPixelIDValue() in (sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkInt8, sitk.sitkInt16):
#          resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     else:
#         resampler.SetInterpolator(sitk.sitkLinear)

#     resampled_image = resampler.Execute(image)
#     logger.info(f'Image resampled to new spacing {new_spacing}')
#     return resampled_image

# def cbct_ac_correction(cbct_image: sitk.Image, 
#                         ct_image: sitk.Image,
#                         ct_skin: sitk.Image,
#                         cbct_skin: sitk.Image,
#                         ct_air_threshold: int = -500,
#                         cbct_air_threshold: int = -300,
#                         roi_dilation_voxels: int = 5,
#                         fill_tissue_value: int = 30,
#                         ) -> sitk.Image:
#     """
#     Corrects air cavities in a warped CT image based on a reference CBCT.

#     This function performs a density override (in-painting) to match the
#     air cavity seen in the CBCT. It assumes warped_ct_image and cbct_image
#     are in the same physical space (i.e., registration is already done).

#     Args:
#         cbct_image: The fixed CBCT image.
#         warped_ct_image: The deformed CT image, resampled to the CBCT's space.
#         ct_air_threshold: HU value to segment air in the CT (e.g., -500).
#         cbct_air_threshold: Intensity value to segment air in the CBCT.
#                            *** This is the most critical parameter to tune. ***
#         roi_dilation_voxels: Amount to dilate the combined air cavities to
#                              create the correction Region of Interest (ROI).
#         fill_tissue_value: The HU value to use when "filling" old air
#                            cavities with new tissue.
    
#     Returns:
#         A new sitk.Image (corrected_ct) with modified air cavities.
#     """
    
#     logger.info("Starting air cavity correction...")
    
#     # Cast all images to uint8
#     ct_skin = sitk.Cast(ct_skin, sitk.sitkUInt8)
#     cbct_skin = sitk.Cast(cbct_skin, sitk.sitkUInt8)
                              
#     # --- Step 1: Segment Air Cavities ---
#     ct_air_mask = ct_image < ct_air_threshold
#     ct_air_mask = sitk.And(ct_air_mask, ct_skin)  # Limit to within skin
#     cbct_air_mask = cbct_image < cbct_air_threshold
#     cbct_air_mask = sitk.And(cbct_air_mask, cbct_skin)  # Limit to within skin

#     ct_tissue_mask = sitk.Not(ct_air_mask)
#     cbct_tissue_mask = sitk.Not(cbct_air_mask)

#     # --- Step 2: Define Correction ROI ---
#     # Combine the two air cavities to find the general area of change
#     combined_air_mask = sitk.Or(ct_air_mask, cbct_air_mask)
    
#     # Dilate this mask to create a slightly larger ROI
#     # This ensures we catch the edges
#     dilation_radius = [roi_dilation_voxels] * cbct_image.GetDimension()
#     correction_roi = sitk.BinaryDilate(combined_air_mask, dilation_radius)

#     # # --- Step 3: (Optional Refinement) Get a plausible tissue fill value ---
#     # # We can improve on the 'fill_tissue_value' by sampling the
#     # # patient's own tissue from the warped CT within the ROI.
#     # try:
#     #     # Get tissue voxels from the CT within our ROI
#     #     ct_tissue_in_roi = sitk.And(ct_tissue_mask, correction_roi)
        
#     #     # --- THIS IS THE CORRECTION ---
#     #     # Use LabelStatisticsImageFilter to get intensity stats
#     #     stats_filter = sitk.LabelStatisticsImageFilter()
#     #     # Pass the intensity image first, then the label mask
#     #     stats_filter.Execute(ct_image, ct_tissue_in_roi)
        
#     #     # Check if the label (1) exists and has voxels
#     #     # The mask ct_tissue_in_roi is binary, so the label is 1.
#     #     if stats_filter.HasLabel(1):
#     #         mean_tissue_val = stats_filter.GetMean(1)
#     #         # Use this dynamic value instead of the fixed one
#     #         fill_tissue_value = mean_tissue_val
#     #         print(f"Using dynamic fill tissue value: {fill_tissue_value:.2f} HU")
#     #     else:
#     #         print(f"ROI has no tissue, using default fill value: {fill_tissue_value} HU")
            
#     # except Exception as e:
#     #     print(f"Could not compute dynamic tissue value, using default {fill_tissue_value} HU. Error: {e}")

#     # --- Step 4: Identify Discrepancy Masks ---
#     # Region 1: Tissue in CT but Air in CBCT -> "Carve" air
#     # (ct_tissue_mask AND cbct_air_mask)
#     mask_to_become_air = sitk.And(ct_tissue_mask, cbct_air_mask)
    
#     # Region 2: Air in CT but Tissue in CBCT -> "Fill" with tissue
#     # (ct_air_mask AND cbct_tissue_mask)
#     mask_to_become_tissue = sitk.And(ct_air_mask, cbct_tissue_mask)

#     # IMPORTANT: Ensure corrections only happen inside our designated ROI
#     mask_to_become_air = sitk.And(mask_to_become_air, correction_roi)
#     mask_to_become_tissue = sitk.And(mask_to_become_tissue, correction_roi)

#     # --- Step 5: Apply Corrections (In-Painting) ---
#     # It's fastest to do this "painting" using NumPy
#     logger.info("Applying corrections...")
    
#     # Create the output image as a numpy array from the warped CT
#     corrected_ct_np = sitk.GetArrayFromImage(ct_image)
    
#     # Get numpy arrays for the masks
#     to_air_np = sitk.GetArrayFromImage(mask_to_become_air)
#     to_tissue_np = sitk.GetArrayFromImage(mask_to_become_tissue)

#     # Apply the corrections
#     # Note: SimpleITK (x,y,z) <-> NumPy (z,y,x)
#     corrected_ct_np[to_air_np == 1] = -1000  # Set to air
#     corrected_ct_np[to_tissue_np == 1] = fill_tissue_value # Set to tissue

#     # --- Step 6: Convert Back to SimpleITK Image ---
#     corrected_ct_image = sitk.GetImageFromArray(corrected_ct_np)
    
#     # CRITICAL: Copy the spacing, origin, and direction from the original
#     corrected_ct_image.CopyInformation(ct_image)
    
#     print("Correction complete.")
#     return corrected_ct_image

# def normalize(image: sitk.Image, new_min: float = 0.0, new_max: float = 1.0) -> sitk.Image:
#     """
#     Normalizes the image intensities to a specified range [new_min, new_max].
#     """
#     stats = sitk.StatisticsImageFilter()
#     stats.Execute(image)
#     old_min = stats.GetMinimum()
#     old_max = stats.GetMaximum()
    
#     logger.info(f'Normalizing image from range [{old_min}, {old_max}] to [{new_min}, {new_max}]')
    
#     # Avoid division by zero
#     if old_max - old_min == 0:
#         logger.warning('Image has zero intensity range. Returning a constant image.')
#         constant_image = sitk.Image(image.GetSize(), image.GetPixelIDValue())
#         constant_image.CopyInformation(image)
#         constant_image = constant_image + new_min
#         return constant_image
    
#     # Normalize
#     normalized_image = sitk.Cast(image, sitk.sitkFloat32)
#     normalized_image = (normalized_image - old_min) / (old_max - old_min)
#     normalized_image = normalized_image * (new_max - new_min) + new_min
    
#     return normalized_image
