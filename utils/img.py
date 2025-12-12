import SimpleITK as sitk
import numpy as np
import logging
from scipy import ndimage
from totalsegmentator.python_api import totalsegmentator
import utils.io as io
from typing import Optional
import configparser as cpars
import pydicom

logger = logging.getLogger(__name__)

def fix_array_order(img: sitk.Image) -> sitk.Image:
    """
    This function ensures the RTK reconstructed image is in a similar configuration as the .SCAN file loaded with xdrt
    """
    img_np = sitk.GetArrayFromImage(img)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = img_np[::-1,:,:]
    
    new_img = sitk.GetImageFromArray(img_np)
    
    old_origin = img.GetOrigin()
    new_origin = (old_origin[0], old_origin[2], old_origin[1])
    
    old_spacing = img.GetSpacing()
    new_spacing = (old_spacing[0], old_spacing[2], old_spacing[1])
    
    new_img.SetOrigin(new_origin)
    new_img.SetSpacing(new_spacing)
    
    return new_img

def allign_images_ini(image: sitk.Image, correction_ini=None) -> sitk.Image:
    """
    alligns Elekta CBCT with corresponding planning CT using parameters written in a .INI file in the Reconstructin folder
    """
    # Load .INI.XVI file (the one with longer file name...)
    with open(correction_ini) as f:
        lines = f.readlines()
    
    correction = [line for line in lines if line.startswith('OnlineToRefTransformCorrection')][0]
    values = correction.split('=')[1].split(' ')
    values = [float(v) for v in values if v != '']
    
    R = np.array([[values[0], values[4], values[8]],
                [values[1], values[5], values[9]],
                [values[2], values[6], values[10]]], dtype=float)
    t = np.array(values[12:15], dtype=float)

    origin = np.array(image.GetOrigin())
    o_new = origin + t*10  # assuming t is in cm, convert to mm
    image.SetOrigin(o_new.tolist())

    return image

import os
import pydicom
import fnmatch


def load_tags_from_file(file_path):
    """Reads the tags configuration file and returns a list of keywords."""
    if not os.path.exists(file_path):
        print(f"Error: Configuration file '{file_path}' not found.")
        return []
    
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def extract_dicom_metadata_to_dict(directory_path, tags_file):
    """
    Extracts specified DICOM metadata tags from the first DICOM file in the given directory.
    """
    target_tags = load_tags_from_file(tags_file)
    extracted_tags = {} 

    if not target_tags:
        return {}

    fn = sorted(fnmatch.filter(os.listdir(directory_path), '*.dcm'))[0]
    
    file_path = os.path.join(directory_path, fn)
    print(f"Processing file: {file_path}")
    try:
        ds = pydicom.dcmread('/data/LMU/patientLMU007/CT_SET/2.16.840.1.114337.1.1.1524060727.0/CT1.dcm', stop_before_pixels=True)
        for tag in target_tags:
            elem = ds.get(tag)
            if elem is not None:
                try:
                    extracted_tags[tag] = elem
                except Exception as e:
                    print(f"Error extracting tag '{tag}': {e}")
            else:
                    extracted_tags[tag] = None
                    
    except Exception as e:
        print(f"Error reading DICOM file '{file_path}': {e}")
    
    return extracted_tags




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
