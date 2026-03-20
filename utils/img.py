import SimpleITK as sitk
import numpy as np
import logging
from scipy import ndimage
from totalsegmentator.python_api import totalsegmentator
import utils.io as io
from typing import Optional
import configparser as cpars
import pydicom
from typing import Any, Tuple
import os
import fnmatch
import yaml
import xml.etree.ElementTree as ET
import utils.seg as seg

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

def mask_image(image: Optional[sitk.Image], mask: Optional[sitk.Image], mask_value=-1024) -> sitk.Image:
    """
    Masks the input image using the provided mask image.
    """
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    masked_image = sitk.Mask(image, mask, outsideValue=mask_value)
    return masked_image

def fill_cavities_by_dilation(ct_img: sitk.Image, cavity_mask: sitk.Image, radius=1) -> sitk.Image:
    """
    Fill cavities in a CT by iteratively propagating neighboring values inward.

    Parameters
    ----------
    ct_img : sitk.Image
        Input CT image.
    cavity_mask : sitk.Image
        Binary mask, 1 where cavities should be filled.
    radius : int
        Neighborhood radius for dilation.

    Returns
    -------
    sitk.Image
        Filled CT image.
    """
    cavity_mask = sitk.Cast(cavity_mask > 0, sitk.sitkUInt8)
    cavity_mask = sitk.BinaryDilate(cavity_mask, (1,1,1))

    min_val = float(sitk.GetArrayViewFromImage(ct_img).min())
    work = sitk.Mask(ct_img, cavity_mask == 0) + sitk.Cast(cavity_mask, ct_img.GetPixelID()) * (min_val - 1)

    remaining = sitk.Image(cavity_mask)

    dilate = sitk.GrayscaleDilateImageFilter()
    dilate.SetKernelRadius(radius)

    prev_remaining_count = -1

    while True:
        remaining_count = int(sitk.GetArrayViewFromImage(remaining).sum())
        if remaining_count == 0 or remaining_count == prev_remaining_count:
            break
        prev_remaining_count = remaining_count
        dilated = dilate.Execute(work)
        work = sitk.Mask(work, remaining == 0) + sitk.Mask(dilated, remaining)
        remaining = sitk.Cast(work <= (min_val - 1), sitk.sitkUInt8)

    work[work < -70] = -70
    smoothed = sitk.DiscreteGaussian(work, variance=1)
    ct_filled = sitk.Mask(ct_img, cavity_mask == 0) + sitk.Mask(smoothed, cavity_mask > 0)
    
    return ct_filled

def insert_air_cavity(
    image: sitk.Image,
    air_mask: sitk.Image,
    air_value: float = -824.0,
    sigma: float = 1.0,
) -> sitk.Image:
    """
    Inserts an air cavity into an image by replacing masked voxels with air_value,
    with a soft Gaussian-blurred boundary transition.

    Args:
        image:     The base image to modify.
        air_mask:  Binary mask indicating the air cavity region.
        air_value: HU value representing air (default: -824.0).
        sigma:     Gaussian blur sigma applied to the air mask for soft blending (default: 1.0).

    Returns:
        Image with the air cavity inserted.
    """
    pixel_type = image.GetPixelID()

    # Resample mask to match image space
    air_mask = sitk.Resample(
        air_mask, image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        defaultPixelValue=0,
    )
    air_mask = sitk.Cast(air_mask, sitk.sitkFloat32)

    # Blur to get soft boundary weights in [0, 1]
    air_mask = sitk.BinaryErode(sitk.Cast(air_mask,sitk.sitkUInt8), (1, 1, 1))   
    air_weight = sitk.SmoothingRecursiveGaussian(air_mask, sigma=sigma, normalizeAcrossScale=True)

    # Blend: weighted sum of air_value and original image
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    combined = air_weight * air_value + (1.0 - air_weight) * image_float
    combined[combined < -1024] = -1024

    return sitk.Cast(combined, pixel_type)


def ac_correction(ct: sitk.Image, cbct: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """Performs air cavity correction on the CT image using the CBCT image."""
    cbct_acs = seg.segment_cbct_ac(cbct)
    ct_acs = seg.segment_ct_ac(ct)
    ct_filled = fill_cavities_by_dilation(ct, ct_acs)
    ct_ac = insert_air_cavity(ct_filled, cbct_acs, air_value=-824.0)
    return ct_ac, ct_acs, cbct_acs
