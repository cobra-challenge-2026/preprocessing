from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import logging
import os
from utils import img
import utils.io as io
import xmltodict
from typing import Optional, Union
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from scipy import ndimage
import numpy as np
import utils.sct_generator as sct_gen


import utils.reg as reg

logger = logging.getLogger(__name__)

def elekta_fov(input: sitk.Image, threshold: float = -1024)->sitk.Image:
    """
    Estimates the CBCT field of view (FOV) by thresholding.
    Only works for Elekta CBCTs where the FOV is clearly distinguishable from the background.
    """
    cbct_fov = sitk.Threshold(input, lower=threshold-1, upper=threshold, outsideValue=0)
    cbct_fov[cbct_fov == 0] = 1
    cbct_fov[cbct_fov != 1] = 0
    cbct_fov = sitk.Cast(cbct_fov, sitk.sitkUInt8)
    cbct_fov = sitk.VotingBinaryIterativeHoleFilling(cbct_fov)
    return cbct_fov

def get_cbct_fov(image: sitk.Image, margin_mm: float = 2.0) -> sitk.Image:
    """
    Create a circular FOV mask for a CBCT image.
    
    The circle is centered in the XY plane with radius derived from the
    image geometry. For edge slices where the FOV shrinks, it detects
    the actual boundary per slice.
    
    Args:
        image: Input CBCT volume (SimpleITK Image).
        margin_mm: Shrink the radius by this amount (mm) to avoid
                   partial-volume edge voxels. Typically 1-2 mm.
    
    Returns:
        Binary SimpleITK Image (UInt8): 1 inside FOV, 0 outside.
    """
    size = image.GetSize()        # (x, y, z)
    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()    # (x, y, z)

    center_x = origin[0] + (size[0] - 1) * spacing[0] / 2.0
    center_y = origin[1] + (size[1] - 1) * spacing[1] / 2.0

    extent_x = size[0] * spacing[0]
    extent_y = size[1] * spacing[1]
    default_radius = min(extent_x, extent_y) / 2 - margin_mm

    arr = sitk.GetArrayFromImage(image)  # shape: (z, y, x)
    n_slices = arr.shape[0]

    radii = np.full(n_slices, default_radius)

    n_check = min(200, n_slices // 2)
    for s in list(range(n_check)) + list(range(n_slices - n_check, n_slices)):
        r = _detect_slice_radius(arr[s], spacing, origin, center_x, center_y)
        if r is not None and r < default_radius:
            radii[s] = r - margin_mm

    mask_arr = np.zeros_like(arr, dtype=np.uint8)

    ix = np.arange(size[0])
    iy = np.arange(size[1])
    gx, gy = np.meshgrid(ix, iy)  # shape: (y, x)

    px = origin[0] + gx * spacing[0]
    py = origin[1] + gy * spacing[1]

    dist_sq = (px - center_x) ** 2 + (py - center_y) ** 2

    for s in range(n_slices):
        mask_arr[s] = (dist_sq <= radii[s] ** 2).astype(np.uint8)

    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(image)
    return mask

def _detect_slice_radius(
    slice_arr: np.ndarray,
    spacing: tuple,
    origin: tuple,
    center_x: float,
    center_y: float,
    threshold: float = -1000.0,
) -> float | None:
    """
    Detect the FOV radius on a single axial slice by finding the
    largest inscribed circle of non-background voxels.
    
    Works by computing the distance of every non-background voxel
    from the center, then taking a high percentile as the radius.
    """
    inside = slice_arr > threshold  # (y, x)

    if inside.sum() < 100:
        return None

    iy, ix = np.where(inside)
    px = origin[0] + ix * spacing[0]
    py = origin[1] + iy * spacing[1]

    dist = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
    return float(np.percentile(dist, 100))


def segment_skin(input:Optional[sitk.Image], modality:str = 'CT', **kwargs)->sitk.Image:
    """
    Segment the skin from the input image using TotalSegmentator.

    Parameters:
    - input (sitk.Image): The input image from which to segment the skin.
    - modality (str): The modality of the input image ('ct' or 'mr'). Default is 'ct'.
    Returns:
    - sitk.Image: The segmented skin image.
    """
    logger.info('Starting skin segmentation using TotalSegmentator...')
    
    input_nib = io.sitk_to_nib(input)
    
    if modality == 'MR':
        ts_output = totalsegmentator(input_nib,output=None,task='body_mr',quiet=True, device= 'gpu:0')
    elif modality == 'CBCT':
        ts_output = totalsegmentator(input_nib,output=None,task='body_mr',quiet=True, device= 'gpu:0')
    elif modality == 'CT':
        ts_output = totalsegmentator(input_nib,output=None,task='body',quiet=True, device= 'gpu:0')
    else:
        raise ValueError(f"Invalid modality '{modality}'. Supported modalities are 'CT' and 'MR'.")

    skin = io.nib_to_sitk(ts_output)
    
    logger.info('Skin segmentation completed.')

    return skin

def segment_lung(input:Optional[sitk.Image], **kwargs)->sitk.Image:

    logger.info('Starting lung segmentation using TotalSegmentator...')
    lung_rois = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    input_nib = io.sitk_to_nib(input)
    ts_output = totalsegmentator(input_nib,output=None,task='total', roi_subset=lung_rois, quiet=True, device= 'gpu:0')
    header = ts_output.header.extensions[0].get_content() # type: ignore
    header = xmltodict.parse(header)
    structures = io.nib_to_sitk(ts_output) 
    return structures, header

def segment_OAR_structures(input:Optional[sitk.Image], modality:str = 'CT', **kwargs) -> tuple[sitk.Image, dict]:
    """
    Segment OAR structures from the input image using TotalSegmentator.

    Parameters:
    - input (sitk.Image): The input image from which to segment OAR structures.
    - modality (str): The modality of the input image ('ct' or 'mr'). Default is 'ct'.
    Returns:
    - dict[str, sitk.Image]: A dictionary of segmented OAR structures.
    """
    logger.info('Starting OAR structure segmentation using TotalSegmentator...')
    
    input_nib = io.sitk_to_nib(input)
    
    if modality == 'MR':
        ts_output = totalsegmentator(input_nib,output=None,task='total_mr',quiet=True, device= 'gpu:0')
    elif modality == 'CBCT':
        ts_output = totalsegmentator(input_nib,output=None,task='total_mr',quiet=True, device= 'gpu:0')
    elif modality == 'CT':
        ts_output = totalsegmentator(input_nib,output=None,task='total',quiet=True, device= 'gpu:0')
    else:
        raise ValueError(f"Invalid modality '{modality}'. Supported modalities are 'CT' and 'MR'.")
    
    header = ts_output.header.extensions[0].get_content() # type: ignore
    header = xmltodict.parse(header)
    structures = io.nib_to_sitk(ts_output)  # Convert to sitk.Image to ensure compatibility
    
    logger.info('OAR structure segmentation completed.')

    return structures, header

def split_multilabel_segmentation(input:sitk.Image, header: dict, save_to_files: bool = False, output_dir: str ='./', **kwargs):
    """
    Split a multi-label segmentation image into individual binary masks. Not all labels may be present in the input image.

    Parameters:
    - input (sitk.Image): The multi-label segmentation image.
    - header (dict): The header information containing label metadata.
    
    Returns:
    - dict[int, sitk.Image]: A dictionary where keys are label values and values are binary mask images.
    """
    #logger.info('Starting to split multi-label segmentation into individual masks...')
    
    label_ids = header['CaretExtension']['VolumeInformation']['LabelTable']['Label']
    split_masks = []
    
    if "float" in input.GetPixelIDTypeAsString(): 
        logger.info(f"Warning: Mask pixel type is float ({input.GetPixelIDTypeAsString()}). Casting to sitk.sitkUInt32.")
        # Casting will truncate float values (e.g., 1.9 -> 1, 2.1 -> 2)
        input = sitk.Cast(input, sitk.sitkUInt32)
    else:
        input = input
        
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(input)

    all_labels = stats.GetLabels()
    logger.info(f"Found labels: {all_labels}")

    for label_value in all_labels:
        binary_mask = (input == label_value)
        new_mask = {'label_value': label_value, 'label_name': label_ids[label_value-1], 'binary_mask': binary_mask}
        split_masks.append(new_mask)

    logger.info(f"Successfully split mask into {len(split_masks)} binary masks.")
    
    if save_to_files:
        for mask_info in split_masks:
            label_value = mask_info['label_value']
            label_name = mask_info['label_name']['#text']
            binary_mask = mask_info['binary_mask']
            output_structure_dir = os.path.join(output_dir, 'ts_structures')
            os.makedirs(output_structure_dir, exist_ok=True)
            filename = os.path.join(output_structure_dir, f"{label_name}.nrrd")
            io.save_image(binary_mask, filename, compression=True)
            logger.info(f"Saved binary mask for label {label_value} ({label_name}) to {filename}")
    
    return split_masks

def warp_planning_structures(planning_structures: dict[str, sitk.Image], disp_field: sitk.Image, save_to_files: bool = False, output_dir: str ='./', invert_dvf = False, **kwargs) -> dict[str, sitk.Image]:
    """
    Warp planning structures using the provided displacement field.

    Parameters:
    - planning_structures (dict[str, sitk.Image]): A dictionary of planning structure images.
    - disp_field (sitk.Image): The displacement field used for warping.

    Returns:
    - dict[str, sitk.Image]: A dictionary of warped planning structure images.
    """
    logger.info('Starting to warp planning structures using the displacement field...')
    if invert_dvf:
        logger.info('Inverting displacement field for warping planning structures...')
        disp_field = reg.invert_displacement_field(disp_field)
    warped_structures = {}
    for structure_name, structure_image in planning_structures.items():
        logger.info(f"Warping structure: {structure_name}")
        
        warped_image = reg.warp_structure(structure_image, disp_field)
        warped_structures[structure_name] = warped_image
        logger.info(f"Completed warping structure: {structure_name}")
        if save_to_files:
            output_structure_dir = os.path.join(output_dir, 'planning_structures')
            os.makedirs(output_structure_dir, exist_ok=True)
            filename = os.path.join(output_structure_dir, f"{structure_name}")
            io.save_image(warped_image, filename, compression=True)
            logger.info(f"Saved warped structure {structure_name} to {filename}")
    
    logger.info('Completed warping all planning structures.')

    return warped_structures

def segment_outline(input:sitk.Image,threshold:float=0.30,log=False)->sitk.Image:
    """
    Segment the outline of a given input image.

    Parameters:
    input (sitk.Image): The input image to segment.
    threshold (float): A relative threshold value for segmentation, 
                       can be used in case holes are appearing in the mask or too much 
                       of surrounding elements are included in the mask. Default is 0.30.

    Returns:
    sitk.Image: The segmented outline image.
    """
    
    # get patient outline segmentation
    input_np = sitk.GetArrayFromImage(input)
    
    #find range of values in image
    background = np.percentile(input_np, 2.5)
    high = np.percentile(input_np, 97.5)

    # create mask
    mask = input_np > background + threshold*(high-background)
    struct_erosion = np.ones((1,10,10))
    struct_dilation = np.ones((1,10,10))
    mask = ndimage.binary_erosion(mask,structure=struct_erosion).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask,structure=struct_dilation).astype(mask.dtype)
    mask = ndimage.binary_erosion(mask,structure=struct_erosion).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask,structure=struct_dilation).astype(mask.dtype)
    mask = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(int)))
    sorted_component_image = sitk.RelabelComponent(mask, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    mask = largest_component_binary_image
    mask = sitk.BinaryMorphologicalClosing(largest_component_binary_image, (8, 8, 8))
    mask = sitk.BinaryFillhole(mask)
    mask.CopyInformation(input)
    mask = sitk.Cast(mask,sitk.sitkUInt8)
    
    # 2D axial hole filling 
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np_filled = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        mask_np_filled[i] = ndimage.binary_fill_holes(mask_np[i])

    mask_filled = sitk.GetImageFromArray(mask_np_filled)
    mask_filled.CopyInformation(mask)
    mask_filled = sitk.Cast(mask_filled, sitk.sitkUInt8)

    if log != False:
        logger.info(f'Patient outline segmented using threshold {threshold}')
    
    return mask_filled

def remove_small_objects(mask, min_size=1000):
    """Remove connected components smaller than min_size"""
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    output_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    output_mask.CopyInformation(mask)
    for label in stats.GetLabels():
        if stats.GetPhysicalSize(label) >= min_size:
            output_mask = output_mask | (cc == label)
    return output_mask
    
def segment_cbct_ac(cbct, sct, fov_mask = None):
    if fov_mask is None:
        cbct_fov = get_cbct_fov(cbct, 2)
    else:
        cbct_fov = fov_mask
    cbct = img.mask_image(sct, cbct_fov)
    cbct_outline = segment_outline(cbct, 0.5)
    cbct_ac = segment_cbct_ac_helper(sct, cbct_outline)
    cbct_ac = sitk.Cast(cbct_ac, sitk.sitkUInt8)
    return cbct_ac

def generate_sct(cbct, device='cuda:0'):
    sct_generator = sct_gen.StandaloneRegressionInference(
    model_path = '/code/configs/checkpoint',
    device = device,
    )
    cbct_np = sitk.GetArrayFromImage(cbct).astype(np.float32)
    cbct_cor_np = sct_generator.predict(
        input_array = cbct_np,
        apply_normalization=True,
        apply_denormalization=True
    )
    cbct_cor_sitk = sitk.GetImageFromArray(cbct_cor_np)
    cbct_cor_sitk.CopyInformation(cbct)
    cbct_cor_sitk = sitk.Cast(cbct_cor_sitk, sitk.sitkInt16)
    return cbct_cor_sitk

def segment_ct_ac(ct, ct_body):
    ct_lung,_ = segment_lung(ct)
    ct_lung = ct_lung > 0
    threshold_air_ct = -250
    volume_threshold = 20

    ct_body.SetOrigin(ct.GetOrigin())
    ct_lung.SetOrigin(ct.GetOrigin())
    ct_body = sitk.Cast(ct_body, sitk.sitkUInt8)
    ct_body_eroded = sitk.BinaryErode(ct_body, (2, 2, 0))
    ct_lung = sitk.Cast(ct_lung, sitk.sitkUInt8)

    air_cavities_ct = ct < threshold_air_ct 
    air_cavities_ct = air_cavities_ct & ct_body_eroded & ~ct_lung
    air_cavities_ct = remove_small_objects(air_cavities_ct, min_size=volume_threshold)
    
    return air_cavities_ct

def segment_cbct_ac_helper(cbct, outline):
    threshold_air_ct = -250
    volume_threshold = 20

    outline.SetOrigin(cbct.GetOrigin())
    outline = sitk.Cast(outline, sitk.sitkUInt8)
    outline = sitk.BinaryErode(outline, (3, 3, 0))

    air_cavities_ct = cbct < threshold_air_ct
    air_cavities_ct = air_cavities_ct & outline
    air_cavities_ct = remove_small_objects(air_cavities_ct, min_size=volume_threshold)
    
    return air_cavities_ct



