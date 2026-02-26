from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import logging
import os
import utils.io as io
import xmltodict
from typing import Optional, Union
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from scipy import ndimage
import numpy as np


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

def segment_outline_improved(input: sitk.Image, 
                             threshold: float = 0.30, 
                             log=False) -> sitk.Image:
    """
    Segment the outline of a given input CBCT image with pre-smoothing and robust morphology.

    Parameters:
    input (sitk.Image): The input image to segment.
    threshold (float): A relative threshold value for segmentation. Default is 0.30.

    Returns:
    sitk.Image: The segmented outline image.
    """
    input_float = sitk.Cast(input, sitk.sitkFloat32)
    
    # --- 1. Pre-processing: Denoise the image FIRST ---
    # This is the most important step for noisy CBCTs.
    # CurvatureAnisotropicDiffusion is excellent at smoothing while preserving edges.
    smoothed_image = sitk.CurvatureAnisotropicDiffusion(
        input_float,
        timeStep=0.0625,
        conductanceParameter=9.0,
        numberOfIterations=5 # You can tune this parameter
    )
    
    # 1. Get the intensity range of interest, just like you already do
    input_np = sitk.GetArrayFromImage(smoothed_image)
    background = np.percentile(input_np, 2.5)
    high = np.percentile(input_np, 97.5)

    # 2. Create a "region of interest" (ROI) mask in SimpleITK
    # This mask includes only the "interesting" pixels, excluding 
    # extreme air and extreme artifacts.
    roi_mask = sitk.And(smoothed_image > background, smoothed_image < high)

    # 3. Calculate Otsu threshold *only* within the ROI mask
    # We tell Otsu to only look at pixels where roi_mask is 1
    dynamic_threshold_value = sitk.OtsuThreshold(
        smoothed_image, 
        roi_mask,
        1
    )

    # 4. Apply the dynamic threshold
    mask = smoothed_image > dynamic_threshold_value

    # --- 3. Morphological Cleaning ---
    # Use the kernel size from your original code, but apply Open -> Close
    # For a more robust solution, use physical units (see suggestion #3)
    kernel_radius_pixels = (8, 8, 8) 
    
    # 3a. Opening: Remove small noise speckles
    mask = sitk.BinaryMorphologicalOpening(mask, kernel_radius_pixels)
    
    # 3b. Closing: Fill small gaps in the main body
    mask = sitk.BinaryMorphologicalClosing(mask, kernel_radius_pixels)

    # --- 4. Keep Largest Component ---
    # This correctly removes disconnected noise/artifacts (e.g., treatment couch)
    mask_cc = sitk.ConnectedComponent(mask)
    sorted_component_image = sitk.RelabelComponent(mask_cc, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    
    # The 3D sitk.BinaryFillhole was redundant with the 2D loop, so it's removed.
    
    # --- 5. 2D Axial Hole Filling (Your robust final step) ---
    mask_np = sitk.GetArrayFromImage(largest_component_binary_image)
    mask_np_filled = np.zeros_like(mask_np)

    for i in range(mask_np.shape[0]):
        # This fills internal holes (e.g., lungs, GI gas) slice by slice
        mask_np_filled[i] = ndimage.binary_fill_holes(mask_np[i])

    mask_filled = sitk.GetImageFromArray(mask_np_filled)
    mask_filled.CopyInformation(input) # Copy metadata from the *original* input
    mask_filled = sitk.Cast(mask_filled, sitk.sitkUInt8)

    if log: # Use 'if log:' which is more Pythonic
        # Use a real logger if available
        # logger.info(f'Patient outline segmented using threshold {threshold}')
        print(f'Patient outline segmented using threshold {threshold}')
    
    return mask_filled