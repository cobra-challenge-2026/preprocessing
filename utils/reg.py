import SimpleITK as sitk
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Union
from scipy.ndimage import map_coordinates
from utils.convex_adam_MIND_gaussian import convex_adam_pt
from convexAdam.convex_adam_utils import validate_image

import utils.io as io
import utils.img as img

logger = logging.getLogger(__name__)

def run_deformable(
    fixed: sitk.Image | None = None,
    moving: sitk.Image | None = None,
    mind_r_c:int = 2,
    mind_d_c:int = 2,
    mind_r_a:int = 2,
    mind_d_a:int = 2,
    disp_hw:int = 6,
    grid_sp:int = 5,
    grid_sp_adam:int = 2,
    selected_smooth:int = 0,
    selected_niter:int = 80,
    lambda_weight:float = 1.6,
    sigma:float = 1.0,
    verbose:bool = True,
    device: torch.device = torch.device('cuda:0'),
    background_value:int = -1024,
    use_mask:bool = False,
    mask_fixed: sitk.Image | None = None,
    mask_moving: sitk.Image | None = None,
)->Tuple[sitk.Image, sitk.Image]:
    """
    Perform deformable image registration using the Convex Adam algorithm.

    Parameters:
    fixed (sitk.Image): The fixed image.
    moving (sitk.Image): The moving image.
    mask (sitk.Image): The mask image.
    mind_r (int): MIND radius parameter.
    mind_d (int): MIND distance parameter.
    disp_hw (int): Displacement half-width.
    grid_sp (int): Grid spacing.
    grid_sp_adam (int): Grid spacing for Adam optimizer.
    selected_smooth (int): Smoothing parameter.
    selected_niter (int): Number of iterations.
    lambda_weight (float): Weighting parameter.
    verbose (bool): Verbosity flag.
    device (str): Device to use for computation.
    
    Details for parameters can be found in the Convex Adam code and in their paper:
    https://pubmed.ncbi.nlm.nih.gov/39283782/
    
    Returns:
    Tuple[sitk.Image, sitk.Image]: The deformed image and the displacement field.
    """
    
    logger.info('Starting deformable registration using Convex Adam...')
    logger.info(f'Parameters: mind_r_c={mind_r_c}, mind_d_c={mind_d_c}, disp_hw={disp_hw}, grid_sp={grid_sp}, grid_sp_adam={grid_sp_adam}, selected_smooth={selected_smooth}, selected_niter={selected_niter}, lambda_weight={lambda_weight}, device={device}')

    if fixed is None or moving is None:
        raise ValueError("Fixed and moving images must be provided for deformable registration.")
    
    if use_mask:
        if mask_fixed is None or mask_moving is None:
            raise ValueError("Masks must be provided when use_mask is True.")
        else:
            disp_field = convex_adam_pt(
                img_fixed=fixed,
                img_moving=moving,
                mind_r_c=mind_r_c,
                mind_d_c=mind_d_c,
                mind_r_a=mind_r_a,
                mind_d_a=mind_d_a,
                disp_hw=disp_hw,
                grid_sp=grid_sp,
                grid_sp_adam=grid_sp_adam,
                selected_smooth=selected_smooth,
                selected_niter=selected_niter,
                lambda_weight=lambda_weight,
                verbose=verbose,
                device=device,
                sigma=sigma,
                use_mask=True,
                mask_fixed=mask_fixed,
                mask_moving=mask_moving,
            )
    else:
        disp_field = convex_adam_pt(
                    img_fixed=fixed,
                    img_moving=moving,
                    mind_r_c=mind_r_c,
                    mind_d_c=mind_d_c,
                    mind_r_a=mind_r_a,
                    mind_d_a=mind_d_a,
                    disp_hw=disp_hw,
                    grid_sp=grid_sp,
                    grid_sp_adam=grid_sp_adam,
                    selected_smooth=selected_smooth,
                    selected_niter=selected_niter,
                    lambda_weight=lambda_weight,
                    verbose=verbose,
                    device=device,
                    sigma=sigma,
                )
    
    logger.info('Deformable registration completed. Applying displacement field to moving image...')
        
    deformed_np = apply_convex(
        disp=disp_field,
        moving=moving,
        background_value=background_value
    )
    deformed_sitk = sitk.GetImageFromArray(deformed_np.astype(np.float32))
    deformed_sitk.CopyInformation(moving)
    
    disp_field_sitk = sitk.GetImageFromArray(disp_field.astype(np.float64), isVector=True)
    disp_field_sitk.CopyInformation(moving)

    return deformed_sitk, disp_field_sitk

def warp_structure(structure:Optional[sitk.Image], disp_field:sitk.Image, interpolator = sitk.sitkNearestNeighbor)->sitk.Image:
    """
    Warp a structure image using the provided displacement field.
    Parameters:
    - structure: The structure image to be warped.
    - disp_field: The displacement field to be applied.
    Returns:
    warped_structure (sitk.Image): The warped structure image.
    """
    logger.info('Warping structure using the provided displacement field...')
    warper = sitk.WarpImageFilter()
    warper.SetInterpolator(interpolator)
    warper.SetOutputParameteresFromImage(structure)
    warped_structure = warper.Execute(structure, disp_field)
    logger.info('Warping finished.')
    return warped_structure

def apply_convex(
    disp: Union[torch.Tensor, np.ndarray, sitk.Image],
    moving: Union[torch.Tensor, np.ndarray, sitk.Image],
    background_value: float = 0.0
) -> np.ndarray:
    """Add background value argument when applying convex adam displacement field"""
    # convert to numpy, if not already
    moving = validate_image(moving).numpy()
    disp = validate_image(disp).numpy()

    d1, d2, d3, _ = disp.shape
    identity = np.meshgrid(np.arange(d1), np.arange(d2), np.arange(d3), indexing='ij')
    warped_image = map_coordinates(moving, disp.transpose(3, 0, 1, 2) + identity, order=1, cval=background_value)
    return warped_image

def invert_displacement_field(disp_field: sitk.Image) -> sitk.Image:
    """
    Invert a displacement field.

    Parameters:
    - disp_field (sitk.Image): The displacement field to be inverted.

    Returns:
    - sitk.Image: The inverted displacement field.
    """
    logger.info('Inverting displacement field...')
    inv_filter = sitk.InvertDisplacementFieldImageFilter()
    inv_filter.SetMaximumNumberOfIterations(100)
    inv_filter.SetMaxErrorToleranceThreshold(1e-4)
    inv_filter.SetMeanErrorToleranceThreshold(1e-6)
    inv_disp_field = inv_filter.Execute(disp_field)
    logger.info('Inversion completed.')
    return inv_disp_field

def CT_MR_params_B_AB():
    """Returns parameters for CT to MR registration for dataset 1THA."""
    params = {
        'use_mask': True,
        'background_value': -1024,
        'mind_r_c': 2,
        'mind_d_c': 2,
        'mind_r_a': 1,
        'mind_d_a': 2,
        'disp_hw': 6,
        'grid_sp': 5,
        'grid_sp_adam': 2,
        'selected_smooth': 0,
        'selected_niter': 200,
        'lambda_weight': 2.5,
        'sigma': 0.2
    }
    return params

def CT_MR_params_B_TH():
    """Returns parameters for CT to MR registration for dataset 1THA."""
    params = {
        'use_mask': True,
        'background_value': -1024,
        'mind_r_c': 2,
        'mind_d_c': 2,
        'mind_r_a': 1,
        'mind_d_a': 2,
        'disp_hw': 6,
        'grid_sp': 5,
        'grid_sp_adam': 2,
        'selected_smooth': 0,
        'selected_niter': 200,
        'lambda_weight': 2.5,
        'sigma': 0.2
    }
    return params

