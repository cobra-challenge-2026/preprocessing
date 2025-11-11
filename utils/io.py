import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def read_image(image_path: str) -> sitk.Image:
    """
    Read an image from the specified image path using SimpleITK. Checks if the path 
    is a directory (DICOM series) or a sitk file (e.g. NIfTI or other formats).
    """
    if os.path.isdir(image_path):
        image = read_dicom_image(image_path)
    elif os.path.isfile(image_path):
        image = sitk.ReadImage(image_path)
    else:
        logger.error(f"Image path not found: {image_path}")
        raise FileNotFoundError(f"Image path not found: {image_path}")
        
    logger.info(f'Image sucessfully read from {image_path}')
    return image

def read_dicom_image(image_path: str) -> sitk.Image:
    """
    Reads a DICOM image series from the specified directory path.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_path)
    if not dicom_names:
        logger.error(f"No DICOM series found in directory: {image_path}")
        raise FileNotFoundError(f"No DICOM series found in directory: {image_path}")
        
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    logger.info(f'DICOM image sucessfuly read from {image_path}')
    return image

def save_image(image:Optional[sitk.Image], image_path:str, compression:bool=True, **kwargs)->None:
    """
    Save the given SimpleITK image to the specified file path.
    
    Args:
        image (sitk.Image): The SimpleITK image to be saved.
        image_path (str): The file path where the image will be saved.
        compression(bool): Whether to use compression when saving the image. Default is True.
        dtype(sitk.PixelIDValueEnum): Default is None. Allowed dtypes: float32 and int16
    """
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        if image.GetPixelIDTypeAsString() != '32-bit float': # type: ignore
            image = sitk.Cast(image,sitk.sitkFloat32)
        image = sitk.Round(image)
        if dtype == 'float32':
            image = sitk.Cast(image,sitk.sitkFloat32)
        elif dtype == 'int16':
            image = sitk.Cast(image,sitk.sitkInt16)
        else:
            raise ValueError('Invalid dtype/not implemented. Allowed dtypes: float32 and int16')
    
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    
    sitk.WriteImage(image, image_path, useCompression=compression) # type: ignore
    logger.info(f'Image saved to {image_path}')

def sitk_to_nib(sitk_image:Optional[sitk.Image]):
    """
    Convert a SimpleITK image to a NIfTI image using nibabel.
    """
    def make_affine(simpleITKImage):
        # get affine transform in LPS
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
            for p in ((1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (0, 0, 0))]
        c = np.array(c)
        affine = np.concatenate([
            np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
            [[0.], [0.], [0.], [1.]]
        ], axis=1)
        affine = np.transpose(affine)
        # convert to RAS to match nibabel
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
        return affine

    affine = make_affine(sitk_image)
    header = nib.Nifti1Header() # type: ignore
    header.set_xyzt_units('mm', 'sec')
    img_nib = nib.Nifti1Image(np.swapaxes(sitk.GetArrayFromImage(sitk_image),2,0), affine, header) # type: ignore

    return img_nib

def nib_to_sitk(nib_image) -> sitk.Image:
    """
    Convert a NIfTI image to a SimpleITK image.

    Args:
        nib_image: The NIfTI image to be converted.

    Returns:
        The converted SimpleITK image.
    """
    img_nib_np = nib_image.get_fdata()
    nib_header = nib_image.header
    img_nib_np = np.swapaxes(img_nib_np, 0, 2)
    img_sitk = sitk.GetImageFromArray(img_nib_np)
    img_sitk.SetSpacing((float(nib_header['pixdim'][1]), float(nib_header['pixdim'][2]), float(nib_header['pixdim'][3])))
    img_sitk.SetOrigin((float(nib_header['srow_x'][3]) * (-1), float(nib_header['srow_y'][3]) * (-1), float(nib_header['srow_z'][3])))
    img_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return img_sitk