import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import logging
import subprocess as sub
import shutil
import tempfile
import configparser as cpars
import xml.etree.ElementTree as ET
from typing import Any, Optional
import itk
from itk import RTK as rtk
import fnmatch
import utils.xim_reader as xim
import xdrt.xdr_reader as xdr_reader

logger = logging.getLogger(__name__)

def read_projections_elekta(projections_path: str, lineint: bool =True) -> sitk.Image:
    """
    Read a projection set from the specified directory
    """
    filenames = sorted(fnmatch.filter(os.listdir(projections_path), "*.his"))
    filenames = [os.path.join(projections_path, f) for f in filenames]
    logger.info(f"Found {len(filenames)} projection files in {projections_path}")

    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    reader = rtk.ProjectionsReader[OutputImageType].New()
    reader.SetFileNames(filenames)
    reader.SetImageIO(itk.HisImageIO.New())
    if lineint:
        reader.SetComputeLineIntegral(True)
    else:
        reader.SetComputeLineIntegral(False)
    reader.Update()
    
    projections = reader.GetOutput()
    projections = itk_to_sitk(projections)
    
    return projections

def read_projections_varian(projections_path: str, lineint: bool = True, header:bool = False) -> sitk.Image:
    filenames = sorted(fnmatch.filter(os.listdir(projections_path), "*.xim"))
    filenames = [os.path.join(projections_path, f) for f in filenames][0:-1] # remove last file as it is often incomplete

    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    reader = rtk.ProjectionsReader[OutputImageType].New()
    reader.SetFileNames(filenames)
    reader.SetImageIO(itk.XimImageIO.New())
    if lineint:
        reader.SetComputeLineIntegral(True)
    else:
        reader.SetComputeLineIntegral(False)
    reader.Update()
    
    projections = reader.GetOutput()
    projections = itk_to_sitk(projections)
    
    # get header info for one projection
    if header:
        _, _, header = xim.read_xim_header(filenames[len(filenames)//2])
        return projections, header
    else:
        return projections

def read_image(image_path: str) -> sitk.Image:
    """
    Read an image from the specified path. Checks if the path is a directory 
    (DICOM series), .SCAN file or sitk file (e.g. .mha, NIfTI or other formats).
    """
    if os.path.isdir(image_path):
        image = read_dicom_image(image_path)
        
    elif os.path.isfile(image_path):
        if os.path.basename(image_path).lower().endswith('.scan'):
            image = xdr_reader.read_as_simpleitk(xdr_reader.read(image_path))
        elif os.path.basename(image_path).lower().endswith('.xim'):
            image,_,_ = xim.read_xim_image(image_path)
        else:
            try:
                image = sitk.ReadImage(image_path)
            except Exception as e:
                logger.error(f"Error reading image file {image_path}: {e}")
                raise
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
    series_ids = reader.GetGDCMSeriesIDs(image_path)

    # find series with most files
    max_files = 0
    longest_series = None
    
    for sid in series_ids:
        filenames = reader.GetGDCMSeriesFileNames(image_path, sid)
        if len(filenames) > max_files:
            max_files = len(filenames)
            longest_series = sid
    
    dicom_names = reader.GetGDCMSeriesFileNames(image_path, longest_series)
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
        elif dtype == 'uint16':
            image = sitk.Cast(image,sitk.sitkUInt16)
        else:
            raise ValueError('Invalid dtype/not implemented. Allowed dtypes: float32, int16 and uint16')
    
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    
    sitk.WriteImage(image, image_path, useCompression=compression) # type: ignore
    logger.info(f'Image saved to {image_path}')

def read_ini_files(reconstruction_dir: str, sections = ['RECONSTRUCTION', 'ALIGNMENT']) -> dict:
    """
    Load and parse all INI files in the specified reconstruction directory (only relevant for Elekta data).
    Args:
        reconstruction_dir (str): Path to the directory containing INI files.
    Returns:
        dict: A dictionary containing parsed INI file data.
    """
    ini_files = fnmatch.filter(os.listdir(reconstruction_dir), '*.INI*')

    if len(ini_files) != 4:
        logger.error(f"Expected 4 INI files, found {len(ini_files)} files.")

    remove_keys = [
        'acquisitiondate', 
        'acquisitiontime', 
        'reconstructiondate', 
        'reconstructiontime',
        'reconstructionoutputfile',
        'alignmentapprovalby',
        'alignmentapprovaldate',
        'alignmentapprovalstatus',
        'alignmentapprovaltime',
        'correctionapprovalby',
        'correctionapprovaldate',
        'correctionapprovalstatus',
        'correctionapprovaltime',
        'datetime',
        'reconstructionprotocolname'
    ]
    reconstruction_dict = {section: {} for section in sections}

    for ini_file in ini_files:
        parser = cpars.ConfigParser()
        parser.read(os.path.join(reconstruction_dir, ini_file))
        
        for section in sections:
            if section not in parser.sections():
                logger.warning(f"Section '{section}' not found in file '{ini_file}'. Skipping.")
                continue
            for key, value in parser.items(section, raw=True):
                if key.lower() in remove_keys:
                    continue
                reconstruction_dict[section][key] = value
    
    return reconstruction_dict

def itk_to_sitk(itk_image):
    array = itk.array_from_image(itk_image)
    
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(list(itk_image.GetSpacing()))
    start_index = itk_image.GetBufferedRegion().GetIndex()
    new_origin = itk_image.TransformIndexToPhysicalPoint(start_index)
    
    sitk_image.SetOrigin(list(new_origin))
    itk_direction = itk_image.GetDirection()
    itk_dims = itk_image.GetImageDimension()
    flat_direction = []
    for i in range(itk_dims):
        for j in range(itk_dims):
            flat_direction.append(itk_direction.GetVnlMatrix().get(i, j))
            
    sitk_image.SetDirection(flat_direction)
    
    return sitk_image

def sitk_to_itk(sitk_image):
    arr = sitk.GetArrayFromImage(sitk_image)
    itk_image = itk.image_view_from_array(arr)

    dimension = sitk_image.GetDimension()

    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())

    sitk_direction = sitk_image.GetDirection()
    direction_np = np.array(sitk_direction, dtype=float).reshape((dimension, dimension))

    itk_image.SetDirection(direction_np)

    return itk_image

def sitk_to_nib(sitk_image:Optional[sitk.Image]):
    def make_affine(simpleITKImage):
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
    img_nib_np = nib_image.get_fdata()
    nib_header = nib_image.header
    img_nib_np = np.swapaxes(img_nib_np, 0, 2)
    img_sitk = sitk.GetImageFromArray(img_nib_np)
    img_sitk.SetSpacing((float(nib_header['pixdim'][1]), float(nib_header['pixdim'][2]), float(nib_header['pixdim'][3])))
    img_sitk.SetOrigin((float(nib_header['srow_x'][3]) * (-1), float(nib_header['srow_y'][3]) * (-1), float(nib_header['srow_z'][3])))
    img_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return img_sitk

def read_geometry(geometry_path:str)->None:
    """
    Read RTK geometry file.
    """
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(geometry_path)
    reader.GenerateOutputInformation()
    return reader.GetGeometry()

def write_geometry(geometry:Any, file_path:str)->None:
    """
    Write RTK geometry to XML file.
    """
    writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    writer.SetFilename(file_path)
    writer.SetObject(geometry)
    writer.WriteFile()

def copy_calibration_dir(source_dir:str, target_dir:str)->None:
    """
    Copy the calibration directory from source to target.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
    logger.info(f"Calibration directory copied from {source_dir} to {target_dir}")
    
