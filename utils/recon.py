import os
import SimpleITK as sitk
import numpy as np
import math
import configparser as cpars
import xml.etree.ElementTree as ET
import itk
from itk import RTK as rtk
import fnmatch
import utils.io as io
import utils.img as img
import utils.xim_reader as xim_reader
import utils.scatter_corrector as sc
from scipy.optimize import curve_fit
from scipy.ndimage import shift 
from tqdm import tqdm

def get_geometry_varian(projections_path: str, xml_path: str) -> sitk.Image:
    """
    Read the geometry from a Varian Scan.xml file.
    """
    filenames = sorted(fnmatch.filter(os.listdir(projections_path), "*.xim"))
    filenames = [os.path.join(projections_path, f) for f in filenames][0:-1] # remove last file as it is often incomplete

    reader = rtk.VarianProBeamGeometryReader.New()
    reader.SetXMLFileName(xml_path)
    reader.SetProjectionsFileNames(filenames)
    reader.UpdateOutputData()

    return reader.GetGeometry()

def get_geometry_elekta(xml_path: str) -> sitk.Image:
    """
    Read the geometry from a Elekta _Frames.xml file.
    """
    reader = rtk.ElektaXVI5GeometryXMLFileReader.New()
    reader.SetFilename(xml_path)
    reader.GenerateOutputInformation()

    return reader.GetGeometry()

def fdk(
        projections: sitk.Image, 
        geometry, 
        gpu: bool = True, 
        size = None, 
        origin = None, 
        spacing = None,
        padding = 0.2,
        hann: float = 1,
        hannY: float = 1,
        ) -> sitk.Image:
    """
    Perform FDK reconstruction using rtk FDKConeBeamReconstructionFilter
    """
    projections = io.sitk_to_itk(projections)
    
    CPUImageType = itk.Image[itk.F, 3]
    if gpu:
        GPUImageType = itk.CudaImage[itk.F, 3]
        ConstantImageSourceType = rtk.ConstantImageSource[GPUImageType]
    else:
        ConstantImageSourceType = rtk.ConstantImageSource[CPUImageType]
    constantImageSource = ConstantImageSourceType.New()
    
    if spacing is None:
        output_spacing = [1,1,1]
    else:
        output_spacing = spacing
    
    sid = geometry.GetSourceToIsocenterDistances()[0]
    sdd = geometry.GetSourceToDetectorDistances()[0]
    mag = sdd / sid
    
    proj_offset_x = geometry.GetProjectionOffsetsX()[0]
   
    proj_size = projections.GetLargestPossibleRegion().GetSize()
    proj_spacing = projections.GetSpacing()
    
    det_width_mm = proj_size[0] * proj_spacing[0]
    det_height_mm = proj_size[1] * proj_spacing[1]
    
    dist_from_center_to_edge = abs(proj_offset_x) + (det_width_mm / 2.0)
    fov_radius_iso = dist_from_center_to_edge / mag
    fov_diameter_iso = fov_radius_iso * 2.0
    fov_height_iso = det_height_mm / mag

    if size is None:
        output_size = itk.Size[3]()
        output_size[0] = math.ceil(fov_diameter_iso / output_spacing[0])
        output_size[2] = math.ceil(fov_diameter_iso / output_spacing[1])
        output_size[1] = math.ceil(fov_height_iso   / output_spacing[2])
    else:
        output_size = size
    
    if origin is None:
        output_origin = itk.Point[itk.F, 3]()
        output_origin[0] = -0.5 * (output_size[0] - 1) * output_spacing[0]
        output_origin[1] = -0.5 * (output_size[1] - 1) * output_spacing[1]
        output_origin[2] = -0.5 * (output_size[2] - 1) * output_spacing[2]
    else:
        output_origin = origin
         
    constantImageSource.SetOrigin(output_origin)
    constantImageSource.SetSpacing(output_spacing)
    constantImageSource.SetSize(output_size)
    constantImageSource.SetConstant(0.0)

    if gpu:
        projections_input = GPUImageType.New()
        projections_input.SetPixelContainer(projections.GetPixelContainer())
        projections_input.CopyInformation(projections)
        projections_input.SetBufferedRegion(projections.GetBufferedRegion())
        projections_input.SetRequestedRegion(projections.GetRequestedRegion())
        FDKGPUType = rtk.CudaFDKConeBeamReconstructionFilter
        feldkamp = FDKGPUType.New()
    else:
        projections_input = projections
        FDKCPUType = rtk.FDKConeBeamReconstructionFilter[CPUImageType]
        feldkamp = FDKCPUType.New()

    if gpu:
        ddf = rtk.CudaDisplacedDetectorImageFilter.New()
        ddf.SetInput(projections_input)
        pssf = rtk.CudaParkerShortScanImageFilter.New()
    else:
        ddf = rtk.DisplacedDetectorForOffsetFieldOfViewImageFilter[
            CPUImageType
        ].New()
        ddf.SetInput(projections_input)
        pssf = rtk.ParkerShortScanImageFilter[CPUImageType].New()

    # Displaced detector weighting
    ddf.SetGeometry(geometry)
    ddf.SetDisable(False)

    # Short scan image filter
    pssf.SetInput(ddf.GetOutput())
    pssf.SetGeometry(geometry)
    pssf.InPlaceOff()
    pssf.SetAngularGapThreshold(20 * 3.14159265359 / 180.0)
    
    feldkamp.SetInput(0, constantImageSource.GetOutput())
    feldkamp.SetInput(1, pssf.GetOutput())
    feldkamp.SetGeometry(geometry)
    feldkamp.GetRampFilter().SetTruncationCorrection(padding)
    feldkamp.GetRampFilter().SetHannCutFrequency(hann)
    feldkamp.GetRampFilter().SetHannCutFrequencyY(hannY)
    feldkamp.Update()

    if gpu:
        recon_gpu = feldkamp.GetOutput()
        recon_cpu = CPUImageType.New()
        recon_cpu.SetPixelContainer(recon_gpu.GetPixelContainer())
        recon_cpu.CopyInformation(recon_gpu)
        recon_cpu.SetBufferedRegion(recon_gpu.GetBufferedRegion())
        recon_cpu.SetRequestedRegion(recon_gpu.GetRequestedRegion())
    else:
        recon_cpu = feldkamp.GetOutput()
        
    recon = io.itk_to_sitk(recon_cpu)
    
    return recon

def read_air_scans(air_scans_path: str, rotation: str = 'CW', return_sitk = True) -> sitk.Image:
    """
    Read a set of 10 air scans and their header from the specified directory using SimpleITK.
    Assumes air scans are stored as a XIM files.
    """
    try:
        filenames = sorted(fnmatch.filter(os.listdir(air_scans_path), f"Filter*_{rotation}_*.xim"))
        if len(filenames) != 10:
            raise ValueError(f"Expected 10 air scan files in {air_scans_path}, found {len(filenames)}")
    except:
        filenames = sorted(fnmatch.filter(os.listdir(air_scans_path), f"Filter*.xim"))
        if len(filenames) < 10:
            raise ValueError(f"Expected 10 air scan files in {air_scans_path}, found {len(filenames)}")
        
    
    imgs = []
    headers = []
    for f in filenames:
        filepath = os.path.join(air_scans_path, f)
        img, _, meta = xim_reader.read_xim_image(filepath)
        if return_sitk:
            imgs.append(img)
        else:
            imgs.append(sitk.GetArrayFromImage(img))
        headers.append(meta)
    
    return imgs, headers

def correct_i0_elekta(projections: sitk.Image, reconstruction_ini: str) -> sitk.Image:
    """
    Perform I0 correction for elekta projections using ini file parameters.
    """
    ma_scan = float(reconstruction_ini['RECONSTRUCTION']['tubema'])
    ms_scan = float(reconstruction_ini['RECONSTRUCTION']['tubekvlength'])
    kvFilter = reconstruction_ini['RECONSTRUCTION']['kvfilter']
    
    if kvFilter == 'F1':
        ma_air = float(reconstruction_ini['RECONSTRUCTION']['floodimagefilterma'])
        ms_air = float(reconstruction_ini['RECONSTRUCTION']['floodimagefilterms'])
        i0_norm = float(reconstruction_ini['RECONSTRUCTION']['floodimagefilternorm'])
    elif kvFilter == 'F0':
        ma_air = float(reconstruction_ini['RECONSTRUCTION']['floodimageopenma'])
        ms_air = float(reconstruction_ini['RECONSTRUCTION']['floodimageopenms'])
        i0_norm = float(reconstruction_ini['RECONSTRUCTION']['floodimageopennorm'])
    else:
        raise ValueError(f"Unknown KVFilter value: {kvFilter}")
    
    i0 = i0_norm * (ma_scan * ms_scan) / (ma_air * ms_air)
    projections_corrected = -1 * sitk.Log(sitk.Divide(projections, i0))
    
    return projections_corrected

def calculate_center_of_mass_u(image_2d):
    """
    Implements Eq (1) from Lindsay et al. 2025 Phiro
    Finds the column index u where the cumulative sum of intensity 
    reaches half the total intensity of the image.
    """
    profile_u = np.sum(image_2d, axis=0)
    total_intensity = np.sum(profile_u)
    half_intensity = total_intensity / 2.0
    cum_sum = np.cumsum(profile_u)
    u_centre = np.argmax(cum_sum >= half_intensity)
    
    return u_centre

def shift_integer(img, shift_val):
    """
    Shifts an image along the horizontal axis (columns) by an integer amount.
    Fills the exposed border with the nearest neighbor (edge value).
    """
    s = int(np.round(shift_val))
    
    if s == 0:
        return img.copy()
    
    h, w = img.shape
    out = np.empty_like(img)
    
    if s > 0: # Shift right
        s = min(s, w)
        out[:, s:] = img[:, :-s]
        out[:, :s] = img[:, 0:1] 
        
    else: # Shift left
        s = abs(s)
        s = min(s, w)
        out[:, :-s] = img[:, s:]
        out[:, -s:] = img[:, -1:]
        
    return out

def correct_i0_varian(
        projections: sitk.Image, 
        projections_header: dict, 
        air_scans_dir: str, 
        geometry, 
        scan_xml_path: str, 
        mode: str = 'nearest', 
        scale_mode: str = 'ma',
        integer_shift: bool = False,
    ) -> sitk.Image:
    """
    Corrects I0 (air scans/flood field) in Varian CBCT projections using a set of 10 air scans 
    acquired before the scan.
    
    Parameters
    ----------
    projections : sitk.Image
        The raw CBCT projections to be corrected.
    projections_header : dict
        Header information for the projections.
    air_scans_dir : str
        Directory containing the air scan images.
    geometry : rtk.ThreeDCircularProjectionGeometry
        The geometry of the CBCT acquisition.
    scan_xml_path : str
        Path to the scan XML file containing acquisition parameters.
    mode : str, optional
        Mode for air scan correction. Options are 'nearest' (default) or 'uniform'.
        'nearest' uses the closest air scan for each projection.
        'uniform' uses the first air scan for all projections.
    scale_mode : str, optional
        Mode for scaling air scans. Options are 'ma' (default) or 'kvnorm
        'ma' scales based on mA and ms settings.
        'kvnorm' scales based on KV normalization chamber readings.
    Returns
    -------
    projections_corrected : sitk.Image
        The I0 corrected CBCT projections.
    """
    #Read projections, air scans and gantry angles
    projections_np = sitk.GetArrayFromImage(projections)
    projection_angles = [gantry_angle * 180.0 / np.pi for gantry_angle in geometry.GetGantryAngles()]
    if projection_angles[20] < projection_angles[30]:
        rotation = 'CW'
    else:
        rotation = 'CC'
    air_imgs, air_header = read_air_scans(air_scans_dir, rotation, return_sitk = False)
    
    if mode == 'nearest':
        air_corr_np = np.zeros_like(projections_np)
        air_scan_angles = [h['GantryRtn'] for h in air_header]
        for i in range(projections_np.shape[0]):
            proj_angle = projection_angles[i]
            #Find index of closest air scan
            closest_idx = np.argmin(np.abs(np.array(air_scan_angles) - proj_angle))
            air_corr_np[i, :, :] = air_imgs[closest_idx][:, :]
    
    elif mode == 'uniform':
        air_corr_np = np.stack([air_imgs[0].astype(np.float32) for i in range(projections_np.shape[0])], axis=0)
    
    elif mode == 'linear':
        air_corr_np = np.zeros_like(projections_np)
        air_scan_angles = [h['GantryRtn'] for h in air_header]
        for i in range(projections_np.shape[0]):
            proj_angle = projection_angles[i]
            p_ang = proj_angle % 360.0
            a_angs = np.array(air_scan_angles) % 360.0
            
            dists = (a_angs - p_ang + 180) % 360 - 180
            
            sorted_indices = np.argsort(np.abs(dists))
            idx1 = sorted_indices[0]
            idx2 = sorted_indices[1]
            
            angle1 = a_angs[idx1]
            angle2 = a_angs[idx2]
            
            diff_total = np.abs((angle2 - angle1 + 180) % 360 - 180)
            
            if diff_total < 1e-6:
                air_corr_np[i, :, :] = air_imgs[idx1]
            else:
                dist1 = np.abs((p_ang - angle1 + 180) % 360 - 180)
                w = dist1 / diff_total
                air_corr_np[i, :, :] = (1 - w) * air_imgs[idx1] + w * air_imgs[idx2]
    
    elif mode == 'sinusoidal':
        air_corr_np = np.zeros_like(projections_np)
        air_scan_angles = [h['GantryRtn'] for h in air_header]
        # Implementation of Lindsay et al. (2025) phiro
        
        # 1. Calculate Center of Mass (u_centre) for all air scans
        u_centres = []
        for img in air_imgs:
            u_c = calculate_center_of_mass_u(img)
            u_centres.append(u_c)
        u_centres = np.array(u_centres)
        
        # 2. Fit sinusoidal curve
        air_rads = np.deg2rad(air_scan_angles)
        p0 = [50, 0, np.mean(u_centres)] 
        
        try:
            def fit_func(x, a, b, c):
                return a * np.sin(x + b) + c
            popt, _ = curve_fit(fit_func, air_rads, u_centres, p0=p0, maxfev=5000)
            print(popt)
        except RuntimeError:
            print("Warning: Sinusoidal fit failed, falling back to 'nearest'.")
            popt = None

        # 3. Reconstruct Air Images
        for i in range(projections_np.shape[0]):
            proj_angle = projection_angles[i]
            proj_rad = np.deg2rad(proj_angle)
            
            # Find nearest air scan (base image)
            diff = np.abs(np.array(air_scan_angles) - proj_angle)
            diff = np.minimum(diff, 360.0 - diff)
            closest_idx = np.argmin(diff)
            base_air_img = air_imgs[closest_idx]
            
            if popt is not None:
                pred_u_centre = fit_func(proj_rad, *popt)
                actual_u_centre = u_centres[closest_idx]
                shift_val = pred_u_centre - actual_u_centre
                
                if integer_shift:
                    shifted_air = shift_integer(base_air_img, shift_val)
                else:
                    shifted_air = shift(base_air_img, shift=[0, shift_val], mode='nearest', order=1)
                
                air_corr_np[i, :, :] = shifted_air
            else:
                air_corr_np[i, :, :] = base_air_img

    else:
        raise ValueError("Invalid mode.")
    
    # convert back to sitk image
    air_corr = sitk.GetImageFromArray(air_corr_np)
    air_corr.CopyInformation(projections)
    
    #Avoid division by zero
    eps = 1e-6
    air_corr = sitk.Maximum(air_corr, eps)
    projections = sitk.Maximum(projections, eps)
    
    #scale air scan to acquisition settings
    if scale_mode == 'ma':
        ma_scan, ms_scan, _  = get_scan_parameters(scan_xml_path)
        ma_air, ms_air = (air_header[0]['KVMilliAmperes'],air_header[0]['KVMilliSeconds'])
        scale_factor = (ma_scan * ms_scan) / (ma_air * ms_air)
    elif scale_mode == 'kvnorm':
        kvnorm_air = air_header[0]['KVNormChamber']
        kvnorm_scan = projections_header['KVNormChamber']
        scale_factor = kvnorm_scan / kvnorm_air
    elif scale_mode == 'constant':
        scale_factor = 24
    air_corr = sitk.Multiply(air_corr, scale_factor)
    
    #raw signals to line integrals
    projections_corrected = -1 * sitk.Log(sitk.Divide(projections, air_corr))
    
    return projections_corrected

def get_scan_parameters(xml_path: str):
    """
    Extract scan parameters (mA, ms, kV) from a Varian Scan.xml file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Remove namespace from tag names
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]  # remove "{namespace}"

    ma = root.find(".//Current").text # type: ignore
    ms = root.find(".//PulseLength").text # type: ignore
    kv = root.find(".//Voltage").text # type: ignore
    
    return float(str(ma)), float(str(ms)), float(str(kv))

def correct_scatter_varian(
        projections: sitk.Image, 
        geometry, 
        scattercorxml_path: str,
        airscans_path: str,
        padding: float = 0.1,
        return_scatter: bool = False,
    ) -> sitk.Image:
    """
    Perform scatter correction on Varian CBCT projections using scatter_corrector module.
    """
    
    spacing = projections.GetSpacing()[0:2]
    
    scattercor = sc.VarianScatterCorrection(
        xml_path=scattercorxml_path,
        pixel_pitch_mm=spacing,
        downsample_factor=8
    )
    
    projections_np = sitk.GetArrayFromImage(projections)
    projections_sc_np = np.zeros_like(projections_np)
    scatter_np = np.zeros_like(projections_np)
    air_corr_np = np.zeros_like(projections_np)
    
    projection_angles = [gantry_angle * 180.0 / np.pi for gantry_angle in geometry.GetGantryAngles()]
    if projection_angles[20] < projection_angles[30]:
        rotation = 'CW'
    else:
        rotation = 'CC'
    air_imgs, air_header = read_air_scans(airscans_path, rotation, return_sitk = False)
    air_scan_angles = [h['GantryRtn'] for h in air_header]

    for i in range(projections_np.shape[0]):
        proj_angle = projection_angles[i]
        closest_idx = np.argmin(np.abs(np.array(air_scan_angles) - proj_angle))
        air_corr_np[i, :, :] = air_imgs[closest_idx][:, :]
    
    for i in tqdm(range(projections_np.shape[0])):
        I_raw = projections_np[i,:,:]
        I_air = air_corr_np[i,:,:]
        I_cor_sc, scatter_estimate = scattercor.correct_projection(
            I_raw,
            I_air,
            iterations=8,
            padding=padding
        )
        projections_sc_np[i,:,:] = I_cor_sc
        scatter_np[i,:,:] = scatter_estimate

    projections_sc_sitk = sitk.GetImageFromArray(projections_sc_np)
    projections_sc_sitk.CopyInformation(projections)

    scatter_sitk = sitk.GetImageFromArray(scatter_np)
    scatter_sitk.CopyInformation(projections)
    
    if return_scatter:
        return projections_sc_sitk, scatter_sitk
    else:       
        return projections_sc_sitk
    
