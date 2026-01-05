import numpy as np
import xml.etree.ElementTree as ET
from scipy.fft import fft2, ifft2
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt

class VarianScatterCorrection:
    def __init__(self, xml_path, pixel_pitch_mm=0.388, downsample_factor=4):
        """
        Args:
            xml_path: Path to scatter correction Calibration.xml
            pixel_pitch_mm: pixel size in mm. Can be scalar (square) or 
                            tuple (row_pitch, col_pitch) for rectangular pixels.
            downsample_factor: Factor to reduce image size for scatter calculation
        """
        if np.isscalar(pixel_pitch_mm):
            self.pixel_pitch_mm = (pixel_pitch_mm, pixel_pitch_mm)
        else:
            self.pixel_pitch_mm = tuple(pixel_pitch_mm)
            
        self.ds_factor = downsample_factor
        
        self.ds_pixel_mm = (self.pixel_pitch_mm[0] * downsample_factor, 
                            self.pixel_pitch_mm[1] * downsample_factor)
        
        # Parse XML
        self.params_path = xml_path
        self.ns = {'ns': 'http://baden.varian.com/cr.xsd'}
        self.params = self.parse_xml(xml_path)
        
        self.kernels = {} 

    def parse_xml(self, xml_path):
        """
        Parses the Varian Calibration.xml file.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = self.ns 

        params = {}
        
        # Global Parameters
        globals_tag = root.find('.//ns:Globals', ns)
        params['mu_water'] = float(globals_tag.find('ns:muH2O', ns).text) # ~0.02 mm^-1
        
        # Detector Scatter Parameters (values seem to be in cm)
        det_model = globals_tag.find('ns:DetectorScatterModel', ns)
        params['det_scatter'] = {
            'a1': float(det_model.find('ns:PScFit0', ns).text),
            'a2': float(det_model.find('ns:PScFit1', ns).text), 
            'a3': float(det_model.find('ns:PScFit2', ns).text),
            'a4': float(det_model.find('ns:PScFit3', ns).text), 
            'a5': float(det_model.find('ns:PScFit4', ns).text), 
            'cover_spr': float(det_model.find('ns:CoverSPR', ns).text)
        }

        # Object Scatter Models
        params['kernels'] = []
        obj_models = root.find('.//ns:ObjectScatterModels', ns)
        
        for model in obj_models.findall('ns:ObjectScatterModel', ns):
            fit = model.find('ns:ObjectScatterFit', ns)
            k_params = {
                'thickness_bound': float(model.find('ns:Thickness', ns).text), # mm
                'A': float(fit.find('ns:A', ns).text),
                'alpha': float(fit.find('ns:alpha', ns).text),
                'beta': float(fit.find('ns:beta', ns).text),
                'B': float(fit.find('ns:B', ns).text),
                'sigma1': float(fit.find('ns:sigma1', ns).text), # cm
                'sigma2': float(fit.find('ns:sigma2', ns).text), # cm
                'gamma': float(fit.find('ns:gamma', ns).text),
                'm': float(fit.find('ns:MagFactor', ns).text),
            }
            # Anti-scatter grid parameters
            grid_eff_node = model.find('.//ns:GridEfficiency/ns:LamellaTransmission', ns)
            k_params['grid_tail'] = float(grid_eff_node.text) if grid_eff_node is not None else 0.3
            params['kernels'].append(k_params)
        
        params['kernels'].sort(key=lambda x: x['thickness_bound'])
        
        return params

    def calc_tau_cm(self, I_p, I_0):
        """
        calculates thickness map in CM.
        """
        ratio = np.divide(I_0, I_p, out=np.ones_like(I_p), where=I_p > 1e-5)
        ratio = np.maximum(ratio, 1.0)
        
        # Equation 9
        tau_mm = (1.0 / self.params['mu_water']) * np.log(ratio)
        return tau_mm / 10.0

    def generate_gaussian_kernel_cm(self, shape, sigma1_cm, sigma2_cm, B):
        """
        Generates the form function g(x,y) in CM taking rectangular pixels into account.
        """
        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[-cy:shape[0]-cy, -cx:shape[1]-cx]
        
        py_mm, px_mm = self.ds_pixel_mm
        py_cm, px_cm = py_mm / 10.0, px_mm / 10.0
        
        r2 = (x * px_cm)**2 + (y * py_cm)**2
        
        s1 = abs(sigma1_cm)
        s2 = abs(sigma2_cm)
        
        # Equation 5
        g = np.exp(-r2 / (2 * s1**2)) + B * np.exp(-r2 / (2 * s2**2))
        return g
    
    def compute_edge_scaling(self, tau_map, pixel_pitch_cm_tuple):
        """
        Computes the edge-modified scaling factor s(x,y) from Section 2.5.
        """
        # Assumption (not really defined in paper): where tau_map > 0.1 cm defines the object region
        object_mask = tau_map > 0.1
        
        dist_cm = distance_transform_edt(object_mask, sampling=pixel_pitch_cm_tuple)
        
        # Linear Ramp from 0.75 to 1.0 over 8 cm (according to paper)
        ramp_distance_cm = 8.0
        s_map = 0.75 + (0.25/ramp_distance_cm * dist_cm)
        s_map = np.clip(s_map, 0.75, 1.0)
        
        return s_map
    
    def generate_grid_response(self, shape, tail_value=0.25):
        """
        Generates the 1D Anti-Scatter Grid weighting function k(y) expanded to 2D.
        Based on paper Section 2.6 and MATLAB code in TIGREVarian.
        Does not seem to make a big difference in results.
        """
        rows, cols = shape
        cy = rows // 2
        
        y = np.arange(rows) - cy
        
        py_mm, _ = self.ds_pixel_mm
        y_cm = y * (py_mm / 10.0)
        
        # Values from TIGREVarian implementation, not sure where they come from
        # https://www.sciencedirect.com/science/article/pii/S1120179722020373
        slope = -0.15
        intercept = 1.0
        
        k_y = slope * np.abs(y_cm) + intercept
        k_y = np.maximum(k_y, tail_value)
        
        k_2d = np.tile(k_y[:, np.newaxis], (1, cols))
        mean_val = np.mean(k_2d)
        if mean_val > 0:
             k_2d = k_2d / mean_val
        
        return k_2d
    
    def generate_detector_kernel(self, shape):
        """
        Generates detector point-scatter function (Eq 22) with rectangular pixels.
        """
        p = self.params['det_scatter']
        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.ogrid[-cy:shape[0]-cy, -cx:shape[1]-cx]
        
        py_mm, px_mm = self.ds_pixel_mm
        py_cm, px_cm = py_mm / 10.0, px_mm / 10.0
        
        r_cm = np.sqrt((x * px_cm)**2 + (y * py_cm)**2)
        
        val1 = -p['a2'] * r_cm
        val2 = -p['a4'] * ((r_cm - p['a5'])**3)
        
        # Equation 22
        h_d = p['a1'] * np.exp(val1) + p['a3'] * np.exp(val2)
        
        # Paper mentions normalization constant a0, calibration XML provides 
        # a 'cover_spr' value which seems to be the total amount of scatter in 
        # the detector 2d integral over h_d should equal cover_spr
        current_sum = np.sum(h_d)
        a0 = (p['cover_spr'] / current_sum)
        if current_sum > 0:
            h_d = h_d * a0
            
        return h_d
    

    def correct_projection(self, projection, open_field, iterations=3, padding=0.1):
        """
        Performs fASKS scatter correction with reflection padding (can be turned off).
        
        Args:
            projection: The raw projection image.
            open_field: The open field (air) image.
            iterations: Number of scatter iterations.
            padding: Fraction of width to pad for edge handling. 
                     Set to 0.0 to disable padding.
        """
        # Downsample Projections (2.8.1)
        I_raw_ds = zoom(projection, 1/self.ds_factor, order=1)
        I_0_ds = zoom(open_field, 1/self.ds_factor, order=1)
        
        # Padd the projections to avoid artifacts at the edges (not mentioned in paper...)
        pad_w = 0
        if padding > 0:
            pad_w = int(I_raw_ds.shape[1] * padding)
        
        # Apply padding if width > 0
        if pad_w > 0:
            I_raw_pad = np.pad(I_raw_ds, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
            I_0_pad = np.pad(I_0_ds, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
        else:
            I_raw_pad = I_raw_ds
            I_0_pad = I_0_ds
        
        rows_pad, cols_pad = I_raw_pad.shape
        
        py_mm, px_mm = self.ds_pixel_mm
        py_cm, px_cm = py_mm / 10.0, px_mm / 10.0
        
        integration_area_factor = py_cm * px_cm
        
        # Detector scatter estimate (2.8.2)
        h_d = self.generate_detector_kernel((rows_pad, cols_pad))
        H_d = fft2(h_d)
        I_ds_pad = np.real(ifft2(fft2(I_raw_pad) * H_d))
        I_ds_pad = np.fft.fftshift(I_ds_pad)
        
        # Initial scatter estimates (2.8.3)
        I_s_pad = np.zeros_like(I_raw_pad)
        I_p_pad = I_raw_pad - I_ds_pad
        I_p_pad = np.maximum(I_p_pad, 1.0)
        
        # Prepare kernels for thickness groups including ASG response 
        # currently only the 1D ASG is supported. Will need modification for 2D grids or maybe just not necessary?
        # after some tests the effect of grid seems negligible on the reconstructed images
        fourier_kernels = []
        grid_response = self.generate_grid_response((rows_pad, cols_pad), tail_value=0.25)
        for k_params in self.params['kernels']:
            g = self.generate_gaussian_kernel_cm((rows_pad, cols_pad), 
                                                 k_params['sigma1'],
                                                 k_params['sigma2'], 
                                                 k_params['B'])
            g_weighted = g * grid_response
            fourier_kernels.append(fft2(g_weighted))

        # Scatter correction iteration loop 
        for it in range(iterations):
            # Thickness map 
            tau_cm_pad = self.calc_tau_cm(I_p_pad, I_0_pad)
            
            # Smooth thickness map
            sigma_pixels = (1.5 / py_cm, 1.5 / px_cm)
            tau_smooth_pad = gaussian_filter(tau_cm_pad, sigma=sigma_pixels)
            
            # Edge scaling map
            s_map = self.compute_edge_scaling(tau_smooth_pad, (py_cm, px_cm))
            # New scatter estimate (2.8.4a)
            I_s_new_pad = np.zeros_like(I_s_pad)
            
            # Convolution for each thickness group (2.8.4b)
            for i, k_params in enumerate(self.params['kernels']):
                # Thickness masking
                tau_mm_check = tau_smooth_pad * 10.0
                t_lower = k_params['thickness_bound']
                
                if i < len(self.params['kernels']) - 1:
                    t_upper = self.params['kernels'][i+1]['thickness_bound']
                    mask = (tau_mm_check >= t_lower) & (tau_mm_check < t_upper)
                else:
                    mask = (tau_mm_check >= t_lower)
                
                if not np.any(mask):
                    continue
                
                R = mask.astype(float)
                
                # Ratio calculation
                ratio_p_0 = np.divide(I_p_pad, I_0_pad, out=np.zeros_like(I_p_pad), where=I_0_pad > 1e-5)
                ratio_0_p = np.divide(I_0_pad, I_p_pad, out=np.zeros_like(I_p_pad), where=I_p_pad > 1e-5)
                
                ratio_p_0 = np.clip(ratio_p_0, 1e-9, 1.0)
                ln_ratio = np.log(np.maximum(ratio_0_p, 1.0))
                
                c = k_params['A'] * (ratio_p_0 ** k_params['alpha']) * (ln_ratio ** k_params['beta'])
                c = c * s_map
                
                # fASKS terms
                term1 = I_p_pad * R * c
                term2 = tau_smooth_pad * I_p_pad * R * c
                
                gamma = k_params['gamma']
                G_fft = fourier_kernels[i]
                
                conv1 = np.real(np.fft.fftshift(ifft2(fft2(term1) * G_fft)))
                conv2 = np.real(np.fft.fftshift(ifft2(fft2(term2) * G_fft)))
                
                part_scatter = ((1 - gamma * tau_smooth_pad) * conv1 + gamma * conv2) * integration_area_factor
                I_s_new_pad += part_scatter
            
            #2.8.4.c
            # Relaxation (new scatter is only partially applied)
            I_s_pad = I_s_pad + 0.6 * (I_s_new_pad - I_s_pad)
            
            # Clipping to ensure a max scatter fraction of 0.95
            sf = np.divide(I_s_pad, I_raw_pad, out=np.zeros_like(I_s_pad), where=I_raw_pad>1.0)
            if np.any(sf > 0.95):
                I_s_pad[sf > 0.95] = I_raw_pad[sf > 0.95] * 0.95
            
            # Update primary estimate
            I_p_pad = I_raw_pad - I_ds_pad - I_s_pad
            # Prevent negative or zero primary
            I_p_pad = np.maximum(I_p_pad, 1.0)

        # Remove Padding
        if pad_w > 0:
            I_s_ds = I_s_pad[pad_w:-pad_w, pad_w:-pad_w]
            I_ds_ds = I_ds_pad[pad_w:-pad_w, pad_w:-pad_w]
        else:
            I_s_ds = I_s_pad
            I_ds_ds = I_ds_pad
        
        # Upsample scatter estimates to original size (2.8.5)
        total_low_res = I_s_ds + I_ds_ds
        zoom_y = projection.shape[0] / total_low_res.shape[0]
        zoom_x = projection.shape[1] / total_low_res.shape[1]
        total_scatter_upsampled = zoom(total_low_res, (zoom_y, zoom_x), order=1)
        
        # Final correction
        max_scatter_allowed = projection * 0.95
        final_total_scatter = np.minimum(total_scatter_upsampled, max_scatter_allowed)
        corrected_projection = projection - final_total_scatter
        corrected_projection = np.maximum(corrected_projection, 0.01)
        
        return corrected_projection, final_total_scatter