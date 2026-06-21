import os
import copy
import shutil
import yaml
import SimpleITK as sitk
import utils.io as io
import utils.recon as recon
from utils.config import load_patient_configs

if __name__ == "__main__":
    configs = load_patient_configs("configs/lmu_config_sets.yaml")
    for patient_id, config in configs.items():
        print(f"Patient ID: {patient_id}")
        if config.general.vendor.lower() == 'varian':
            cbct_projections = io.read_image(os.path.join(config.data.output, f'projections.mha'))
            cbct_geometry = io.read_geometry(os.path.join(config.data.output, f'geometry.xml'))

            cbct_projections = sitk.Cast(cbct_projections, sitk.sitkFloat32)
            cbct_projections_cor = copy.deepcopy(cbct_projections)

            if cbct_projections_cor.GetSize()[0] * cbct_projections_cor.GetSpacing()[0] > 850:
                padding_val = 0
            else:
                padding_val = 0.1

            cbct_projections_cor = recon.correct_scatter_varian(
                        projections = cbct_projections,
                        geometry = cbct_geometry,
                        scattercorxml_path = config.data.scattercorxml,
                        airscans_path = config.data.airscans,
                        padding = padding_val
                        )

            cbct_projections_cor = recon.correct_i0_varian(
                        projections = cbct_projections_cor, 
                        air_scans_dir = config.data.airscans,
                        geometry = cbct_geometry,
                        scan_xml_path= config.data.scanxml,
                        )
            
            cbct_projections_simulated = io.read_image(os.path.join(config.data.output, f'projections_simulated.mha'))
            cbct_projections_simulated_float = sitk.Cast(cbct_projections_simulated, sitk.sitkFloat32)
            cbct_projections_simulated_log = -1 * sitk.Log(sitk.Divide(cbct_projections_simulated_float, 1))
        
        elif config.general.vendor.lower() == 'elekta':
            cbct_geometry = io.read_geometry(os.path.join(config.data.output, f'geometry.xml'))
            cbct_projections = io.read_image(os.path.join(config.data.output, f'projections.mha'))
            cbct_projections = sitk.Cast(cbct_projections, sitk.sitkFloat32)
            reconstruction_ini = yaml.safe_load(open(os.path.join(config.data.output, f'reconstruction.yaml'), 'r'))
            
            cbct_projections_cor = recon.correct_i0_elekta(
                cbct_projections, 
                reconstruction_ini
                )
            
            cbct_projections_simulated_log = io.read_image(os.path.join(config.data.output, f'projections_simulated.mha'))
            cbct_projections_simulated_log = sitk.Cast(cbct_projections_simulated_log, sitk.sitkFloat32)
            cbct_projections_simulated_log = -1 * sitk.Log(sitk.Divide(cbct_projections_simulated_log, 1))
            
        if not os.path.isdir(os.path.join(config.data.output, 'GC_data')):
            os.makedirs(os.path.join(config.data.output, 'GC_data'))
        
        io.save_image(cbct_projections_cor, os.path.join(config.data.output, 'GC_data', f'projections.mha'), use_compression=True)
        io.save_image(cbct_projections_simulated_log, os.path.join(config.data.output, 'GC_data', f'projections_simulated.mha'), use_compression=True)
        io.write_geometry(cbct_geometry, os.path.join(config.data.output, 'GC_data', f'geometry.xml'))
        shutil.copy(os.path.join(config.data.output, f'metadata.yaml'), os.path.join(config.data.output, 'GC_data', f'metadata.yaml'))
        shutil.copy(os.path.join(config.data.output, f'fov_cbct_nocouch.mha'), os.path.join(config.data.output, 'GC_data', f'fov_cbct.mha'))
        shutil.copy(os.path.join(config.data.output, f'ct_def_masked.mha'), os.path.join(config.data.output, 'GC_data', f'ct_def_masked.mha'))