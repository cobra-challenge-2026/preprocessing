import os
import logging
import SimpleITK as sitk
from typing import TYPE_CHECKING, Optional, Any

import torch

import utils.io as io
import utils.recon as recon
import utils.scatter_corrector as sc
import utils.img as img

if TYPE_CHECKING:
    from utils.config import PatientConfig

class PreProcessor:
    def __init__(self, patient_id: str, config: 'PatientConfig', device: torch.device = torch.device('cuda:0')):
        self.id = patient_id
        self.config = config
        self.logger = logging.getLogger(f'PreProcessor.{self.id}')
        self.device = device

        # Data placeholders
        self.cbct_projections: Optional[sitk.Image] = None
        self.cbct_geometry: Optional[Any] = None 
        self.cbct_clinical: Optional[sitk.Image] = None
        self.cbct_rtk: Optional[sitk.Image] = None
        self.ct: Optional[sitk.Image] = None
        self.metadata: dict = {}
    
    ### define filenames for preprocessing files ###
    def cbct_projections_path(self) -> str:
        return os.path.join(self.config.data.output, f'projections.mha')
    
    def cbct_geometry_path(self) -> str:
        return os.path.join(self.config.data.output, f'geometry.xml')
    
    def cbct_clinical_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_clinical.mha')
    
    def cbct_rtk_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_rtk.mha')
    
    def ct_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct.mha')
    
    def metadata_path(self) -> str:
        return os.path.join(self.config.data.output, f'metadata.yaml')
    
    def overview_path(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}.png')

    ### main preprocessing function ###
    def run_preprocessing(self):
        self.logger.info("Starting preprocessing...")
        # if self.patient_complete():
        #     self.logger.info("All preprocessing files already exist. Skipping patient...")
        # else:
        self.load_data()
        self.recon_cbct()
        self.generate_overview()
        self.write_data()
        self.logger.info("Preprocessing completed.")

    ### individual preprocessing steps ###
    def patient_complete(self) -> bool:
        """Checks if all essential files for the patient exist."""
        files_to_check = [
            self.ct_path(),
            self.cbct_geometry_path(),
            self.cbct_rtk_path(),
            self.cbct_clinical_path(),
        ]
        return all(os.path.isfile(f) for f in files_to_check)
    
    def load_data(self):
        self.logger.info("Loading data...")
        self.ct = io.read_image(self.config.data.ct)
        if self.config.general.vendor.lower() == 'elekta':
            self.logger.info("Reading Elekta projections...")
            self.cbct_projections = io.read_projections_elekta(
                self.config.data.projections, 
                lineint = False
                )
            self.logger.info("Loading geometry...")
            self.cbct_geometry = recon.get_geometry_elekta(
                self.config.data.framesxml
                )
            self.logger.info("Correcting I0...")
            self.cbct_projections = recon.correct_i0_elekta(
                self.cbct_projections, 
                ini_file=self.config.data.inifile
                )
            self.logger.info("Load clinical reconstruction...")
            self.cbct_clinical = io.read_image(self.config.data.clinical_recon)
            if self.config.settings.correct_orientation and self.config.data.correctionini is not None:
                self.logger.info("Correcting clinical reconstruction orientation using ini file...")
                self.cbct_clinical = img.allign_images_ini(self.cbct_clinical, self.config.data.correctionini)
            
            
        elif self.config.general.vendor.lower() == 'varian':
            self.logger.info("Reading Varian projections...")
            self.cbct_projections, header = io.read_projections_varian(
                self.config.data.projections, 
                lineint = False, 
                header = True
                )
            self.logger.info("Loading geometry...")
            self.cbct_geometry = recon.get_geometry_varian(
                self.config.data.projections, 
                self.config.data.scanxml
                )
            self.logger.info("Correcting scatter...")
            self.cbct_projections = recon.correct_scatter_varian(
                projections = self.cbct_projections,
                geometry = self.cbct_geometry,
                scattercorxml_path = self.config.data.scattercorxml,
                airscans_path = self.config.data.airscans
                )
            self.logger.info("Correcting I0...")
            self.cbct_projections = recon.correct_i0_varian(
                projections = self.cbct_projections, 
                projections_header = header,
                air_scans_dir = self.config.data.airscans,
                geometry = self.cbct_geometry,
                scan_xml_path= self.config.data.scanxml,
                )
        else:
            self.logger.error(f"Unsupported vendor: {self.config.general.vendor}")
            raise ValueError(f"Unsupported vendor: {self.config.general.vendor}")
        
        
        self.logger.info("Images loaded.")
        
    def recon_cbct(self):
        self.logger.info("Reconstructing CBCT from projections...")
        if self.config.general.vendor.lower() == 'elekta':
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections, 
                geometry = self.cbct_geometry,
                gpu = True,
                spacing = [1.0, 1.0, 1.0],
                size = [410, 264, 410],
                padding = 0.2,
                hann = 1,
                hannY = 1,
            )
            if self.config.settings.correct_orientation and self.config.data.correctionini is not None:
                self.cbct_rtk = img.fix_array_order(self.cbct_rtk)
                self.cbct_rtk = img.allign_images_ini(self.cbct_rtk, self.config.data.correctionini)
        elif self.config.general.vendor.lower() == 'varian':
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections, 
                geometry = self.cbct_geometry,
                gpu = True,
                padding = 0.2,
                hann = 0.5,
                hannY = 0.5,
            )
        self.logger.info("CBCT reconstruction completed.")
    
    def generate_overview(self):
        self.logger.info("Generating overview visualization...")
        self.logger.info("Overview visualization generated.")

    def write_data(self):
        self.logger.info("Saving preprocessed data to files...")
        if not os.path.exists(self.config.data.output):
            os.makedirs(self.config.data.output)
        sitk.WriteImage(self.cbct_projections, self.cbct_projections_path())
        io.write_geometry(self.cbct_geometry, self.cbct_geometry_path())
        sitk.WriteImage(self.cbct_rtk, self.cbct_rtk_path())
        sitk.WriteImage(self.ct, self.ct_path())
        sitk.WriteImage(self.cbct_clinical, self.cbct_clinical_path())
        self.logger.info("Preprocessed data saved.")
    
    # def run_deformation(self):
    #     self.logger.info("Running deformable registration...")
    #     if self.config.region == 'AB':
    #         params = reg.CT_MR_params_B_AB()
    #     if self.config.region == 'TH':
    #         params = reg.CT_MR_params_B_TH()
    #     self.ct_deformed, self.ct_disp_field = reg.run_deformable(
    #         fixed = self.input_image, 
    #         moving = self.ct_image, 
    #         mask_fixed= self.sr_mask,
    #         mask_moving= self.sr_mask,
    #         use_mask= params['use_mask'],
    #         background_value = params['background_value'],
    #         mind_r_c = params['mind_r_c'],
    #         mind_d_c = params['mind_d_c'],
    #         mind_r_a = params['mind_r_a'],
    #         mind_d_a = params['mind_d_a'],
    #         disp_hw = params['disp_hw'],
    #         grid_sp = params['grid_sp'],
    #         grid_sp_adam = params['grid_sp_adam'],
    #         selected_smooth = params['selected_smooth'],
    #         selected_niter = params['selected_niter'],
    #         lambda_weight = params['lambda_weight'], 
    #         sigma = params['sigma'],
    #         device= self.device
    #     )
        
    #     self.ct_deformed = img.mask_image(self.ct_deformed, self.sr_fov, mask_value=-1024)
    #     io.save_image(self.ct_deformed, self.ct_def_path())
    #     io.save_image(self.ct_disp_field, self.ct_dvf_path())
    #     vis.generate_overview_dir(self.ct_image, self.input_image, self.ct_deformed, self.config.output, self.id)
    #     self.logger.info("Deformable registration completed.")
        
    # def run_segmentation(self):
    #     self.logger.info("Running segmentation...")
        
    #     self.ct_skin = seg.segment_skin(self.ct_deformed, modality='CT')
    #     io.save_image(self.ct_skin, self.ct_body_path())
        
    #     self.input_skin = seg.segment_skin(self.input_image, modality=self.input_type_str)
    #     io.save_image(self.input_skin, self.input_body_path())
        
    #     self.ts_structures = seg.segment_OAR_structures(self.ct_deformed, modality='CT')
    #     seg.split_multilabel_segmentation(self.ts_structures[0], self.ts_structures[1], save_to_files=True, output=self.config.output)

    #     if self.ct_disp_field is not None and self.config.modality == 'CBCT':
    #         self.planning_structures = seg.warp_planning_structures(self.planning_structures, self.ct_disp_field, save_to_files=True, output=self.config.output)
            
    #     self.logger.info("Segmentation completed.")

        
        
        
        

