import os
import logging
import SimpleITK as sitk
from typing import TYPE_CHECKING, Optional, Any

import torch

import utils.io as io
import utils.img as img
import utils.reg as reg
import utils.seg as seg
import utils.vis as vis

if TYPE_CHECKING:
    from utils.config import PatientConfig

class PreProcessor:
    def __init__(self, patient_id: str, config: 'PatientConfig', device: torch.device = torch.device('cuda:0')):
        self.id = patient_id
        self.config = config
        self.logger = logging.getLogger(f'PreProcessor.{self.id}')
        self.device = device

        # Original data from SynthRAD training dataset
        self.ct_image: Optional[sitk.Image] = None
        self.input_image: Optional[sitk.Image] = None
        self.sr_mask: Optional[sitk.Image] = None
        self.sr_fov: Optional[sitk.Image] = None
        self.planning_structures: dict[str, sitk.Image] = {}
        
        # Preprocessed data
        self.ct_deformed: Optional[sitk.Image] = None
        self.ct_disp_field: Optional[sitk.Image] = None
        self.ct_skin: Optional[sitk.Image] = None
        self.input_skin: Optional[sitk.Image] = None
        self.ts_structures: tuple[sitk.Image, dict] = (sitk.Image(), {})

    ### define filenames for preprocessing files ###
    @property
    def input_type_str(self) -> str:
        """Returns 'MR' or 'CBCT' based on the task."""
        return "MR" if self.config.modality == 'MR' else "CBCT"

    def ct_path(self) -> str:
        return os.path.join(self.config.output_dir, f'ct.mha')
    
    def input_path(self) -> str:
        return os.path.join(self.config.output_dir, f'{self.input_type_str.lower()}.mha')
    
    def sr_mask_path(self) -> str:
        return os.path.join(self.config.output_dir, f'sr_mask.mha')
    
    def sr_fov_path(self) -> str:
        return os.path.join(self.config.output_dir, f'sr_fov.mha')

    def planning_structures_path(self, structure_name: str) -> str:
        return os.path.join(self.config.output_dir, 'planning_structures',f'{structure_name}')
    
    def ct_def_path(self) -> str:
        return os.path.join(self.config.output_dir, f'ct_def.mha')

    def ct_dvf_path(self) -> str:
        return os.path.join(self.config.output_dir, f'ct_dvf.mha')

    def ct_body_path(self) -> str:
        return os.path.join(self.config.output_dir, f'ct_body.mha')

    def input_body_path(self) -> str:
        return os.path.join(self.config.output_dir, f'{self.input_type_str.lower()}_body.mha')

    def ts_structures_path(self, structure_name: str) -> str:
        return os.path.join(self.config.output_dir, f'ts_structures')
    
    ### main preprocessing function ###
    def run_preprocessing(self):
        self.logger.info("Starting preprocessing...")
        if self.patient_complete():
            self.logger.info("All preprocessing files already exist. Skipping patient...")
        else:
            self.load_images()
            self.transfer_sr_images()
            self.transfer_planning_structures()
            self.run_deformation()
            self.run_segmentation()
            self.logger.info("Preprocessing completed.")
    
    ### individual preprocessing steps ###
    def patient_complete(self) -> bool:
        """Checks if all essential files for the patient exist."""
        files_to_check = [
            self.ct_path(),
            self.input_path(),
            self.sr_mask_path(),
            self.sr_fov_path(),
            self.ct_def_path(),
            self.ct_dvf_path(),
            self.ct_body_path(),
            self.input_body_path()
        ]
        return all(os.path.isfile(f) for f in files_to_check)
    
    def load_images(self):
        self.logger.info("Loading images...")
        self.ct_image = io.read_image(self.config.ct_path)
        self.input_image = io.read_image(self.config.input_path)
        if self.config.sr_mask:
            self.sr_mask = io.read_image(self.config.sr_mask)
        if self.config.sr_fov:
            self.sr_fov = io.read_image(self.config.sr_fov)
        self.logger.info("Images loaded.")

    def transfer_sr_images(self):
        self.logger.info("Copying files from synthrad to doserad dataset...")
        io.save_image(self.ct_image, self.ct_path())
        io.save_image(self.input_image, self.input_path())
        io.save_image(self.sr_mask, self.sr_mask_path())
        io.save_image(self.sr_fov, self.sr_fov_path())
    
    def transfer_planning_structures(self):
        self.logger.info("Copying planning structures from synthrad to doserad dataset...")
        structure_names = os.listdir(self.config.sr_structures)
        for structure_name in structure_names:
            if structure_name.endswith('_s2.nrrd'):
                self.logger.info(f"Transferring structure: {structure_name}")
                structure_path = os.path.join(self.config.sr_structures, structure_name)
                structure_image = io.read_image(structure_path)
                self.planning_structures[structure_name] = structure_image
                io.save_image(structure_image, self.planning_structures_path(structure_name))

    def run_deformation(self):
        self.logger.info("Running deformable registration...")
        if self.config.region == 'AB':
            params = reg.CT_MR_params_B_AB()
        if self.config.region == 'TH':
            params = reg.CT_MR_params_B_TH()
        self.ct_deformed, self.ct_disp_field = reg.run_deformable(
            fixed = self.input_image, 
            moving = self.ct_image, 
            mask_fixed= self.sr_mask,
            mask_moving= self.sr_mask,
            use_mask= params['use_mask'],
            background_value = params['background_value'],
            mind_r_c = params['mind_r_c'],
            mind_d_c = params['mind_d_c'],
            mind_r_a = params['mind_r_a'],
            mind_d_a = params['mind_d_a'],
            disp_hw = params['disp_hw'],
            grid_sp = params['grid_sp'],
            grid_sp_adam = params['grid_sp_adam'],
            selected_smooth = params['selected_smooth'],
            selected_niter = params['selected_niter'],
            lambda_weight = params['lambda_weight'], 
            sigma = params['sigma'],
            device= self.device
        )
        
        self.ct_deformed = img.mask_image(self.ct_deformed, self.sr_fov, mask_value=-1024)
        io.save_image(self.ct_deformed, self.ct_def_path())
        io.save_image(self.ct_disp_field, self.ct_dvf_path())
        vis.generate_overview_dir(self.ct_image, self.input_image, self.ct_deformed, self.config.output_dir, self.id)
        self.logger.info("Deformable registration completed.")
        
    def run_segmentation(self):
        self.logger.info("Running segmentation...")
        
        self.ct_skin = seg.segment_skin(self.ct_deformed, modality='CT')
        io.save_image(self.ct_skin, self.ct_body_path())
        
        self.input_skin = seg.segment_skin(self.input_image, modality=self.input_type_str)
        io.save_image(self.input_skin, self.input_body_path())
        
        self.ts_structures = seg.segment_OAR_structures(self.ct_deformed, modality='CT')
        seg.split_multilabel_segmentation(self.ts_structures[0], self.ts_structures[1], save_to_files=True, output_dir=self.config.output_dir)

        if self.ct_disp_field is not None and self.config.modality == 'CBCT':
            self.planning_structures = seg.warp_planning_structures(self.planning_structures, self.ct_disp_field, save_to_files=True, output_dir=self.config.output_dir)
            
        self.logger.info("Segmentation completed.")

        
        
        
        

