import os
import shutil
import logging
import copy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

import numpy as np
import SimpleITK as sitk
import yaml

import utils.io as io
import utils.recon as recon
import utils.scatter_corrector as sc
import utils.virtual_ct as vc
import utils.img as img
import utils.vis as vis
import utils.reg as reg
import utils.seg as seg

if TYPE_CHECKING:
    from utils.config import PatientConfig

class PreProcessor:
    def __init__(self, patient_id: str, config: 'PatientConfig', device: str = 'cuda:0', skip_recon: bool = False):
        self.id = patient_id
        self.config = config
        self.logger = logging.getLogger(f'PreProcessor.{self.id}')
        self.device = device
        self.skip_recon = skip_recon

        # Data placeholders
        self.cbct_projections: Optional[sitk.Image] = None
        self.cbct_projections_cor: Optional[sitk.Image] = None
        self.cbct_geometry: Optional[Any] = None 
        self.cbct_clinical: Optional[sitk.Image] = None
        self.cbct_clinical_rigid: Optional[sitk.Image] = None
        self.cbct_rtk: Optional[sitk.Image] = None
        self.ct: Optional[sitk.Image] = None
        self.ct_def: Optional[sitk.Image] = None
        self.ct_def_masked: Optional[sitk.Image] = None
        self.metadata: dict = {}
        self.reconstruction_ini: dict = {}
        self.fov_cbct: Optional[sitk.Image] = None
        self.rigid_transform: Optional[Any] = None
        self.dvf: Optional[sitk.Image] = None
    
    ### define filenames for preprocessing files ###
    def cbct_projections_path(self) -> str:
        return os.path.join(self.config.data.output, f'projections.mha')
    
    def cbct_geometry_path(self) -> str:
        return os.path.join(self.config.data.output, f'geometry.xml')
    
    def cbct_clinical_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_clinical.mha')
    
    def cbct_clinical_rigid_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_clinical_rigid.mha')
    
    def cbct_rtk_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_rtk.mha')
    
    def ct_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct.mha')
    
    def reconstruction_ini_path(self) -> str:
        return os.path.join(self.config.data.output, f'reconstruction.yaml')
    
    def metadata_path(self) -> str:
        return os.path.join(self.config.data.output, f'metadata.yaml')
    
    def overview_path(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}.png')
    
    def overview_path_s2(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}_deformed.png')
    
    def fov_cbct_path(self) -> str:
        return os.path.join(self.config.data.output, f'fov_cbct.mha')

    def overview_path_s3(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}_simulated.png')

    def overview_path_s3_projections(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}_projections.png')

    def overview_path_final(self) -> str:
        return os.path.join(self.config.data.output, f'overview_{self.id}_final.png')

    def cbct_simulated_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_simulated.mha')

    def projections_simulated_path(self) -> str:
        return os.path.join(self.config.data.output, f'projections_simulated.mha')
    
    def ct_def_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct_def.mha')
    
    def ct_def_masked_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct_def_masked.mha')
    
    def rigid_transform_path(self) -> str:
        return os.path.join(self.config.data.output, f'rigid_transform.txt')
    
    def cbct_clinical_rigid_transform_path(self) -> str:
        return os.path.join(self.config.data.output, f'cbct_clinical_rigid_transform.txt')
    
    def dvf_path(self) -> str:
        return os.path.join(self.config.data.output, f'dvf.mha')

    def _is_empty(self, img: sitk.Image) -> bool:
        """Check if a file is all -1024 (empty image)"""
        arr = sitk.GetArrayFromImage(img)
        return np.all(arr == -1024)

    ### main preprocessing function for stage1 ###
    def run_stage1(self):
        self.logger.info("Starting preprocessing...")
        if self.patient_complete():
            self.logger.info("All preprocessing files already exist. Skipping patient...")
        else:
            self.load_data()
            if not self.skip_recon:
                self.recon_cbct()
            else:
                self.placeholder_cbct()
            self.extract_metadata()
            self.generate_overview()
            self.write_data()
            self.logger.info("Preprocessing completed.")
    
    def run_stage_recon(self):
        self.logger.info("Starting reconstruction stage...")
        self.load_data_recon()
        self.recon_cbct()
        self.write_data_recon()

    def run_stage2(self):
        self.logger.info("Starting stage 2 preprocessing...")
        if self.patient_complete_s2():
            self.logger.info("All preprocessing files already exist. Skipping patient...")
        else:
            if self._is_empty(self.cbct_rtk):
                self.logger.warning("Reconstructed CBCT is empty. Skipping stage 2 preprocessing...")
                return
            self.logger.info("Performing deformable registration...")
            self.load_data_s2()
            self.run_deformable()
            self.logger.info("Postprocessing deformed CT...")
            self.postprocess_deformed()
            self.generate_overview_s2()
            self.write_data_s2()
            self.logger.info("Stage 2 preprocessing completed.")

    def generate_overview_image(self):
        self.logger.info("Generating overview image...")
        self.load_data_s1()
        if self.cbct_clinical is None or self.cbct_rtk is None or self.ct is None:
            self.logger.error("Clinical CBCT, RTK CBCT, or CT not loaded. Cannot generate overview.")
            return
        vis.generate_overview(
            cbct_clinical = copy.deepcopy(self.cbct_clinical),
            cbct_rtk = copy.deepcopy(self.cbct_rtk),
            ct = copy.deepcopy(self.ct),
            output_path = self.overview_path(),
            patient_ID = self.id,
            metadata=self.metadata
        )
        self.logger.info("Overview image generated.")
    
    
    def generate_overview_deformed(self):
        self.logger.info("Generating overview image...")
        self.load_data_overview()
        if self.cbct_clinical is None or self.ct_def_masked is None:
            self.logger.error("Clinical CBCT or deformed CT not loaded. Cannot generate overview.")
            return
        vis.generate_overview_deformed(
            cbct_clinical = copy.deepcopy(self.cbct_clinical),
            ct_deformed = copy.deepcopy(self.ct_def_masked),
            fov = copy.deepcopy(self.fov_cbct),
            output_path = self.overview_path_s2(),
            patient_ID = self.id,
            metadata=self.metadata,
            checkerboard_tile = 24
        )
        self.logger.info("Overview image generated.")
    ### individual preprocessing steps ###
    
    ### --- Load data from various sources --- ###
    def load_data(self):
        self.logger.info("Loading data...")
        self.ct = io.read_image(self.config.data.ct)
        
        if self.config.general.vendor.lower() == 'elekta':
            self.logger.info("Reading Elekta projections...")
            self.reconstruction_ini = io.read_ini_files(self.config.data.reconstruction_dir)
            self.cbct_projections = io.read_projections_elekta(
                self.config.data.projections, 
                lineint = False
                )
            self.logger.info("Loading geometry...")
            self.cbct_geometry = recon.get_geometry_elekta(
                self.config.data.framesxml
                )
            self.logger.info("Correcting I0...")
            self.cbct_projections_cor = recon.correct_i0_elekta(
                self.cbct_projections, 
                self.reconstruction_ini
                )
            self.logger.info("Load clinical reconstruction...")
            cbct_clinical_temp = io.read_image(self.config.data.clinical_recon)
            self.cbct_clinical = sitk.ShiftScale(cbct_clinical_temp, shift=-1024.0, scale=1.0)
            
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
            if self.config.settings.correct_scatter:
                self.logger.info("Correcting scatter...")
                self.cbct_projections_cor = recon.correct_scatter_varian(
                    projections = self.cbct_projections,
                    geometry = self.cbct_geometry,
                    scattercorxml_path = self.config.data.scattercorxml,
                    airscans_path = self.config.data.airscans
                    )
                self.logger.info("Correcting I0...")
                self.cbct_projections_cor = recon.correct_i0_varian(
                    projections = self.cbct_projections_cor,
                    air_scans_dir = self.config.data.airscans,
                    geometry = self.cbct_geometry,
                    scan_xml_path= self.config.data.scanxml,
                    )
            else:
                self.logger.info("Correcting I0...")
                self.cbct_projections_cor = recon.correct_i0_varian(
                    projections = self.cbct_projections,
                    air_scans_dir = self.config.data.airscans,
                    geometry = self.cbct_geometry,
                    scan_xml_path= self.config.data.scanxml,
                    )
            self.cbct_clinical = io.read_image(self.config.data.clinical_recon)
        
        else:
            self.logger.error(f"Unsupported vendor: {self.config.general.vendor}")
            raise ValueError(f"Unsupported vendor: {self.config.general.vendor}")
        
        
        self.logger.info("Images loaded.")
    
    def load_data_recon(self):
        self.logger.info("Loading data for reconstruction...")
        
        if self.config.general.vendor.lower() == 'elekta':
            self.logger.info("Loading geometry...")
            self.cbct_geometry = io.read_geometry(os.path.join(self.config.data.output, f'geometry.xml'))
            self.cbct_projections = io.read_image(os.path.join(self.config.data.output, f'projections.mha'))
            self.cbct_projections = sitk.Cast(self.cbct_projections, sitk.sitkFloat32)
            self.reconstruction_ini = yaml.safe_load(open(os.path.join(self.config.data.output, f'reconstruction.yaml'), 'r'))
            self.logger.info("Correcting I0...")
            self.cbct_projections_cor = recon.correct_i0_elekta(
                self.cbct_projections, 
                self.reconstruction_ini
                )
            self.logger.info("Load clinical reconstruction...")
            self.cbct_clinical = io.read_image(os.path.join(self.config.data.output, f'cbct_clinical.mha'))
            self.fov_cbct = seg.get_cbct_fov(self.cbct_clinical, 2)
            self.ct = io.read_image(os.path.join(self.config.data.output, f'ct.mha'))
        
        elif self.config.general.vendor.lower() == 'varian':
            self.ct = io.read_image(os.path.join(self.config.data.output, f'ct.mha'))
            self.cbct_clinical = io.read_image(os.path.join(self.config.data.output, f'cbct_clinical.mha'))
            self.cbct_projections = io.read_image(os.path.join(self.config.data.output, f'projections.mha'))
            self.cbct_projections = sitk.Cast(self.cbct_projections, sitk.sitkFloat32)
            self.cbct_geometry = io.read_geometry(os.path.join(self.config.data.output, f'geometry.xml'))
            self.cbct_projections_cor = copy.deepcopy(self.cbct_projections)
            if self.cbct_projections_cor.GetSize()[0] * self.cbct_projections_cor.GetSpacing()[0] > 850:
                padding_val = 0
            else:
                padding_val = 0.1
            self.cbct_projections_cor = recon.correct_scatter_varian(
                        projections = self.cbct_projections,
                        geometry = self.cbct_geometry,
                        scattercorxml_path = self.config.data.scattercorxml,
                        airscans_path = self.config.data.airscans,
                        padding = padding_val
                        )
            self.cbct_projections_cor = recon.correct_i0_varian(
                        projections = self.cbct_projections_cor, 
                        air_scans_dir = self.config.data.airscans,
                        geometry = self.cbct_geometry,
                        scan_xml_path= self.config.data.scanxml,
                        )
            self.cbct_projections_cor = sitk.Cast(self.cbct_projections_cor, sitk.sitkFloat32)
        
    
    ### --- Reconstruct CBCT using RTK FDK --- ###
    def recon_cbct(self):
        self.logger.info("Reconstructing CBCT from projections...")
        if self.config.general.vendor.lower() == 'elekta':
            spacing = self.cbct_clinical.GetSpacing()
            size = self.cbct_clinical.GetSize()
            recon_order = (0, 2, 1)
            recon_spacing = tuple(spacing[i] for i in recon_order)
            recon_size = tuple(size[i] for i in recon_order)
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections_cor, 
                geometry = self.cbct_geometry,
                gpu = False if self.device == 'cpu' else True,
                spacing = recon_spacing,
                size = recon_size,
                padding = 0.2,
                hann = 0.99,
                hannY = 0.99,
            )
            if self.config.settings.correct_orientation and self.reconstruction_ini is not None:
                self.cbct_rtk = img.fix_array_order(self.cbct_rtk, order=(1,2,0), flip=(0,))
                try:
                    self.cbct_rtk = img.allign_images_ini(self.cbct_rtk, self.reconstruction_ini)
                    self.cbct_rtk = img.rtk_to_HU(self.cbct_rtk)
                    self.cbct_clinical = img.allign_images_ini(self.cbct_clinical, self.reconstruction_ini)
                except Exception as e:
                    self.logger.warning(f"Could not align images based on INI file: {e}")
                    translation = reg.rigid_registration(
                            fixed = self.ct,
                            moving = self.cbct_rtk,
                            translation_only = True
                        )
                    self.cbct_rtk = reg.shift_origin(self.cbct_rtk, translation)
                    self.cbct_rtk = img.rtk_to_HU(self.cbct_rtk)
                    self.cbct_clinical = reg.shift_origin(self.cbct_clinical, translation) 
                                    
        elif self.config.general.vendor.lower() == 'varian':
            spacing = self.cbct_clinical.GetSpacing()
            size = self.cbct_clinical.GetSize()
            recon_order = (0, 2, 1)
            recon_spacing = tuple(spacing[i] for i in recon_order)
            recon_size = tuple(size[i] for i in recon_order)
            # fix recon for large projections which violate parker weighting
            if self.cbct_projections_cor.GetSize()[0] * self.cbct_projections_cor.GetSpacing()[0] > 850:
                self.cbct_projections_cor = self.cbct_projections_cor[128:-128,:,:]
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections_cor, 
                geometry = self.cbct_geometry,
                size = recon_size,
                spacing = recon_spacing,
                gpu = False if self.device == 'cpu' else True,
                padding = 0.5,
                hann = 0.4,
                hannY = 0.4,
            )
            self.fov_cbct = seg.get_fov_rtk(self.cbct_rtk, self.cbct_projections_cor, self.cbct_geometry)
            self.cbct_rtk = img.fix_array_order(self.cbct_rtk, order=(1,0,2), flip=(0,1))
            self.fov_cbct = img.fix_array_order(self.fov_cbct, order=(1,0,2), flip=(0,1))
            self.cbct_rtk = img.rtk_to_HU(self.cbct_rtk)
            iso_offset = reg.rigid_registration(
                fixed = self.cbct_rtk,
                moving = self.cbct_clinical,
                translation_only = True
            )
            iso_residual = reg.registration_residual(iso_offset, self.cbct_rtk, self.cbct_clinical)
            self.logger.info(f"Clinical-to-RTK offset: {np.round(iso_offset.GetOffset(), 2)} mm "
                             f"(residual after grid centering: {np.round(iso_residual, 2)} mm)")
            if np.linalg.norm(iso_residual) > 10.0:
                self.logger.warning("Clinical-to-RTK residual offset exceeds 10 mm. "
                                    "Check geometry parsing or registration result.")
            self.cbct_clinical = reg.shift_origin(self.cbct_clinical, iso_offset)
            translation = reg.rigid_registration(
                fixed = self.ct,
                moving = self.cbct_clinical,
                translation_only = True
            )
            self.cbct_rtk = reg.shift_origin(self.cbct_rtk, translation)
            self.fov_cbct = reg.shift_origin(self.fov_cbct, translation)
            self.cbct_rtk = img.mask_image(self.cbct_rtk, self.fov_cbct)
            self.cbct_clinical = reg.shift_origin(self.cbct_clinical, translation)
        
        self.logger.info("CBCT reconstruction completed.")
    
    def placeholder_cbct(self):
        self.logger.info("Creating placeholder CBCT...")
        if self.config.general.vendor.lower() == 'elekta':
            self.logger.error("Placeholder CBCT is not implemented for Elekta. Please run with --skip_recon only for Varian data.")
                        
        elif self.config.general.vendor.lower() == 'varian':
            translation = reg.rigid_registration(
                fixed = self.ct,
                moving = self.cbct_clinical,
                translation_only = True
            )
            self.cbct_clinical = reg.shift_origin(self.cbct_clinical, translation)
            # create a placeholder CBCT with the same size and spacing as the clinical CBCT, but with all values set to 0
            self.cbct_rtk = sitk.Image(self.cbct_clinical.GetSize(), sitk.sitkInt16)
            self.cbct_rtk.SetSpacing(self.cbct_clinical.GetSpacing())
            self.cbct_rtk.SetOrigin(self.cbct_clinical.GetOrigin())
            self.cbct_rtk.SetDirection(self.cbct_clinical.GetDirection())
        
        self.logger.info("CBCT reconstruction completed.")
    
    ### --- Extract metadata from CT and CBCT --- ###
    def extract_metadata(self):
        self.logger.info("Extracting metadata...")
        ct_metadata = img.extract_dicom_metadata(
            self.config.data.ct,
            tags_file = './configs/tags_CT.txt'
        )
        if self.config.general.vendor.lower() == 'elekta':
            if self.config.data.clinical_recon.endswith('.SCAN'):
                cbct_metadata = img.extract_elekta_metadata(
                    tags_yaml = './configs/tags_CBCT.yaml',
                    reconstruction_ini = self.reconstruction_ini,
                    projections = self.cbct_projections,
                    cbct_clinical = self.cbct_clinical,
                    geometry = self.cbct_geometry
                )
        elif self.config.general.vendor.lower() == 'varian':
            cbct_metadata = img.extract_varian_metadata(
                tags_yaml = './configs/tags_CBCT.yaml',
                scan_xml = self.config.data.scanxml,
                projections = self.cbct_projections,
                cbct_clinical = self.cbct_clinical,
                geometry = self.cbct_geometry
            )
        self.metadata = {
            'ct': ct_metadata,
            'cbct': cbct_metadata
        }
        self.logger.info("Metadata extracted.")
    
    ### --- Generate overview visualization --- ###
    def generate_overview(self):
        self.logger.info("Generating overview visualization...")
        vis.generate_overview(
            cbct_clinical = copy.deepcopy(self.cbct_clinical),
            cbct_rtk = copy.deepcopy(self.cbct_rtk),
            ct = copy.deepcopy(self.ct),
            output_path = self.overview_path(),
            patient_ID = self.id,
            metadata=self.metadata
        )
        self.logger.info("Overview visualization generated.")

    def load_data_s1(self):
        self.logger.info("Loading data for overview...")
        self.ct = io.read_image(self.ct_path())
        self.cbct_clinical = io.read_image(self.cbct_clinical_path())
        self.cbct_rtk = io.read_image(self.cbct_rtk_path())
        self.fov_cbct = io.read_image(self.fov_cbct_path())
        self.metadata = yaml.safe_load(open(self.metadata_path(), 'r'))
        self.logger.info("Data for overview loaded.")
    
    ### --- Save preprocessed data to files --- ###
    def write_data(self):
        self.logger.info("Saving preprocessed data to files...")
        if not os.path.exists(self.config.data.output):
            os.makedirs(self.config.data.output)
        io.save_image(self.cbct_projections, self.cbct_projections_path(), dtype='uint16')
        io.write_geometry(self.cbct_geometry, self.cbct_geometry_path())
        io.save_image(self.cbct_rtk, self.cbct_rtk_path(), dtype='int16')
        io.save_image(self.ct, self.ct_path(), dtype='int16')
        io.save_image(self.cbct_clinical, self.cbct_clinical_path(), dtype='int16')
        io.save_image(self.fov_cbct, self.fov_cbct_path(), dtype='uint16')
        if self.config.general.vendor.lower() == 'elekta':
            yaml.dump(self.reconstruction_ini, open(self.reconstruction_ini_path(), 'w'))
        yaml.dump(self.metadata, open(self.metadata_path(), 'w'))
        if self.config.general.vendor.lower() == 'varian':
            io.copy_calibration_dir(
                self.config.data.calibration_dir, 
                os.path.join(self.config.data.output, 'Calibrations')
            )
        self.logger.info("Preprocessed data saved.")
    
    def write_data_recon(self):
        self.logger.info("Saving preprocessed data to files...")
        if not os.path.exists(self.config.data.output):
            os.makedirs(self.config.data.output)
        io.save_image(self.cbct_rtk, self.cbct_rtk_path(), dtype='int16')
        io.save_image(self.fov_cbct, self.fov_cbct_path(), dtype='uint16')
        self.logger.info("Preprocessed data saved.")
    
    ### --- Check if all necessary files exist after preprocessing --- ###
    def patient_complete(self) -> bool:
        """Checks if all essential files for the patient exist."""
        files_to_check = [
            self.ct_path(),
            self.cbct_geometry_path(),
            self.cbct_rtk_path(),
            self.cbct_clinical_path(),
            self.cbct_projections_path(),
            self.metadata_path(),
        ]
        return all(os.path.isfile(f) for f in files_to_check)
    
    ### --- Load data from stage 1 --- ###
    def load_data_s2(self):
        self.logger.info("Loading data for stage 2...")
        self.ct = io.read_image(self.ct_path())
        self.cbct_clinical = io.read_image(self.cbct_clinical_path())
        self.cbct_rtk = io.read_image(self.cbct_rtk_path())
        self.fov_cbct = io.read_image(self.fov_cbct_path())
        self.logger.info("Data for stage 2 loaded.")
        
    def load_data_overview(self):
        self.logger.info("Loading data for overview...")
        self.ct = io.read_image(self.ct_path())
        self.cbct_clinical = io.read_image(self.cbct_clinical_path())
        self.ct_def_masked = io.read_image(self.ct_def_masked_path())
        self.fov_cbct = io.read_image(self.fov_cbct_path())
        self.logger.info("Data for overview loaded.")
    
    ### --- Run deformable registration between clinical CBCT and CT --- ###
    def run_deformable(self):
        self.logger.info("Running deformable registration...")
        if self.config.general.center == 'F':
            try:
                self.ct_rigid, self.rigid_transform = reg.rigid_elastix(
                                fixed = self.cbct_rtk,
                                moving = self.ct,
                                parameter_file='/code/configs/rigid_params_ukk.txt',
                                default_value=-1024,
                                output_dir=self.config.data.output
                            )
            except:
                self.ct_rigid, self.rigid_transform = reg.rigid_elastix(
                                fixed = self.cbct_rtk,
                                moving = self.ct,
                                parameter_file='/code/configs/rigid_params.txt',
                                default_value=-1024,
                                output_dir=self.config.data.output
                            )
        else:
            self.ct_rigid, self.rigid_transform = reg.rigid_elastix(
                            fixed = self.cbct_rtk,
                            moving = self.ct,
                            parameter_file='/code/configs/rigid_params.txt',
                            default_value=-1024,
                            output_dir=self.config.data.output
                        )
            
        self.ct_def_small, _, self.dvf = reg.deformable_impact(
                        fixed = self.cbct_rtk,
                        moving = self.ct_rigid,
                        output_dir = self.config.data.output
                    )
        self.ct_rigid_full = sitk.Resample(self.ct, self.ct, self.rigid_transform, sitk.sitkLinear, -1024)  
        
        if self.config.general.vendor.lower() == 'varian':
            self.fov_cbct = self.fov_cbct
            self.cbct_clinical_rigid, self.cbct_clinical_rigid_transform = reg.rigid_elastix(
                        fixed = self.cbct_rtk,
                        moving = self.cbct_clinical,
                        parameter_file='/code/configs/rigid_params.txt',
                        default_value=-1024,
                        output_dir=self.config.data.output
                    )
            sitk.WriteTransform(self.cbct_clinical_rigid_transform, self.cbct_clinical_rigid_transform_path())
            sitk.WriteImage(self.cbct_clinical_rigid, self.cbct_clinical_rigid_path())
        else:
            self.cbct_clinical_rigid = sitk.Resample(self.cbct_clinical, self.cbct_rtk, sitk.Transform(), sitk.sitkLinear, -1024)
            self.fov_cbct = seg.get_cbct_fov(self.cbct_clinical_rigid, 2)
            self.cbct_rtk = img.mask_image(self.cbct_rtk, self.fov_cbct) 
            sitk.WriteImage(self.cbct_clinical_rigid, self.cbct_clinical_rigid_path())
            sitk.WriteImage(self.cbct_rtk, self.cbct_rtk_path())
        self.logger.info("Deformable registration completed.")
    
    def postprocess_deformed(self):
        self.logger.info("Postprocessing deformed CT...")
        if self.fov_cbct is None:
            raise ValueError("FOV mask for CBCT is not available. Cannot postprocess deformed CT.")
            # if self.config.general.vendor.lower() == 'varian':
            #     self.fov_cbct = seg.get_cbct_fov_varian(self.cbct_clinical, 2)
            # else:
            #     self.fov_cbct = seg.get_cbct_fov(self.cbct_clinical, 2)
    
        if self.config.general.vendor.lower() == 'varian':     
            vct_gen = vc.VirtualCTCreator(correct_cbct_before_virtual_ct=True, sct_model_path='/code/configs/checkpoint', sct_max_copy_hu=100, air_threshold_hu=-200, blend_margin_mm=3)
            sitk.WriteImage(self.ct_def_small, os.path.join(self.config.data.output, f'ct_def_small.mha'))
            self.ct_def_masked, cbct_for_blending = vct_gen.create(
                deformed_ct = self.ct_def_small,
                cbct = self.cbct_rtk,
                cbct_fov = self.fov_cbct,
                use_cbct_body = True,
                avoid_bone_region = True,
                cbct_clinical_rigid = self.cbct_clinical_rigid
            )
        else:
            vct_gen = vc.VirtualCTCreator(correct_cbct_before_virtual_ct=True, sct_model_path='/code/configs/checkpoint', sct_max_copy_hu=100, air_threshold_hu=-400, blend_margin_mm=2)
            self.ct_def_masked, cbct_for_blending = vct_gen.create(
                deformed_ct = self.ct_def_small,
                cbct = self.cbct_clinical_rigid,
                cbct_fov = self.fov_cbct,
                use_cbct_body = True,
                avoid_bone_region = True,
                cbct_clinical_rigid = self.cbct_clinical_rigid
            )
        
        sitk.WriteImage(cbct_for_blending, os.path.join(self.config.data.output, f'cbct_for_blending.mha'))
        self.dvf = reg.extend_vector_field_outside_mask(
            self.dvf,
            self.fov_cbct,
            reference_image=self.ct,
        )
        dvf_for_transform = sitk.Image(self.dvf)   # deep copy
        dfield_tx = sitk.DisplacementFieldTransform(dvf_for_transform)
        fov_cbct_full = sitk.Resample(self.fov_cbct, self.ct_rigid_full, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        fov_cbct_deformed = sitk.Resample(fov_cbct_full, fov_cbct_full, dfield_tx, sitk.sitkNearestNeighbor, 0)
        fov_intersection_full = sitk.And(fov_cbct_full, fov_cbct_deformed)
        fov_intersection_full = sitk.Cast(fov_intersection_full, sitk.sitkInt16)
        fov_intersection_full = sitk.BinaryErode(fov_intersection_full, (1,1,1))
        sitk.WriteImage(fov_intersection_full, os.path.join(self.config.data.output, f'fov_intersection_full.mha'))
        ct_def_masked_full = sitk.Resample(self.ct_def_masked, self.ct_rigid_full, sitk.Transform(), sitk.sitkLinear, -1024)
        ct_def_extended = reg.apply_transforms(
            ct=self.ct,
            rigid_transform=self.rigid_transform,
            dvf=self.dvf,
            default_value=-1024,
            interpolator=sitk.sitkLinear,
        )
        self.ct_def_masked = sitk.Mask(self.ct_def_masked, self.fov_cbct, outsideValue=-1024)
        self.ct_def = fov_intersection_full * ct_def_masked_full + (1 - fov_intersection_full) * ct_def_extended
        self.ct_def = sitk.Clamp(self.ct_def, lowerBound=-1024, upperBound=3071)
        self.ct_def_masked = sitk.Clamp(self.ct_def_masked, lowerBound=-1024, upperBound=3071)
        self.cbct_clinical = sitk.Clamp(self.cbct_clinical, lowerBound=-1024, upperBound=3071)
        self.logger.info("Postprocessing completed.")

    ### --- Save deformed CT, FOV mask, and DVF to files --- ###
    def write_data_s2(self):
        self.logger.info("Saving preprocessed data to files...")
        io.save_image(self.ct_def, self.ct_def_path(), dtype='int16')
        io.save_image(self.fov_cbct, self.fov_cbct_path(), dtype='uint16')
        io.save_image(self.ct_def_masked, self.ct_def_masked_path(), dtype='int16')
        sitk.WriteImage(self.dvf, self.dvf_path())
        sitk.WriteTransform(self.rigid_transform, self.rigid_transform_path())
        self.logger.info("Preprocessed data saved.")
    
    def generate_overview_s2(self):
        self.logger.info("Generating overview visualization...")
        vis.generate_overview_deformed(
            cbct_clinical = copy.deepcopy(self.cbct_clinical if self.cbct_clinical_rigid is None else self.cbct_clinical_rigid),
            ct_deformed = copy.deepcopy(self.ct_def_masked),
            output_path = self.overview_path_s2(),
            patient_ID = self.id,
            fov = copy.deepcopy(self.fov_cbct),
            checkerboard_tile = 24
        )
        self.logger.info("Overview visualization generated.")
    
        
    ### --- Stage 3: simulate projections from deformed CT --- ###

    def run_stage3(self):
        self.logger.info("Starting stage 3 preprocessing...")
        if self.patient_complete_s3():
            self.logger.info("All stage 3 files already exist. Skipping patient...")
        else:
            if self._is_empty(self.cbct_rtk):
                self.logger.warning("Reconstructed CBCT is empty. Skipping stage 3 preprocessing...")
                return
            self.load_data_s3()
            self.simulate_projections()
            self.generate_overview_s3()
            self.generate_overview_s3_projections()
            self.logger.info("Stage 3 preprocessing completed.")

    def load_data_s3(self):
        self.logger.info("Loading data for stage 3...")
        self.ct_def = io.read_image(self.ct_def_path())
        self.cbct_clinical = io.read_image(self.cbct_clinical_path())
        # the rigid clinical recon defines the projector isocenter (its grid center
        # marks the acquisition isocenter); without it the projector infers the
        # isocenter from the CT and the simulated geometry will not match the real scan
        if os.path.isfile(self.cbct_clinical_rigid_path()):
            self.cbct_clinical_rigid = io.read_image(self.cbct_clinical_rigid_path())
        else:
            self.cbct_clinical_rigid = None
            self.logger.warning("Rigid clinical CBCT not found. Projector isocenter will be "
                                "inferred from the CT and simulated projections will not be "
                                "geometrically identical to the real ones.")
        self.logger.info("Data for stage 3 loaded.")

    def simulate_projections(self):
        from simcbctgenerator import (
            MotionConfig, Patient, PatientConfig, ProjectionPipeline, Vendor,
        )

        self.logger.info("Generating simulated projections...")
        output_dir = Path(self.config.data.output)
        pipeline = ProjectionPipeline(
            vendor=Vendor.from_value(self.config.general.vendor),
            correct_contrast_media=self.config.settings.correct_contrast_media,
            gpu=self.device != 'cpu',
        )
        simulator = pipeline._create_simulator()

        # the library computes recon_origin as -n*s/2, displacing the simulated
        # recon by half a voxel from the isocenter and breaking the origin
        # permutation in its export step; -(n-1)*s/2 is the true centered origin
        system_config = simulator.build_system_config(
            geometry_xml=self.cbct_geometry_path(),
            metadata_yaml=self.metadata_path(),
        )
        system_config = system_config.with_reconstruction_volume(
            recon_origin=[
                -0.5 * (n - 1) * s
                for n, s in zip(system_config.recon_size, system_config.recon_spacing)
            ]
        )

        # shared random-motion settings for both the rigid and inferred-iso paths
        motion_kwargs = dict(
            random_motion_type=MotionConfig.MotionType.PELVIS,
            random_motion_amplitude_range=(2.0, 7.0),    # motion displacement +/- mm
            random_motion_frequency_range=(12.0, 20.0),  # breathing frequency
            random_motion_uncertainty_range=(0.01, 0.02),  # random perturbation
        )

        if self.cbct_clinical_rigid is not None:
            # Build the patient from the CT at native resolution and drive the
            # simulator directly: the upstream package builds the patient
            # internally (resampling the CT onto the reference grid) and has no
            # patient= passthrough on generate_projections, so we construct it
            # here with reference_cbct=None to keep the CT native. Work on a copy
            # of the CT so re-origining below does not mutate self.ct_def.
            patient = Patient.from_images(
                ct_image=sitk.Image(self.ct_def),
                config=PatientConfig(
                    plan_dir=".", ct_dir=".", cbct_dir=".",
                    export_structures=[], priority=[],
                    image_modality="synrad", use_totalsegmentator=False,
                ),
                reference_cbct=None,
                patient_id="projection_pipeline",
            )
            # position the CT for projection with an exact continuous shift: the
            # library snaps the isocenter to whole voxels and lands it one voxel
            # off (0,0,0); the grid center of the rigid clinical recon marks the
            # acquisition isocenter
            iso_center = np.array(self.cbct_clinical_rigid.TransformContinuousIndexToPhysicalPoint(
                [(s - 1) / 2.0 for s in self.cbct_clinical_rigid.GetSize()]))
            patient.shifted_origin = np.array(self.ct_def.GetOrigin()) - iso_center
            patient.ct_image.SetOrigin(patient.shifted_origin.tolist())
            patient.iso_center = np.zeros(3)

            # run() generates the projections, reconstructs via FDK and writes
            # both projections_simulated.mha and cbct_simulated.mha to output_dir
            result = simulator.run(
                patient=patient,
                system_config=system_config,
                output_dir=output_dir,
                cbct_image=self.cbct_clinical_rigid,
                cleanup_temp=False,
                **motion_kwargs,
            )
            system_config = result.system_config
        else:
            # no rigid recon: let the pipeline infer the isocenter from the CT
            _, system_config = pipeline.generate_projections(
                ct_image=self.ct_def,
                output_dir=output_dir,
                cbct_image=self.cbct_clinical_rigid,
                system_config=system_config,
                **motion_kwargs,
            )

        proj_dir = output_dir / "drr_temp"

        missing_outputs = [
            str(path) for path in (
                Path(self.projections_simulated_path()),
                Path(self.cbct_simulated_path()),
            ) if not path.is_file()
        ]
        if missing_outputs:
            raise FileNotFoundError(
                "simcbctgenerator did not produce the expected stage 3 outputs: "
                + ", ".join(missing_outputs)
            )

        # the simulated recon is exported in the isocenter frame; it is
        # reconstructed on the same CBCT geometry as the RTK recon, so it is
        # voxel-congruent with cbct_rtk. Copy cbct_rtk's metadata to give the
        # simulated CBCT the same origin as cbct_rtk (stage 4 then shifts both
        # so the isocenter lands at (0,0,0)).
        if os.path.isfile(self.cbct_rtk_path()):
            cbct_rtk = io.read_image(self.cbct_rtk_path())
            cbct_simulated = io.read_image(self.cbct_simulated_path())
            if cbct_simulated.GetSize() == cbct_rtk.GetSize():
                cbct_simulated.CopyInformation(cbct_rtk)
                sitk.WriteImage(cbct_simulated, self.cbct_simulated_path())
            else:
                self.logger.warning("Simulated CBCT grid does not match the RTK grid. "
                                    "Leaving the simulated CBCT in the isocenter frame.")
        else:
            self.logger.warning("cbct_rtk not found; leaving the simulated CBCT in the "
                                "isocenter frame.")

        if proj_dir.is_dir():
            shutil.rmtree(proj_dir)

        self.logger.info("Simulated projections and CBCT saved.")

    def generate_overview_s3(self):
        self.logger.info("Generating stage 3 overview visualization...")
        cbct_simulated = io.read_image(self.cbct_simulated_path())
        # the rigid clinical recon shares the RTK grid, whose center marks the
        # isocenter; the clinical recon grid carries the measured residual offset
        if self.cbct_clinical_rigid is not None:
            cbct_simulated.CopyInformation(self.cbct_clinical_rigid)
        else:
            cbct_simulated.CopyInformation(self.cbct_clinical)
        vis.generate_overview_deformed(
            cbct_clinical=copy.deepcopy(self.cbct_clinical if self.cbct_clinical_rigid is None else self.cbct_clinical_rigid),
            ct_deformed=copy.deepcopy(cbct_simulated),
            output_path=self.overview_path_s3(),
            patient_ID=self.id,
            image2_label="CBCT simulated",
        )
        self.logger.info("Stage 3 overview visualization generated.")

    def generate_overview_s3_projections(self):
        self.logger.info("Generating stage 3 projection overview...")
        geometry = io.read_geometry(self.cbct_geometry_path())
        gantry_angles = [a * 180.0 / np.pi for a in geometry.GetGantryAngles()]

        proj_real = sitk.GetArrayFromImage(io.read_image(self.cbct_projections_path())).astype(np.float32)
        proj_sim = sitk.GetArrayFromImage(io.read_image(self.projections_simulated_path())).astype(np.float32)

        if self.config.general.vendor.lower() == 'varian':
            rotation = 'CW' if gantry_angles[20] < gantry_angles[30] else 'CC'
            air_imgs, _ = recon.read_air_scans(self.config.data.airscans, rotation=rotation, return_sitk=False)
            air = air_imgs[0].astype(np.float32)
            eps = 1e-6
            proj_real = -np.log(np.clip(proj_real / np.clip(air, eps, None), eps, None))
            proj_sim = -np.log(np.clip(proj_sim, eps, None))

        vis.generate_overview_projections(
            proj_real=proj_real,
            proj_sim=proj_sim,
            gantry_angles=gantry_angles,
            output_path=self.overview_path_s3_projections(),
            patient_ID=self.id,
        )
        self.logger.info("Stage 3 projection overview generated.")

    def patient_complete_s3(self) -> bool:
        files_to_check = [
            self.cbct_simulated_path(),
            self.projections_simulated_path(),
        ]
        return all(os.path.isfile(f) for f in files_to_check)

    def cleanup_s2(self):
        files = [
            self.ct_def_path(),
            self.ct_def_masked_path(),
            self.fov_cbct_path(),
            self.dvf_path(),
            self.rigid_transform_path(),
            self.overview_path_s2(),
        ]
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
                self.logger.info(f"Deleted {f}")

    def cleanup_s3(self):
        files = [
            self.cbct_simulated_path(),
            self.projections_simulated_path(),
            self.overview_path_s3(),
            self.overview_path_s3_projections(),
        ]
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
                self.logger.info(f"Deleted {f}")

    ### --- Check if all necessary files exist after stage2 --- ###
    def patient_complete_s2(self) -> bool:
        """Checks if all essential files for the patient exist."""
        files_to_check = [
            self.ct_def_path(),
            self.fov_cbct_path(),
            self.ct_def_masked_path(),
            self.dvf_path(),
            self.rigid_transform_path(),
        ]
        return all(os.path.isfile(f) for f in files_to_check)
    
    ### --- Stage 4: postprocessing and clean up --- ###

    def fov_cbct_nocouch_path(self) -> str:
        return os.path.join(self.config.data.output, f'fov_cbct_nocouch.mha')

    def generate_final_overview(self):
        self.logger.info("Generating final overview visualization...")

        # reference grid for the volumetric panels
        cbct_rtk = io.read_image(self.cbct_rtk_path())
        cbct_clinical = (
            io.read_image(self.cbct_clinical_rigid_path())
            if os.path.isfile(self.cbct_clinical_rigid_path())
            else io.read_image(self.cbct_clinical_path())
        )
        ct_def_masked = io.read_image(self.ct_def_masked_path())
        fov_cbct = io.read_image(self.fov_cbct_nocouch_path())

        # the simulated recon shares the rigid clinical / RTK grid; restore info
        cbct_simulated = io.read_image(self.cbct_simulated_path())

        # projections (same I0/log handling as the stage 3 projection overview)
        geometry = io.read_geometry(self.cbct_geometry_path())
        gantry_angles = [a * 180.0 / np.pi for a in geometry.GetGantryAngles()]
        proj_real = sitk.GetArrayFromImage(io.read_image(self.cbct_projections_path())).astype(np.float32)
        proj_sim = sitk.GetArrayFromImage(io.read_image(self.projections_simulated_path())).astype(np.float32)

        if self.config.general.vendor.lower() == 'varian':
            rotation = 'CW' if gantry_angles[20] < gantry_angles[30] else 'CC'
            air_imgs, _ = recon.read_air_scans(self.config.data.airscans, rotation=rotation, return_sitk=False)
            air = air_imgs[0].astype(np.float32)
            eps = 1e-6
            proj_real = -np.log(np.clip(proj_real / np.clip(air, eps, None), eps, None))
            proj_sim = -np.log(np.clip(proj_sim, eps, None))

        vis.generate_final_overview(
            cbct_clinical=cbct_clinical,
            cbct_rtk=cbct_rtk,
            cbct_simulated=cbct_simulated,
            ct_def_masked=ct_def_masked,
            fov_cbct=fov_cbct,
            proj_real=proj_real,
            proj_sim=proj_sim,
            gantry_angles=gantry_angles,
            output_path=self.overview_path_final(),
            patient_ID=self.id,
        )
        self.logger.info("Final overview visualization generated.")

    def run_stage4(self):
        self.logger.info("Starting stage 4 postprocessing...")
        
        ### Remove couch ###
        cbct_rtk = io.read_image(self.cbct_rtk_path()) if os.path.isfile(self.cbct_rtk_path()) else None
        fov = io.read_image(self.fov_cbct_path()) if os.path.isfile(self.fov_cbct_path()) else None
        if cbct_rtk is None or fov is None:
            self.logger.warning("CBCT or FOV mask not found. Skipping stage 4 postprocessing...")
            return
        self.logger.info("Detecting couch on image...")
        result = img.generate_couch_masks_from_image(cbct_rtk, fov)
        fov_nocouch = img.remove_couch_from_fov(fov, result["behind_mask"])
        self.logger.info("Removed couch from FOV.")
        io.save_image(fov_nocouch, self.fov_cbct_nocouch_path(), dtype='uint16')
        
        ### Load images for re-origining ###
        specs = [
            (self.ct_def_masked_path(), 'int16'),
            (self.ct_def_path(), 'int16'),
            (self.cbct_rtk_path(), 'int16'),
            (self.cbct_simulated_path(), 'int16'),
            (self.cbct_clinical_rigid_path(), 'int16'),
            (self.fov_cbct_path(), 'uint16'),
            (self.fov_cbct_nocouch_path(), 'uint16')
        ]
        images = {}
        for path, _ in specs:
            if path == self.cbct_rtk_path():
                images[path] = cbct_rtk
            elif path == self.fov_cbct_nocouch_path():
                images[path] = fov_nocouch
            elif os.path.isfile(path):
                images[path] = io.read_image(path)

        # the simulated recon is reconstructed on the same CBCT geometry as the
        # RTK recon, so it is voxel-congruent with cbct_rtk but may still carry a
        # stale origin from the clinical frame (e.g. when the stage-3 rigid stamp
        # was skipped). Re-stamp it onto the RTK grid so it shares cbct_rtk's
        # origin before the common isocenter shift below.
        sim_path = self.cbct_simulated_path()
        if sim_path in images:
            if images[sim_path].GetSize() == cbct_rtk.GetSize():
                images[sim_path].CopyInformation(cbct_rtk)
            else:
                self.logger.warning("Simulated CBCT grid does not match the RTK grid; "
                                    "leaving its origin unchanged.")

        ### change origin to be consistent with direct RTK recon ###
        center = np.array(cbct_rtk.TransformContinuousIndexToPhysicalPoint(
            [(s - 1) / 2.0 for s in cbct_rtk.GetSize()]))
        if np.allclose(center, 0.0, atol=1e-3):
            self.logger.info("cbct_rtk is already centered on the isocenter; "
                             "skipping origin shift.")
        else:
            self.logger.info(f"Shifting origins by {np.round(-center, 2)} mm so the "
                             "isocenter lands at (0,0,0).")
            for image in images.values():
                image.SetOrigin((np.array(image.GetOrigin()) - center).tolist())

        ### write the files ###
        for path, dtype in specs:
            if path in images:
                if dtype is None:
                    sitk.WriteImage(images[path], path, useCompression=True)
                else:
                    io.save_image(images[path], path, dtype=dtype)
        
        self.generate_final_overview()
        
        self.logger.info("Stage 4 completed.")
        
        
