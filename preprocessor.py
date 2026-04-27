import os
import logging
import SimpleITK as sitk
from typing import TYPE_CHECKING, Optional, Any
import yaml

import torch

import utils.io as io
import utils.recon as recon
import utils.scatter_corrector as sc
import utils.virtual_ct as vc
import utils.img as img
import utils.vis as vis
import utils.reg as reg
import utils.seg as seg

import yaml
import copy

if TYPE_CHECKING:
    from utils.config import PatientConfig

class PreProcessor:
    def __init__(self, patient_id: str, config: 'PatientConfig', device: torch.device = torch.device('cuda:0'), skip_recon: bool = False):
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
    
    def ct_def_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct_def.mha')
    
    def ct_def_masked_path(self) -> str:
        return os.path.join(self.config.data.output, f'ct_def_masked.mha')
    
    def rigid_transform_path(self) -> str:
        return os.path.join(self.config.data.output, f'rigid_transform.txt')
    
    def dvf_path(self) -> str:
        return os.path.join(self.config.data.output, f'dvf.mha')

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
    
    def run_stage2(self):
        self.logger.info("Starting stage 2 preprocessing...")
        if self.patient_complete_s2():
            self.logger.info("All preprocessing files already exist. Skipping patient...")
        else:
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
                    projections_header = header,
                    air_scans_dir = self.config.data.airscans,
                    geometry = self.cbct_geometry,
                    scan_xml_path= self.config.data.scanxml,
                    )
            else:
                self.logger.info("Correcting I0...")
                self.cbct_projections_cor = recon.correct_i0_varian(
                    projections = self.cbct_projections, 
                    projections_header = header,
                    air_scans_dir = self.config.data.airscans,
                    geometry = self.cbct_geometry,
                    scan_xml_path= self.config.data.scanxml,
                    )
            self.cbct_clinical = io.read_image(self.config.data.clinical_recon)
        
        else:
            self.logger.error(f"Unsupported vendor: {self.config.general.vendor}")
            raise ValueError(f"Unsupported vendor: {self.config.general.vendor}")
        
        
        self.logger.info("Images loaded.")
    
    ### --- Reconstruct CBCT using RTK FDK --- ###
    def recon_cbct(self):
        self.logger.info("Reconstructing CBCT from projections...")
        if self.config.general.vendor.lower() == 'elekta':
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections_cor, 
                geometry = self.cbct_geometry,
                gpu = False if self.device.type == 'cpu' else True,
                spacing = [1.0, 1.0, 1.0],
                size = [410, 264, 410],
                padding = 0.2,
                hann = 0.99,
                hannY = 0,
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
            self.cbct_rtk = recon.fdk(
                projections = self.cbct_projections_cor, 
                geometry = self.cbct_geometry,
                gpu = False if self.device.type == 'cpu' else True,
                padding = 0.5,
                hann = 0.5,
                hannY = 0.5,
            )
            self.cbct_rtk = img.fix_array_order(self.cbct_rtk, order=(1,0,2), flip=(0,1))
            translation = reg.rigid_registration(
                fixed = self.ct,
                moving = self.cbct_clinical,
                translation_only = True
            )
            self.cbct_rtk = reg.shift_origin(self.cbct_rtk, translation)
            self.cbct_rtk = img.rtk_to_HU(self.cbct_rtk)
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
        if self.config.general.vendor.lower() == 'elekta':
            yaml.dump(self.reconstruction_ini, open(self.reconstruction_ini_path(), 'w'))
        yaml.dump(self.metadata, open(self.metadata_path(), 'w'))
        if self.config.general.vendor.lower() == 'varian':
            io.copy_calibration_dir(
                self.config.data.calibration_dir, 
                os.path.join(self.config.data.output, 'Calibrations')
            )
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
        self.ct_rigid, self.rigid_transform = reg.rigid_elastix(
                        fixed = self.cbct_clinical,
                        moving = self.ct,
                        parameter_file='/code/configs/rigid_params.txt',
                        default_value=-1024
                    )
        self.ct_def_small, _, self.dvf = reg.deformable_impact(
                        fixed = self.cbct_clinical,
                        moving = self.ct_rigid,
                        output_dir = self.config.data.output
                    )
        self.ct_rigid_full = sitk.Resample(self.ct, self.ct, self.rigid_transform, sitk.sitkLinear, -1024)  
        self.fov_cbct = seg.get_cbct_fov(self.cbct_clinical, 2)
        self.logger.info("Deformable registration completed.")
    
    def postprocess_deformed(self):
        self.logger.info("Postprocessing deformed CT...")
        if self.fov_cbct is None:
            self.fov_cbct = seg.get_cbct_fov(self.cbct_clinical, 2)
        vct_gen = vc.VirtualCTCreator(correct_cbct_before_virtual_ct=True, sct_model_path='/code/configs/checkpoint',sct_max_copy_hu=100)
        self.ct_def_masked, cbct_for_blending = vct_gen.create(
            deformed_ct = self.ct_def_small,
            cbct = self.cbct_clinical,
            cbct_fov = self.fov_cbct
        )
        self.dvf = sitk.Cast(self.dvf, sitk.sitkVectorFloat64)
        dvf_for_transform = sitk.Image(self.dvf)   # deep copy
        dfield_tx = sitk.DisplacementFieldTransform(dvf_for_transform)
        fov_cbct_deformed = sitk.Resample(self.fov_cbct, self.fov_cbct, dfield_tx, sitk.sitkNearestNeighbor, 0)
        fov_intersection = sitk.And(self.fov_cbct, fov_cbct_deformed)
        fov_intersection = sitk.Cast(fov_intersection, sitk.sitkInt16)
        fov_intersection_full = sitk.Resample(fov_intersection, self.ct_rigid_full, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        fov_intersection_full = sitk.BinaryErode(fov_intersection_full, (1,1,1))
        ct_def_masked_full = sitk.Resample(self.ct_def_masked, self.ct_rigid_full, sitk.Transform(), sitk.sitkLinear, -1024)
        self.ct_def_masked = sitk.Mask(self.ct_def_masked, self.fov_cbct, outsideValue=-1024)
        self.ct_def = fov_intersection_full * ct_def_masked_full + (1 - fov_intersection_full) * self.ct_rigid_full
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
            cbct_clinical = copy.deepcopy(self.cbct_clinical),
            ct_deformed = copy.deepcopy(self.ct_def_masked),
            output_path = self.overview_path_s2(),
            patient_ID = self.id,
            fov = copy.deepcopy(self.fov_cbct),
            checkerboard_tile = 24
        )
        self.logger.info("Overview visualization generated.")
    
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
        

