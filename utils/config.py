import yaml
from pydantic import BaseModel
from typing import Optional, Dict

class DataConfig(BaseModel):
    # general data paths
    clinical_recon: str
    ct: str
    projections: str
    output: str

    # elekta specific paths
    reconstruction_dir: Optional[str] = None
    framesxml: Optional[str] = None
    
    # varian specific paths
    scanxml: Optional[str] = None
    airscans: Optional[str] = None
    scattercorxml: Optional[str] = None
    calibration_dir: Optional[str] = None

class GeneralConfig(BaseModel):
    center: str
    vendor: str 
    
class SettingsConfig(BaseModel):
    correct_orientation: bool = False
    correct_scatter: bool = False

class PatientConfig(BaseModel):
    data: DataConfig
    general: GeneralConfig
    settings: SettingsConfig = SettingsConfig()

def load_patient_configs(config_path: str) -> Dict[str, PatientConfig]:
    with open(config_path, "r") as f:
        raw_configs = yaml.safe_load(f)

    validated_configs: Dict[str, PatientConfig] = {}

    for patient_id, config_block in raw_configs.items():
        try:
            validated_configs[patient_id] = PatientConfig(**config_block)
        except Exception as e:
            print(f"Error validating config for {patient_id}: {e}")

    return validated_configs