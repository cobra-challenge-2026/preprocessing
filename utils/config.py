import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PatientConfig(BaseModel):
    modality: str
    region: str
    ct_path: str
    input_path: str
    sr_mask: str
    sr_fov: str
    output_dir: str
    ts_segmentation: bool
    sr_structures: str

def load_patient_configs(config_path: str) -> dict[str, PatientConfig]:
    with open(config_path, 'r') as f:
        raw_configs = yaml.safe_load(f)
    
    validated_configs = {}
    for patient_id, config in raw_configs.items():
        try:
            validated_configs[patient_id] = PatientConfig(**config)
        except Exception as e:
            print(f"Error validating config for {patient_id}: {e}")
            
    return validated_configs