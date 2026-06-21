import os
from utils.config import load_patient_configs

### OUTPUT CONFIGURATION ###
output_dir = "/data/RELEASE/test_lukas"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## FILES TO INCLUDE IN THE RELEASE ###
files = [
    'cbct_rtk.mha',
    'cbct_clinical_rigid.mha', 
    'cbct_simulated.mha',
    'fov_cbct.mha', 
    'fov_cbct_nocouch.mha',
    'ct_def.mha',
    'ct_def_masked.mha',
    'projections.mha',
    'projections_simulated.mha',
    'metadata.yaml',
    'reconstruction.yaml',
    'geometry.xml',
    'Scan.xml',
    'Calibrations',
    'GC_data'
    ]

files_extra = [
    r'overview_{}_final.png'
]

if __name__ == "__main__":
    configs = load_patient_configs("configs/lmu_config_sets.yaml")
    for patient_id, config in configs.items():
        print(f"Patient ID: {patient_id}")
        if not os.path.exists(os.path.join(output_dir, patient_id)):
            os.makedirs(os.path.join(output_dir, patient_id))
        for file in files:
            src = os.path.join(config.data.output, file)
            dst = os.path.join(output_dir, patient_id, file)
            if os.path.isfile(src):
                os.system(f"cp {src} {dst}")
                print(f"Copied {src} to {dst}")
            elif os.path.isdir(src):
                os.system(f"cp -r {src} {dst}")
                print(f"Copied directory {src} to {dst}")
            else:
                print(f"File {src} does not exist and was skipped.")
        for file in files_extra:
            src = os.path.join(config.data.output, file.format(patient_id))
            dst = os.path.join(output_dir, patient_id, file.format(patient_id))
            if os.path.isfile(src):
                os.system(f"cp {src} {dst}")
                print(f"Copied {src} to {dst}")
            else:
                print(f"File {src} does not exist and was skipped.")