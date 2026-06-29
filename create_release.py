import os
from utils.config import load_patient_configs

def load_patient_split(patient_split_csv: str) -> list:
    with open(patient_split_csv, "r") as f:
        lines = f.readlines()
    patient_split = [line.strip().split(',')[0:2] for line in lines][1:]
    patient_split_dict = {patient_id: split.lower() for patient_id, split in patient_split}
    return patient_split_dict

#SETTINGS
split_paths = {
    'A': '/data/UMCU/patient_split.csv',
    'B': '/data/MUW/patient_split.csv',
    'C': '/data/LYON/patient_split.csv',
    'D': '/data/LMU/patient_split.csv',
    'F': '/data/UKK/patient_split.csv',
    'G': '/data/MUG/patient_split.csv'
}

config_paths = {
    'A': 'configs/umcu_config_s2.yaml',
    'B': 'configs/muw_config.yaml',
    'C': 'configs/lyon_config_s2.yaml',
    'D': 'configs/lmu_config_sets.yaml',
    'F': 'configs/ukk_config.yaml',
    'G': 'configs/mug_config.yaml'
}

SELECTED_CASE = ['G000']
SET = 'train'
CENTER = 'G'
PATIENT_SPLIT = split_paths[CENTER]
CONFIG = config_paths[CENTER]


### OUTPUT CONFIGURATION ###
output_dir = f"/data/RELEASE/{CENTER}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## FILES TO INCLUDE IN THE RELEASE ###
# input - output filename mapping
files = {
    # "cbct_rtk.mha": "cbct_rtk.mha",
    # "cbct_simulated.mha": "cbct_simulated.mha",
    "cbct_clinical_rigid.mha": "cbct_clinical.mha",
    "fov_cbct.mha": "fov_cbct.mha",
    "fov_cbct_nocouch.mha": "fov_cbct_nocouch.mha",
    "ct.mha": "ct_original.mha",
    "ct_def.mha": "ct_def.mha",
    "ct_def_masked.mha": "ct_def_masked.mha",
    "projections.mha": "projections.mha",
    "projections_simulated.mha": "projections_simulated.mha",
    "metadata.yaml": "metadata.yaml",
    "reconstruction.yaml": "reconstruction.yaml",
    "geometry.xml": "geometry.xml",
    "Scan.xml": "Scan.xml",
    "Calibrations": "Calibrations"
}

files_extra = {
    r'overview_{}_final.png': r'overview_{}.png',
}

set = 'train'

if __name__ == "__main__":
    patient_sets = load_patient_split(PATIENT_SPLIT)
    configs = load_patient_configs(CONFIG)
    for case in patient_sets.keys():
        print(f"Processing case: {case}, set: {patient_sets[case]}")
        if patient_sets[case] == set and (case in SELECTED_CASE or len(SELECTED_CASE) == 0):
            config = configs[case]
            print(f"Processing patient {case}")
            if not os.path.exists(os.path.join(output_dir, case)):
                os.makedirs(os.path.join(output_dir, case))
            for file in files.keys():
                src = os.path.join(config.data.output, file)
                dst = os.path.join(output_dir, case, files[file])
                if os.path.isfile(src):
                    os.system(f"cp {src} {dst}")
                    print(f"Copied {src} to {dst}")
                elif os.path.isdir(src):
                    os.system(f"cp -r {src} {dst}")
                    print(f"Copied directory {src} to {dst}")
                else:
                    print(f"File {src} does not exist and was skipped.")
            for file in files_extra.keys():
                src = os.path.join(config.data.output, file.format(case))
                dst = os.path.join(output_dir, case, files_extra[file].format(case))
                if os.path.isfile(src):
                    os.system(f"cp {src} {dst}")
                    print(f"Copied {src} to {dst}")
                else:
                    print(f"File {src} does not exist and was skipped.")
            