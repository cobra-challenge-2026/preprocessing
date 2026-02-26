import os
import fnmatch
import yaml

# This script needs to be modified for each center/dataset

data_dir = "/data_autoseg/projections/Train"
output_dir = "/data/LMU/"
patients = sorted(os.listdir(data_dir))
patients = fnmatch.filter(patients, 'Pelvis_*')
config_yaml = '/code/configs/lmu_config.yaml'

if os.path.exists(config_yaml):
    os.remove(config_yaml)

for patient in patients:
    print(f'Processing patient: {patient}')
    projections_path = os.path.join(data_dir, patient, 'IMAGES', os.listdir(os.path.join(data_dir, patient, 'IMAGES'))[0])
    ct_dir = os.path.join(data_dir, patient, 'CT_SET')
    ct_folder = [dirname for dirname in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, dirname))]
    ct_path = os.path.join(ct_dir, ct_folder[0])
    id = f'D{patient.strip("Pelvis_")}'
    
    config = {
        id: {
            'general': { 
                'center': 'D',
                'vendor': 'Elekta',
                },
            'data': {
                'projections': projections_path,
                'framesxml': os.path.join(projections_path, '_Frames.xml'),
                'reconstruction_dir': os.path.join(projections_path, 'Reconstruction'),
                'clinical_recon': os.path.join(projections_path, 'Reconstruction', fnmatch.filter(os.listdir(os.path.join(projections_path, 'Reconstruction')), '*.SCAN')[0]),
                'ct': ct_path,
                'output': os.path.join(output_dir, id),
                },
            'settings': {
                'correct_orientation': True,
                }
        }
    }
    
    with open(config_yaml, 'a') as f:
        yaml.dump(config, f)