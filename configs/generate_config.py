import os
import fnmatch
import yaml

data_dir = "/data/LMU/"
patients = sorted(os.listdir(data_dir))
config_yaml = '/code/configs/lmu_config.yaml'

if os.path.exists(config_yaml):
    os.remove(config_yaml)

for patient in patients:
    projections_path = os.path.join(data_dir, patient, 'IMAGES', os.listdir(os.path.join(data_dir, patient, 'IMAGES'))[0])
    ct_path = os.path.join(data_dir, patient, 'CT_SET', fnmatch.filter(os.listdir(os.path.join(data_dir, patient, 'CT_SET')),'2.*')[0])
    id = patient.strip('patient')
    
    config = {
        id: {
            'general': { 
                'center': 'A',
                'vendor': 'Elekta',
                },
            'data': {
                'projections': projections_path,
                'framesxml': os.path.join(projections_path, '_Frames.xml'),
                'inifile': os.path.join(projections_path, 'Reconstruction', sorted(fnmatch.filter(os.listdir(os.path.join(projections_path, 'Reconstruction')), '*.INI.XVI'))[-1]),
                'correctionini': os.path.join(projections_path, 'Reconstruction', fnmatch.filter(os.listdir(os.path.join(projections_path, 'Reconstruction')), '*.SCAN')[0].replace('.SCAN', '.INI.XVI')),
                'clinical_recon': os.path.join(projections_path, 'Reconstruction', fnmatch.filter(os.listdir(os.path.join(projections_path, 'Reconstruction')), '*.SCAN')[0]),
                'ct': ct_path,
                'output': os.path.join(data_dir, patient, 'output'),
                },
            'settings': {
                'correct_orientation': True,
                }
        }
    }
    
    with open(config_yaml, 'a') as f:
        yaml.dump(config, f)