# Configuration options

To run the preprocesssing pipeline for a set of patients, a .yaml configuration file is needed. Assign a unique identifer for each patient. For each patient the configuration file is structured into three main sections: `data`, `general`, and `settings`.

An example to automatically generate a .yaml configuration file for a set of patients can be found [here](generate_config.py).

## Data section
The `data` section contains paths to the necessary input files and output directories. The following keys are required:
- `projections`: Path to the directory containing the projection images (either .xim for Varian or .his for Elekta).
- `clinical_recon`: Path to the clinically reconstructed CBCT image. For Elekta data the .SCAN file (stored in the 'Reconstruction' directory) is preferred, but .DCM files can also be used.
- `ct`: Path to the corresponding planning CT image (can be .DCM or any SimpleITK supported format).
- `output`: Path to the output directory where results will be saved.

Elekta only:
- `framesxml`: Path to the _Frames.xml file (containing geometry information).
- `reconstruction_dir`: Path to the directory containing .INI files containing metadata of the acquisition.

Varian only:
- `scanxml`: Path to the Scan.xml file.
- `airscans`: Path to the directory containing air scan calibration data. Usually found in the calibration directory, in a subdirectory called "AIR-Bowtie-*". If multiple AIR directories are present, choose the one where the "Current" subdirectory is present and not the ones with "Factory".
- `scattercorxml`: Path to the Scatter Correction Calibration.xml file. Usually found in the calibration directory, in a subdirectory called "SC-*-*".
- `calibration_dir`: Path to the directory containing calibration data.


## General section
The `general` section includes general information about the patient and imaging system. The following keys are required:
- `center`: The center where the data was acquired (e.g., A, B, C,...).
- `vendor`: The vendor of the imaging system/Linac (Elekta or Varian).


## Settings section
The `settings` section contains various preprocessing options. The following keys are available:
- `correct_orientation`: Boolean flag indicating whether to correct the orientation of the images (true or false). For Elekta data, this usaully needs to be set to true. For Varian data, this usually needs to be set to false.
- `correct_scatter`: Only applicable for Varian data. Boolean flag indicating whether to apply scatter correction (true or false). Does not work with the newest Varian systems having very large flat panel detectors.


## Example configuration file
### Elekta:

```yaml
A001:
    data:
        projections: /data/LMU/patientLMU007/IMAGES/img_1.3.46.423632
        clinical_recon: /data/LMU/patientLMU007/IMAGES/recon.SCAN
        ct: /data/LMU/patientLMU007/CT_SET/2.16.840.1.114337.1.1.1524060727.0
        output: /data/LMU/patientLMU007/output
        framesxml: /data/LMU/patientLMU007/IMAGES/img_1.3.46.423632/_Frames.xml
        reconstruction_dir: /data/LMU/patientLMU007/IMAGES/img_1.3.46.423632/Reconstruction
    general:
        center: A
        vendor: Elekta
    settings:
        correct_orientation: true
```

### Varian:

```yaml
C123:
  data:
    projections: /data/samples/Acquisitions/875743021
    clinical_recon: /data/samples/TrueBeam/25-07-04_07-59-05
    ct: /data/samples/cologne/TrueBeam/pCT
    output: /data/samples/cologne/TrueBeam/output
    scanxml: /data/samples/cologne/TrueBeam/Scan.xml
    airscans: /data/samples/cologne/TrueBeam/Calibrations/AIR-Full-Bowtie-100KV-PaxScan4030CB/Current
    scattercorxml: /data/samples/cologne/TrueBeam/Calibrations/SC-100kV/Factory/Calibration.xml
    calibration_dir: /data/samples/cologne/TrueBeam/Calibrations
  general:
    center: C
    vendor: Varian
  settings:
    correct_orientation: false
    correct_scatter: true
```
