# COBRA2026 Preprocessing Pipeline

The preprocessing pipeline used for the COBRA2026 dataset. This pipeline prepares CBCT projections and corresponding CT images for the COBRA2026 reconstruction challenge.

Currently preprocessing for Elekta and Varian CBCT systems is supported.

## Running the Preprocessing

To run the preprocessing pipeline, use the `run_preprocessing.py` script with a configuration `.yaml` file.

The configuration files should contain the following sections:

`data`: Paths to the input CBCT projections, CT images, metadata and output directory.

* `ct`: Path to CT images (can be a directory or a single file)
* `clinical_recon`: Path to reconstructed CBCT images (can be a dicom directory or a single file, for Elekta .SCAN files are recommended)
* `projections`: Path to CBCT projections directory
* `output`: Path to the output directory where preprocessed data will be saved
* `framesxml`: Path to the frames XML file (only for Elekta systems)
* `reconstruction_dir`: Path to the 'Reconstruction' directory saved with projections (only for Elekta systems)
* `scanxml`: Path to the Scan.xml file (only for Varian systems)
* `airscans`: Path to the directory containing relevant air scans (only for Varian systems)
* `scattercorxml`: Path to the scatter correction XML file (only for Varian systems)
* `calibration_dir`: Path to the entire calibration folder (only for Varian systems)

`general`: General settings such as center and vendor.

* `center`: Single character representing the center (e.g. A, B, C, ...)
* `vendor`: Elekta or Varian

`settings`: Specific preprocessing settings.

* `correct_orientation`: Whether to correct the orientation of the clinical/reconstructed images.

Some example configuration files are located in the `configs/` directory.

Furthermore, you can specify the device to use for preprocessing (e.g. `cpu` for CPU, `cuda:0` for GPU 0) and the preprocessing stage to run (1 or 2). Currently, only stage 1 is implemented and GPU acceleration is only used for reconstruction using RTK.

### Usage

To run stage 1 on the CPU, execute the following command:
```bash
python3 run_preprocessing.py -c ./configs/lmu_config.yaml -d 'cpu' -s 1 
```
To run stage 1 on GPU 0, execute the following command:
```bash
python3 run_preprocessing.py -c ./configs/lmu_config.yaml -d 'cuda:0' -s 1 
```

## Preprocessing Steps

**Stage1 consists of the following pre-processing steps**:
1. Load CBCT projections, clinically reconstructed CBCT, CT, and some metadata files necessary for reconstruction (e.g geometry).
2. Apply some basic preprocessing to the CBCT projections, e.g. i0 correction, scatter correction (Varian only).
3. Perform a basic CBCT reconstruction using RTK to test if the projections and geometry are correct. Apply some image orientation corrections if necessary and align the reconstructed CBCT with the CT images (no resampling/interpolation).
4. Extract relevant metadata from CT, CBCT and projections.
5. Generate an overview .png image showing CT, clinical CBCT and reconstructed CBCT slices side by side for visual inspection.
6. Save the raw projections (without any corrections applied), the clinically reconstructed CBCT, the CT image and metadata files required for reconstruction.

## Requirements

A dockerfile is provided to set up the required environment. The docker image can be built using the following command:

```bash
docker build -t cobra_preprocessing -f Dockerfile .
```

An examplary docker compose file is also provided to simplify running the docker container and mounting some directories. To run the docker container and start a bash shell use the following commands:

```bash
docker compose -f docker/docker-compose.yml up -d
docker exec -it cobra_preprocessing bash
```

If docker is not available, the required packages can be also installed locally using pip:

```bash
pip install -r requirements.txt
``` 