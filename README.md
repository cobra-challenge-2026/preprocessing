# COBRA2026 Preprocessing Pipeline

This preprocessing pipeline prepares CBCT projections and corresponding CT images for the COBRA2026 reconstruction dataset/challenge.

Currently preprocessing for Elekta and Varian CBCT systems is supported.

## Running the Preprocessing

To run the preprocessing pipeline, use the `run_preprocessing.py` script with a configuration `.yaml` file. Details about the configuration options and format can be found here: [Configuration Options](configs/README.md).

An example script to generate a configuration file can be found here: [LMU Config Generator](configs/generate_config.py).

Furthermore, you can specify the device to use for preprocessing (e.g. `cpu` for CPU, `cuda:0` for GPU 0) and the preprocessing stage to run (1 or 2). Currently, only stage 1 is implemented and GPU acceleration is only used for CBCT reconstruction using RTK.

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

In the docker directory a `Dockerfile` and `docker-compose.yaml` file are provided to set up the required environment. The docker image can be built using the following command:

```bash
cd docker
docker compose build
```

To start a docker container modify the `docker-compose.yml` to mount your data directory and then run the following command within the docker directory:

```bash
docker compose up -d
docker exec -it cobra_preprocessing bash
```

If docker is not available, the required packages can be also installed locally using pip:

```bash
pip install -r requirements.txt
``` 