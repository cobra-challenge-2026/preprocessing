# required for nvcc 
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# CuPy needs to find the CUDA toolkit at runtime for JIT kernel compilation
ENV CUDA_PATH=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential software-properties-common libxcb1 libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.13 python3.13-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY docker/requirements.txt .
RUN uv pip install --system --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system --no-cache-dir -r requirements.txt

ARG USE_CUDA=true
ENV USE_CUDA=${USE_CUDA}

RUN if [ "$USE_CUDA" = "true" ]; then \
        uv pip install --system --no-cache-dir itk-rtk-cuda124 ; \
    else \
        uv pip install --system --no-cache-dir itk-rtk ; \
    fi

RUN uv pip uninstall --system -y SimpleITK SimpleITK-SimpleElastix simpleitk simpleitk-simpleelastix
RUN uv pip install --system --no-cache-dir SimpleITK-SimpleElastix
WORKDIR /workspace
COPY . .
CMD ["bash"]
