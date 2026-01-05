# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

# COPY --from=build /opt/itk /opt/itk
COPY requirements.txt ./
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt update
RUN apt install python3 python3-pip -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
CMD ["bash"]