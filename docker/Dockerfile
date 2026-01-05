# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt update
RUN apt install python3 python3-pip -y
RUN apt install cmake -y
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
COPY . .
CMD ["bash"]
