FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        pkg-config \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG ITK_VERSION="v5.4.4"
ENV PREFIX=/opt

RUN git clone --branch ${ITK_VERSION} --depth 1 https://github.com/InsightSoftwareConsortium/ITK.git /tmp/ITK 

RUN mkdir -p /tmp/ITK-build && cd /tmp/ITK-build && \
    cmake /tmp/ITK \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${PREFIX}/itk \
        -DBUILD_SHARED_LIBS=ON \
        -DModule_RTK=ON \
        -DModule_CudaCommon=ON \
        -DITK_USE_CUDA=ON \
        -DRTK_USE_CUDA=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
        -DRTK_BUILD_APPLICATIONS=ON \
        -DRTK_PROBE_EACH_FILTER=ON &&\
    make -j"$(nproc)"

RUN cd /tmp/ITK-build && make install

ENV LD_LIBRARY_PATH=/opt/itk/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/itk/bin:${PATH}

WORKDIR /workspace
CMD ["bash"]

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

COPY --from=build /opt/itk /opt/itk
COPY requirements.txt ./
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt update
RUN apt install python3 python3-pip python3-venv -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment for RTK execution
ENV LD_LIBRARY_PATH=/opt/itk/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/itk/bin:${PATH}

WORKDIR /workspace
CMD ["bash"]