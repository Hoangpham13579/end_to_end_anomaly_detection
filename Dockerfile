# # Jetpack 4.6.0 (Jetson Nano)
# FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3
# RUN echo "l4t-ml"

# L4T Pytorch 2.0 Jetpack 5.1.1 (Jetson Orin Nano)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
# RUN echo "l4t-pytorch"

ARG project_dir=/app
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR $project_dir
# ENTRYPOINT ["#!/bin/sh"]  # Comment this line when install with camera or Multi-threading

ADD main.py $project_dir
ADD Dockerfile $project_dir
ADD py2trt_i3d.py $project_dir
ADD py2trt_rtfm.py $project_dir
ADD temp.py $project_dir

ADD advanTech ${project_dir}/advanTech
ADD ckpt ${project_dir}/ckpt
ADD configs ${project_dir}/configs
ADD data ${project_dir}/data
ADD output ${project_dir}/output
ADD run_scripts ${project_dir}/run_scripts
ADD slowfast ${project_dir}/slowfast
ADD TensorRT ${project_dir}/TensorRT

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip
RUN pip3 uninstall -y torchaudio
RUN pip3 install --upgrade pip
RUN apt-get upgrade -y \
&& apt-get install -y \
    git \
    nano \
    cmake \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    gcc 
RUN pip3 install tqdm \
    simplejson \
    iopath \
    fvcore \
    psutil --ignore-installed


# BEGINING SCRIPT AFTER INSTALLING THE CODE ON JETSON ORIN NANO VIA DOCKER!!!!
# Check whether exist any docker container
# ...

# Increase Swap Memory to 4GB
# Jetson configration
# RUN sudo nvpmodel -n 0 \
#     && sudo jetson_clocks

# # Install bazel
# RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-arm64 \
#     && chmod +700 bazelisk-linux-arm64 \
#     && ./bazelisk-linux-arm64 \
#     && mv bazelisk-linux-arm64 bazel \
#     && mv bazel /usr/local/bin/

# # Install & setup Torch-TensorRT
# RUN echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
# # RUN source ~/.bashrc
# WORKDIR $project_dir/TensorRT
# RUN bazel build //:libtorchtrt --platforms //toolchains:jetpack_5.0
# WORKDIR $project_dir/TensorRT/py
# RUN python3 setup.py install --use-cxx11-abi --jetpack-version 5.0 
# WORKDIR $project_dir

# # Convert ResNet50 I3D Non-local to TensorRT
# RUN python3 py2trt_i3d.py --cfg configs/Convert_RTFM_I3D_slowfast.yaml

# # Convert RTFM to TensorRT
# RUN python3 py2trt_rtfm.py --cfg configs/Convert_RTFM_I3D_slowfast.yaml

# # Install dependencies for "advanTech" python folder
# ...

# # DOCKER Command to run the container on Jetson Orin Nano
# sudo docker run -i -t --runtime nvidia --network host --device /dev/video0:/dev/video1 --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unit:/tmp/.X11-unix:rw" [Docker Username]/rtfm_i3d-nonlocal_jetson:r35.1.0-pth1.11-py3-orinnano bash

# # DOCKER command to build the image on Jetson Orin Nano
# sudo docker buildx build --platform linux/arm64/v8 -f Dockerfile -t [Docker Username]/rtfm_i3d-nonlocal_jetson:r35.1.0-pth1.13-py3-orinnano --push .