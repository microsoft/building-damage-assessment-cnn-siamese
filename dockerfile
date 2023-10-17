# syntax=docker/dockerfile:1
FROM nvidia/cuda-arm64:11.1.1-cudnn8-devel-ubuntu18.04

# Fetch cuda signing key
# See https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/3bf863cc.pub

# Miniconda archive to install
ARG miniconda_version="4.9.2"
ARG miniconda_checksum="b6fbba97d7cef35ebee8739536752cd8b8b414f88e237146b11ebf081c44618f"
ARG conda_version="4.9.2"
ARG PYTHON_VERSION=default

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

ENV MINICONDA_VERSION="${miniconda_version}" \
    CONDA_VERSION="${conda_version}"

# General OS dependencies 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    wget \
    apt-utils \
    unzip \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    unattended-upgrades \
    run-one \
    nano \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda installation
WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_${MINICONDA_VERSION}-Linux-aarch64.sh && \
    echo "${miniconda_checksum} *Miniconda3-py38_${MINICONDA_VERSION}-Linux-aarch64.sh" | shasum -a 256 - && \
    /bin/bash Miniconda3-py38_${MINICONDA_VERSION}-Linux-aarch64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py38_${MINICONDA_VERSION}-Linux-aarch64.sh && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes "conda=${CONDA_VERSION}" && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn

# Inference code
WORKDIR /inference_files
COPY . /inference_files/

RUN /opt/conda/bin/conda init bash
RUN conda env create --file environment.yml

ENTRYPOINT ["/opt/conda/envs/nlrc/bin/python", "./inference/inference.py"]
