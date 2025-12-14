#!/usr/bin/env bash

# Exit on error
set -e

ENV_NAME="opencv_env"
PYTHON_VERSION="3.11"

echo "---------------------------------------------"
echo " Creating conda environment: $ENV_NAME"
echo "---------------------------------------------"

# Create environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate environment (bash)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "---------------------------------------------"
echo " Installing compilers (GCC/G++) from conda-forge"
echo "---------------------------------------------"

# GCC 12 is compatible with CUDA 12.x
conda install -y -c conda-forge compilers=2 gcc_linux-64=12 gxx_linux-64=12 libgcc-devel_linux-64=12

echo "---------------------------------------------"
echo " Installing PyTorch (GPU), NVCC, & Git"
echo "---------------------------------------------"

# 1. 'pytorch-cuda=12.1': Installs the CUDA runtime for PyTorch
# 2. 'cuda-nvcc': Installs the CUDA compiler (REQUIRED to build Detectron2 from source)
# 3. 'git': Required for the pip install git+ command
conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch \
    torchvision \
    pytorch-cuda=12.1 \
    cuda-nvcc \
    git \
    opencv

echo "---------------------------------------------"
echo " Installing build tools & Jupyter kernel"
echo "---------------------------------------------"

conda install -y -c conda-forge \
    ipython \
    ipykernel \
    scipy \
    pybind11 \
    matplotlib

echo "---------------------------------------------"
echo " Installing Pip Packages (building Detectron2)"
echo "---------------------------------------------"

# Ensure the build process uses the Conda-installed CUDA compiler
export CUDA_HOME=$CONDA_PREFIX

pip install \
    bm3d \
    pyiqa \
    fiftyone \
    ultralytics \
    'git+https://github.com/facebookresearch/detectron2.git'

echo "---------------------------------------------"
echo " Registering Jupyter kernel: $ENV_NAME"
echo "---------------------------------------------"

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "---------------------------------------------"
echo " DONE! Environment created and ready to use."
echo " To activate it:  conda activate $ENV_NAME"
echo "---------------------------------------------"