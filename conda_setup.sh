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

conda install -y -c conda-forge compilers

echo "---------------------------------------------"
echo " Installing OpenCV from conda-forge"
echo "---------------------------------------------"

conda install -y -c conda-forge opencv

echo "---------------------------------------------"
echo " Installing build tools & Jupyter kernel"
echo "---------------------------------------------"

conda install -y -c conda-forge \
#    cmake \
#    make \
    ipython \
    ipykernel \
#    numpy \
    scipy \
    pybind11 \
    matplotlib

pip install \
    bm3d
echo "---------------------------------------------"
echo " Registering Jupyter kernel: $ENV_NAME"
echo "---------------------------------------------"

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "---------------------------------------------"
echo " DONE! Environment created and ready to use."
echo " To activate it:  conda activate $ENV_NAME"
echo "---------------------------------------------"
