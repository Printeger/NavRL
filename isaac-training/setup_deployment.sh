#!/bin/bash

# Exit immediately if a command fails
set -e

# Define environment name
ENV_NAME="NavRL"

# Load Conda environment handling
eval "$(conda shell.bash hook)"


# Step 1: Create conda env with python3.10
echo "Setting up conda env..."
conda create -n $ENV_NAME python=3.8
# conda create -n $ENV_NAME
conda activate $ENV_NAME
# pip install numpy==1.26.4
# pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip3 install "pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2"
pip3 install imageio-ffmpeg==0.4.9
pip3 install moviepy==1.0.3
pip3 install hydra-core --upgrade
pip3 install einops
pip3 install pyyaml
pip3 install rospkg
pip3 install matplotlib

# Step 2: Install TensorDict and dependencies
echo "Installing TensorDict dependencies..."
pip3 uninstall -y tensordict
pip3 uninstall -y tensordict
pip3 install tomli  # If missing 'tomli'
cd ./third_party/tensordict
python3 setup.py develop --user


# Step 3: Install TorchRL
echo "Installing TorchRL..."
cd ../rl
python3 setup.py develop --user

# Check which torch is being used
python3 -c "import torch; print(torch.__path__)"

echo "Setup completed successfully!"