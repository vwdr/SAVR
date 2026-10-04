# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n specprune python=3.10 -y
conda activate specprune

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git # skip
cd SpecPrune-VLA/openvla-oft
pip install -e .
```

Note: when runing the model, the `modeling_llama.py` in the environment won't be executed. Local file `openvla-oft/prismatic/extern/hf/modeling_llama.py` will be executed instead. So you don't need to modifiy the package in your environment. 

```bash
# Install LIBERO, follow openvla-oft/LIBERO.md

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

