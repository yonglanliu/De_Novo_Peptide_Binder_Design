# Installation Guide for Peptide Design

This guide will walk you through setting up the complete environment for cyclic peptide design targeting your homotetramer.

## Prerequisites

- Linux or macOS (Windows users: use WSL2)
- At least 50GB free disk space
- GPU recommended (NVIDIA with CUDA) but not required
- Conda or Miniconda installed

### Install Conda (if not already installed)

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, then reload shell
source ~/.bashrc
```

## Installation Methods

Choose one:

### Method 1: Automated Setup (Recommended)

```bash
# Run the setup script
chmod +x setup_environment.sh
./setup_environment.sh
```

This will:
- Create conda environment
- Install all dependencies
- Clone RFDiffusion and ProteinMPNN
- Set up project structure
- Create activation script

### Method 2: Manual Setup (Step-by-Step)

If you prefer to install manually or the automated script fails:

#### Step 1: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n peptide_design python=3.10 -y
conda activate peptide_design
```

#### Step 2: Install Basic Dependencies

```bash
# Scientific computing
conda install -y numpy scipy pandas matplotlib seaborn -c conda-forge

# Bioinformatics
pip install biopython
```

#### Step 3: Install PyTorch

**For GPU (CUDA 11.8):**
```bash
conda install pytorch==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

**For CPU only:**
```bash
conda install pytorch cpuonly -c pytorch
```

**For different CUDA versions:**
Check https://pytorch.org/get-started/locally/

#### Step 4: Install RFDiffusion Dependencies

```bash
pip install hydra-core==1.3.2
pip install pyrsistent==0.19.3
pip install icecream
pip install prody
```

#### Step 5: Clone and Setup RFDiffusion

```bash
# Clone repository
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion


# Install
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../../
pip install -e .

# Download model weights
mkdir -p models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
cd ../..
```

#### Step 6: Clone ProteinMPNN

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
```

