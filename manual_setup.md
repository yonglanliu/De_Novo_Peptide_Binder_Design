# Installation Guide for Cyclic Peptide Design

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

#### Step 7: Install ColabFold (Optional)

**Option A: For local predictions (requires ~2.2TB databases)**
```bash
pip install "colabfold[alphafold]@git+https://github.com/sokrypton/ColabFold"

# Setup databases (this takes a while!)
mkdir -p /path/to/databases
colabfold_setup_databases /path/to/databases
```

**Option B: Use ColabFold web interface (Recommended for beginners)**
- No local installation needed
- Go to: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
- Upload your FASTA files directly

#### Step 8: Create Project Structure

```bash
mkdir -p data/{raw_pdbs,cleaned_pdbs}
mkdir -p results/{rfdiffusion,filtered,sequences,af2_validation}
mkdir -p scripts
mkdir -p logs
```

#### Step 9: Create Activation Script

Create `activate_env.sh`:

```bash
cat > activate_env.sh << 'EOF'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cyclic_peptide_design

export RFDIFFUSION_PATH="$(pwd)/RFdiffusion"
export PROTEINMPNN_PATH="$(pwd)/ProteinMPNN"
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${RFDIFFUSION_PATH}:${PROTEINMPNN_PATH}:${PYTHONPATH}"

echo "✓ Environment activated"
EOF

chmod +x activate_env.sh
```

## Verify Installation

```bash
# Activate environment
source activate_env.sh

# Test imports
python << 'EOF'
import numpy
import scipy
import pandas
import Bio
import torch
print("✓ All basic packages imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```

## Troubleshooting

### Issue: PyTorch CUDA not working

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda install pytorch pytorch-cuda=XX.X -c pytorch -c nvidia
```

### Issue: RFDiffusion import errors

```bash
cd RFdiffusion
pip install -e . --force-reinstall
```

### Issue: Out of memory during RFDiffusion

- Reduce number of designs: `NUM_DESIGNS=100`
- Use CPU instead of GPU (slower): Set `CUDA_VISIBLE_DEVICES=""`
- Use smaller chain subset instead of full tetramer

### Issue: ProteinMPNN not found

```bash
# Check path
ls ProteinMPNN/protein_mpnn_run.py

# If not found, re-clone
git clone https://github.com/dauparas/ProteinMPNN.git
```

## Directory Structure After Setup

```
cyclic_peptide_design/
├── activate_env.sh
├── setup_environment.sh
├── environment_info.txt
├── requirements.txt
├── HOMOTETRAMER_QUICKSTART.md
├── INSTALLATION_GUIDE.md
├── data/
│   ├── raw_pdbs/
│   └── cleaned_pdbs/
├── results/
│   ├── rfdiffusion/
│   ├── filtered/
│   ├── sequences/
│   └── af2_validation/
├── scripts/
│   ├── prepare_structure.py
│   ├── analyze_homotetramer_pockets.py
│   ├── analyze_multi_ligand_pocket.py
│   ├── filter_for_cyclization.py
│   ├── prepare_af2_validation.py
│   ├── run_rfdiffusion_cyclic.sh
│   ├── run_proteinmpnn.sh
│   └── complete_pipeline.sh
├── RFdiffusion/
└── ProteinMPNN/
```

## Quick Test Run

After installation, test with a small run:

```bash
# Activate
source activate_env.sh

# Copy your PDB
cp /path/to/your/structure.pdb data/raw_pdbs/

# Analyze (replace with your chain IDs and ligands)
python scripts/analyze_homotetramer_pockets.py \
  --pdb data/raw_pdbs/structure.pdb \
  --chains A B C D \
  --ligands E:1 F:1

# Test with small run
cd scripts
# Edit run_rfdiffusion_cyclic.sh: set NUM_DESIGNS=10
./run_rfdiffusion_cyclic.sh
```

## Minimum Install (Analysis Only)

If you only want to analyze structures without running RFDiffusion:

```bash
conda create -n peptide_analysis python=3.10 -y
conda activate peptide_analysis
pip install biopython numpy scipy matplotlib
```

This allows you to run:
- `prepare_structure.py`
- `analyze_homotetramer_pockets.py`
- `analyze_multi_ligand_pocket.py`

## Next Steps

1. ✓ Environment installed
2. → Read `HOMOTETRAMER_QUICKSTART.md`
3. → Analyze your structure
4. → Run design pipeline
5. → Validate with AlphaFold2

## Resources

- **RFDiffusion:** https://github.com/RosettaCommons/RFdiffusion
- **ProteinMPNN:** https://github.com/dauparas/ProteinMPNN
- **ColabFold:** https://github.com/sokrypton/ColabFold
- **PyTorch:** https://pytorch.org/

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed: `conda list`
3. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`
4. Try with CPU-only version first
5. Start with smaller test runs (10 designs)

Good luck with your cyclic peptide design!
