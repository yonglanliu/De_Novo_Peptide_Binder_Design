# De novo Peptide Binder Design

## Overview
This pipline is for *De Novo* design of peptides for a target. 
The target can be a monomer, dimer, or multimer.


## Step 0: Understand Your Structure (Optional)

Let's first make a folder to save log files

```bash
makedir -p logs
```

First, you need to know:
1. **Chain IDs** - What are your protein chains called? (Usually A, B, C, D)
2. **Ligand locations** - Which chain and residue number for each ligand?
3. **Which interface** - Does the pocket involve 1 chain, 2 chains (dimer), 3 chains (trimer), or 4 chains?

### Quick Analysis

```bash
# Analyze your structure
python scripts/analyze_pocket.py \
  --pdb ./data/pdb/raw/example.pdb \
  --chains A B C D \
  --ligands ligand_chain_ID:ligand_residue_number >& logs/analyze_pocket.out
```

This will tell you:
- Which chains form the binding pocket
- Whether you should use the full multimer or just a subset
- Recommended peptide length

## Step 1: Prepare Structure

```bash
# Clean the structure and identify binding residues
python scripts/prepare_structure.py \
  --input_pdb ./data/pdb/raw/example.pdb \
  --output_pdb ./data/pdb/clean/example_clean.pdb \
  --ligands ligand_chain_ID:ligand_residue_number  \
  --keep_chains A B \
  --distance_cutoff 3.2 >& logs/prepare_structure.out
```

**Note the output** - it will show:
- Chain lengths
- Binding pocket residues
- Hotspot residues to use in RFDiffusion

## Step 2: Detailed Pocket Analysis

```bash
# Get comprehensive pocket analysis
python scripts/pocket_visual.py \
  --pdb ./data/pdb/raw/example.pdb \
  --ligands ligand_chain_ID:ligand_residue_number \
  --output_report logs/pocket_report.txt \
  --output_pymol pymol_visual/visualize.pml

# Visualize in PyMOL
pymol ./pymol_visual/visualize.pml
```
This generates:
- Pocket dimensions
- Recommended peptide length
- PyMOL visualization

## Step 3: Configure RFDiffusion

Edit `./bash/run_rfdiffusion.sh`:

```bash
#!/bin/bash
# RFDiffusion script for peptide binder design
# Adjust parameters based on your specific target structure

# Activate RFdiffusion environment
source activate rfdiffusion
cd ./RFdiffusion

# =============================================================================
# CONFIGURATION - EDIT THESE
INPUT_PDB="./data/pdb/clean/example_clean.pdb "

# 4-8: design a peptide containing 10-20 residues; A20-500/0: residue range 20-500 in chain A; C20-500/0: residue range 20-500 in chain C
# It means, you want to keep chain A and C and design a peptide with a length range of 20-20
CONTIG="10-20 A20-500/0 C20-500/0"

# Checkpoint .pt of RFdiffusion model
CHECKPOINT_PT="./models/Complex_base_ckpt.pt"

# Diffuser timesteps
T=50

# Design cyclic peptide or not
CYCLIC=False

# Inference code to run in RFdiffusion package
RUN_INFER="./scripts/run_inference.py"

# Number of designs to generate
NUM_DESIGNS=50

# Output directory
OUTPUT_DIR="../results/rfdiffusion_aa_10-20"
mkdir -p "${OUTPUT_DIR}"

# Hotspot residues (optional)
HOTSPOT_RES="A300,A489,B200" # Officially recommend 3-6 hotspots
```


### Run RFDiffusion

```bash
source bash/run_rfdiffusion.sh
```

## Step 4: Filter for Cyclization (if design cyclic peptide)

```bash
python scripts/filter_cyclic.py \
  --input_dir results/rfdiffusion_pdb \
  --output_dir results/filtered_pdb \
  --max_distance 4 \
  --min_distance 2.5
```

## Step 5: Design Sequences with ProteinMPNN

Edit `run_proteinmpnn.sh`:

```bash

folder_with_pdbs="./results/rfdiffusion_pdb"
output_dir="./results/proteinmpnn_sequences"

# Chain to design (peptide)
CHAIN_TO_DESIGN="B"
```

### Run ProteinMPNN

```bash
./bash/run_proteinmpnn.sh
```

