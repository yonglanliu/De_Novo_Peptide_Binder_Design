#!/usr/bin/env bash
#SBATCH --job-name=sample
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --mem=100g
#SBATCH --gres=gpu:p100:1,lscratch:100
#SBATCH -p gpu

# Load conda
source ~/bin/myconda
conda activate peptide_design

# Input pdb directory
# Peptide chain is the first chain (A), target chains can be B, C, D, ....
# It is better to remove cap residues, ACE, NME, or NMA
input_pdb="input.pdb"

# Extra parameters to be added
# If there is a residue that not included the rosetta database, you need to custom the parameters for that residue
extra_param="1.params,2.params,3.params"

# Working directory
outdir="outdir"
mkdir -p ${outdir}

# Anchor rosetta residue number. 
# For example, if you have two anchor residues, after extending one residue at the N-terminal and one residue at the C-termineal, 
# the residue number of that two anchors turns to "2,3"
anchors="2,3"

# Number of extend residues at the two side of anchors
# "1" indicates you are gonna extend 1 residue at the N-terminal and one residue at the C-terminal
# The N-terminal extended residue number is 1 and the C-terminal extended residue is 4 (with 2 residue anchors)
num_res=1

# residue number of the mutate residue to check if different score and metrics
mut=1 

# chain id of peptide. Recommend put the peptide as the first chain, for example, chain A
chain_id=1

python src/rosetta/sample_interface_residue.py \
        -i ${input_pdb} \
        -o ${outdir} \
        -n 1 -m 1 -c 1 \
        -ep ${extra_param} \
        -ar ${anchors}
