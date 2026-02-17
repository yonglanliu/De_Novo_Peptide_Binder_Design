#!/bin/bash

# RFDiffusion script for peptide binder design
# Adjust parameters based on your specific target structure

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

# If run cluster and the cluster has already compiled the RFdiffusion model, you can use module load
# module purge
# module load RFdiffusion
conda activate rfdiffusion
cd ./RFdiffusion

# Input PDB (cleaned, protein only)
INPUT_PDB="../data/pdb/clean/example.pdb"

# Contig map
CONTIG="10-20 A20-500/0 B20-500/0"

# Checkpoint .pt
CHECKPOINT_PT="./models/Complex_base_ckpt.pt"

# Diffuser timesteps
T=50

# Design cyclic peptide or not
CYCLIC=False

# Inference code to run
RUN_INFER="./scripts/run_inference.py"

# Number of designs to generate
NUM_DESIGNS=50

# Output directory
OUTPUT_DIR="../results/rfdiffusion_aa_10-20"
mkdir -p "${OUTPUT_DIR}"

# Hotspot residues (optional)
HOTSPOT_RES="A300,A2300,B400"

# =============================================================================
# FIX: Writable schedules directory (avoids /opt/conda read-only error)
# =============================================================================
export HYDRA_FULL_ERROR=1

# =============================================================================
# GUIDING POTENTIALS (binder design)
# =============================================================================
GUIDE_SCALE=2.0
GUIDE_DECAY="quadratic"

POTENTIALS_STR="potentials.guiding_potentials=[
\"type:binder_ROG,weight:1.0\"
]"

# =============================================================================
# SANITY CHECKS
# =============================================================================
if [[ ! -f "$INPUT_PDB" ]]; then
  echo "ERROR: INPUT_PDB not found: $INPUT_PDB"
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PT" ]]; then
  echo "ERROR: CHECKPOINT_PT not found: $CHECKPOINT_PT"
  exit 1
fi

# =============================================================================
# LOGGING
# =============================================================================
echo "=== RFDiffusion Peptide Binder Design ==="
echo "Input PDB:    $INPUT_PDB"
echo "Contig map:   $CONTIG"
echo "Checkpoint:   $CHECKPOINT_PT"
echo "Designs:      $NUM_DESIGNS"
echo "Cyclic:       $CYCLIC"
if [[ -n "${HOTSPOT_RES}" ]]; then
  echo "Hotspots:     $HOTSPOT_RES"
else
  echo "Hotspots:     (none)"
fi
echo "Guide scale:  $GUIDE_SCALE"
echo "Guide decay:  $GUIDE_DECAY"
echo "Potentials:   $POTENTIALS_STR"
echo ""

# =============================================================================
# RUN RFDIFFUSION
# =============================================================================
CMD=(python-rfd "$RUN_INFER"
  "inference.input_pdb=$INPUT_PDB"
  "inference.output_prefix=$OUTPUT_DIR/design"
  "inference.ckpt_override_path=${CHECKPOINT_PT}"
  "contigmap.contigs=[$CONTIG]"
  "inference.num_designs=$NUM_DESIGNS"
  "denoiser.noise_scale_ca=0"
  "denoiser.noise_scale_frame=0"
  "diffuser.T=$T"
  "inference.cyclic=$CYCLIC"
  "$POTENTIALS_STR"
  "potentials.guide_scale=$GUIDE_SCALE"
  "potentials.guide_decay=$GUIDE_DECAY"
)


# Only add hotspots if provided
if [[ -n "${HOTSPOT_RES}" ]]; then
  CMD+=("ppi.hotspot_res=[$HOTSPOT_RES]")
fi

echo "Running RFDiffusion..."
printf '%q ' "${CMD[@]}"
echo -e "\n"

"${CMD[@]}"

echo ""
echo "=== RFDiffusion Complete ==="
echo "Designs saved to: $OUTPUT_DIR"
echo ""
