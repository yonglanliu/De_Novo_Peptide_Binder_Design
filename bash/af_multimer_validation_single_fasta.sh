#!/bin/bash
#SBATCH --job-name=5F91_af
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:128
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=100:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# If run HPC
# module purge
# module load alphafold2/2.3.2
conda activate alphafold

# ---- User inputs ----
FINAL_OUT_DIR="./results/AF_predict"
FASTA_PATH="/results/AF_fasta/example.fa"
NUM_PRED_PER_MODEL=5 
MAX_TEMPLATE_DATE="2021-11-01"

# ---- Threading / oversubscription control ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- Use fast local scratch for heavy I/O ----
SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "${SCRATCH}/af_out"
mkdir -p "${FINAL_OUT_DIR}"

# Keep a stable target name (AF uses fasta basename to create subdir)
FASTA_BASENAME="$(basename "${FASTA_PATH}")"
SCRATCH_FASTA="${SCRATCH}/${FASTA_BASENAME}"
cp -f "${FASTA_PATH}" "${SCRATCH_FASTA}"

echo "Running AlphaFold from scratch: ${SCRATCH}"
echo "Final output dir: ${FINAL_OUT_DIR}"
echo "FASTA: ${SCRATCH_FASTA}"

# ---- If you expect precomputed features, verify they exist in FINAL_OUT_DIR ----
# AlphaFold only skips searches if it finds features.pkl under:
#   <output_dir>/<target_name>/features.pkl
TARGET_NAME="${FASTA_BASENAME%.*}"
FEATURES_PKL="${FINAL_OUT_DIR}/${TARGET_NAME}/features.pkl"

if [[ -f "${FEATURES_PKL}" ]]; then
  echo "Found precomputed features.pkl: ${FEATURES_PKL}"
else
  echo "WARNING: features.pkl not found at ${FEATURES_PKL}"
  echo "AlphaFold may re-run MSA searches even with --use_precomputed_msas."
fi

# ---- Run ----
run_singularity \
  --model_preset=multimer \
  --fasta_paths="${SCRATCH_FASTA}" \
  --max_template_date="${MAX_TEMPLATE_DATE}" \
  --num_multimer_predictions_per_model "${NUM_PRED_PER_MODEL}" \
  --use_precomputed_msas \
  --output_dir="${SCRATCH}/af_out" \
  --models_to_relax=none

# ---- Copy results back ----
# If you want to preserve the per-target subdir structure:
rsync -av "${SCRATCH}/af_out/" "${FINAL_OUT_DIR}/"

echo "Done. Results copied to ${FINAL_OUT_DIR}"
