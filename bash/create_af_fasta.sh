#!/usr/bin/env bash

PDB_PATH="./data/pdb/clean/example_clean.pdb"
CHAINS_IN_PDB="A B C D"   # space-separated
INPUT_SEQ_DIR="./results/proteinmpnn_sequences/seqs"
OUTPUT_SEQ_DIR="./results/AF_fasta"
RUN_SCRIPT="./scripts/create_alphafold_multimer_fasta.py"

mkdir -p "$OUTPUT_SEQ_DIR"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Error: run script not found: $RUN_SCRIPT" >&2
  exit 2
fi

python "$RUN_SCRIPT" \
  --pdb "$PDB_PATH" \
  --chains $CHAINS_IN_PDB \
  --indir "$INPUT_SEQ_DIR" \
  --outdir "$OUTPUT_SEQ_DIR" \
  --score_threshold 0.7
