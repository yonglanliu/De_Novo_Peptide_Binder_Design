#!/usr/bin/env python3
"""
Thread (assign) multiple peptide-binder sequences onto a specific chain of a complex PDB using PyRosetta,
then write one output PDB per sequence.

- Input:  complex PDB (receptor + peptide binder as one chain)
- Input:  multi-FASTA with many candidate peptide sequences (top-ranked from ProteinMPNN, etc.)
- Output: PDBs with the binder chain mutated to each sequence (backbone coordinates unchanged)

IMPORTANT:
- This does NOT do packing/minimization. It only threads residues (sidechains will look "ugly" until you pack/relax).
- Sequence length must match the number of residues in the binder chain in the PDB.

Usage:
  python thread_binder_sequences.py \
    --pdb complex.pdb \
    --fasta top_sequences.fasta \
    --chain B \
    --outdir threaded_pdbs

Optional:
  --init "-beta_nov16 -mute all -use_terminal_residues true"
"""

import argparse
import os
import re
from typing import List, Tuple
from pathlib import Path

from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta import core


AA_1_TO_3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}

VALID_AA = set(AA_1_TO_3.keys())


def read_multifasta(path: str) -> List[Tuple[str, str]]:
    """Return list of (header, sequence). Header excludes leading '>'."""
    recs: List[Tuple[str, str]] = []
    header = None
    seq_lines: List[str] = []

    def flush():
        nonlocal header, seq_lines
        if header is not None:
            seq = "".join(seq_lines).replace(" ", "").replace("\t", "").upper()
            recs.append((header, seq))
        header = None
        seq_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                seq_lines.append(line)
    flush()
    return recs


def sanitize_name(s: str, maxlen: int = 80) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return (s[:maxlen] if len(s) > maxlen else s) or "seq"


def pose_positions_for_chain(pose, chain_id: str) -> List[int]:
    """Return pose residue indices (1-based) for residues whose PDB chain ID == chain_id."""
    pdbinfo = pose.pdb_info()
    if pdbinfo is None:
        raise RuntimeError("Pose has no PDBInfo; cannot map chain IDs. Load from PDB, not silent.")

    idxs = [i for i in range(1, pose.total_residue() + 1) if pdbinfo.chain(i) == chain_id]
    if not idxs:
        raise ValueError(f"No residues found for chain '{chain_id}'. Check chain IDs in your PDB.")
    return idxs


def thread_sequence_onto_chain(pose, chain_positions: List[int], seq: str) -> None:
    """Replace residue identities along chain_positions to match seq. Keeps backbone coordinates."""
    if len(seq) != len(chain_positions):
        raise ValueError(f"Sequence length {len(seq)} != chain length {len(chain_positions)}")

    # Validate AA alphabet
    bad = sorted(set([aa for aa in seq if aa not in VALID_AA]))
    if bad:
        raise ValueError(f"Sequence contains unsupported AA letters: {bad}. Only standard 20 AAs are allowed.")

    rsd_set = pose.residue_type_set_for_pose(core.chemical.FULL_ATOM_t)

    for aa, pose_idx in zip(seq, chain_positions):
        name3 = AA_1_TO_3[aa]
        new_res = core.conformation.ResidueFactory.create_residue(rsd_set.name_map(name3))
        # True => keep backbone, idealize sidechain geometry
        pose.replace_residue(pose_idx, new_res, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="Input complex PDB")
    ap.add_argument("--fasta", required=True, help="Multi-FASTA with binder sequences")
    ap.add_argument("--chain", required=True, help="Binder chain ID in the PDB (e.g., B)")
    ap.add_argument("--outdir", default="threaded_outputs", help="Output directory for threaded PDBs")
    ap.add_argument("--init", default="-beta_nov16 -mute all -use_terminal_residues true",
                    help="Extra PyRosetta init flags")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    init(args.init)

    base_pose = pose_from_pdb(args.pdb)
    chain_positions = pose_positions_for_chain(base_pose, args.chain)
    chain_len = len(chain_positions)

    records = read_multifasta(args.fasta)
    if not records:
        raise RuntimeError("No FASTA records found.")

    print(f"Loaded PDB: {args.pdb}")
    print(f"Binder chain: {args.chain} (length {chain_len} residues)")
    print(f"FASTA records: {len(records)}")

    ok = 0
    skipped = 0

    for i, (hdr, seq) in enumerate(records, start=1):
        try:
            pose = base_pose.clone()
            thread_sequence_onto_chain(pose, chain_positions, seq)

            tag = sanitize_name(hdr)
            outname = f"{os.path.splitext(os.path.basename(args.pdb))[0]}_chain{args.chain}_rank{i}.pdb"
            outpath = os.path.join(args.outdir, outname)

            pose.dump_pdb(outpath)
            ok += 1

        except Exception as e:
            skipped += 1
            print(f"[SKIP] record {i} ({hdr}): {e}")

    print(f"Done. Wrote {ok} PDB(s) to: {args.outdir}. Skipped: {skipped}.")


if __name__ == "__main__":
    main()
