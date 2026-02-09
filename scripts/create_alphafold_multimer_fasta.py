import argparse
from pathlib import Path
from typing import List, Optional, Iterable
import logging

from Bio import PDB, SeqIO
from Bio.PDB.Polypeptide import PPBuilder
import re

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# shared 3-letter -> 1-letter map
AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}


def extract_sequence_from_pdb(pdb_path: Path, chain_id: str, raise_on_unknown: bool = False) -> Optional[str]:
    """
    Extract a single-chain amino-acid sequence (one-letter) from a PDB file.
    Returns None if chain not found.
    Uses PPBuilder to build polypeptides (handles breaks).
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    # use first model by default
    model = next(iter(structure), None)
    if model is None:
        logger.warning("No model found in %s", pdb_path)
        return None

    # find requested chain
    chain = model[chain_id] if chain_id in model else None
    if chain is None:
        logger.warning("Chain %r not found in %s", chain_id, pdb_path)
        return None

    # build peptides (handles breaks / missing atoms)
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:  # empty chain or only HETATMs
        logger.warning("No polypeptides found for chain %r in %s", chain_id, pdb_path)
        return None

    # concatenate peptide fragments
    seq_str = "".join(str(p.get_sequence()) for p in peptides)
    # Ensure standard residues only (optional remapping if needed)
    # Convert three-letter map only if you had 3-letter residues; PPBuilder already gives one-letter

    # Optionally verify characters
    invalid = set(seq_str) - set("ACDEFGHIKLMNPQRSTVWY")
    if invalid:
        logger.warning("Non-standard residues detected in chain %r: %s", chain_id, invalid)
        if raise_on_unknown:
            raise ValueError(f"Non-standard residues {invalid} in {pdb_path}:{chain_id}")

    return seq_str


def create_protein_fasta(seqs: Iterable[Optional[str]], chain_names: Iterable[str]) -> str:
    """
    Create a multi-FASTA string from a list of sequences and chain names.
    Filters out None sequences.
    """
    entries = []
    for seq, name in zip(seqs, chain_names):
        if seq is None:
            logger.warning("Skipping chain %r because sequence is None", name)
            continue
        entries.append(f">protein_chain_{name}\n{seq}\n")
    return "".join(entries)

# -------------------------------------------
# Helper Function to parse ProteinMPNN score
# -------------------------------------------
_SCORE_RE = re.compile(r"\bscore=([-\d\.eE]+)")

def parse_mpnn_score(desc: str) -> Optional[float]:
    """Parse ProteinMPNN 'score=' value from FASTA description line."""
    if not desc:
        return None
    m = _SCORE_RE.search(desc)
    return float(m.group(1)) if m else None


def create_af_multimer_fasta(
    protein_fa: str,
    peptide_fasta_path: Path,
    outdir: Path,
    score_threshold: Optional[float] = None,
) -> List[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(str(peptide_fasta_path), "fasta"))
    if not records:
        logger.warning("No records found in %s", peptide_fasta_path)
        return []

    created_files: List[Path] = []
    design_id = records[0].id.replace(",", "")

    kept = 0
    skipped = 0
    missing = 0

    for i, r in enumerate(records, start=0):
        if i == 0:
            continue  # keep your original behavior

        score = parse_mpnn_score(r.description)
        if score is None:
            missing += 1
            # conservative default: skip records without score if filtering is on
            if score_threshold is not None:
                continue
        if score_threshold is not None and score is not None and score >= score_threshold:
            skipped += 1
            continue

        out_name = f"{design_id}_complex_pep_{i}.fa"
        out_path = outdir / out_name
        with out_path.open("w") as fh:
            fh.write(protein_fa)
            fh.write(f">peptide_chain_P, design_id={design_id}, peptide_id={i}, length={len(r.seq)}, score={score}\n{str(r.seq)}\n")

        created_files.append(out_path)
        kept += 1

    logger.info(
        "Processed %s | kept=%d skipped=%d missing_score=%d threshold=%s",
        peptide_fasta_path.name, kept, skipped, missing, score_threshold
    )
    return created_files


def create_af_fasta_for_single_peptide_fa(pdb_path: Path, pdb_chain_list: List[str],
                                          peptide_fasta_file_path: Path, outdir: Path,
                                          score_threshold: Optional[float] = None) -> List[Path]:
    """
    Extract sequences from pdb_path for all chains in pdb_chain_list,
    create receptor FASTA and then create AF multimer FASTA files for peptides in peptide_fasta_file_path.
    Returns list of created files.
    """
    seqs = [extract_sequence_from_pdb(Path(pdb_path), c) for c in pdb_chain_list]
    protein_fa = create_protein_fasta(seqs, pdb_chain_list)
    return create_af_multimer_fasta(protein_fa, Path(peptide_fasta_file_path), Path(outdir),
                                    score_threshold=score_threshold)


def create_af_fasta_for_all_fasta_in_dir(pdb_path: Path, pdb_chain_list: List[str],
                                         indir: Path, outdir: Path, recursive: bool = False,
                                         score_threshold: Optional[float] = None) -> List[Path]:
    """
    Iterate over all .fa/.fasta files in indir and run the single-file creator.
    Returns list of all created files across all inputs.
    """
    indir = Path(indir)
    patterns = ["*.fa", "*.fasta"]
    files = []
    for pat in patterns:
        files.extend(indir.rglob(pat) if recursive else indir.glob(pat))

    all_created = []
    for fa in sorted(files):
        created = create_af_fasta_for_single_peptide_fa(
            Path(pdb_path), pdb_chain_list, fa, Path(outdir),
            score_threshold=score_threshold
        )
        all_created.extend(created)
    return all_created

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AlphaFold-Multimer FASTA files from a receptor PDB and peptide FASTAs"
    )

    parser.add_argument(
        "--pdb",
        required=True,
        type=Path,
        help="Receptor PDB file"
    )

    parser.add_argument(
        "--chains",
        required=True,
        nargs="+",
        help="Chain IDs to extract from the PDB (e.g. A B)"
    )

    parser.add_argument(
        "--indir",
        required=True,
        type=Path,
        help="Directory containing peptide FASTA files (.fa or .fasta)"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for AlphaFold FASTA files"
    )
    
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Keep only peptide records with ProteinMPNN score < threshold (e.g. 0.9). "
             "If unset, keep all."
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for FASTA files in indir"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if non-standard amino acids are found in the PDB"
    )

    return parser.parse_args()

# Example usage:
if __name__ == "__main__":
    args = parse_args()

    # optional strict behavior
    def _extract(chain):
        return extract_sequence_from_pdb(
            args.pdb, chain, raise_on_unknown=args.strict
        )

    seqs = [_extract(c) for c in args.chains]
    protein_fa = create_protein_fasta(seqs, args.chains)

    create_af_fasta_for_all_fasta_in_dir(
        pdb_path=args.pdb,
        pdb_chain_list=args.chains,
        indir=args.indir,
        outdir=args.outdir,
        recursive=args.recursive,
        score_threshold=args.score_threshold,
    )


