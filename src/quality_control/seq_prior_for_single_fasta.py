import argparse
import logging
from pathlib import Path
from typing import Dict, List

from quality_control.seq_prioritization import read_fasta, write_fasta, parse_score, FastaRec


def setup_logger(verbosity: int = 0) -> logging.Logger:
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logger = logging.getLogger("seq_prioritization")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input FASTA from ProteinMPNN")
    ap.add_argument("-o", "--output", required=True, help="Output FASTA with top sequences")
    ap.add_argument("-n", "--top_n", type=int, default=50, help="Number of top sequences to keep")
    ap.add_argument("--dedup", action="store_true", help="Remove exact duplicate sequences (keep best score)")
    ap.add_argument("--min_len", type=int, default=1, help="Minimum sequence length to keep")
    ap.add_argument("--max_len", type=int, default=10**9, help="Maximum sequence length to keep")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)")
    args = ap.parse_args()

    logger = setup_logger(args.verbose)

    in_fa = Path(args.input)
    out_fa = Path(args.output)
    out_fa.parent.mkdir(parents=True, exist_ok=True)

    raw = read_fasta(in_fa)

    parsed: List[FastaRec] = []
    skipped_no_score = 0
    skipped_len = 0

    for h, s in raw:
        seqlen = len(s)
        if not (args.min_len <= seqlen <= args.max_len):
            skipped_len += 1
            continue

        sc = parse_score(h)
        if sc is None:
            skipped_no_score += 1
            continue

        parsed.append(FastaRec(header=h, seq=s, score=sc))

    if args.dedup:
        best_by_seq: Dict[str, FastaRec] = {}
        for r in parsed:
            prev = best_by_seq.get(r.seq)
            if prev is None or r.score < prev.score:
                best_by_seq[r.seq] = r
        parsed = list(best_by_seq.values())

    parsed.sort(key=lambda r: r.score)
    top = parsed[: max(0, args.top_n)]

    out: List[FastaRec] = []
    for idx, r in enumerate(top, start=1):
        new_header = f"rank={idx} score={r.score:.6f} | {r.header}"
        out.append(FastaRec(header=new_header, seq=r.seq, score=r.score))

    write_fasta(out_fa, out)

    # Summary (logger)
    logger.info("Processed %s", in_fa.name)
    logger.info("Read records: %d", len(raw))
    logger.info("Kept (scored) records: %d", len(parsed))
    if args.dedup:
        logger.info("After dedup: %d", len(parsed))
    logger.info("Wrote top N: %d -> %s", len(out), out_fa)

    if skipped_no_score:
        logger.warning("Skipped (no score in header): %d", skipped_no_score)
    if skipped_len:
        logger.warning("Skipped (length filter): %d", skipped_len)


if __name__ == "__main__":
    main()
