from typing import Dict, List, Tuple, Sequence, Optional
from alphafold.common import residue_constants
import numpy as np
import os
import collections
from collections import OrderedDict

import jax
import jax.numpy as jnp

from pyrosetta import init, Pose, pose_from_pdb, pose_from_file
from rosetta import core

# -----------------------------
# AF2/Rosetta utility functions
# -----------------------------
def pose_chain_ranges(pose: Pose) -> Dict[int, Tuple[int, int]]:
    """Rosetta chain index (1..n) -> (start_resi, end_resi), 1-indexed residue numbers."""
    conf = pose.conformation()
    n = pose.num_chains()
    ranges: Dict[int, Tuple[int, int]] = {}
    start = 1
    for ch in range(1, n + 1):
        end = conf.chain_end(ch)
        ranges[ch] = (start, end)
        start = end + 1
    return ranges


def pose_chain_letters(pose: Pose) -> Dict[int, str]:
    """Rosetta chain index (1..n) -> PDB chain letter (e.g., 'A','B','C')."""
    info = pose.pdb_info()
    if info is None:
        # Fallback if no PDBInfo: assign A,B,C,...
        letters = {}
        for ch in range(1, pose.num_chains() + 1):
            letters[ch] = chr(ord("A") + ch - 1)
        return letters

    letters: Dict[int, str] = {}
    ranges = pose_chain_ranges(pose)
    for ch, (s, _) in ranges.items():
        letters[ch] = info.chain(s)
    return letters


def build_residue_mask_by_chain(
    pose: Pose,
    fixed_chains: Sequence[str],
    design_chains: Sequence[str],
) -> List[bool]:
    """
    residue_mask length L:
      True  -> templated (fixed)
      False -> not templated (free / designed)
    """
    L = pose.size()
    residue_mask = [False] * L  # default free unless fixed

    fixed_set = set([c.strip() for c in fixed_chains if c.strip()])
    design_set = set([c.strip() for c in design_chains if c.strip()])

    ranges = pose_chain_ranges(pose) # {1: (1, 13), 2: (14, 470), 3: (471, 927)}
    letters = pose_chain_letters(pose) # {1: 'B', 2: 'A', 3: 'C'}

    for ch, (s, e) in ranges.items():
        letter = letters[ch]
        if letter in fixed_set:
            for resi in range(s, e + 1):
                residue_mask[resi - 1] = True
        if letter in design_set:
            for resi in range(s, e + 1):
                residue_mask[resi - 1] = False  # override to free

    return residue_mask


def build_target_mask_by_chain(pose: Pose, target_chains: Sequence[str]) -> np.ndarray:
    """Boolean mask length L where True indicates target residues."""
    L = pose.size()
    tmask = np.zeros(L, dtype=bool)
    target_set = set([c.strip() for c in target_chains if c.strip()])

    ranges = pose_chain_ranges(pose)
    letters = pose_chain_letters(pose)
    for ch, (s, e) in ranges.items():
        if letters[ch] in target_set:
            tmask[s - 1 : e] = True
    return tmask


def chainbreak_indices_zero_based(pose: Pose) -> List[int]:
    """0-indexed break positions to insert AFTER each chain end (excluding last chain)."""
    conf = pose.conformation()
    n = pose.num_chains()
    breaks = []
    for ch in range(1, n):  # no break after last chain
        endpos = conf.chain_end(ch)
        breaks.append(endpos - 1)  # convert to 0-index boundary index
    return breaks


def insert_truncations(residue_index: np.ndarray, break_indices: Sequence[int]) -> np.ndarray:
    """Add +200 offset to residue_index after each break index."""
    idx_res = residue_index.copy()
    for break_i in break_indices:
        idx_res[break_i + 1 :] += 200  # break after break_i
    return idx_res


def generate_template_features(
    seq: str,
    all_atom_positions: np.ndarray,  # (L,37,3)
    all_atom_masks: np.ndarray,      # (L,37)
    residue_mask: Sequence[bool],
    confidence_value: int = 9,
) -> Dict[str, np.ndarray]:
    """
    Build AF2 template dict for 1 template.
    residue_mask[i]=True -> templated; False -> free.
    """
    L = len(seq)
    if all_atom_positions.shape != (L, residue_constants.atom_type_num, 3):
        raise ValueError(f"all_atom_positions shape {all_atom_positions.shape} != (L,37,3) with L={L}")
    if all_atom_masks.shape != (L, residue_constants.atom_type_num):
        raise ValueError(f"all_atom_masks shape {all_atom_masks.shape} != (L,37) with L={L}")
    if len(residue_mask) != L:
        raise ValueError(f"residue_mask len {len(residue_mask)} != L={L}")

    keep = np.asarray(residue_mask, dtype=bool)

    templ_pos = np.zeros((L, residue_constants.atom_type_num, 3), dtype=np.float32)
    templ_msk = np.zeros((L, residue_constants.atom_type_num), dtype=np.float32)

    templ_seq_chars = np.full((L,), "-", dtype="<U1")
    templ_seq_chars[keep] = np.array(list(seq), dtype="<U1")[keep]
    templ_seq = "".join(templ_seq_chars.tolist())

    conf_scores = np.full((L,), -1, dtype=np.int32)
    conf_scores[keep] = int(confidence_value)

    templ_pos[keep, :, :] = all_atom_positions[keep, :, :]
    templ_msk[keep, :] = all_atom_masks[keep, :]

    templ_aatype = residue_constants.sequence_to_onehot(templ_seq, residue_constants.HHBLITS_AA_TO_ID)

    return {
        "template_all_atom_positions": templ_pos[None, ...],      # (1,L,37,3)
        "template_all_atom_masks": templ_msk[None, ...],          # (1,L,37)
        "template_sequence": [templ_seq.encode()],
        "template_aatype": templ_aatype[None, ...],              # (1,L,22)
        "template_confidence_scores": conf_scores[None, ...],     # (1,L)
        "template_domain_names": [b"none"],
        "template_release_date": [b"none"],
    }


def parse_initial_guess(all_atom_positions: np.ndarray) -> jnp.ndarray:
    """Convert numpy (L,37,3) to JAX (L,37,3) for initial guess."""
    return jnp.array(all_atom_positions, dtype=jnp.float32)


def af2_get_atom_positions_from_pose(pose: Pose, tmp_fn: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dump pose -> PDB -> parse AF2 atom arrays.
    Returns:
      all_positions      (L,37,3)
      all_positions_mask (L,37)
    """
    pose.dump_pdb(tmp_fn)
    with open(tmp_fn, "r") as f:
        lines = f.readlines()
    os.remove(tmp_fn)

    # Residue numbers in order of CA appearance
    idx_s = [int(l[22:26]) for l in lines if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    num_res = len(idx_s)

    all_positions = np.zeros((num_res, residue_constants.atom_type_num, 3), dtype=np.float32)
    all_masks = np.zeros((num_res, residue_constants.atom_type_num), dtype=np.int64)

    residues = collections.defaultdict(list)
    for l in lines:
        if not l.startswith("ATOM"):
            continue
        resNo = int(l[22:26])
        atom_name = l[12:16].strip()
        resname = l[17:20].strip()
        x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        residues[resNo].append((atom_name, resname, (x, y, z)))

    for resNo, atom_list in residues.items():
        pos = np.zeros((residue_constants.atom_type_num, 3), dtype=np.float32)
        msk = np.zeros((residue_constants.atom_type_num,), dtype=np.float32)

        for atom_name, resname, (x, y, z) in atom_list:
            if atom_name in residue_constants.atom_order:
                aidx = residue_constants.atom_order[atom_name]
                pos[aidx] = (x, y, z)
                msk[aidx] = 1.0
            elif atom_name.upper() == "SE" and resname == "MSE":
                # Put Selenium into SD slot
                aidx = residue_constants.atom_order["SD"]
                pos[aidx] = (x, y, z)
                msk[aidx] = 1.0

        try:
            idx = idx_s.index(resNo)
        except ValueError:
            continue
        all_positions[idx] = pos
        all_masks[idx] = msk.astype(np.int64)

    return all_positions, all_masks


def check_residue_distances(
    all_positions: np.ndarray,
    all_masks: np.ndarray,
    max_amide_distance: float,
) -> List[int]:
    """
    Detect large gaps between prev C and current N; return break indices (0-indexed at current residue).
    If break at i, it means insert chainbreak before residue i (i.e., between i-1 and i).
    """
    breaks: List[int] = []
    c_idx = residue_constants.atom_order["C"]
    n_idx = residue_constants.atom_order["N"]

    prev_is_unmasked = False
    prev_c = None

    for i, (coords, mask) in enumerate(zip(all_positions, all_masks)):
        this_is_unmasked = bool(mask[c_idx]) and bool(mask[n_idx])
        if this_is_unmasked:
            this_n = coords[n_idx]
            if prev_is_unmasked and prev_c is not None:
                dist = np.linalg.norm(this_n - prev_c)
                if dist > max_amide_distance:
                    breaks.append(i)  # break before residue i
            prev_c = coords[c_idx]
        prev_is_unmasked = this_is_unmasked

    # Convert "break before i" into "break after i-1" indices:
    # insert_truncations expects break-after indices.
    break_after = sorted(set([i - 1 for i in breaks if i > 0]))
    return break_after


def subset_rmsd(
    xyz1: np.ndarray,
    align1: np.ndarray,
    calc1: np.ndarray,
    xyz2: np.ndarray,
    align2: np.ndarray,
    calc2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Kabsch alignment on align*, RMSD on calc*."""
    assert xyz1[align1].shape == xyz2[align2].shape
    assert xyz1[calc1].shape == xyz2[calc2].shape

    xyz1c = xyz1 - xyz1[align1].mean(0)
    xyz2c = xyz2 - xyz2[align2].mean(0)

    C = xyz2c[align2].T @ xyz1c[align1]
    V, S, Wt = np.linalg.svd(C)
    d = np.ones((3, 3))
    d[:, -1] = np.sign(np.linalg.det(V) * np.linalg.det(Wt))
    U = (d * V) @ Wt

    xyz2r = xyz2c @ U

    divL = xyz2r[calc2].shape[0]
    diff = xyz2r[calc2] - xyz1c[calc1]
    return float(np.sqrt(np.sum(diff * diff) / (divL + eps)))


def calculate_rmsds(
    init_crds: np.ndarray,   # (L,37,3) or (L,*,3) but CA must be at index 1 if using 37
    pred_crds: np.ndarray,   # (L,37,3)
    target_mask: np.ndarray  # (L,) True for target residues
) -> Dict[str, float]:
    """
    RMSDs using CA atoms (index 1 = CA in AF2 atom order).
    - binder_aligned_rmsd: align on designed (non-target) residues, compute RMSD on designed residues
    - target_aligned_rmsd: align on target residues, compute RMSD on designed residues
    """
    init_ca = init_crds[:, residue_constants.atom_order["CA"], :]
    pred_ca = pred_crds[:, residue_constants.atom_order["CA"], :]

    binder_mask = ~target_mask

    rmsds = {
        "binder_aligned_rmsd": subset_rmsd(
            xyz1=init_ca, align1=binder_mask, calc1=binder_mask,
            xyz2=pred_ca, align2=binder_mask, calc2=binder_mask
        ),
        "target_aligned_rmsd": subset_rmsd(
            xyz1=init_ca, align1=target_mask, calc1=binder_mask,
            xyz2=pred_ca, align2=target_mask, calc2=binder_mask
        ),
    }
    return rmsds


def pae_interaction_from_masks(pae: np.ndarray, design_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """Mean pAE between design residues and target residues (both directions averaged)."""
    if design_mask.sum() == 0 or target_mask.sum() == 0:
        return float("nan")
    a = float(np.mean(pae[design_mask][:, target_mask]))
    b = float(np.mean(pae[target_mask][:, design_mask]))
    return (a + b) / 2.0


def get_final_dict(score_dict: Optional[dict], string_dict: Optional[dict]) -> OrderedDict:
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)
    all_keys = keys_score + keys_string
    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]
        if idx < len(keys_score):
            final_dict[key] = "%8.3f" % (score_dict[key])
        else:
            final_dict[key] = string_dict[key]
    return final_dict


def add2scorefile(
    tag: str,
    scorefilename: str,
    write_header: bool = False,
    score_dict: Optional[dict] = None,
    string_dict: Optional[dict] = None,
) -> None:
    with open(scorefilename, "a") as f:
        final_dict = get_final_dict(score_dict, string_dict)
        if write_header:
            f.write("SCORE:     %s description\n" % (" ".join(final_dict.keys())))
        scores_string = " ".join(final_dict.values())
        f.write("SCORE:     %s        %s\n" % (scores_string, tag))


def insert_rosetta_chain_endings(pose: Pose, chain_end_positions: Sequence[int]) -> Pose:
    """
    Insert chain endings after each position in chain_end_positions (1-indexed residue indices).
    """
    conf = pose.conformation()
    for endpos in sorted(chain_end_positions):
        conf.insert_chain_ending(int(endpos))
    pose.set_new_conformation(conf)

    splits = pose.split_by_chain()
    newpose = splits[1]
    for i in range(2, len(splits) + 1):
        newpose.append_pose_by_jump(splits[i], newpose.size())

    info = core.pose.PDBInfo(newpose, True)
    newpose.pdb_info(info)
    return newpose
