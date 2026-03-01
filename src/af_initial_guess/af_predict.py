#!/usr/bin/env python3
"""
AF2 single-seq runner with template + optional initial guess, supporting >2 chains.

Typical use (your case: fix A and C, design B):
  python af_predict.py \
    -pdbdir in_pdbs \
    -outpdbdir outputs \
    --fixed_chains A,C \
    --design_chains B \
    --target_chains A,C \
    -recycle 3

Notes:
- This script uses AlphaFold "model_1_ptm" weights stored in ./model_weights
- It builds a 1-row "MSA" (no MSA search).
- It builds a template where fixed chains are templated, design chains are left free.
- It inserts chain breaks via residue_index jumps (+200) at chain boundaries (and optionally bad amide distances).
"""

import os
import sys
import glob
import uuid
import argparse


from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
from jax.lib import xla_bridge


from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data as af2_data
from alphafold.model import config as af2_config
from alphafold.model import model as af2_model

from pyrosetta import init, Pose, pose_from_pdb, pose_from_file
from rosetta import core
from af_initial_guess.af_utils import *

# -----------------------------
# Optional silent-tools support
# -----------------------------
SILENT_TOOLS_AVAILABLE = False
try:
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(parent, "include"))
    from silent_tools import silent_tools  # type: ignore

    SILENT_TOOLS_AVAILABLE = True
except Exception:
    SILENT_TOOLS_AVAILABLE = False


def range1(size: int):
    return range(1, size + 1)


# -----------------------------
# Feature holder / AF2 runner
# -----------------------------
class FeatureHolder:
    def __init__(self, pose: Pose, tag: str):
        self.pose = pose
        self.tag = tag
        self.outtag = f"{tag}_af2pred"

        self.seq = pose.sequence()
        self.initial_all_atom_positions: Optional[np.ndarray] = None
        self.initial_all_atom_masks: Optional[np.ndarray] = None

        self.residue_mask: Optional[List[bool]] = None

        self.outpose: Optional[Pose] = None
        self.plddt_array: Optional[np.ndarray] = None
        self.score_dict: Optional[dict] = None


class AF2Runner:
    def __init__(self, args, struct_manager):
        self.args = args
        self.struct_manager = struct_manager
        self.max_amide_dist = args.max_amide_dist
        self.model_name = "model_1_multimer_v3"

        model_cfg = af2_config.model_config(self.model_name) # initial model configuration
        model_cfg.data.eval.num_ensemble = 1  # Run the model only once per recycle.
        model_cfg.data.common.num_recycle = args.recycle
        model_cfg.model.num_recycle = args.recycle

        # without initial guess, af predict from scratch (MSA)
        # with initial guess, af use the template structure and refine around it
        model_cfg.model.embeddings_and_evoformer.initial_guess = (not args.no_initial_guess) # if use initial guess


        # Keep single-seq "MSA" tiny
        model_cfg.data.common.max_extra_msa = 5
        model_cfg.data.eval.max_msa_clusters = 5

        # Load model
        params_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_weights")
        model_params = af2_data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)

        self.model_runner = af2_model.RunModel(model_cfg, model_params)
        self.t0 = None

    def featurize(self, feat_holder: FeatureHolder) -> Tuple[dict, jnp.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          feature_dict (processed),
          initial_guess (jax),
          target_mask (np bool),
          design_mask (np bool),
          breaks (np list)
        """

        # all_pos (L,37,3)
        # all_mask (L,37)
        all_pos, all_msk = af2_get_atom_positions_from_pose(feat_holder.pose, self.struct_manager.tmp_fn)
        feat_holder.initial_all_atom_positions = all_pos
        feat_holder.initial_all_atom_masks = all_msk

        # Convert numpy (L,37,3) to JAX (L,37,3) for initial guess.
        initial_guess = parse_initial_guess(all_pos)

        # Chain-based masks
        fixed_set = self.struct_manager.fixed_chains
        design_set = self.struct_manager.design_chains
        target_set = self.struct_manager.target_chains

        feat_holder.residue_mask = build_residue_mask_by_chain(feat_holder.pose, fixed_set, design_set)
        target_mask = build_target_mask_by_chain(feat_holder.pose, target_set)
        design_mask = ~target_mask  # design region = non-target by default

        # Template features
        template_dict = generate_template_features(
            seq=feat_holder.seq,
            all_atom_positions=all_pos,
            all_atom_masks=all_msk,
            residue_mask=feat_holder.residue_mask,
        )

        # Single-seq features (no MSA search)
        feature_dict = {
            **pipeline.make_sequence_features(
                sequence=feat_holder.seq, description="none", num_res=len(feat_holder.seq)
            ),
            **pipeline.make_msa_features(
                msas=[[feat_holder.seq]],
                deletion_matrices=[[[0] * len(feat_holder.seq)]],
            ),
            **template_dict,
        }

        # Breaks: chain boundaries + optional "bad amide" gaps
        breaks = chainbreak_indices_zero_based(feat_holder.pose)  # find chain break indices
        if not self.args.disable_distance_breaks:
            breaks += check_residue_distances(all_pos, all_msk, self.max_amide_dist)
        breaks = sorted(set(breaks))

        feature_dict["residue_index"] = insert_truncations(feature_dict["residue_index"], breaks)

        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)
        return feature_dict, initial_guess, target_mask, design_mask, np.array(breaks, dtype=int)

    def process_output(
        self,
        feat_holder: FeatureHolder,
        feature_dict: dict,
        prediction_result: dict,
        target_mask: np.ndarray,
        design_mask: np.ndarray,
    ) -> None:
        structure_module = prediction_result["structure_module"]
        this_protein = protein.Protein(
            aatype=feature_dict["aatype"][0],
            atom_positions=structure_module["final_atom_positions"][...],
            atom_mask=structure_module["final_atom_mask"][...],
            residue_index=feature_dict["residue_index"][0] + 1,
            b_factors=np.zeros_like(structure_module["final_atom_mask"][...]),
        )

        # Confidences
        confidences = {}
        confidences["plddt"] = confidence.compute_plddt(prediction_result["predicted_lddt"]["logits"][...])
        if "predicted_aligned_error" in prediction_result:
            confidences.update(
                confidence.compute_predicted_aligned_error(
                    prediction_result["predicted_aligned_error"]["logits"][...],
                    prediction_result["predicted_aligned_error"]["breaks"][...],
                )
            )

        feat_holder.plddt_array = confidences["plddt"]
        plddt_total = float(np.mean(confidences["plddt"]))
        plddt_design = float(np.mean(confidences["plddt"][design_mask])) if design_mask.any() else float("nan")
        plddt_target = float(np.mean(confidences["plddt"][target_mask])) if target_mask.any() else float("nan")

        pae = confidences.get("predicted_aligned_error", None)
        if pae is None:
            pae_design = pae_target = pae_interaction = float("nan")
        else:
            pae_design = float(np.mean(pae[design_mask][:, design_mask])) if design_mask.any() else float("nan")
            pae_target = float(np.mean(pae[target_mask][:, target_mask])) if target_mask.any() else float("nan")
            pae_interaction = pae_interaction_from_masks(pae, design_mask, target_mask)

        # RMSDs
        rmsds = calculate_rmsds(
            init_crds=feat_holder.initial_all_atom_positions,
            pred_crds=this_protein.atom_positions,
            target_mask=target_mask,
        )

        elapsed = timer() - self.t0
        score_dict = {
            "plddt_total": plddt_total,
            "plddt_design": plddt_design,
            "plddt_target": plddt_target,
            "pae_design": pae_design,
            "pae_target": pae_target,
            "pae_interaction": pae_interaction,
            "binder_aligned_rmsd": rmsds["binder_aligned_rmsd"],
            "target_aligned_rmsd": rmsds["target_aligned_rmsd"],
            "time": elapsed,
        }
        feat_holder.score_dict = score_dict

        # Write PDB so Rosetta can read
        pdb_str = protein.to_pdb(this_protein)
        with open(self.struct_manager.tmp_fn, "w") as f:
            f.write(pdb_str)

        feat_holder.outpose = pose_from_file(self.struct_manager.tmp_fn)
        os.remove(self.struct_manager.tmp_fn)

        # Record scores + dump pose
        self.struct_manager.record_scores(feat_holder.outtag, score_dict, None)
        self.struct_manager.dump_pose(feat_holder)

        print(score_dict)
        print(f"Tag: {feat_holder.outtag} success in {elapsed:.1f}s")

    def process_struct(self, struct_path_or_tag: str) -> None:
        self.t0 = timer()
        pose, tag = self.struct_manager.load_pose(struct_path_or_tag)

        feat_holder = FeatureHolder(pose, tag)
        print(f"Processing: {tag}")

        feature_dict, initial_guess, target_mask, design_mask, breaks = self.featurize(feat_holder)

        start = timer()
        print(f"Running {self.model_name} (recycles={self.args.recycle}, initial_guess={not self.args.no_initial_guess})")

        prediction_result = self.model_runner.apply(
            self.model_runner.params,
            jax.random.PRNGKey(0),
            feature_dict,
            initial_guess,
        )

        print(f"Tag: {tag} finished AF2 in {timer() - start:.1f}s")
        self.process_output(feat_holder, feature_dict, prediction_result, target_mask, design_mask)


# -----------------------------
# Struct manager (I/O)
# -----------------------------
class StructManager:
    def __init__(self, args):
        self.args = args
        self.score_fn = args.scorefilename
        self.tmp_fn = f"tmp_{uuid.uuid4()}.pdb"

        self.fixed_chains = [c.strip() for c in args.fixed_chains.split(",") if c.strip()]
        self.design_chains = [c.strip() for c in args.design_chains.split(",") if c.strip()]
        self.target_chains = [c.strip() for c in args.target_chains.split(",") if c.strip()]

        self.pdb = (args.pdbdir != "")
        self.silent = (args.silent != "")

        assert self.pdb ^ self.silent, "Set exactly one of -pdbdir or -silent."

        # Iterator over structs
        if self.pdb:
            self.pdbdir = args.pdbdir
            self.outpdbdir = args.outpdbdir
            self.struct_iterator = glob.glob(os.path.join(self.pdbdir, "*.pdb"))

            if args.runlist:
                with open(args.runlist, "r") as f:
                    runset = set([line.strip() for line in f if line.strip()])
                self.struct_iterator = [
                    p for p in self.struct_iterator
                    if ".".join(os.path.basename(p).split(".")[:-1]) in runset
                ]
                print(f"After runlist filtering: {len(self.struct_iterator)} structures")

        if self.silent:
            if not SILENT_TOOLS_AVAILABLE:
                raise RuntimeError("silent_tools not available but -silent was provided.")
            self.struct_iterator = silent_tools.get_silent_index(args.silent)["tags"]
            self.sfd_in = core.io.silent.SilentFileData(core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)
            self.sfd_out = core.io.silent.SilentFileData(
                args.outsilent, False, False, "binary", core.io.silent.SilentFileOptions()
            )
            self.outsilent = args.outsilent

        # Checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()
        if os.path.isfile(self.chkfn):
            with open(self.chkfn, "r") as f:
                self.finished_structs = set([line.strip() for line in f if line.strip()])

    def record_checkpoint(self, tag: str) -> None:
        with open(self.chkfn, "a") as f:
            f.write(f"{tag}\n")

    def iterate(self):
        for struct in self.struct_iterator:
            if self.pdb:
                tag = ".".join(os.path.basename(struct).split(".")[:-1])
            else:
                tag = struct
            if tag in self.finished_structs:
                print(f"{tag} already done, skipping")
                continue
            yield struct

    def record_scores(self, tag: str, score_dict: dict, string_dict: Optional[dict]):
        write_header = not os.path.isfile(self.score_fn)
        add2scorefile(tag, self.score_fn, write_header, score_dict, string_dict)

    def load_pose(self, struct_path_or_tag: str) -> Tuple[Pose, str]:
        if self.pdb:
            pose = pose_from_pdb(struct_path_or_tag)
            tag = ".".join(os.path.basename(struct_path_or_tag).split(".")[:-1])
            return pose, tag

        if self.silent:
            pose = Pose()
            tag = struct_path_or_tag
            self.sfd_in.get_structure(tag).fill_pose(pose)
            return pose, tag

        raise RuntimeError("Neither pdb nor silent mode is active.")

    def dump_pose(self, feat_holder: FeatureHolder) -> None:
        assert feat_holder.outpose is not None
        assert feat_holder.plddt_array is not None
        assert feat_holder.score_dict is not None

        pose = feat_holder.outpose

        # Rebuild chain endings to match input pose chain structure
        # Use chain ends from input pose:
        conf_in = feat_holder.pose.conformation()
        chain_end_positions = [conf_in.chain_end(ch) for ch in range(1, feat_holder.pose.num_chains())]
        if chain_end_positions:
            pose = insert_rosetta_chain_endings(pose, chain_end_positions)

        # Add pLDDT as b-factors
        info = pose.pdb_info()
        if info is None:
            info = core.pose.PDBInfo(pose)
        for resi in range1(pose.size()):
            p = float(feat_holder.plddt_array[resi - 1])
            info.add_reslabel(resi, f"{p:.2f}")
            for atom_i in range1(pose.residue(resi).natoms()):
                info.bfactor(resi, atom_i, p)
        pose.pdb_info(info)

        if self.pdb:
            os.makedirs(self.outpdbdir, exist_ok=True)
            outpath = os.path.join(self.outpdbdir, feat_holder.outtag + ".pdb")
            pose.dump_pdb(outpath)

        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, feat_holder.outtag)
            for scorename, value in feat_holder.score_dict.items():
                struct.add_energy(scorename, float(value), 1)
            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outsilent)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--pdbdir", type=str, default="", help="Directory of input pdbs")
    parser.add_argument("--silent", type=str, default="", help="Input silent file")
    parser.add_argument("--outpdbdir", type=str, default="outputs", help="Output PDB directory (pdb mode)")
    parser.add_argument("--outsilent", type=str, default="out.silent", help="Output silent file (silent mode)")
    parser.add_argument("--runlist", type=str, default="", help="Optional list of tags to run (pdb mode)")
    parser.add_argument("--checkpoint_name", type=str, default="check.point", help="Checkpoint file")
    parser.add_argument("--scorefilename", type=str, default="out.sc", help="Scorefile name")
    parser.add_argument("--debug", action="store_true", default=False, help="Crash on error")

    # AF2 behavior
    parser.add_argument("--max_amide_dist", type=float, default=3.0, help="Amide C-N distance cutoff")
    parser.add_argument("--recycle", type=int, default=3, help="AF2 recycles")
    parser.add_argument("--no_initial_guess", action="store_true", default=False, help="Disable initial guess")
    parser.add_argument("--disable_distance_breaks", action="store_true", default=False,
                        help="Do not add breaks from amide distance gaps; only chain boundaries")

    # Multi-chain control
    parser.add_argument("--fixed_chains", type=str, default="A,C",
                        help="Comma-separated PDB chain IDs to FIX (templated). Example: A,C")
    parser.add_argument("--design_chains", type=str, default="B",
                        help="Comma-separated PDB chain IDs to DESIGN (not templated). Example: B")
    parser.add_argument("--target_chains", type=str, default="A,C",
                        help="Comma-separated PDB chain IDs considered TARGET for scoring (pAE/RMSD). Example: A,C")

    args = parser.parse_args()

    # PyRosetta init
    init("-in:file:silent_struct_type binary -mute all")

    # Device info
    device = xla_bridge.get_backend().platform
    if device == "gpu":
        print("/" * 60)
        print("Found GPU and will use it to run AF2")
        print("/" * 60)
    else:
        print("/" * 60)
        print("WARNING: No GPU detected; running AF2 on CPU")
        print("/" * 60)

    struct_manager = StructManager(args)
    af2_runner = AF2Runner(args, struct_manager)

    for struct in struct_manager.iterate():
        t0 = timer()
        try:
            af2_runner.process_struct(struct)
        except KeyboardInterrupt:
            sys.exit("Killed by Ctrl+C")
        except Exception as e:
            if args.debug:
                raise
            secs = int(timer() - t0)
            print(f"Struct {struct} failed in {secs}s with error: {type(e).__name__}: {e}")

        # Record checkpoint using tag (not path)
        if struct_manager.pdb:
            tag = ".".join(os.path.basename(struct).split(".")[:-1])
        else:
            tag = struct
        struct_manager.record_checkpoint(tag)


if __name__ == "__main__":
    main()
