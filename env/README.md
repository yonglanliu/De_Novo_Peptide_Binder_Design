## Three conda environments need to be set up.
1. [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
2. ProteinMPNN + FastRelax Environment
3. af2_binder_design

---

### 1. RFdiffusion Set up

Using RFdiffusion to generate the backbone of peptides, giving hotspot residues 

```bash
conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
```

---

Thanks for **Nathaniel R. Bennett** and [his paper](https://www.nature.com/articles/s41467-023-38328-5) to for Improving de novo protein binder design with deep learning.
To setup "proteinmpnn_binder_design" and "af2_binder_design", please use two file <code>./env/proteinmpnn_fastrelax.yml</code> and <code>af2_binder_design.yml</code>
The environment file is adapted from the following repository: [dl_binder_design](https://github.com/nrbennet/dl_binder_design). 

Only **minor modifications** were made to simplify installation.

### 2. ProteinMPNN + FastRelax Environment

<code>proteinmpnn_binder_design</code>

* This environment is used for:

* ProteinMPNN sequence design

* Rosetta / PyRosetta FastRelax

* Interface scoring and refinement


```bash
conda env create -f env/proteinmpnn_fastrelax.yml
conda activate proteinmpnn_binder_design
```

--- 

### 3. AlphaFold2 Validation Environment

This environment runs a modified AlphaFold2 model for validating the final peptide–protein complexes.

<cdoe>proteinmpnn_binder_design</code>

```bash
conda env create -f env/af2_binder_design.yml
conda activate af2_binder_design
```

## Final Environment Summary

After setup, you should have **three conda environments**:

### SE3nv

* RFdiffusion
* Backbone generation of peptide binders with hotspot constraints

### proteinmpnn_binder_design

* ProteinMPNN
* PyRosetta / FastRelax
* Sequence design and structural refinement

### af2_binder_design
* Modified AlphaFold2
* Final structural validation of peptide–protein complexes
