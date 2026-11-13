# PhaseResSAGE

PhaseResSAGE is the runnable implementation of the paper model. It represents
each protein as a residue graph and applies a four-head Graph Attention Network
followed by two residual mean-aggregation GraphSAGE layers. The same backbone
is trained for two tasks:

1. LLPS versus Non-LLPS, with LLPS as the positive class.
2. Scaffold versus Client inside the LLPS set, with Scaffold as the positive class.

The code does not use PyTorch Geometric. Graph batching and message passing are
implemented with PyTorch tensor operations, so the only graph-specific runtime
dependency is PyTorch itself.

## Files

`run_phase_res_sage.py` is the complete training and evaluation entry point. It
contains FASTA parsing, ESM2 residue embeddings, physicochemical and disorder
proxy features, DSSP features, sequence/contact graph construction, PhaseResSAGE,
sqrt-frequency weighted BCE, focal-loss ablation, threshold selection, metrics,
checkpoints, prediction CSV files, and residue-level importance output.

`prepare_phaseom_data.py` converts the uploaded PhaseOM-derived FASTA files into
the stable files expected by the runner. The IDR/non-IDR files are audited but
are not appended as extra samples. `prepare_structures.py` searches RCSB PDB,
then the AlphaFold Protein Structure Database when an accession is available,
and writes a structure manifest. Unresolved records are reported instead of
being silently replaced by a proxy structure.

## Environment

Use a Python environment containing PyTorch, Biopython, SciPy, scikit-learn,
fair-esm, and PyDSSP. The repository machine was tested with:

```powershell
C:\Users\27873\anaconda3\python.exe -c "import torch, Bio, scipy, sklearn, esm, pydssp; print(torch.__version__)"
```

Install the packages listed in `requirements.txt` after installing the desired
PyTorch build.

## Prepare the PhaseOM-derived data

The default source directory is the directory containing the uploaded files.
Use an explicit source directory on another machine:

```powershell
python experiments\phase_res_sage\prepare_phaseom_data.py `
  --source-dir C:\path\to\phaseom_fastas `
  --out-root data\phaseom_attachment
```

The output contains `stage1_llps.fasta`, `stage1_nonllps.fasta`, and
`stage2_roles.fasta`, together with `prepare_manifest.json`.

## Prepare PDB-first structures

```powershell
python experiments\phase_res_sage\prepare_structures.py `
  --data-root data\phaseom_attachment `
  --out-root data\phaseom_attachment\structure_manifest
```

The formal structure mode requires every record used by the experiment to have
a usable PDB or AlphaFold2-derived PDB path. Records marked
`requires_local_alphafold2` need a sequence-to-UniProt mapping or a local
AlphaFold2/ColabFold prediction before they can be included in the formal
structure experiment.

## Run the lightweight, reproducible mode

ESM2-t6 is the default and fits machines with limited memory. The proxy mode is
useful when the structure manifest is incomplete; its contact edges are ESM2
contact proxies and must be reported as such.

```powershell
python experiments\phase_res_sage\run_phase_res_sage.py `
  --data-root data\phaseom_attachment `
  --structure-mode esm_proxy `
  --encoder esm2_t6_8M_UR50D `
  --loss sqrt_bce `
  --epochs 60 `
  --patience 12 `
  --seeds 42 43 44 `
  --device cpu `
  --out-root results\phase_res_sage_attachment_t6_proxy `
  --cache-root results\phase_res_sage_attachment_t6_proxy
```

## Run the formal PDB-first/AlphaFold2-fallback mode

```powershell
python experiments\phase_res_sage\run_phase_res_sage.py `
  --data-root data\phaseom_attachment `
  --structure-mode pdb_or_alphafold `
  --structure-manifest data\phaseom_attachment\structure_manifest\structure_manifest.json `
  --encoder esm2_t6_8M_UR50D `
  --loss sqrt_bce `
  --epochs 60 `
  --patience 12 `
  --seeds 42 43 44 `
  --device cuda `
  --out-root results\phase_res_sage_attachment_pdb_af2 `
  --cache-root results\phase_res_sage_attachment_pdb_af2
```

Use `--encoder esm2_t33_650M_UR50D` only when the machine has enough GPU memory
and disk space. The code automatically reads the checkpoint's final hidden
layer and adapts the input dimension.

## Outputs

Each run writes `data_audit.json`, `results.json`, stage-specific checkpoints,
and prediction CSV files. The CSV files contain protein ID, split, label,
prediction score, selected threshold, and predicted role. The JSON result file
contains Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, AUROC, AUPRC,
confusion matrices, threshold, and Scaffold Recall for the second task.

The runner selects the checkpoint by validation AUPRC. The test set is used only
after checkpoint and threshold selection. The default main loss is square-root
frequency weighted BCE; `--loss focal` runs the class-balanced focal-loss
ablation.

## Important reporting boundary

The lightweight `esm_proxy` mode is a valid runnable experiment, but it is not
the same as the formal RCSB PDB plus AlphaFold2 fallback and DSSP workflow. Do
not report proxy results as experimental-structure results. The runner records
the structure mode and encoder in `data_audit.json` and `results.json` for this
reason.
