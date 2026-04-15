# predPPIGreMLIN

predPPIGreMLIN models protein-protein interfaces as graphs, mines conserved interaction patterns, and exports hotspot and visualization artifacts for downstream analysis.

This repository now includes a reproducible inference workflow for new protein inputs together with example input and output files:

- Inference script: `main.py`
- Example input: `test_even2.txt`
- Example output directory: `results_m8_check/`

## Repository contents

- `main.py`: end-to-end pipeline for new protein inputs.
- `common.py`, `contacts.py`, `graphprocessing.py`, `graphmining.py`, `clustering.py`, `eigen_gap.py`, `find_better_eigen.py`: core graph construction, clustering, mining, and analysis modules.
- `atom_types.csv`, `interactions.csv`, `typenames.json`: interaction and atom-type definitions used by the pipeline.
- `gSpan/`: bundled `gSpan-64` executable and graph-related resources.
- `pdbfiles/`: example raw PDB structures used by the included sample run.
- `results_m8_check/`: committed example outputs for reproducibility.
- `requirements.txt`: pinned Python dependency list.
- `README`: original plain-text usage notes kept for reference.

## Requirements

The project was originally developed for Python 3.4.3+, but a modern Python 3 environment is recommended.

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input format

The input file must contain one PDB identifier followed by chain pairs on each line, separated by commas.

Example:

```text
1EJM,A,B,C,D,E,F
```

This describes chain pairs `(A,B)`, `(C,D)`, and `(E,F)` for the same structure.

If a PDB structure is not already available locally in `pdbfiles/`, the code will try to retrieve it from the PDB.

## How to run

From the repository root:

```bash
python main.py <pdbidsfile> <results_dir> [baseline_results_dir]
```

Arguments:

- `<pdbidsfile>`: input file listing new protein structures and chain pairs.
- `<results_dir>`: output directory to create.
- `[baseline_results_dir]`: optional regression-comparison directory.

Concrete example:

```bash
python main.py test_even2.txt results_example
```

## Example inference workflow for new proteins

The repository includes a ready-to-inspect example that supports reproducibility and ease of use.

Example input:

```bash
test_even2.txt
```

Example output:

```bash
results_m8_check/
```

The committed `results_m8_check/` folder contains 58 generated files, including:

- `graphs.txt`
- `count_matrix.csv`
- `clusters.csv`
- `gSpan.fp`
- `maximal.json`
- `pattern_hotspots_summary.csv`
- `pattern_occurrences.json`
- `pattern_occurrences_residue.json`
- `regression_validation.md`
- `visualization/visualization_manifest.json`

This means the repository already contains:

> an inference script for new protein inputs and a usage example with input and output files to facilitate reproducibility and ease of use.

## Outputs

Running `main.py` generates the main mining and analysis artifacts inside the chosen results directory, including:

- graph files and count matrices,
- clustering outputs,
- gSpan frequent subgraph mining results,
- maximal pattern summaries,
- residue-level hotspot summaries,
- visualization-ready JSON and CSV files,
- regression validation reports.

## Notes

- `results_m8_check/` is intentionally versioned in this repository as a reproducibility example.
- `pdbs/` remains ignored because it is regenerated chain-level output.
- The original legacy instructions are preserved in `README`.
