# predPPIGreMLIN

This repository combines graph-based protein interface mining code, conserved-pattern analysis workflows, a multilabel notebook, and benchmark data bundles from MaSIF and ScanNet.

The main reproducibility update in this repository is that we added an inference-oriented workflow for new protein inputs together with concrete example input and output files:

- `graphmedeling_conseved_pattern/main.py` runs the original ppiGReMLIN-style pipeline on a new list of PDB-chain inputs.
- `graphmedeling_conseved_pattern/test_even2.txt` is an example input file for new protein structures.
- `graphmedeling_conseved_pattern/results_m8_check/` is an example output directory showing the expected artifacts.
- `searchPattern/main.py` runs pattern loading and feature generation for new graph inputs.
- `searchPattern/phase1_output/` contains example output files for the pattern-learning and inference stages.

## Repository layout

- `Dataset/`: reference datasets and benchmark scripts.
- `graphmedeling_conseved_pattern/`: end-to-end graph mining pipeline for protein interface pattern discovery from PDB inputs.
- `multilabel/`: notebook-based multilabel and multiclass prediction experiment.
- `searchPattern/`: pattern-learning and inference pipeline that converts graph sets into reproducible feature tables.

## Environment

The repository does not provide a single root environment file. The closest pinned dependency list is `graphmedeling_conseved_pattern/requirements.txt`.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r graphmedeling_conseved_pattern/requirements.txt
pip install pandas
```

Notes:

- `searchPattern/main.py` needs `pandas`, `networkx`, and local helper modules.
- Several scripts under `Dataset/masif/` expect an external MaSIF checkout with a `source/` tree and use `git rev-parse --show-toplevel` to resolve it.
- The bundled `gSpan/gSpan-64` binaries in `graphmedeling_conseved_pattern/` and `searchPattern/` are expected by their respective pipelines.

## How to run each top-level folder except `Dataset`

### 1. `graphmedeling_conseved_pattern/`

Purpose: run graph construction, clustering, frequent subgraph mining, hotspot summarization, visualization export, and regression validation for protein interfaces.

Run from inside the folder:

```bash
cd graphmedeling_conseved_pattern
python main.py test_even2.txt results_example
```

Arguments:

- `test_even2.txt`: input list of PDB IDs and chain pairs.
- `results_example`: output folder to create.
- Optional third argument: baseline results directory for regression comparison.

Input format:

Each line contains one PDB ID followed by chain pairs, for example:

```text
1EJM,A,B,C,D,E,F
```

Existing reproducibility example:

- Input file: `graphmedeling_conseved_pattern/test_even2.txt`
- Example outputs: `graphmedeling_conseved_pattern/results_m8_check/graphs.txt`, `maximal.json`, `pattern_hotspots_summary.csv`, `pattern_occurrences.json`, `regression_validation.md`

This is the clearest inference script for new protein inputs in the repository because it accepts a fresh PDB-chain list and generates all downstream outputs in one command.

### 2. `searchPattern/`

Purpose: learn positive and negative graph patterns from training graphs, then generate reproducible feature tables for test or new graph inputs.

Run from the repository root:

```bash
python searchPattern/main.py \
  --graph-path searchPattern/gSpan/gSpan-64 \
  --positive-train-graphs masif_training_graphs.txt \
  --corpus-graphs corpus_filtertest_masif_graphs.txt \
  --test-positive-graphs masif_testing_graphs.txt \
  --output-dir searchPattern/phase1_output
```

Inference on new graphs using previously learned patterns:

```bash
python searchPattern/main.py \
  --graph-path searchPattern/gSpan/gSpan-64 \
  --load-patterns-dir searchPattern/phase1_output \
  --test-positive-graphs masif_testing_graphs.txt \
  --output-dir searchPattern/inference_example
```

What the script writes:

- `phase1_patterns_positive.json`
- `phase1_patterns_negative.json`
- `phase1_train_features.csv`
- `phase1_train_graph_features.csv`
- `phase1_train_features_with_graph_features.csv`
- `phase2_test_features.csv`
- `phase2_test_graph_features.csv`
- `phase2_test_features_with_graph_features.csv`

Existing reproducibility example:

- Input graph files: `searchPattern/gSpan/gSpan-64/masif_training_graphs.txt`, `masif_testing_graphs.txt`, `corpus_filtertest_masif_graphs.txt`
- Example outputs: `searchPattern/phase1_output/`

This folder provides the second inference-style entry point in the repository: once Phase 1 patterns exist, the same script can score additional graph files and export a new feature table.

### 3. `multilabel/`

Purpose: notebook-based multilabel and multiclass prediction analysis.

Contents:

- `multilabel_multiclass_prediction.ipynb`
- `multilabel_dataset.csv`

Run:

```bash
jupyter notebook multilabel/multilabel_multiclass_prediction.ipynb
```





