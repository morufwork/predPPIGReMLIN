# Graph-Based Protein Structure Analysis Repository

This repository contains supplementary materials for a PhD thesis on graph-based modeling and analysis of protein-protein interactions and conserved structures. It includes various Python scripts and tools for processing protein structures, generating interaction graphs, performing pattern mining, feature extraction, and machine learning classification for protein interaction analysis.

## Project Structure

### Datasets
- **Dataset/**: Collection of protein datasets used in the analyses.
  - **camp/**: CAMP (Cysteine-rich Antimicrobial Peptides) dataset.
  - **multi_class/**: Multi-class protein interaction dataset.
  - **sars-cov2-ace2/**: SARS-CoV-2 and ACE2 protein interaction dataset.
  - **yeast/**: Yeast protein interaction dataset.

### Analysis Modules

#### 1. binary_classification/
Binary classification module for distinguishing between positive and negative protein interaction patterns using graph-based features and machine learning.

**Key Files:**
- `main.py`: Main script for binary classification analysis
- `all_features.csv`: Extracted features for all graphs
- `cluster_threshold_summary_result.csv`: Clustering results summary
- Various evaluation plots (ROC curves, PR curves)

**How to Run:**
```bash
cd binary_classification
python main.py
```

**Requirements:**
- Requires `dgraphs.txt`, `qgraphs.txt`, `tgraphs.txt` in the `gSpan/` subdirectory
- Dependencies: biopython, networkx, scikit-learn, numpy, matplotlib, pandas

**Output:**
- Feature extraction and classification results
- ROC and PR curves for different clustering thresholds
- Summary CSV files with performance metrics

#### 2. multi_class_classification/
Multi-class classification module for protein-protein interaction analysis using surface residue features and graph-based representations.

**Key Files:**
- `main.py`: Main script for multi-class classification
- `classical_residue_features.csv`: Extracted residue features
- Amino acid and interaction type mappings

**How to Run:**
```bash
cd multi_class_classification
python main.py
```

**Requirements:**
- Requires `file_final.csv` with protein pairs
- Requires `files/` directory containing PDB files referenced in `file_final.csv`
- Dependencies: biopython, networkx, numpy, pandas, matplotlib

**Output:**
- Feature vectors for surface, adjacent, and adjacent-adjacent residues
- CSV file with summed features for protein pairs

#### 3. pattern_search/
Pattern search and graph matching module for identifying conserved interaction patterns in protein structures, including docking analysis.

**Key Files:**
- `main.py`: Main script for pattern search and docking analysis
- `result.csv`: Pattern matching results
- `dock_best.csv` and `dock_top.csv`: Best docking predictions
- `atom_types.csv`: Atom type definitions
- `interactions.csv`: Interaction criteria
- `target_chain.csv`: Target chain mappings

**How to Run:**
```bash
cd pattern_search
python main.py
```

**Requirements:**
- Requires `dgraphs.txt`, `graphs.txt` in `gSpan/` subdirectory
- Requires `target_chain.csv` for chain mappings
- Dependencies: biopython, networkx, numpy, pandas, matplotlib

**Output:**
- Pattern matching results between query and data graphs
- Docking predictions with scoring metrics
- CSV files with mapping densities and similarity scores

#### 4. graph_modeling_and_conserved_structures/
Core pipeline for ppiGReMLIN - a scalable graph-based strategy for modeling protein-protein interfaces. Includes graph generation, clustering, and frequent subgraph mining using gSpan.

**Key Files:**
- Graph generation and processing scripts
- Clustering algorithms
- gSpan integration for frequent subgraph mining

**How to Run:**
Refer to individual scripts in the directory for specific usage.

## Requirements

- Python 3.6 or higher
- Key dependencies:
  - biopython (for PDB structure parsing)
  - networkx (for graph operations)
  - scikit-learn (for machine learning)
  - numpy (for numerical computations)
  - matplotlib (for plotting)
  - pandas (for data manipulation)

Install dependencies:
```bash
pip install biopython networkx scikit-learn numpy matplotlib pandas
```

## Data Format

The project primarily uses:
- **PDB files**: Protein Data Bank format files for 3D structures
- **Graph files**: NetworkX-compatible graph representations (.txt format)
- **CSV files**: Feature matrices and result summaries
- **JSON files**: Configuration and type mappings

## Usage Overview

1. **Data Preparation**: Place PDB files and graph files in appropriate directories
2. **Feature Extraction**: Run individual modules to extract graph-based features
3. **Analysis**: Use classification modules for prediction tasks
4. **Pattern Mining**: Apply pattern search for conserved structure identification

## Key Algorithms

- **Graph Generation**: Convert protein structures to interaction graphs
- **Feature Extraction**: Atom types, interaction types, surface residue analysis
- **Clustering**: Spectral clustering with eigen gap analysis
- **Classification**: Random Forest with cross-validation
- **Pattern Mining**: gSpan algorithm for frequent subgraph mining
- **Graph Matching**: Subgraph isomorphism for pattern detection

## Output Files

- **Feature CSVs**: Numerical features for machine learning
- **Result CSVs**: Classification and matching results
- **Plot Images**: ROC curves, PR curves, clustering visualizations
- **Graph Files**: Generated interaction graphs

## Citation

This is supplementary material for academic research. Please cite the associated PhD thesis if used in publications.

## Contact

For questions or issues, refer to the individual script documentation or contact the corresponding author of the manuscript.