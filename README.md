# Graph-Based Protein Structure Analysis

This repository contains supplementary materials for a PhD thesis on graph-based modeling and analysis of protein-protein interactions and conserved structures. It includes various Python scripts and tools for processing protein structures, generating interaction graphs, performing pattern mining, and feature extraction for machine learning applications.

## Project Structure

- **camp/**: Contains datasets and graph files for protein interaction analysis.
- **graph_modeling_and_conserved_structures/**: Core pipeline for ppiGReMLIN - a scalable graph-based strategy for modeling protein-protein interfaces. Includes graph generation, clustering, and frequent subgraph mining using gSpan.
- **graph_surface_residue/**: Scripts for analyzing surface residues in protein structures, computing residue-residue interactions, and extracting features for classification tasks.
- **multi_class/**: CSV files containing interaction type classifications.
- **pattern_search/**: Tools for pattern searching and graph matching in protein structures, including docking analysis and graph mining algorithms.

## Requirements

- Python 3.4.3 or higher
- Key dependencies include:
  - biopython
  - networkx
  - scikit-learn
  - numpy
  - matplotlib
  - pandas

See individual directory README files or requirements.txt files for specific dependencies.

## Usage

Each subdirectory contains its own main.py script with specific usage instructions. For example:

- In `graph_modeling_and_conserved_structures/`: Run `python main.py <pdbidsfile> <directory>` to process PDB structures and generate graphs.
- In `graph_surface_residue/`: Run `python main.py` to extract features from protein pairs.

Refer to the README files in each subdirectory for detailed instructions.

## Data

The project uses PDB (Protein Data Bank) files as input. Some datasets include PDB IDs for serine proteases, BCL-2 family proteins, and other protein complexes.

## License

This is supplementary material for academic research. Please cite appropriately if used in publications.