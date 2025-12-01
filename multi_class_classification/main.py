import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import numpy as np
import csv
import os


# Define interaction criteria
interaction_criteria = {
    "ARM_STACK": {"atomic_type1": "ARM", "atomic_type2": "ARM", "min_dist": 1.5, "max_dist": 3.5},
    "H_BOND": {"atomic_type1": "ACP", "atomic_type2": "DON", "min_dist": 2.0, "max_dist": 3.0},
    "HYDROPHOBIC": {"atomic_type1": "HPB", "atomic_type2": "HPB", "min_dist": 2.0, "max_dist": 3.8},
    "REPULSIVE": {"atomic_type1": "POS", "atomic_type2": "POS", "min_dist": 2.0, "max_dist": 6.0},
    "REPULSIVE": {"atomic_type1": "NEG", "atomic_type2": "NEG", "min_dist": 2.0, "max_dist": 6.0},
    "SALT_BRIDGE": {"atomic_type1": "POS", "atomic_type2": "NEG", "min_dist": 2.0, "max_dist": 6.0},
    "SS_BRIDGE": {"atomic_type1": "SG", "atomic_type2": "SG", "min_dist": 2.0, "max_dist": 2.2},
}

# Full names mapping
type_full_names = {
    "ACP": "Acceptor",
    "DON": "Donor",
    "POS": "Positive",
    "NEG": "Negative",
    "HPB": "Hydrophobic",
    "ARM": "Aromatic",
    "HYDROPHOBIC": "Hydrophobic",
    "SALT_BRIDGE": "Salt bridge",
    "ARM_STACK": "Aromatic",
    "H_BOND": "Hydrogen bond",
    "REPULSIVE": "Repulsive",
    "SS_BRIDGE": "Disulfide Bridge"
}

# Amino acid to atom types mapping (same as given in the provided data)
# Amino acid to atom types mapping
amino_acid_atoms = {
    "ALA": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB"},
    "ARG": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", "CD": None, 
            "NE": "POS,DON", "CZ": "POS", "NH1": "POS,DON", "NH2": "POS,DON"},
    "ASN": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": None, 
            "OD1": "ACP", "ND2": "DON"},
    "ASP": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": None, 
            "OD1": "NEG,ACP", "OD2": "NEG,ACP"},
    "CYS": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "SG": "DON,ACP"},
    "GLN": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "CD": None, "OE1": "ACP", "NE2": "DON"},
    "GLU": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "CD": None, "OE1": "NEG,ACP", "OE2": "NEG,ACP"},
    "GLY": {"N": "DON", "CA": None, "C": None, "O": "ACP"},
    "HIS": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "ARM", 
            "ND1": "ARM,POS", "CD2": "ARM", "CE1": "ARM", "NE2": "ARM,POS"},
    "ILE": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG1": "HPB", 
            "CG2": "HPB", "CD1": "HPB"},
    "LEU": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "CD1": "HPB", "CD2": "HPB"},
    "LYS": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "CD": "HPB", "CE": None, "NZ": "POS,DON"},
    "MET": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "SD": "ACP", "CE": "HPB"},
    "PHE": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB,ARM", 
            "CD1": "HPB,ARM", "CD2": "HPB,ARM", "CE1": "HPB,ARM", "CE2": "HPB,ARM", 
            "CZ": "HPB,ARM"},
    "PRO": {"N": None, "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB", 
            "CD": None},
    "SER": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": None, "OG": "DON,ACP"},
    "THR": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": None, "OG1": "DON,ACP", 
            "CG2": "HPB"},
    "TRP": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB,ARM", 
            "CD1": "ARM", "CD2": "HPB,ARM", "NE1": "ARM,DON", "CE2": "ARM", 
            "CE3": "HPB,ARM", "CZ2": "HPB,ARM", "CZ3": "HPB,ARM", "CH2": "HPB,ARM"},
    "TYR": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", "CG": "HPB,ARM", 
            "CD1": "HPB,ARM", "CD2": "HPB,ARM", "CE1": "HPB,ARM", "CE2": "HPB,ARM", 
            "CZ": "ARM", "OH": "DON,ACP"},
    "VAL": {"N": "DON", "CA": None, "C": None, "O": "ACP", "CB": "HPB", 
            "CG1": "HPB", "CG2": "HPB"},
}



# 1. **Surface Area Residue Calculation using SASA**
# Define helper functions for SASA calculation
def calculate_sasa(pdb_file):
    """
    Calculate the Solvent Accessible Surface Area (SASA) for residues in a PDB file.
    Uses the Shrake-Rupley method.
    """
    parser = PDBParser()
    structure = parser.get_structure(pdb_file, pdb_file)
    
    # Step 2: Initialize the ShrakeRupley class and compute SASA
    sr = ShrakeRupley()
    sr.compute(structure, level="A")  # Compute at the atom level
    
    surface_residues = []
    residues_dict = {}

    for chain in structure[0]:  # Iterate over chains in the first model
        for residue in chain:
            # Skip any residues that are not standard (e.g., water, ligands)
            if residue.get_resname() in ["WAT", "HOH", "ACE"]:
                continue

            # Accumulate SASA for all atoms in the residue
            residue_sasa = sum(atom.sasa for atom in residue if hasattr(atom, 'sasa'))
            id = f"{residue.get_resname()} {residue.id[1]}"
            residues_dict[residue] = residue_sasa
            
    # # Step 3:Identify the maximum SASA: Find the maximum SASA value across all residues in the protein structure
    max_sasa = max(residues_dict.values())
        
    # Step 4: Set a threshold for identifying surface residues
    surface_threshold = 0.25 * max_sasa
    
    # Step 5: Identify and collect surface residues
    for sasa_residue in residues_dict.items():
        if sasa_residue[1] >= surface_threshold:
            surface_residues.append(sasa_residue[0])
            #print(sasa_residue)

    return surface_residues


# Step 1: Parse PDB File
def parse_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file, pdb_file)
    residues = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                res_id = residue.get_id()[1]  # Residue sequence number
                res_key = f"{res_name}_{res_id}"
                if res_key not in residues:
                    residues[res_key] = {
                        'residue': res_name,
                        'res_id': res_id,
                        'atoms': []
                    }
                for atom in residue:
                    atom_name = atom.get_name()
                    element = atom.element
                    coord = atom.get_coord()
                    atom_type = None
                    if res_name in amino_acid_atoms:
                        atom_type_info = amino_acid_atoms[res_name].get(atom_name)
                        if atom_type_info:
                            atom_types = atom_type_info.split(',')
                            atom_type = atom_types  # list
                    residues[res_key]['atoms'].append({
                        'atom_name': atom_name,
                        'element': element,
                        'coord': coord,
                        'atom_type': atom_type
                    })
    return residues

# Step 2: Compute Residue-Residue Interactions
def compute_residue_interactions(residues, interaction_criteria):
    interactions = []
    residue_keys = list(residues.keys())
    for i in range(len(residue_keys)):
        for j in range(i+1, len(residue_keys)):
            res1 = residues[residue_keys[i]]
            res2 = residues[residue_keys[j]]
            # Compute all atom pairs between res1 and res2
            for atom1 in res1['atoms']:
                for atom2 in res2['atoms']:
                    distance = np.linalg.norm(atom1['coord'] - atom2['coord'])
                    # Check each interaction type
                    for interaction, criteria in interaction_criteria.items():
                        type1 = criteria['atomic_type1']
                        type2 = criteria['atomic_type2']
                        min_dist = criteria['min_dist']
                        max_dist = criteria['max_dist']
                        
                        # Check if atom types match
                        if atom1['atom_type'] and atom2['atom_type']:
                            # Since atom_type can be a list, check intersection
                            if type1 in atom1['atom_type'] and type2 in atom2['atom_type']:
                                if min_dist <= distance <= max_dist:
                                    interactions.append({
                                        'res1': residue_keys[i],
                                        'res2': residue_keys[j],
                                        'interaction_type': interaction,
                                        'distance': distance
                                    })
    return interactions


# Step 3: Generate Graph Network
def generate_residue_graph(residues, residue_interactions):
    G = nx.Graph()
    # Add residues as nodes
    for res_key, res_info in residues.items():
        G.add_node(res_key, residue=res_info['residue'], res_id=res_info['res_id'], 
                   atom_types=[atom['atom_type'] for atom in res_info['atoms'] if atom['atom_type']])
    
    # Add edges based on interactions
    for interaction in residue_interactions:
        res1 = interaction['res1']
        res2 = interaction['res2']
        interaction_type = interaction['interaction_type']
        distance = interaction['distance']
        G.add_edge(res1, res2, interaction_type=interaction_type, distance=distance)
    
    return G


# Step 1: Map surface residues to nodes in the graph
def map_surface_residues_to_graph(graph, surface_residues):
    mapped_surface_nodes = []
    for surface_res in surface_residues:
        res_name = surface_res.get_resname()
        res_id = surface_res.id[1]
        surface_node_key = f"{res_name}_{res_id}"
        
        # Check if this residue is in the graph
        if surface_node_key in graph.nodes:
            mapped_surface_nodes.append(surface_node_key)
    return mapped_surface_nodes

# Step 2: Retrieve adjacent nodes and edges
def get_adjacent_nodes_and_edges(graph, surface_node):
    # Get the adjacent residues (1-hop neighbors)
    neighbors = list(graph.neighbors(surface_node))
    edges = list(graph.edges(surface_node, data=True))  # Edges with data (interaction_type, distance, etc.)
    
    return neighbors, edges

# Step 3: Retrieve adjacent-adjacent nodes and edges
def get_adjacent_adjacent_nodes_and_edges(graph, neighbor_nodes):
    adjacent_adjacent_nodes = set()  # To avoid duplicate entries
    adjacent_adjacent_edges = []
    
    for neighbor in neighbor_nodes:
        # Get neighbors of the neighbor (2-hop neighbors)
        next_neighbors = list(graph.neighbors(neighbor))
        next_edges = list(graph.edges(neighbor, data=True))
        
        adjacent_adjacent_nodes.update(next_neighbors)  # Add next level of neighbors
        adjacent_adjacent_edges.extend(next_edges)      # Add next level of edges
    
    return list(adjacent_adjacent_nodes), adjacent_adjacent_edges

# Step 4: Form a subgraph network
def form_subgraph(graph, surface_nodes, adjacent_nodes, adjacent_adjacent_nodes):
    # Combine all nodes of interest
    subgraph_nodes = set(surface_nodes) | set(adjacent_nodes) | set(adjacent_adjacent_nodes)
    
    # Create subgraph
    subgraph = graph.subgraph(subgraph_nodes).copy()
    return subgraph


def map_surface_residues_to_graph(graph, surface_residues):
    """
    Map surface residues to nodes in the graph.
    Returns the mapped surface node keys and a dictionary to retrieve residue objects.
    """
    mapped_surface_nodes = []
    residue_dict = {}

    for surface_res in surface_residues:
        res_name = surface_res.get_resname()
        res_id = surface_res.id[1]
        surface_node_key = f"{res_name}_{res_id}"

        # Check if this residue is in the graph
        if surface_node_key in graph.nodes:
            mapped_surface_nodes.append(surface_node_key)
            residue_dict[surface_node_key] = surface_res  # Store residue for later retrieval

    return mapped_surface_nodes, residue_dict


def get_adjacent_residues(graph, node, residue_dict):
    """
    Get adjacent and adjacent-adjacent residues for a given node.
    Returns adjacent residues and adjacent-adjacent residues.
    """
    adjacent_residues = []
    adjacent_adjacent_residues = []

    # Get adjacent nodes
    adjacent_nodes = list(graph.neighbors(node))

    for adj_node in adjacent_nodes:
        if adj_node in residue_dict:
            adjacent_residues.append(residue_dict[adj_node])

        # Get adjacent-adjacent nodes
        adjacent_adjacent_nodes = list(graph.neighbors(adj_node))
        for adj_adj_node in adjacent_adjacent_nodes:
            if adj_adj_node in residue_dict:
                adjacent_adjacent_residues.append(residue_dict[adj_adj_node])

    return adjacent_residues, adjacent_adjacent_residues




def compute_atom_type_vector(residue, atom_type_categories):
    """
    Compute the atom type vector for a given residue.
    The vector counts atoms for each category in atom_type_categories using the amino_acid_atoms mapping.
    """
    atom_type_vector = {category: 0 for category in atom_type_categories}

    # Get the residue name (three-letter code, e.g., "ALA", "ARG")
    residue_name = residue.get_resname()
    #print(f"Residue {residue_name}")
    
    # Fetch the atom-category mappings for this residue
    if residue_name not in amino_acid_atoms:
        #print(f"Residue {residue_name} not found in amino_acid_atoms dictionary")
        return np.zeros(len(atom_type_categories))  # Return zero vector if residue is not mapped

    atom_categories = amino_acid_atoms[residue_name]
    #print(f"atom_categories {atom_categories}")
    
    for atom in residue:
        atom_name = atom.get_name()
        #print(f"atom {atom_name}")
        
        # Check if the atom is in the mapping for this residue
        if atom_name in atom_categories:
            categories = atom_categories[atom_name]
            if categories:  # Some atoms may not belong to any category (None)
                # Categories can be multiple, so split by comma and increment each category
                for category in categories.split(","):
                    if category in atom_type_categories:
                        #print(f"cat {category}")
                        atom_type_vector[category] += 1

    return np.array(list(atom_type_vector.values()))



def compute_interaction_vector(residue, interaction_categories, graph, residue_dict):
    """
    Compute the interaction vector for a given residue.
    The vector counts interactions for each category in interaction_categories.
    """
    interaction_vector = {category: 0 for category in interaction_categories}
    
    residue_node = f"{residue.get_resname()}_{residue.id[1]}"
    
    if residue_node in graph:
        # Get interactions (edges) of the residue with other residues
        for neighbor_node in graph.neighbors(residue_node):
            edge_data = graph.get_edge_data(residue_node, neighbor_node)
            interaction_type = edge_data.get("interaction_type", None)  # You should have interaction_type defined in the graph

            # Increment the interaction category count
            if interaction_type in interaction_categories:
                interaction_vector[interaction_type] += 1

    return np.array(list(interaction_vector.values()))


def compute_feature_vector(residue, atom_type_categories, interaction_categories, graph, residue_dict):
    """
    Compute the feature vector for a given residue.
    Combines atom type vector and interaction vector.
    """
    atom_type_vector = compute_atom_type_vector(residue, atom_type_categories)
    interaction_vector = compute_interaction_vector(residue, interaction_categories, graph, residue_dict)

    # Concatenate atom type vector and interaction vector
    #print(f"atom type vectors {atom_type_vector}")
    #print(f"interaction vectors {interaction_vector}")
    feature_vector = np.concatenate([atom_type_vector, interaction_vector])

    return feature_vector


def sum_feature_vectors(feature_vectors, atom_type_categories, interaction_categories):
    """
    Sum all feature vectors in the list.
    Returns a single summed feature vector and the count of residues.
    """
    if len(feature_vectors) == 0:
        return np.zeros(len(atom_type_categories) + len(interaction_categories)), 0  # Return zero vector and count of 0
    feature_sum = np.sum(feature_vectors, axis=0)
    return feature_sum, len(feature_vectors)  # Return the summed vector and the count of residues



def append_feature_sums_to_csv(protein1, protein2, surface_sum1, surface_count1, adjacent_sum1, adjacent_count1, adjacent_adjacent_sum1, adjacent_adjacent_count1, surface_sum2, surface_count2, adjacent_sum2, adjacent_count2, adjacent_adjacent_sum2, adjacent_adjacent_count2, atom_type_categories, interaction_categories, output_file="demo_residue_features2.csv"):
    """
    Append the summed feature vectors and residue counts for surface, adjacent, and adjacent-adjacent residues to a CSV file.
    """
    # Check if the file exists to write the header only once
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:  # If the file doesn't exist, write the header
            header = ['', ''] + ['Surface_P1']*13 + ['Adjacent_P1']*13 + ['Adjacent-Adjacent_p1']*13 + ['Surface_P2']*13 + ['Adjacent_P2']*13 + ['Adjacent-Adjacent_p2']*13
            subheader = ['P1', 'P2', 'Sur_Res_Count_P1'] + atom_type_categories + interaction_categories+ [ 'Adj_Res_Count_P1'] + atom_type_categories + interaction_categories + ['Adj_Adj_Res_Count_P1'] + atom_type_categories + interaction_categories + ['Sur_Res_Count_P2'] + atom_type_categories + interaction_categories+ [ 'Adj_Res_Count_P2'] + atom_type_categories + interaction_categories + ['Adj_Adj_Res_Count_P2'] + atom_type_categories + interaction_categories
            writer.writerow(header)
            writer.writerow(subheader)

        # Append the feature vectors side-by-side
        row = [protein1, protein2, surface_count1] + list(surface_sum1) + [adjacent_count1] + list(adjacent_sum1) + [adjacent_adjacent_count1] + list(adjacent_adjacent_sum1) + [surface_count2] + list(surface_sum2) + [adjacent_count2] + list(adjacent_sum2) + [adjacent_adjacent_count2] + list(adjacent_adjacent_sum2)
        writer.writerow(row)

def main():
    #  **Feature Extraction and Vectorization**
    # Helper functions to extract features from the protein structure graph
    atom_type_categories = ["ACP", "DON", "POS", "NEG", "HPB", "ARM"]
    interaction_categories = ["HYDROPHOBIC", "SALT_BRIDGE", "ARM_STACK", "H_BOND", "REPULSIVE", "SS_BRIDGE"]

    #  **Generate Feature Vectors from CSV Dataset (`file_final.csv`)**
    # Load the CSV dataset containing protein pairs and class labels
    df_pairs = pd.read_csv('file_final.csv')  # Load your dataset here (replace demo.csv with actual file)

    # Iterate over protein pairs and generate feature vectors
    feature_vectors = []
    for index, row in df_pairs.iterrows():
        protein1 = row['P1']  # Assuming 'P1' is the protein 1 column
        protein2 = row['P2']  # Assuming 'P2' is the protein 2 column
        
        # P1
        surface_residues_p1 = calculate_sasa('files/' + protein1)
        residues_p1 = parse_pdb('files/' + protein1)
        interactions_p1 = compute_residue_interactions(residues_p1, interaction_criteria)
        graph_p1 = generate_residue_graph(residues_p1, interactions_p1)
        mapped_surface_nodes_p1, residue_dict_p1 = map_surface_residues_to_graph(graph_p1, surface_residues_p1)
        
        # Initialize lists to store feature vectors for each category of residues
        surface_feature_vectors_p1 = []
        adjacent_feature_vectors_p1 = []
        adjacent_adjacent_feature_vectors_p1 = []
        
        # Loop over all mapped surface nodes
        for surface_node_p1 in mapped_surface_nodes_p1:
            # Get the adjacent and adjacent-adjacent residues for the current surface node
            adjacent_residues_p1, adjacent_adjacent_residues_p1 = get_adjacent_residues(graph_p1, surface_node_p1, residue_dict_p1)
            
            # Retrieve the corresponding surface residue object from residue_dict
            surface_residue_p1 = residue_dict_p1[surface_node_p1]
            #print(surface_residue_p1)
            #print(graph_p1)
            #print(residue_dict_p1)
            
            # Compute the feature vector for the surface residue
            surface_feature_vector_p1 = compute_feature_vector(surface_residue_p1, atom_type_categories, interaction_categories, graph_p1, residue_dict_p1)
            surface_feature_vectors_p1.append(surface_feature_vector_p1)

            

            #print(surface_feature_vectors_p1)

            # Compute feature vectors for adjacent residues
            for adj_residue_p1 in adjacent_residues_p1:
                adj_feature_vector_p1 = compute_feature_vector(adj_residue_p1, atom_type_categories, interaction_categories, graph_p1, residue_dict_p1)
                adjacent_feature_vectors_p1.append(adj_feature_vector_p1)

            # Compute feature vectors for adjacent-adjacent residues
            for adj_adj_residue_p1 in adjacent_adjacent_residues_p1:
                adj_adj_feature_vector_p1 = compute_feature_vector(adj_adj_residue_p1, atom_type_categories, interaction_categories, graph_p1, residue_dict_p1)
                adjacent_adjacent_feature_vectors_p1.append(adj_adj_feature_vector_p1)
            
        
        
        # P2
        surface_residues_p2 = calculate_sasa('files/' + protein2)
        residues_p2 = parse_pdb('files/' + protein2)
        interactions_p2 = compute_residue_interactions(residues_p2, interaction_criteria)
        graph_p2 = generate_residue_graph(residues_p2, interactions_p2)
        mapped_surface_nodes_p2, residue_dict_p2 = map_surface_residues_to_graph(graph_p2, surface_residues_p2)
        
        # Initialize lists to store feature vectors for each category of residues
        surface_feature_vectors_p2 = []
        adjacent_feature_vectors_p2 = []
        adjacent_adjacent_feature_vectors_p2 = []
        
        
        # Loop over all mapped surface nodes
        for surface_node_p2 in mapped_surface_nodes_p2:
            # Get the adjacent and adjacent-adjacent residues for the current surface node
            adjacent_residues_p2, adjacent_adjacent_residues_p2 = get_adjacent_residues(graph_p2, surface_node_p2, residue_dict_p2)

            # Retrieve the corresponding surface residue object from residue_dict
            surface_residue_p2 = residue_dict_p2[surface_node_p2]

            # Compute the feature vector for the surface residue
            surface_feature_vector_p2 = compute_feature_vector(surface_residue_p2, atom_type_categories, interaction_categories, graph_p2, residue_dict_p2)
            surface_feature_vectors_p2.append(surface_feature_vector_p2)

            # Compute feature vectors for adjacent residues
            for adj_residue_p2 in adjacent_residues_p2:
                adj_feature_vector_p2 = compute_feature_vector(adj_residue_p2, atom_type_categories, interaction_categories, graph_p2, residue_dict_p2)
                adjacent_feature_vectors_p2.append(adj_feature_vector_p2)

            # Compute feature vectors for adjacent-adjacent residues
            for adj_adj_residue_p2 in adjacent_adjacent_residues_p2:
                adj_adj_feature_vector_p2 = compute_feature_vector(adj_adj_residue_p2, atom_type_categories, interaction_categories, graph_p2, residue_dict_p2)
                adjacent_adjacent_feature_vectors_p2.append(adj_adj_feature_vector_p2)
        
        
        # Sum the feature vectors and get residue counts
        surface_sum_p1, surface_count_p1 = sum_feature_vectors(surface_feature_vectors_p1, atom_type_categories, interaction_categories)
        adjacent_sum_p1, adjacent_count_p1 = sum_feature_vectors(adjacent_feature_vectors_p1, atom_type_categories, interaction_categories)
        adjacent_adjacent_sum_p1, adjacent_adjacent_count_p1 = sum_feature_vectors(adjacent_adjacent_feature_vectors_p1, atom_type_categories, interaction_categories)
        
        # Sum the feature vectors and get residue counts
        surface_sum_p2, surface_count_p2 = sum_feature_vectors(surface_feature_vectors_p2, atom_type_categories, interaction_categories)
        adjacent_sum_p2, adjacent_count_p2 = sum_feature_vectors(adjacent_feature_vectors_p2, atom_type_categories, interaction_categories)
        adjacent_adjacent_sum_p2, adjacent_adjacent_count_p2 = sum_feature_vectors(adjacent_adjacent_feature_vectors_p2, atom_type_categories, interaction_categories)

        
        # Save the results to a CSV file
        append_feature_sums_to_csv(protein1, protein2, surface_sum_p1, surface_count_p1, adjacent_sum_p1, adjacent_count_p1, adjacent_adjacent_sum_p1, adjacent_adjacent_count_p1, surface_sum_p2, surface_count_p2, adjacent_sum_p2, adjacent_count_p2, adjacent_adjacent_sum_p2, adjacent_adjacent_count_p2, atom_type_categories, interaction_categories, output_file="classical_residue_features.csv")
        print(index)
        
            
    print("residue_feature_sums is saved in residue_feature_sums5")


if __name__ == "__main__":
    main()