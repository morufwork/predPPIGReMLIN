import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBParser, NeighborSearch
import freesasa
import gc
import csv

# --- Feature utilities ---
def one_hot_encode(resname):
    residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'VAL']
    encoding = [0] * len(residues)
    if resname in residues:
        encoding[residues.index(resname)] = 1
    return encoding

def get_hydrophobicity(resname):
    hydrophobicity_scale = {
        'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
        'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
        'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
        'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
    }
    return hydrophobicity_scale.get(resname, 0.0)

def get_residue_charge(resname):
    charge_map = {
        'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0
    }
    return charge_map.get(resname, 0.0)


# --- Graph generation from PDB ---
def pdb_to_graph(pdb_path, sasa_threshold=23.5, radius=6.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]

    atoms = [atom for atom in model.get_atoms() if atom.element != 'H']
    ns = NeighborSearch(atoms)

    sasa_structure = freesasa.Structure(pdb_path)
    sasa_result_raw = freesasa.calc(sasa_structure).residueAreas()

    sasa_dict = {}
    for chain_id, res_dict in sasa_result_raw.items():
        for resnum, area_obj in res_dict.items():
            key = f"{chain_id}:{resnum}"
            sasa_dict[key] = area_obj.total

    residues = [res for res in model.get_residues() if res.get_id()[0] == ' ']
    x_list, coord_list, residue_names = [], [], []
    edge_index = []
    surface_indices = []
   

    for i, res in enumerate(residues):
        if 'CA' not in res:
            continue
        resname = res.get_resname()
        chain_id = res.get_parent().id
        resnum = res.get_id()[1]
        sasa_key = f"{chain_id}:{resnum}"
        sasa = sasa_dict.get(sasa_key, 0.0)
        if sasa > sasa_threshold:
            surface_indices.append(i)
        coord = res['CA'].get_coord()
        charge = get_residue_charge(resname)
        hydro = get_hydrophobicity(resname)
        feat = one_hot_encode(resname) + [sasa, hydro, charge] + list(coord)
        x_list.append(feat)
        

 
    num = len(x_list)
    #all_data.append(num)
    # Assuming x_list is a list of equal-length feature vectors
    x_array = np.array(x_list)        # Convert list of lists to NumPy array
    mean_vector = np.mean(x_array, axis=0)  # Compute mean across all residues (rows)
    mean_list = mean_vector.tolist()
    
    # Add num to the list
    mean_list.append(num)
    
    
    return mean_list



# --- Load and test dataset ---
if __name__ == "__main__":
    # Load your DataFrame
    df_pairs = pd.read_csv('file_final.csv')

    # CSV output file path
    csv_file = 'output.csv'

    # Step 1: Prepare and write header if file doesn't exist
    if not os.path.exists(csv_file):
        header = [f'f{i+1}' for i in range(54)] + ['label']
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Step 2: Iterate and append data
    for index, row in df_pairs.iterrows():
        protein1 = row['P1']
        protein2 = row['P2']
        class_label = row['class']

        # Extract features (27 each expected)
        test1 = pdb_to_graph(os.path.join('files', protein1 + '.pdb'))
        test2 = pdb_to_graph(os.path.join('files', protein2 + '.pdb'))


        if test1 is None or test2 is None:
            print(f"Skipping pair {protein1}, {protein2} due to invalid structure.")
            continue

        if len(test1) != 27 or len(test2) != 27:
            print(f"Feature vector length mismatch in {protein1} or {protein2}")
            continue


        final_row = test1 + test2 + [class_label]


        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(final_row)

        print(f"Appended data for: {protein1}, {protein2}")

