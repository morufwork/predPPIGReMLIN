import contacts as ct
import graphprocessing as gp
import graphmining as gm
import common as cm
import logging
import graphmatcher as matcher
import networkx as nx
import sys
import json
from pathlib import Path
import numpy as np
import os
from networkx.algorithms import isomorphism
import csv
import pandas as pd

GSPANPATH = os.getcwd() + "/gSpan/gSpan-64"

if __name__ == '__main__':

    # ---------- READ INPUT FILES ----------
    interactions, int_list = ct.readInteractions("interactions.csv")
    a_types, a_type_list = ct.readAtom_Types("atom_types.csv")
    typeCode = cm.TypeCode(a_type_list, int_list)
    typenames = cm.TypeMap("typenames.json")

    path = Path("gSpan")
    path.mkdir(parents=True, exist_ok=True)
    path = str(path)

    logging.basicConfig(level=logging.DEBUG)

    logging.info("--- Read graphs file <<DataGraphs>> ---")
    dgraphs, dnode_labels, dedge_labels = gp.read_graphs('dgraphs.txt', path=path)

    logging.info("--- Read graphs file <<QueryGraphs>> ---")
    qgraphs, qnode_labels, qedge_labels = gp.read_graphs('graphs.txt', path=path)

    # ---------- READ TARGET CHAIN CSV ----------
    logging.info("--- Load target chains ---")
    csv_file_path = "target_chain.csv"
    data_dict = {}

    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:
                data_dict[row[0]] = row[1]

    # A list to store generated DataFrames
    dataframes = []

    # ---------- Helper functions ----------
    def get_key_position(dictionary, key):
        keys = list(dictionary.keys())
        return keys.index(key) if key in keys else -1

    def get_key_value_at_position(dictionary, position):
        keys_list = list(dictionary.keys())
        if position < 0 or position >= len(keys_list):
            raise IndexError("Position out of range")
        key = keys_list[position]
        value = dictionary[key]
        return key, value

    id = 0
    i = 0

    # ---------- MAIN LOOP ----------
    logging.info("--- Pattern search and chain mapping ---")

    for D in dgraphs:

        for Q in qgraphs:

            # Only process query graphs with >= 6 nodes
            if Q.number_of_nodes() >= 6:

                pdbid = Q.graph.get('pdbid', None)
                TargSize = Q.number_of_nodes()
                LigSize = D.number_of_nodes()

                # Build line graphs
                graph_dict = {"graph": D, "l_graph": gp.line_graph(D)}
                pattern_dict = {"graph": Q, "l_graph": gp.line_graph(Q)}

                nm = lambda x, y: x['type'] >= y['type'] and (x['type'] & y['type'])
                em = nx.isomorphism.numerical_node_match("type", 0)

                m = nx.isomorphism.GraphMatcher(
                    graph_dict["l_graph"],
                    pattern_dict["l_graph"],
                    edge_match=em,
                    node_match=nm
                )

                if m.subgraph_is_isomorphic():

                    pdbid2 = D.graph.get('pdbid', None)

                    # Chain mapping
                    chain_mappings = {}

                    for sub_edge, graph_edge in m.mapping.items():

                        u, v = sub_edge
                        u2, v2 = graph_edge

                        sub_chain = D.nodes[u].get('chain') or D.nodes[v].get('chain')
                        graph_chain = Q.nodes[u2].get('chain') or Q.nodes[v2].get('chain')

                        if graph_chain not in chain_mappings:
                            chain_mappings[graph_chain] = []

                        chain_mappings[graph_chain].append(((u, v), (u2, v2)))

                    # Convert mappings into map_dict
                    map_dict = {}
                    nmap = 0

                    for chain, mappings in chain_mappings.items():
                        temp_list = []
                        for sub_edge, graph_edge in mappings:
                            u, v = sub_edge
                            s_chain = D.nodes[u].get('chain') or D.nodes[v].get('chain')
                            temp_list.append(s_chain)
                            nmap += 1
                        map_dict[chain] = temp_list

                    # Get target chain from CSV
                    key_to_find = data_dict.get(pdbid)

                    if key_to_find is None:
                        continue  # skip if pdbid not in CSV

                    position = get_key_position(map_dict, key_to_find)

                    # Choose alternative chain if needed
                    if position == 0 and len(map_dict) > 1:
                        alt_position = 1
                    elif position != 0:
                        alt_position = 0
                    else:
                        alt_position = 0

                    key, value = get_key_value_at_position(map_dict, alt_position)

                    # ---------- Build DataFrame ----------
                    data = {
                        'S/N': [i],
                        'PDBID_Lig': [pdbid2],
                        'PDBID_Lig_Size': [LigSize],
                        'PDBID_Targ': [pdbid],
                        'PDBID_Targ_Size': [TargSize],
                        'No_Map': [nmap],
                        'Targ_Chain': [key_to_find],
                        'Lig_Chain': [value[0]]
                    }

                    df = pd.DataFrame(data)
                    dataframes.append(df)
                    id += 1

        #i += 1

        # Stop after 2 D-graphs
        #if i == 2:
            #break

    # ---------- Final Output ----------
    final_df = pd.concat(dataframes, ignore_index=True)

    output_file_path = "result.csv"
    final_df.to_csv(output_file_path, index=False)

    print("CSV file saved successfully:", output_file_path)

    # ---------- Selecting Best Ligand per Target ----------
    logging.info("--- Selecting best ligand per target ---")

    final_df = final_df[final_df["No_Map"] >= 6].copy()

    # Step 2: scoring
    final_df["map_density"] = final_df["No_Map"] / final_df["PDBID_Targ_Size"].clip(lower=1)

    final_df["size_similarity"] = (
        final_df[["PDBID_Lig_Size", "PDBID_Targ_Size"]].min(axis=1)
        / final_df[["PDBID_Lig_Size", "PDBID_Targ_Size"]].max(axis=1)
    )

    final_df["cross_chain"] = (
        (final_df["Lig_Chain"] != final_df["Targ_Chain"]).astype(float)
        .replace({0.0: 0.9, 1.0: 1.0})
    )

    final_df["pick_score"] = (
        0.6 * final_df["map_density"]
        + 0.4 * final_df["size_similarity"]
    )

    # Step 3: Best ligand per (target, target_chain)
    final_df_best = (
        final_df.sort_values("pick_score", ascending=False)
                .groupby(["PDBID_Targ", "Targ_Chain"], as_index=False)
                .head(100)
                .reset_index(drop=True)
    )

    # Optional: top 10 per target chain
    final_df_top10 = (
        final_df.sort_values("pick_score", ascending=False)
                .groupby(["PDBID_Targ", "Targ_Chain"], as_index=False)
                .head(100)
                .reset_index(drop=True)
    )

    # Save results
    final_df_best.to_csv("dock_best.csv", index=False)
    final_df_top10.to_csv("dock_top.csv", index=False)
