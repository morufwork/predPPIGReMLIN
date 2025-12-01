# Function to generating graph features and annotation
import networkx as nx                 # For graph creation and analysis
import numpy as np                    # For numerical operations like mean
from collections import Counter       # For counting occurrences (residues, node types, edge types)
import logging                        # For logging messages (logging.info)

#logging.info("---Function to generating graph features and anotation ---")

def collect_all_residue_names(graphs):
    residue_names = set()
    for G in graphs:
        for n in G.nodes:
            res = G.nodes[n].get("residueName")
            if res:
                residue_names.add(res)
    return sorted(list(residue_names))

def collect_all_edge_types(graphs):
    edge_types = set()
    for G in graphs:
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type")
            if edge_type:
                edge_types.add(edge_type)
    return sorted(list(edge_types))


def collect_all_node_types(graphs):
    node_types = set()
    for G in graphs:
        for _, data in G.nodes(data=True):
            node_type = data.get("type")
            if node_type:
                node_types.add(node_type)
    return sorted(list(node_types))



def graph_to_feature_row(G, all_residues, all_edge_types, all_node_types):
    # Graph-level
    feats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": int(nx.is_connected(G.to_undirected())),
        "avg_clustering": nx.average_clustering(G.to_undirected()),
        "transitivity": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
    }

    degrees = dict(G.degree())
    feats["max_degree"] = max(degrees.values())
    feats["min_degree"] = min(degrees.values())
    feats["avg_degree"] = np.mean(list(degrees.values()))

    # Centrality
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    #ec = nx.eigenvector_centrality(G, max_iter=1000)
    #ec = nx.eigenvector_centrality_numpy(G)

    feats["avg_betweenness"] = np.mean(list(bc.values()))
    feats["avg_closeness"] = np.mean(list(cc.values()))
    #feats["avg_eigenvector"] = np.mean(list(ec.values()))

    # Diameter and radius (only if connected)
    try:
        feats["diameter"] = nx.diameter(G.to_undirected())
        feats["radius"] = nx.radius(G.to_undirected())
    except:
        feats["diameter"] = None
        feats["radius"] = None

    # === Node metadata aggregation ===
    node_types = [G.nodes[n].get("type") for n in G.nodes() if "type" in G.nodes[n]]
    is_ligands = [G.nodes[n].get("isLigand", 0) for n in G.nodes()]
    residue_names = [G.nodes[n].get("residueName") for n in G.nodes()]

    feats["avg_is_ligand"] = np.mean(is_ligands)
    feats["num_unique_residues"] = len(set(residue_names))

    # Most common residue
    """residue_counts = Counter(residue_names)
    for res, count in residue_counts.items():
        feats[f"residue_{res}_count"] = count"""
        
    # Count residues
    residue_names = [G.nodes[n].get("residueName") for n in G.nodes()]
    residue_counts = Counter(residue_names)

    # One-hot-like encoding: count or binary presence
    for res in all_residues:
        feats[f"residue_{res}_count"] = residue_counts.get(res, 0)  # or just int(res in residue_counts)

    # Node type count
    node_type_list = [G.nodes[n].get("type") for n in G.nodes() if "type" in G.nodes[n]]
    type_counts = Counter(node_type_list)

    for typename in all_node_types:
        feats[f"node_type_{typename}_count"] = type_counts.get(typename, 0)
    

    # === Edge attribute aggregation ===
    distances = [edata.get("distance", 0) for _, _, edata in G.edges(data=True)]
    edge_types = [edata.get("type") for _, _, edata in G.edges(data=True)]

    feats["avg_edge_distance"] = np.mean(distances)
    feats["min_edge_distance"] = np.min(distances)
    feats["max_edge_distance"] = np.max(distances)

    edge_type_counts = Counter(edge_types)
    for etype in all_edge_types:
        feats[f"edge_type_{etype}_count"] = edge_type_counts.get(etype, 0)

    return feats