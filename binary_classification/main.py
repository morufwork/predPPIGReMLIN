import contacts as ct
import graphprocessing as gp
import clustering as cl
import graphmining as gm
import common as cm
import logging
import graphmatcher as matcher
import networkx as nx
import sys
import json
from pathlib import Path
import pandas as pd        
import numpy as np
import os
from networkx.algorithms import isomorphism
import networkx as nx
import metrics_utils as mu
import graph_feature_extraction as gfe
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import numpy as np
import matplotlib.pyplot as plt

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
#logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.info("---Clustering analysis with 5-fold CV on each cluster---")


GSPANPATH = os.getcwd() + "/gSpan/gSpan-64"
if __name__ == '__main__': 
    
    interactions,int_list = ct.readInteractions("interactions.csv")
    a_types,a_type_list = ct.readAtom_Types("atom_types.csv")
    typeCode = cm.TypeCode(a_type_list,int_list)
    typenames = cm.TypeMap("typenames.json")
    path = Path("gSpan/") 
    path.mkdir(parents=True,exist_ok=True)
    path = str(path)
    logging.basicConfig(level=logging.DEBUG)
    
    
    ############## Data Graph
    logging.info("---Read graphs file <<DataGraphs>>---")
    dgraphs,dnode_labels,dedge_labels = gp.read_graphs('dgraphs.txt',path=path)

    ############## Query Graph
    logging.info("---Read graphs file <<DataGraphs>>---")
    qgraphs,qnode_labels,qedge_labels = gp.read_graphs('qgraphs.txt',path=path)


    ############## Test Graph
    logging.info("---Read graphs file <<TestGraphs>>---")
    tgraphs,tnode_labels,tedge_labels = gp.read_graphs('tgraphs.txt',path=path)

    # qgraphs on data graphs = all positive dataGraphPattern and all negative dataGraphPattern
    logging.info("--- qgraphs on data graphs = all positive dataGraphPattern and all negative dataGraphPattern ---")
    dgraphs1 = dgraphs
    qgraphs1 = qgraphs

    notIso = []  
    notIsoD = dict()
    Iso = []
    IsoD = dict()

    print("dgraphs:", len(dgraphs1))
    for query in qgraphs1:
        for graph in dgraphs1:
            g1, g2 = graph, query
            
            if g1.number_of_nodes() < g2.number_of_nodes():
                g1, g2 = g2, g1
            
            GM = isomorphism.GraphMatcher(g1, g2)
            is_isomorphic = GM.is_isomorphic()

            if is_isomorphic:
                Iso.append(graph)
                IsoD[graph.graph['id']] = graph
            else:
                notIso.append(graph)
                notIsoD[graph.graph['id']] = graph
            
        
        isoD2 = notIsoD.copy()
        notIsoD.clear()
        dgraphs1 = []
        dgraphs1 = notIso
        notIso = []


    allposdict = {}
    allposlist = []
    allnegdict = {}
    allneglist = []
    allposdict.update(IsoD)
    allposlist = Iso
    allnegdict.update(isoD2)
    allneglist = dgraphs1

    print("All Positive Dictionary:", len(allposdict))
    print("All Positive List:", len(allposlist))
    print("All Negative Dictionary:", len(allnegdict))
    print("All Negative List:", len(allneglist))


    # Keep the raw process file
    logging.info("---Keep the raw process file---")
    allposdict2 = {}
    allposlist2 = []
    allnegdict2 = {}
    allneglist2 = []
    allposdict2.update(allposdict)
    allposlist2= allposlist
    allnegdict2.update(allnegdict)
    allneglist2 = allneglist

    # test graphs on all positive dataGraphPattern = TP + FN
    logging.info("---Test graphs on all positive dataGraphPattern = TP + FN ---")
    PnotIso = []  
    PnotIsoD = dict()
    PIso = []
    PIsoD = dict()
    tgraphs1 = tgraphs

    print("allpositive:", len(allposdict2))

    for test in tgraphs1:
        for k, pattern in allposdict2.items():
            g1, g2 = pattern, test
            
            if g1.number_of_nodes() < g2.number_of_nodes():
                g1, g2 = g2, g1
                
            
            # Check for subgraph isomorphism
            GM = isomorphism.GraphMatcher(g1, g2)
            is_isomorphic = GM.is_isomorphic()

            if is_isomorphic:
                PIso.append(pattern)
                PIsoD[k] = pattern
            else:
                PnotIso.append(pattern)
                PnotIsoD[k] = pattern

        PnotIsoD2 = PnotIsoD.copy()
        allposdict2.clear()
        allposdict2.update(PnotIsoD)
        PnotIsoD.clear()

    TP = len(PIso)
    FN = len(PnotIsoD2)

    print("TP:", TP)
    print("FN:", FN)


    # test graphs on all negative dataGraphPattern = TN + FP
    logging.info("---Test graphs on all negative dataGraphPattern = TN + FP ---")
    PnotIso = []  
    PnotIsoD = dict()
    PIso = []
    PIsoD = dict()
    tgraphs1 = tgraphs

    print("allnegative:", len(allnegdict2))

    for test in tgraphs1:
        for k, pattern in allnegdict2.items():
            g1, g2 = pattern, test
            
            if g1.number_of_nodes() < g2.number_of_nodes():
                g1, g2 = g2, g1
            
            
            # Check for subgraph isomorphism
            GM = isomorphism.GraphMatcher(g1, g2)
            is_isomorphic = GM.is_isomorphic()

            if is_isomorphic:
                PIso.append(pattern)
                PIsoD[k] = pattern
            else:
                PnotIso.append(pattern)
                PnotIsoD[k] = pattern

        PnotIsoD2 = PnotIsoD.copy()
        allnegdict2.clear()
        allnegdict2.update(PnotIsoD)
        PnotIsoD.clear()
        

    FP = len(PIso)
    TN = len(PnotIsoD2)

    print("FP:", FP)
    print("TN:", TN)       

    logging.info("---Evaluate the strategy ---")
    mu.print_metrics(TP, FP, TN, FN)   


    # Generate graph features and anotate for all positive graphpattern and all negative graph pattern
    logging.info("---Generate graph features and annotate for all positive graphpattern and all negative graph pattern ---")

    class_graph1 = allposlist2
    class_graph0 = allneglist2

    #get all possible residues
    all_residues = gfe.collect_all_residue_names(class_graph1 + class_graph0)
    print(all_residues)

    #get all node types
    all_node_types = gfe.collect_all_node_types(class_graph1 + class_graph0)
    print(all_node_types)

    #get all edge types
    all_edge_types = gfe.collect_all_edge_types(class_graph1 + class_graph0)
    print(all_edge_types)

    #Extract features from each graph for all positive graph pattern and all negative graph pattern
    data1 = [gfe.graph_to_feature_row(g, all_residues, all_edge_types, all_node_types) for g in class_graph1]
    data0 = [gfe.graph_to_feature_row(g, all_residues, all_edge_types, all_node_types) for g in class_graph0]

    #Annotate for all positive graph features
    df1 = pd.DataFrame(data1)
    df1["class"] = 1 

    #Annotate for all negative graph features
    df2 = pd.DataFrame(data0)
    df2["class"] = 0 

    # Merge the dataset
    logging.info("---Merge the dataset ---")

    merged_df = pd.concat([df1, df2], ignore_index=True)

    # save to CSV
    merged_df.to_csv("all_features.csv", index=False)
    print("Merged and saved as all_graph_features.csv")


    # === Load dataset ===
    logging.info("---Machine learning classifier for ROCAUC and PRAUC ---")
    df = pd.read_csv("all_features.csv")
    df.fillna(0, inplace=True)

    # === Prepare X and y ===
    X_all = df.drop(columns=["class"])
    y_all = df["class"].values

    # === Thresholds for clustering ===
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results_summary = []

    # To store the best for curve generation
    best_auc = -1
    best_curves = {}

    # === Main loop ===
    for threshold in thresholds:
        n_clusters = int(threshold * 10)
        logging.info(f"Clustering with threshold {threshold} → {n_clusters} clusters")

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_all)

        all_y_true = []
        all_y_scores = []

        # Loop through each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            X_cluster = X_all[cluster_mask]
            y_cluster = y_all[cluster_mask]

            if len(np.unique(y_cluster)) < 2 or len(y_cluster) < 10:
                # Skip if too small or single-class
                continue

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, test_idx in skf.split(X_cluster, y_cluster):
                X_train, X_test = X_cluster.iloc[train_idx], X_cluster.iloc[test_idx]
                y_train, y_test = y_cluster[train_idx], y_cluster[test_idx]

                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)

                y_scores = clf.predict_proba(X_test)[:, 1]

                all_y_true.extend(y_test)
                all_y_scores.extend(y_scores)

        # After all clusters are processed
        if len(all_y_true) == 0 or len(np.unique(all_y_true)) < 2:
            continue

        roc_auc = roc_auc_score(all_y_true, all_y_scores)
        pr_auc = average_precision_score(all_y_true, all_y_scores)
        logging.info(f"Threshold {threshold}: ROC-AUC={roc_auc:.6f}, PR-AUC={pr_auc:.6f}")

        results_summary.append({
            "threshold": threshold,
            "n_clusters": n_clusters,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

        # Save best for plotting
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_curves = {
                "threshold": threshold,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "y_true": all_y_true.copy(),
                "y_scores": all_y_scores.copy()
            }

    # === Convert results to DataFrame ===
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("cluster_threshold_summary_result.csv", index=False)
    print(summary_df)


    # === Plot & Save ROC and PR Curves ===
    y_true = best_curves["y_true"]
    y_scores = best_curves["y_scores"]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {best_curves['roc_auc']:.6f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Best Threshold = {best_curves['threshold']})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_best_threshold.png", dpi=300)
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {best_curves['pr_auc']:.6f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Best Threshold = {best_curves['threshold']})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pr_curve_best_threshold.png", dpi=300)
    plt.close()

    logging.info("Figures saved as 'roc_curve_best_threshold.png' and 'pr_curve_best_threshold.png'")

    # === 1. Load summary ===
    df = pd.read_csv("cluster_threshold_summary_result.csv")

    # === 2. Plot ROC-AUC vs Threshold ===
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["roc_auc"], marker='o', label="ROC-AUC")
    plt.xlabel("Cluster Threshold")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC vs Cluster Threshold")
    plt.grid(True)
    plt.xticks(df["threshold"])
    plt.ylim(0.998, 1.0001)
    plt.legend()
    plt.tight_layout()

    # Save PNG
    plt.savefig("roc_auc_vs_threshold.png", dpi=300)
    plt.close()

    # === 3. Plot PR-AUC vs Threshold ===
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["pr_auc"], marker='o', label="PR-AUC")
    plt.xlabel("Cluster Threshold")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC vs Cluster Threshold")
    plt.grid(True)
    plt.xticks(df["threshold"])
    plt.ylim(0.998, 1.0001)
    plt.legend()
    plt.tight_layout()

    # Save PNG
    plt.savefig("pr_auc_vs_threshold.png", dpi=300)
    plt.close()

    print("Saved: roc_auc_vs_threshold.png and pr_auc_vs_threshold.png")

            
        