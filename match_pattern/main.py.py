import argparse
import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms import isomorphism

import graph_feature_extraction as gfe
import graphprocessing as gp


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

NODE_MATCH = isomorphism.numerical_node_match("type", 0)
EDGE_MATCH = isomorphism.numerical_edge_match("type", 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 graph pattern learning and Phase 2 test feature creation."
    )
    parser.add_argument(
        "--positive-train-graphs",
        default="masif_training_graphs.txt",
        help="Training graphs for interacting protein pairs.",
    )
    parser.add_argument(
        "--negative-train-graphs",
        default=None,
        help="Optional training graphs for non-interacting or mismatched pairs.",
    )
    parser.add_argument(
        "--corpus-graphs",
        default="corpus_filtertest_masif_graphs.txt",
        help="Filtered corpus graphs used for pattern discovery.",
    )
    parser.add_argument(
        "--graph-path",
        default="gSpan/gSpan-64",
        help="Directory containing the graph files.",
    )
    parser.add_argument(
        "--min-graph-size",
        type=int,
        default=3,
        help="Minimum number of nodes allowed during pattern discovery.",
    )
    parser.add_argument(
        "--output-dir",
        default="phase1_output",
        help="Directory for learned patterns and the training feature matrix.",
    )
    parser.add_argument(
        "--test-positive-graphs",
        default="masif_testing_graphs.txt",
        help="Test graphs for interacting protein pairs.",
    )
    parser.add_argument(
        "--test-negative-graphs",
        default=None,
        help="Optional test graphs for non-interacting or mismatched pairs.",
    )
    parser.add_argument(
        "--load-patterns-dir",
        default=None,
        help="Existing output directory containing saved Phase 1 pattern JSON files.",
    )
    return parser.parse_args()


def read_graphs_checked(filename, path):
    graph_path = Path(path) / filename
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graphs, _, _ = gp.read_graphs(filename, path=path)
    return graphs


def filter_by_size(graphs, min_graph_size):
    return [graph for graph in graphs if graph.number_of_nodes() >= min_graph_size]


def to_line_graph(graph):
    line_graph = nx.line_graph(graph)
    transformed = nx.Graph()
    transformed.graph.update(graph.graph)

    for edge_node in line_graph.nodes():
        edge_data = graph.get_edge_data(*edge_node) or {}
        transformed.add_node(edge_node, type=edge_data.get("type", 0))

    for left_node, right_node in line_graph.edges():
        transformed.add_edge(left_node, right_node, type=1)

    return transformed


def is_subgraph_match(target_graph, pattern_graph):
    target_line_graph = to_line_graph(target_graph)
    pattern_line_graph = to_line_graph(pattern_graph)

    if target_line_graph.number_of_nodes() < pattern_line_graph.number_of_nodes():
        return False
    matcher = isomorphism.GraphMatcher(
        target_line_graph,
        pattern_line_graph,
        node_match=NODE_MATCH,
        edge_match=EDGE_MATCH,
    )
    return matcher.subgraph_is_isomorphic()


def is_same_graph(left_graph, right_graph):
    left_line_graph = to_line_graph(left_graph)
    right_line_graph = to_line_graph(right_graph)

    matcher = isomorphism.GraphMatcher(
        left_line_graph,
        right_line_graph,
        node_match=NODE_MATCH,
        edge_match=EDGE_MATCH,
    )
    return matcher.is_isomorphic()


def deduplicate_graphs(graphs):
    unique_graphs = []
    for graph in graphs:
        if not any(is_same_graph(graph, existing) for existing in unique_graphs):
            unique_graphs.append(graph)
    return unique_graphs


def discover_patterns(query_graphs, corpus_graphs):
    matched = []
    unmatched = []

    for corpus_graph in corpus_graphs:
        matched_query = any(is_same_graph(corpus_graph, query_graph) for query_graph in query_graphs)
        if matched_query:
            matched.append(corpus_graph)
        else:
            unmatched.append(corpus_graph)

    return matched, unmatched


def strip_graph(graph):
    clean_graph = nx.Graph()
    clean_graph.graph.update(
        {
            "id": graph.graph.get("id"),
            "title": graph.graph.get("title"),
            "pdbid": graph.graph.get("pdbid"),
            "source": graph.graph.get("source"),
            "target": graph.graph.get("target"),
        }
    )

    for node, data in graph.nodes(data=True):
        clean_graph.add_node(node, type=data.get("type", 0))

    for source, target, data in graph.edges(data=True):
        clean_graph.add_edge(source, target, type=data.get("type", 0))

    return clean_graph


def save_patterns(patterns, output_path, pattern_class):
    rows = []
    for index, graph in enumerate(patterns, start=1):
        rows.append(
            {
                "pattern_id": f"{pattern_class.upper()}_{index}",
                "pattern_class": pattern_class,
                "source_graph_id": graph.graph.get("id"),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "graph_data": nx.node_link_data(strip_graph(graph)),
            }
        )
    output_path.write_text(json.dumps(rows, indent=2))


def load_patterns(patterns_path):
    rows = json.loads(patterns_path.read_text())
    patterns = []
    for row in rows:
        graph = nx.node_link_graph(row["graph_data"])
        patterns.append(graph)
    return patterns


def graph_pair_id(graph, fallback_prefix):
    return str(
        graph.graph.get("title")
        or graph.graph.get("pair_id")
        or graph.graph.get("id")
        or f"{fallback_prefix}_{id(graph)}"
    )


def pattern_column_name(index, pattern):
    source_graph_id = pattern.graph.get("id")
    if source_graph_id is None:
        return f"Pattern_{index}"
    return f"Pattern_{index}_{source_graph_id}"


def compute_majority_label(pattern_values):
    match_count = sum(pattern_values)
    return int(match_count > (len(pattern_values) / 2.0))


def compute_positive_threshold_label(positive_match_count, total_patterns, threshold=0.10):
    if total_patterns <= 0:
        return 0
    return int((positive_match_count / float(total_patterns)) >= threshold)


def compute_match_threshold_labels(matched_patterns_total, total_patterns):
    labels = {"Label_gt_0": int(matched_patterns_total > 0)}
    if total_patterns <= 0:
        for percent in range(1, 11):
            labels[f"Label_gt_{percent}pct"] = 0
        return labels

    match_ratio = matched_patterns_total / float(total_patterns)
    for percent in range(1, 11):
        labels[f"Label_gt_{percent}pct"] = int(match_ratio > (percent / 100.0))
    return labels


def build_feature_row(graph, positive_patterns, negative_patterns, pair_id, label=None):
    row = {"Pair_ID": pair_id}

    positive_match_count = 0

    all_patterns = list(positive_patterns) + list(negative_patterns)
    pattern_values = []
    for index, pattern in enumerate(all_patterns, start=1):
        match_value = int(is_same_graph(graph, pattern))
        row[pattern_column_name(index, pattern)] = match_value
        pattern_values.append(match_value)

        if index <= len(positive_patterns):
            positive_match_count += match_value

    matched_patterns_total = sum(pattern_values)
    negative_match_count = len(all_patterns) - matched_patterns_total
    row["Positive_Match_Count"] = positive_match_count
    row["Negative_Match_Count"] = negative_match_count
    row["Matched_Patterns_Total"] = matched_patterns_total
    row.update(compute_match_threshold_labels(matched_patterns_total, len(all_patterns)))
    row["Majority_Label"] = compute_majority_label(pattern_values)
    row["Label"] = compute_positive_threshold_label(positive_match_count, len(all_patterns))
    return row


def build_feature_matrix_from_groups(graph_groups, positive_patterns, negative_patterns):
    rows = []
    for graphs, prefix, label in graph_groups:
        for graph in graphs:
            pair_id = graph_pair_id(graph, prefix)
            rows.append(build_feature_row(graph, positive_patterns, negative_patterns, pair_id, label=label))
    return pd.DataFrame(rows)


def build_graph_feature_dataframe(graph_groups):
    all_graphs = []
    for graphs, _, _ in graph_groups:
        all_graphs.extend(graphs)

    if not all_graphs:
        return pd.DataFrame(columns=["Pair_ID"])

    all_residues = gfe.collect_all_residue_names(all_graphs)
    all_node_types = gfe.collect_all_node_types(all_graphs)
    all_edge_types = gfe.collect_all_edge_types(all_graphs)

    rows = []
    for graphs, prefix, _ in graph_groups:
        for graph in graphs:
            row = {"Pair_ID": graph_pair_id(graph, prefix)}
            row.update(gfe.graph_to_feature_row(graph, all_residues, all_edge_types, all_node_types))
            rows.append(row)

    return pd.DataFrame(rows)


def load_phase1_pattern_set(patterns_dir):
    positive_patterns = load_patterns(patterns_dir / "phase1_patterns_positive.json")
    negative_patterns = load_patterns(patterns_dir / "phase1_patterns_negative.json")
    return positive_patterns, negative_patterns


def feature_label_columns():
    return ["Label_gt_0"] + [f"Label_gt_{percent}pct" for percent in range(1, 11)] + ["Label"]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_patterns_dir:
        patterns_dir = Path(args.load_patterns_dir)
        positive_patterns, negative_patterns = load_phase1_pattern_set(patterns_dir)
        all_patterns = positive_patterns + negative_patterns
        logging.info("Loaded %s saved patterns from %s", len(all_patterns), patterns_dir)
    else:
        logging.info("Loading positive training graphs from %s", args.positive_train_graphs)
        positive_train_graphs = read_graphs_checked(args.positive_train_graphs, args.graph_path)
        positive_train_graphs = filter_by_size(positive_train_graphs, args.min_graph_size)

        logging.info("Loading corpus graphs from %s", args.corpus_graphs)
        corpus_graphs = read_graphs_checked(args.corpus_graphs, args.graph_path)
        corpus_graphs = filter_by_size(corpus_graphs, args.min_graph_size)

        negative_train_graphs = []
        if args.negative_train_graphs:
            logging.info("Loading negative training graphs from %s", args.negative_train_graphs)
            negative_train_graphs = read_graphs_checked(args.negative_train_graphs, args.graph_path)
            negative_train_graphs = filter_by_size(negative_train_graphs, args.min_graph_size)

        logging.info("Discovering positive patterns using training graphs only")
        positive_patterns, corpus_remainder = discover_patterns(positive_train_graphs, corpus_graphs)

        if negative_train_graphs:
            logging.info("Discovering negative patterns from negative training graphs")
            negative_patterns, _ = discover_patterns(negative_train_graphs, corpus_remainder)
        else:
            logging.info("No negative training graph file provided; using unmatched corpus graphs as mismatched negatives")
            negative_patterns = corpus_remainder

        negative_patterns = deduplicate_graphs(negative_patterns)
        all_patterns = positive_patterns + negative_patterns
        train_groups = [
            (positive_train_graphs, "TRAIN_POS", 1),
            (negative_train_graphs, "TRAIN_NEG", 0),
        ]
        feature_df = build_feature_matrix_from_groups(
            train_groups,
            positive_patterns,
            negative_patterns,
        )
        train_graph_feature_df = build_graph_feature_dataframe(train_groups)
        merged_train_df = train_graph_feature_df.merge(
            feature_df[["Pair_ID"] + feature_label_columns()],
            on="Pair_ID",
            how="left",
        )

        save_patterns(positive_patterns, output_dir / "phase1_patterns_positive.json", "positive")
        save_patterns(negative_patterns, output_dir / "phase1_patterns_negative.json", "negative")
        feature_df.to_csv(output_dir / "phase1_train_features.csv", index=False)
        train_graph_feature_df.to_csv(output_dir / "phase1_train_graph_features.csv", index=False)
        merged_train_df.to_csv(output_dir / "phase1_train_features_with_graph_features.csv", index=False)

        metadata = {
            "positive_train_graphs_file": args.positive_train_graphs,
            "negative_train_graphs_file": args.negative_train_graphs,
            "corpus_graphs_file": args.corpus_graphs,
            "graph_path": args.graph_path,
            "min_graph_size": args.min_graph_size,
            "num_positive_training_graphs": len(positive_train_graphs),
            "num_negative_training_graphs": len(negative_train_graphs),
            "num_corpus_graphs": len(corpus_graphs),
            "num_positive_patterns": len(positive_patterns),
            "num_negative_patterns": len(negative_patterns),
            "num_total_patterns": len(all_patterns),
            "used_test_data": False,
        }
        (output_dir / "phase1_metadata.json").write_text(json.dumps(metadata, indent=2))

        logging.info("Phase 1 artifacts written to %s", output_dir)
        logging.info("Positive patterns: %s", len(positive_patterns))
        logging.info("Negative patterns: %s", len(negative_patterns))
        logging.info("Training matrix shape: %s", feature_df.shape)

    if args.test_positive_graphs or args.test_negative_graphs:
        phase2_patterns_dir = Path(args.load_patterns_dir) if args.load_patterns_dir else output_dir
        positive_patterns, negative_patterns = load_phase1_pattern_set(phase2_patterns_dir)
        all_patterns = positive_patterns + negative_patterns
        logging.info(
            "Phase 2 will match test graphs against %s saved Phase 1 patterns from %s",
            len(all_patterns),
            phase2_patterns_dir,
        )
        test_rows = []

        if args.test_positive_graphs:
            logging.info("Loading positive test graphs from %s", args.test_positive_graphs)
            positive_test_graphs = read_graphs_checked(args.test_positive_graphs, args.graph_path)
            positive_test_graphs = filter_by_size(positive_test_graphs, args.min_graph_size)
            test_rows.extend(
                build_feature_matrix_from_groups(
                    [(positive_test_graphs, "TEST_POS", None)],
                    positive_patterns,
                    negative_patterns,
                ).to_dict("records")
            )

        if args.test_negative_graphs:
            logging.info("Loading negative test graphs from %s", args.test_negative_graphs)
            negative_test_graphs = read_graphs_checked(args.test_negative_graphs, args.graph_path)
            negative_test_graphs = filter_by_size(negative_test_graphs, args.min_graph_size)
            test_rows.extend(
                build_feature_matrix_from_groups(
                    [(negative_test_graphs, "TEST_NEG", None)],
                    positive_patterns,
                    negative_patterns,
                ).to_dict("records")
            )

        test_df = pd.DataFrame(test_rows)
        test_groups = []
        if args.test_positive_graphs:
            test_groups.append((positive_test_graphs, "TEST_POS", None))
        if args.test_negative_graphs:
            test_groups.append((negative_test_graphs, "TEST_NEG", None))
        test_graph_feature_df = build_graph_feature_dataframe(test_groups)
        merged_test_df = test_graph_feature_df.merge(
            test_df[["Pair_ID"] + feature_label_columns()],
            on="Pair_ID",
            how="left",
        )
        test_df.to_csv(output_dir / "phase2_test_features.csv", index=False)
        test_graph_feature_df.to_csv(output_dir / "phase2_test_graph_features.csv", index=False)
        merged_test_df.to_csv(output_dir / "phase2_test_features_with_graph_features.csv", index=False)
        logging.info("Phase 2 test matrix written to %s", output_dir / "phase2_test_features.csv")
        logging.info("Test matrix shape: %s", test_df.shape)


if __name__ == "__main__":
    main()
