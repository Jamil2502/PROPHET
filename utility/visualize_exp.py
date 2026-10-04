import sys
import os
import pickle
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

LOG_NAME = sys.argv[1]
EXAMPLE_ID = int(sys.argv[2])

TEST_GRAPH_PATH = (
    f"heterographs_tracenode/{LOG_NAME}_test.db"
)

W2V_PATH = f"w2v/{LOG_NAME}"

OUTPUT_DIR = f"utility/visualizations/{LOG_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graphs_from_hdf5(filename):

    graphs = []
    labels = []

    with h5py.File(filename, "r") as f:

        for key in f.keys():

            pickled_graph = f[key][()]
            obj = pickle.loads(pickled_graph)

            graphs.append(obj["graph"])
            labels.append(obj["label"])

    return graphs, labels


# ============================================================
# LOAD EXPLANATION
# ============================================================

def load_feature_explanation(log_name, node_type):

    filename = (
        f"utility/{log_name}_feature_{node_type}.pickle"
    )

    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)


def load_edge_explanation(log_name, edge_type):

    filename = (
        f"utility/{log_name}_edge_{edge_type}.pickle"
    )

    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD W2V DICTIONARY
# ============================================================

def load_dictionary(log_name, node_type):

    filename = (
        f"w2v/{log_name}/{node_type}.pkl"
    )

    if not os.path.exists(filename):
        return {}

    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# CONVERT NODE FEATURE TO HUMAN-READABLE VALUE
# ============================================================

def decode_node_value(graph, node_type, node_id):

    dictionary = load_dictionary(
        LOG_NAME,
        node_type
    )

    if not dictionary:
        return f"{node_type}:{node_id}"

    try:

        # -----------------------------------------------
        # Get feature stored in graph
        # -----------------------------------------------

        value = graph.nodes[node_type].data[node_type][node_id]

        if isinstance(value, torch.Tensor):

            value = (
                value
                .detach()
                .cpu()
                .numpy()
            )

        value = np.asarray(value).squeeze()

        # -----------------------------------------------
        # Compare with dictionary values
        # -----------------------------------------------

        best_key = None
        best_distance = float("inf")

        for original_value, encoded_value in dictionary.items():

            encoded_value = np.asarray(
                encoded_value
            ).squeeze()

            if encoded_value.shape != value.shape:
                continue

            try:

                distance = np.linalg.norm(
                    encoded_value.astype(float)
                    -
                    value.astype(float)
                )

            except Exception:
                continue

            if distance < best_distance:

                best_distance = distance
                best_key = original_value

        if best_key is not None:

            return str(best_key)

    except Exception:
        pass

    return f"{node_type}:{node_id}"


# ============================================================
# GET LABEL
# ============================================================

def decode_label(label):

    label = np.asarray(label)

    if label.ndim > 0:

        return int(np.argmax(label))

    return int(label)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    graph,
    example_id
):

    results = []

    for node_type in graph.ntypes:

        explanation = load_feature_explanation(
            LOG_NAME,
            node_type
        )

        if explanation is None:
            continue

        if example_id >= len(explanation):
            continue

        mask = explanation[example_id]

        if isinstance(mask, torch.Tensor):

            mask = (
                mask
                .detach()
                .cpu()
                .numpy()
            )

        mask = np.asarray(mask)

        # ------------------------------------------------
        # Flatten feature mask
        # ------------------------------------------------

        mask = np.abs(mask)

        if mask.ndim == 0:

            score = float(mask)

        else:

            score = float(
                np.mean(mask)
            )

        results.append(
            (
                node_type,
                score
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ============================================================
# EDGE IMPORTANCE
# ============================================================

def get_edge_importance(
    graph,
    example_id
):

    results = []

    for edge_type in graph.canonical_etypes:

        explanation = load_edge_explanation(
            LOG_NAME,
            edge_type
        )

        if explanation is None:
            continue

        if example_id >= len(explanation):
            continue

        mask = explanation[example_id]

        if isinstance(mask, torch.Tensor):

            mask = (
                mask
                .detach()
                .cpu()
                .numpy()
            )

        mask = np.asarray(mask)

        mask = np.abs(mask)

        if mask.size == 0:
            continue

        score = float(
            np.mean(mask)
        )

        results.append(
            (
                edge_type,
                score
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ============================================================
# VISUALIZE FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    graph,
    example_id
):

    results = get_feature_importance(
        graph,
        example_id
    )

    if not results:

        print(
            "No feature explanations found."
        )

        return

    # ------------------------------------------------
    # Take top 15
    # ------------------------------------------------

    results = results[:15]

    names = [
        x[0]
        for x in results
    ]

    scores = [
        x[1]
        for x in results
    ]

    names = names[::-1]
    scores = scores[::-1]

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        names,
        scores
    )

    plt.xlabel(
        "Mean absolute feature importance"
    )

    plt.ylabel(
        "Node type"
    )

    plt.title(
        f"PROPHET Feature Importance\n"
        f"{LOG_NAME} — Example {example_id}"
    )

    plt.tight_layout()

    filename = (
        f"{OUTPUT_DIR}/"
        f"example_{example_id}_features.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Feature visualization saved:"
    )

    print(
        f"  {filename}"
    )


# ============================================================
# VISUALIZE GRAPH
# ============================================================

def plot_graph(
    graph,
    label,
    example_id
):

    G = nx.MultiDiGraph()

    # ------------------------------------------------
    # Add nodes
    # ------------------------------------------------

    for node_type in graph.ntypes:

        count = graph.num_nodes(
            node_type
        )

        for node_id in range(count):

            value = decode_node_value(
                graph,
                node_type,
                node_id
            )

            node_name = (
                f"{node_type}\n"
                f"{value}"
            )

            G.add_node(
                f"{node_type}:{node_id}",
                node_type=node_type,
                value=value,
                label=node_name
            )

    # ------------------------------------------------
    # Add edges
    # ------------------------------------------------

    for edge_type in graph.canonical_etypes:

        src_type, relation, dst_type = (
            edge_type
        )

        src, dst = graph.edges(
            etype=edge_type
        )

        src = src.cpu().numpy()
        dst = dst.cpu().numpy()

        for s, d in zip(src, dst):

            G.add_edge(
                f"{src_type}:{s}",
                f"{dst_type}:{d}",
                relation=relation
            )

    # ------------------------------------------------
    # Layout
    # ------------------------------------------------

    if len(G.nodes) == 0:
        return

    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.5,
        iterations=100
    )

    plt.figure(
        figsize=(20, 16)
    )

    # ------------------------------------------------
    # Draw nodes
    # ------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1800,
        alpha=0.9
    )

    # ------------------------------------------------
    # Draw edges
    # ------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.35,
        arrows=True,
        arrowsize=12
    )

    # ------------------------------------------------
    # Labels
    # ------------------------------------------------

    labels = {}

    for node in G.nodes:

        labels[node] = G.nodes[node]["label"]

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=7
    )

    # ------------------------------------------------
    # Edge relation labels
    # ------------------------------------------------

    edge_labels = {}

    for u, v, data in G.edges(
        data=True
    ):

        edge_labels[
            (u, v)
        ] = data["relation"]

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=5
    )

    predicted_label = "N/A"

    plt.title(
        f"PROPHET Explanation Graph\n"
        f"Dataset: {LOG_NAME} | "
        f"Example: {example_id} | "
        f"True class: {predicted_label}",
        fontsize=16
    )

    plt.axis("off")

    plt.tight_layout()

    filename = (
        f"{OUTPUT_DIR}/"
        f"example_{example_id}_graph.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graph visualization saved:"
    )

    print(
        f"  {filename}"
    )


# ============================================================
# PRINT TOP FEATURES
# ============================================================

def print_top_features(
    graph,
    example_id
):

    print()
    print("=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)

    results = get_feature_importance(
        graph,
        example_id
    )

    for i, (name, score) in enumerate(
        results[:10],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"{name:30s} "
            f"{score:.6f}"
        )


# ============================================================
# PRINT TOP EDGES
# ============================================================

def print_top_edges(
    graph,
    example_id
):

    print()
    print("=" * 70)
    print("TOP EDGE IMPORTANCE")
    print("=" * 70)

    results = get_edge_importance(
        graph,
        example_id
    )

    for i, (edge_type, score) in enumerate(
        results[:15],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"{edge_type} "
            f"score={score:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def visualize():

    print("=" * 70)
    print("PROPHET EXPLANATION VISUALIZATION")
    print("=" * 70)

    print(
        f"Dataset       : {LOG_NAME}"
    )

    print(
        f"Example       : {EXAMPLE_ID}"
    )

    # ------------------------------------------------
    # Load test graphs
    # ------------------------------------------------

    X_test, y_test = (
        load_graphs_from_hdf5(
            TEST_GRAPH_PATH
        )
    )

    if EXAMPLE_ID >= len(X_test):

        raise IndexError(
            f"Example {EXAMPLE_ID} does not exist. "
            f"Only {len(X_test)} examples available."
        )

    graph = X_test[EXAMPLE_ID]
    label = y_test[EXAMPLE_ID]

    print(
        f"Label         : {label}"
    )

    print(
        f"Node types    : {graph.ntypes}"
    )

    print(
        f"Number nodes  : "
        f"{sum(graph.num_nodes(n) for n in graph.ntypes)}"
    )

    print(
        f"Number edges  : "
        f"{sum(graph.num_edges(e) for e in graph.canonical_etypes)}"
    )

    # ------------------------------------------------
    # Check explanation files
    # ------------------------------------------------

    print()
    print("Checking explanation files...")

    feature_files = []

    for node_type in graph.ntypes:

        filename = (
            f"utility/"
            f"{LOG_NAME}_feature_{node_type}.pickle"
        )

        if os.path.exists(filename):

            feature_files.append(
                node_type
            )

    edge_files = []

    for edge_type in graph.canonical_etypes:

        filename = (
            f"utility/"
            f"{LOG_NAME}_edge_{edge_type}.pickle"
        )

        if os.path.exists(filename):

            edge_files.append(
                edge_type
            )

    print(
        f"Feature explanation files: "
        f"{len(feature_files)}"
    )

    print(
        f"Edge explanation files: "
        f"{len(edge_files)}"
    )

    # ------------------------------------------------
    # Print important features
    # ------------------------------------------------

    print_top_features(
        graph,
        EXAMPLE_ID
    )

    # ------------------------------------------------
    # Print important edges
    # ------------------------------------------------

    print_top_edges(
        graph,
        EXAMPLE_ID
    )

    # ------------------------------------------------
    # Feature plot
    # ------------------------------------------------

    plot_feature_importance(
        graph,
        EXAMPLE_ID
    )

    # ------------------------------------------------
    # Graph plot
    # ------------------------------------------------

    plot_graph(
        graph,
        label,
        EXAMPLE_ID
    )

    # ------------------------------------------------
    # Done
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    visualize()