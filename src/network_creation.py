import pandas as pd
import itertools
from collections import Counter
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import networkx as nx

def build_edge_df(data_df, id_col='author', post_col='link_id', min_weight=5, max_authors=50):
    """
    Builds an edge dataframe between users who commented on the same post.
    Keeps only edges with co-comment count >= min_weight.
    """

    df = data_df.dropna(subset=[id_col, post_col]).copy()
    df[post_col] = df[post_col].str.replace(r'^t\\d_', '', regex=True)

    edge_counts = Counter()

    for _, group in df.groupby(post_col):
        authors = group[id_col].unique()

        if len(authors) >= max_authors:
            continue

        for a1, a2 in itertools.combinations(authors, 2):
            edge_counts[tuple(sorted((a1, a2)))] += 1

    # Convert to dataframe
    edges = [(a1, a2, w) for (a1, a2), w in edge_counts.items() if w >= min_weight]
    edge_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])

    return edge_df

def build_pyg_graph(edge_df, feature_df, id_col='author', feature_cols=None, normalize=True, weighted=True):
    """
    Converts an edge dataframe and feature dataframe to a pytorch geometric object.
    """

    # Create node mapping
    nodes = pd.Index(sorted(set(edge_df['source']).union(edge_df['target'])))
    node_map = {node: i for i, node in enumerate(nodes)}

    # Convert edges to integer indices
    edge_index = torch.tensor(
        [[node_map[s], node_map[t]] for s, t in zip(edge_df['source'], edge_df['target'])],
        dtype=torch.long
    ).T

    # Edge weights
    edge_weight = torch.tensor(edge_df['weight'].values, dtype=torch.float) if weighted else None

    # Feature processing
    if feature_cols is None:
        feature_cols = [c for c in feature_df.columns if c != id_col]

    feats = feature_df.set_index(id_col).reindex(nodes).fillna(0)

    if normalize:
        scaler = StandardScaler()
        feats[feature_cols] = scaler.fit_transform(feats[feature_cols])

    x = torch.tensor(feats[feature_cols].values, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.node_order = list(nodes)

    print(f"Graph created: {data.num_nodes:,} nodes, {edge_index.shape[1]:,} edges")

    return data


def build_networkx_from_edgelist(edge_df, author_index):
    G = nx.from_pandas_edgelist(edge_df, "src", "dst", "weight")
    mapping = {i: author_index[i] for i in range(len(author_index))}
    return nx.relabel_nodes(G, mapping)