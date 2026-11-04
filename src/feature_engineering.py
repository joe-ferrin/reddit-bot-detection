import pandas as pd
import numpy as np
import networkx as nx
import re
from itertools import combinations
import random

# =====================================================================
# Create network graph
# =====================================================================
def build_networkx_graph(df, id_col='author', post_col='link_id'):
    """
    Build NetworkX graph using co-comment edges.
    """
    post_groups = df.groupby(post_col)[id_col].apply(list)

    G = nx.Graph()
    for authors in post_groups:
        authors = list(set(authors))
        if len(authors) > 1:
            for a, b in combinations(authors, 2):
                a, b = sorted((a, b))

                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G

def build_networkx_from_edge_df(edge_df):
    """
    Convert your lightweight edge_df into a NetworkX graph.
    This keeps memory small.
    """
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    print('--Graph built')
    return G
# =====================================================================
# Graph features
# =====================================================================
def approx_clustering(G, sample_size=5000, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    if len(nodes) <= sample_size:
        return nx.clustering(G)

    sample = rng.choice(nodes, size=sample_size, replace=False)

    # Compute clustering only on sample
    sample_clust = nx.clustering(G, nodes=sample)

    # Fill the rest with 0 (or average? up to you)
    clustering = {n: sample_clust.get(n, 0) for n in nodes}
    return clustering

def fast_pagerank_sampling(G, seeds=200, alpha=0.85, seed=0):
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    seed_nodes = rng.choice(nodes, size=min(seeds, len(nodes)), replace=False)

    pr_total = {n: 0 for n in nodes}

    for s in seed_nodes:
        pr = nx.pagerank(G, alpha=alpha, personalization={s: 1})
        for n,v in pr.items():
            pr_total[n] += v

    # average results
    for n in pr_total:
        pr_total[n] /= seeds

    return pr_total

# Not used
def fast_betweenness(G, sample_size=3000, seed=0):
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)

    # Choose random *sources only*
    sources = rng.choice(nodes, size=min(sample_size, len(nodes)), replace=False)

    # Compute betweenness only using sampled sources
    return nx.betweenness_centrality_subset(G, sources, nodes, normalized=True)


def compute_graph_features(G=None, features=None, seed=0):



    degree = {}
    weighted_degree = {}

    if 'degree' in features:
        degree = dict(G.degree())
        print('--degree completed')


    if "weighted_degree" in features:
        weighted_degree = dict(G.degree(weight="weight"))
        print("--weighted_degree completed")



    '''
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, max_iter=100)

    # Betweenness on sample for speed
    sample_nodes = random.sample(list(G.nodes()), min(5000, G.number_of_nodes()))
    betweenness = nx.betweenness_centrality(G.subgraph(sample_nodes))
    '''


    ### I am using nx approximation to make these features faster ###

    clustering = {}
    pagerank = {}
    betweenness = {}

    if 'clustering' in features:
        clustering = approx_clustering(G, sample_size=5000, seed=seed)
        print('--clustering completed')
    if 'pagerank' in features:
        pagerank = fast_pagerank_sampling(G, seed=seed)
        print('--pagerank completed')
    if 'betweenness' in features:
        betweenness = nx.betweenness_centrality(G, k=500, seed=seed)
        print('--betweenness completed')


    graph_data = {}
    for node in G.nodes():
        graph_data[node] = {
            "degree": degree.get(node, 0),
            "weighted_degree": weighted_degree.get(node, 0),
            "clustering": clustering.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness": betweenness.get(node, 0),
        }

    graph_df = pd.DataFrame.from_dict(graph_data, orient="index")
    graph_df.index.name = "author"
    return graph_df


# =====================================================================
# Temporal Features
# =====================================================================

def hour_spread(hours):
    return len(set(hours)) / 24

def active_days_ratio(dates):
    dates = pd.Series(dates)
    total_days = (dates.max() - dates.min()).days + 1
    return dates.nunique() / total_days if total_days > 0 else 0

def mean_time_gap(datetimes):
    times = sorted(datetimes)
    if len(times) < 2: return 0
    gaps = np.diff(times).astype("timedelta64[s]").astype(float)
    return gaps.mean() / 3600

def std_time_gap(datetimes):
    times = sorted(datetimes)
    if len(times) < 3: return 0
    gaps = np.diff(times).astype("timedelta64[s]").astype(float)
    return gaps.std() / 3600


# =====================================================================
# Helper Functions
# =====================================================================
def duplication_ratio(texts):
    if len(texts) < 2: return 0
    return 1 - (len(set(texts)) / len(texts))

def url_ratio(texts):
    pattern = r"(https?://|www\.)"
    return sum(bool(re.search(pattern, str(t))) for t in texts) / len(texts)

def no_caps_ratio(texts):
    total = sum(c.isalpha() for t in texts for c in str(t))
    caps = sum(c.isupper() for t in texts for c in str(t))
    return 1 - (caps / total) if total > 0 else 0

def suspicious_username(u):
    """
    heuristic for detecting bot-like or autogenerated usernames.
    """
    u = str(u).lower()
    if re.search(r'\d{4,}$', u):
        return 1
    vowels = sum(c in "aeiou" for c in u)
    if len(u) > 5 and vowels / len(u) < 0.2:
        return 1
    return 0


# =====================================================================
# Create a dataframe with all the features
# =====================================================================
def create_all_features(data_df, edge_df, features=None, min_comments=5, seed=0):
    # -------------------------------------------------------------
    # Build networkx graph
    # -------------------------------------------------------------
    print("Building NetworkX graph...")
    G = build_networkx_from_edge_df(edge_df)

    print("Computing graph features...")
    graph_features = compute_graph_features(G, features, seed=seed)

    # Clean timestamp
    df = data_df.copy()
    df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["date"] = df["datetime"].dt.date

    # -------------------------------------------------------------
    # filter active users
    # -------------------------------------------------------------
    active_users = df["author"].value_counts()
    active_users = active_users[active_users >= min_comments].index
    df = df[df["author"].isin(active_users)]

    # -------------------------------------------------------------
    # temporal
    # -------------------------------------------------------------
    print("Computing temporal features...")
    temporal = pd.DataFrame(index=active_users)

    temporal["hour_spread"] = df.groupby("author")["hour"].apply(hour_spread)
    temporal["dow_variance"] = df.groupby("author")["dow"].apply(lambda x: x.value_counts().var())
    temporal["active_days_ratio"] = df.groupby("author")["date"].apply(active_days_ratio)
    temporal["mean_hours_between"] = df.groupby("author")["datetime"].apply(mean_time_gap)
    temporal["std_hours_between"] = df.groupby("author")["datetime"].apply(std_time_gap)

    # -------------------------------------------------------------
    # content features
    # -------------------------------------------------------------
    print("Computing content features...")
    content = pd.DataFrame(index=active_users)

    content["text_similarity"] = df.groupby("author")["body"].apply(duplication_ratio)
    content["avg_comment_length"] = df.groupby("author")["body"].apply(lambda x: np.mean([len(str(t)) for t in x]))
    content["comment_length_variance"] = df.groupby("author")["body"].apply(lambda x: np.var([len(str(t)) for t in x]))
    content["url_ratio"] = df.groupby("author")["body"].apply(url_ratio)
    content["no_caps_ratio"] = df.groupby("author")["body"].apply(no_caps_ratio)

    # -------------------------------------------------------------
    # metadata
    # -------------------------------------------------------------
    print("Computing metadata features...")
    meta = pd.DataFrame(index=active_users)

    meta["comment_count"] = df.groupby("author").size()
    meta["avg_score"] = df.groupby("author")["score"].mean()
    meta["score_variance"] = df.groupby("author")["score"].var()
    meta["suspicious_username"] = [suspicious_username(u) for u in meta.index]

    if "author_flair_text" in df.columns:
        meta["has_flair"] = df.groupby("author")["author_flair_text"].apply(
            lambda x: 1 if any(isinstance(v, str) and v not in ("", "None") for v in x) else 0
        )
    else:
        meta["has_flair"] = 0

    if "gilded" in df.columns:
        meta["gilded_count"] = df.groupby("author")["gilded"].sum()
    else:
        meta["gilded_count"] = 0

    # -------------------------------------------------------------
       # 7. COMBINE ALL FEATURES INTO ONE TABLE
    # -------------------------------------------------------------
    print("Combining all features...")
    final = (
        graph_features
        .join(temporal, how="outer")
        .join(content, how="outer")
        .join(meta, how="outer")
        .fillna(0)
        .reset_index()
    )

    return final, G