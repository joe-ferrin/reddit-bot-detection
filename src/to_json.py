import pandas as pd
import json
from itertools import combinations

def build_graph_json(edge_df, comments_df, anomaly_df, features_df,
                     top_n=500, output_path="graph.json"):


    # --------------------------------------------------------------------
    # 1. Get top-N anomalous users
    # --------------------------------------------------------------------
    top_users = (
        anomaly_df.sort_values("anomaly_score", ascending=False)
                  .head(top_n)["author"]
                  .unique()
    )
    top_users = set(top_users)

    # Filter comments to top users only
    df = comments_df[comments_df["author"].isin(top_users)]
    df = df[["author", "link_id", "subreddit"]]

    # --------------------------------------------------------------------
    # 2. Build subreddit list per user
    # --------------------------------------------------------------------
    user_subs = (
        df.groupby("author")["subreddit"]
          .apply(lambda x: sorted(set(x)))
          .to_dict()
    )

    # --------------------------------------------------------------------
    # 3. Build features lookup
    # --------------------------------------------------------------------
    features_lookup = (
        features_df.merge(anomaly_df, on="author", how="left")
                   .set_index("author")
                   .to_dict(orient="index")
    )

    # --------------------------------------------------------------------
    # 4. Build global nodes list
    # --------------------------------------------------------------------
    nodes = []
    for u in top_users:
        feats = features_lookup.get(u, {})
        nodes.append({
            "id": u,
            "subreddits": user_subs.get(u, []),
            "features": {k: v for k, v in feats.items() if k != "anomaly_score"},
            "anomaly_score": feats.get("anomaly_score", None)
        })

    # --------------------------------------------------------------------
    # 5. Build global edges with subreddit list
    # --------------------------------------------------------------------
    # Structure: (u,v) → set([sub1, sub2])
    edge_map = {}

    # Build edges per post, only among top users
    for post_id, group in df.groupby("link_id"):
        users = list(group["author"])
        sub = group["subreddit"].iloc[0]   # post only belongs to one subreddit

        for u, v in combinations(users, 2):
            key = tuple(sorted((u, v)))

            if key not in edge_map:
                edge_map[key] = set()

            edge_map[key].add(sub)

    # Convert to list
    edges = [
        {"source": u, "target": v, "subreddits": sorted(list(subs))}
        for (u, v), subs in edge_map.items()
    ]

    # All subreddits a top user appears in
    subreddit_list = sorted({s for subs in user_subs.values() for s in subs})

    # --------------------------------------------------------------------
    # 6. Save final JSON
    # --------------------------------------------------------------------
    result = {
        "nodes": nodes,
        "edges": edges,
        "subreddits": subreddit_list
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path
