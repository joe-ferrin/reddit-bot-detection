import duckdb
import pandas as pd
import os

# ==========================================================
# Connection
# ==========================================================
con = duckdb.connect("../data/database.sqlite")

# ==========================================================
# Query
# ==========================================================
def simple_query(query, params=None):
    return con.execute(query, params or []).fetchdf()

# ==========================================================
# Data clean
# ==========================================================
def decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x

def clean_dataframe(df):
    """Decode byte columns, convert timestamps, and remove bad authors."""
    for col in df.columns:
        df[col] = df[col].apply(decode)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df = df[df["author"].notnull()]
    df = df[~df["author"].isin(["[deleted]", "AutoModerator"])]
    return df

# ==========================================================
# Data retrieval
# ==========================================================
def load_subreddit_data(subreddits, path="/subreddit_data", reload=False, load=False):
    """
    Retrieves and cleans data for one or more subreddits.
    Caches parquet files to avoid re-querying.
    Returns combined DataFrame.
    """
    if subreddits is None:
        # load every *.parquet file in folder
        subs = []
        for fn in os.listdir(path):
            if fn.endswith("_subreddit.parquet"):
                subs.append(fn.replace("_subreddit.parquet", ""))
        subreddits = subs

    if isinstance(subreddits, str):
        subreddits = [subreddits]

    dfs = []
    count = 0
    for sub in subreddits:
        data_path = path + f"/{sub}_subreddit.parquet"
        if os.path.exists(data_path) and not reload:
            print(f"Loading cached data for {sub}...")
            df = pd.read_parquet(data_path)
            dfs.append(df)
            count+=1
        else:
            if load:
                print(f"Querying data for {sub}...")
                query = "SELECT * FROM May2015 WHERE subreddit = ?"
                df = simple_query(query, [sub])
                df = clean_dataframe(df)
                df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
                dfs.append(df)
                count+=1
            else:
                print(f"{sub} parquet not found")


    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {combined.shape[0]} rows from {count} subreddit(s).")
    return combined