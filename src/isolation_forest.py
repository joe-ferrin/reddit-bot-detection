import numpy as np
from sklearn.ensemble import IsolationForest

class IF_Detector:
    def __init__(self, n_estimators=200, contamination="auto", random_state=0):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        """Return anomaly score (higher = more anomalous)."""
        raw_scores = self.model.decision_function(X)
        return -raw_scores

    def fit_score(self, X):
        self.model.fit(X)
        return self.score(X)


    def predict(self, X, threshold=None):
        """Return binary anomaly predictions."""
        scores = self.score(X)

        if threshold is None:
            threshold = np.percentile(scores, 95)  # default top 5%

        return (scores >= threshold).astype(int)


def run_iso_forest(X, index=None, n_estimators=200, contamination="auto", seed=0):
    x_features = X.copy()
    if index is not None:
        x_features = x_features.drop(index, axis=1).values

    model = IF_Detector(n_estimators, contamination, random_state=seed)
    scores = model.fit_score(x_features)

    result = X[[index]].copy()
    result['anomaly_score'] = scores
    result = result.sort_values('anomaly_score', ascending=False)

    return result

def combine_scores(df1,df2):
    df = df1.merge(df2, on="author", how="inner")
    df["anomaly_score"] = (df["anomaly_score_x"] + df["anomaly_score_y"])/2
    df = df.sort_values("anomaly_score", ascending=False)
    return df[['author','anomaly_score']]