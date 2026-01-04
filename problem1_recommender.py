from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


def _parse_minutes(duration: str) -> Optional[float]:
    if not isinstance(duration, str):
        return None
    m = re.search(r"(\d+)\s*min", duration.lower())
    if m:
        return float(m.group(1))
    return None


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["director", "cast", "country", "rating", "listed_in", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df["release_year"] = pd.to_numeric(df.get("release_year"), errors="coerce")
    df["duration_min"] = df.get("duration").apply(_parse_minutes) if "duration" in df.columns else np.nan
    return df


@dataclass
class RecommenderArtifacts:
    df_movies: pd.DataFrame
    X: sparse.spmatrix
    X_red: np.ndarray


def build_feature_matrix(df: pd.DataFrame, svd_components: int = 100, random_state: int = 42) -> RecommenderArtifacts:
    df_movies = df[df["type"].astype(str).str.lower() == "movie"].copy().reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2), max_features=20000)
    X_desc = tfidf.fit_transform(df_movies["description"].astype(str))

    genres = df_movies["listed_in"].astype(str).apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_gen = mlb.fit_transform(genres)

    def top_k(series: pd.Series, k: int) -> pd.Series:
        vc = series.value_counts(dropna=False)
        top = set(vc.head(k).index.astype(str))
        return series.astype(str).apply(lambda x: x if x in top else "Other")

    df_c = df_movies.copy()
    df_c["country_red"] = top_k(df_c["country"], 25) if "country" in df_c.columns else "Unknown"
    df_c["director_red"] = top_k(df_c["director"], 40) if "director" in df_c.columns else "Unknown"
    df_c["rating_red"] = top_k(df_c["rating"], 25) if "rating" in df_c.columns else "Unknown"

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df_c[["country_red", "director_red", "rating_red"]])

    num = df_c[["release_year", "duration_min"]].copy()
    num = num.fillna(num.median(numeric_only=True))
    scaler = StandardScaler(with_mean=False)
    X_num = scaler.fit_transform(sparse.csr_matrix(num.values))

    X = sparse.hstack([X_desc, X_gen, X_cat, X_num], format="csr")

    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    X_red = svd.fit_transform(X)

    return RecommenderArtifacts(df_movies=df_movies, X=X, X_red=X_red)


def topn_similar(art: RecommenderArtifacts, target_show_id: str, topn: int = 10) -> pd.DataFrame:
    dfm = art.df_movies
    if target_show_id not in set(dfm["show_id"].astype(str)):
        raise ValueError("El id objetivo no está en el subconjunto de películas.")
    idx = dfm.index[dfm["show_id"].astype(str) == target_show_id][0]
    sims = cosine_similarity(art.X[idx], art.X).ravel()
    sims[idx] = -1
    top_idx = np.argsort(-sims)[:topn]
    out = dfm.loc[top_idx, ["show_id", "title", "release_year", "listed_in", "rating"]].copy()
    out["similarity"] = sims[top_idx]
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


def run_clustering(X_red: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    km = KMeans(n_clusters=12, n_init="auto", random_state=random_state)
    lab_km = km.fit_predict(X_red)

    agg = AgglomerativeClustering(n_clusters=12, linkage="ward")
    lab_agg = agg.fit_predict(X_red)

    db = DBSCAN(eps=2.2, min_samples=8)
    lab_db = db.fit_predict(X_red)

    return {"kmeans": lab_km, "hierarchical": lab_agg, "dbscan": lab_db}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="s5485")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    df = load_dataset(args.csv)
    art = build_feature_matrix(df)

    recs = topn_similar(art, args.target, args.topn)
    print(recs.to_string(index=False))

    labs = run_clustering(art.X_red)
    print({k: int(np.unique(v).size) for k, v in labs.items()})


if __name__ == "__main__":
    main()
