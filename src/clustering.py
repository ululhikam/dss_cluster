"""Clustering K-Means beserta penentuan jumlah cluster optimal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from . import config


def evaluate_k(
    matrix,
    *,
    k_range=config.K_RANGE,
    n_init: int = config.KMEANS_N_INIT,
    random_state: int = config.RANDOM_STATE,
    max_iter: int = config.KMEANS_MAX_ITER,
    sample_size: int | None = 5000,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Menghitung inertia, silhouette, dan Davies-Bouldin untuk setiap k."""
    rows = []
    models: dict[int, KMeans] = {}
    n_samples = matrix.shape[0]

    for k in k_range:
        if k >= n_samples:
            continue
        km = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
            max_iter=max_iter,
        )
        labels = km.fit_predict(matrix)
        models[k] = km

        if len(set(labels)) > 1:
            if sample_size and n_samples > sample_size:
                sil = silhouette_score(
                    matrix, labels, sample_size=sample_size,
                    random_state=random_state,
                )
            else:
                sil = silhouette_score(matrix, labels)
            dbi = davies_bouldin_score(matrix.toarray() if hasattr(matrix, "toarray") else matrix, labels)
        else:
            sil, dbi = -1.0, np.inf

        sizes = np.bincount(labels)
        rows.append(
            {
                "k": k,
                "Inertia": round(float(km.inertia_), 2),
                "Silhouette": round(float(sil), 4),
                "Davies-Bouldin": round(float(dbi), 4),
                "Ukuran Terkecil": int(sizes.min()),
                "Ukuran Terbesar": int(sizes.max()),
            }
        )

    eval_df = pd.DataFrame(rows)
    eval_df["Penurunan Inertia"] = eval_df["Inertia"].diff().fillna(0).round(2)
    return eval_df, models


def pick_best_k(
    eval_df: pd.DataFrame,
    *,
    n_samples: int | None = None,
    min_cluster_share: float = 0.03,
    tie_tolerance: float = 0.005,
) -> tuple[int, str]:
    """Memilih k optimal.

    Silhouette pada teks ulasan pendek biasanya sangat kecil dan selisih antar k
    berada dalam kisaran noise. Karena itu pemilihan dilakukan bertahap:

    1. Buang k yang menghasilkan cluster sangat kecil (< ``min_cluster_share``
       dari data), karena cluster seperti itu tidak dapat diinterpretasi.
    2. Pilih silhouette tertinggi; jika beberapa k selisihnya kurang dari
       ``tie_tolerance``, pilih k yang paling dekat dengan titik siku (elbow).
    """
    candidates = eval_df.copy()
    if n_samples:
        min_size = max(2, int(np.ceil(n_samples * min_cluster_share)))
        filtered = candidates[candidates["Ukuran Terkecil"] >= min_size]
        if not filtered.empty:
            candidates = filtered

    best_sil = float(candidates["Silhouette"].max())
    near = candidates[candidates["Silhouette"] >= best_sil - tie_tolerance]
    if len(near) > 1:
        knee = elbow_knee(eval_df)
        near = near.assign(jarak_siku=(near["k"] - knee).abs())
        chosen = near.sort_values(["jarak_siku", "k"]).iloc[0]
    else:
        chosen = near.iloc[0]

    k = int(chosen["k"])
    reason = (
        f"Silhouette tertinggi pada k={k} sebesar {chosen['Silhouette']:.4f} "
        f"(Davies-Bouldin {chosen['Davies-Bouldin']:.4f}, cluster terkecil "
        f"{int(chosen['Ukuran Terkecil'])} ulasan). "
        f"Silhouette seluruh k sangat kecil karena ulasan pendek dan berbagi kosakata, "
        f"sehingga k dipilih dari kandidat dengan silhouette setara yang paling dekat "
        f"dengan titik siku (k={elbow_knee(eval_df)})."
    )
    return k, reason


def elbow_knee(eval_df: pd.DataFrame) -> int:
    """Titik siku (knee) kurva inertia dengan metode jarak ke garis diagonal."""
    x = eval_df["k"].to_numpy(dtype=float)
    y = eval_df["Inertia"].to_numpy(dtype=float)
    if len(x) < 3:
        return int(x[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # jarak titik ke garis dari titik pertama ke terakhir
    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    line = p2 - p1
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(x[0])
    distances = np.abs(np.cross(line, p1 - np.column_stack([xn, yn]))) / line_norm
    return int(x[int(np.argmax(distances))])


def fit_final(matrix, k: int, **kwargs) -> tuple[KMeans, np.ndarray]:
    km = KMeans(
        n_clusters=k,
        n_init=kwargs.get("n_init", config.KMEANS_N_INIT),
        random_state=kwargs.get("random_state", config.RANDOM_STATE),
        max_iter=kwargs.get("max_iter", config.KMEANS_MAX_ITER),
    )
    labels = km.fit_predict(matrix)
    return km, labels


def silhouette_per_cluster(matrix, labels: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import silhouette_samples

    values = silhouette_samples(matrix, labels)
    rows = []
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        rows.append(
            {
                "Cluster": int(cluster),
                "Ukuran": int(mask.sum()),
                "Silhouette Rata-rata": round(float(values[mask].mean()), 4),
                "Silhouette Minimum": round(float(values[mask].min()), 4),
                "Silhouette Maksimum": round(float(values[mask].max()), 4),
            }
        )
    return pd.DataFrame(rows)


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    vc = pd.Series(labels).value_counts().sort_index()
    out = pd.DataFrame({"Cluster": vc.index.astype(int), "Jumlah": vc.values})
    out["Persentase"] = (out["Jumlah"] / out["Jumlah"].sum() * 100).round(2)
    return out


def top_terms_per_cluster(
    matrix,
    vectorizer,
    labels: np.ndarray,
    *,
    n: int = 15,
) -> pd.DataFrame:
    """Term dengan rata-rata bobot TF-IDF tertinggi pada tiap cluster."""
    names = vectorizer.get_feature_names_out()
    rows = []
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        mean_tfidf = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:n]
        for rank, i in enumerate(top_idx, start=1):
            if mean_tfidf[i] <= 0:
                continue
            rows.append(
                {
                    "Cluster": int(cluster),
                    "Peringkat": rank,
                    "Term": names[i],
                    "Bobot TF-IDF": round(float(mean_tfidf[i]), 4),
                }
            )
    return pd.DataFrame(rows)


def top_terms_strings(matrix, vectorizer, labels: np.ndarray, n: int = 8) -> dict[int, str]:
    tf = top_terms_per_cluster(matrix, vectorizer, labels, n=n)
    return (
        tf.groupby("Cluster")["Term"]
        .apply(lambda s: ", ".join(s))
        .to_dict()
    )


def run_clustering(matrix, vectorizer, *, k: int | None = None):
    eval_df, models = evaluate_k(matrix)
    best_k_sil, reason_sil = pick_best_k(eval_df, n_samples=matrix.shape[0])
    knee = elbow_knee(eval_df)
    chosen = int(k) if k is not None else best_k_sil

    km, labels = fit_final(matrix, chosen)
    sil_final = silhouette_score(matrix, labels) if len(set(labels)) > 1 else -1.0
    dbi_final = davies_bouldin_score(matrix.toarray(), labels)

    return {
        "evaluation": eval_df,
        "models": models,
        "best_k_silhouette": best_k_sil,
        "best_k_elbow": knee,
        "chosen_k": chosen,
        "reason": reason_sil,
        "model": km,
        "labels": labels,
        "silhouette_final": round(float(sil_final), 4),
        "dbi_final": round(float(dbi_final), 4),
        "per_cluster": silhouette_per_cluster(matrix, labels),
        "sizes": cluster_sizes(labels),
        "top_terms": top_terms_per_cluster(matrix, vectorizer, labels),
        "top_terms_str": top_terms_strings(matrix, vectorizer, labels),
    }