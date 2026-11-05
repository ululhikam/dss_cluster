"""Evaluasi kuantitatif: klasifikasi sentimen vs rating dan kualitas cluster."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["Positif", "Netral", "Negatif"]


def confusion_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix sentimen teks (prediksi) vs rating (acuan)."""
    cm = confusion_matrix(df["sentimen_rating"], df["sentimen_teks"], labels=LABELS)
    return pd.DataFrame(
        cm,
        index=[f"Aktual (rating): {l}" for l in LABELS],
        columns=[f"Prediksi (teks): {l}" for l in LABELS],
    ).reset_index(names=" ")


def classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["sentimen_rating"]
    y_pred = df["sentimen_teks"]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )[2]

    rows = [
        {"Metrik": "Accuracy", "Nilai": round(float(acc), 4)},
        {"Metrik": "Macro Precision", "Nilai": round(float(macro_p), 4)},
        {"Metrik": "Macro Recall", "Nilai": round(float(macro_r), 4)},
        {"Metrik": "Macro F1", "Nilai": round(float(macro_f1), 4)},
        {"Metrik": "Weighted F1", "Nilai": round(float(weighted_f1), 4)},
        {"Metrik": "Cohen's Kappa", "Nilai": round(float(cohen_kappa_score(y_true, y_pred)), 4)},
    ]
    for label, pi, ri, fi, si in zip(LABELS, p, r, f1, s):
        rows.append({"Metrik": f"Precision {label}", "Nilai": round(float(pi), 4)})
        rows.append({"Metrik": f"Recall {label}", "Nilai": round(float(ri), 4)})
        rows.append({"Metrik": f"F1 {label}", "Nilai": round(float(fi), 4)})
        rows.append({"Metrik": f"Support {label}", "Nilai": int(si)})
    return pd.DataFrame(rows)


def per_class_frame(df: pd.DataFrame) -> pd.DataFrame:
    rep = classification_report(
        df["sentimen_rating"], df["sentimen_teks"], labels=LABELS,
        output_dict=True, zero_division=0,
    )
    rows = []
    for label in LABELS:
        d = rep.get(label, {})
        rows.append(
            {
                "Kelas": label,
                "Precision": round(d.get("precision", 0), 4),
                "Recall": round(d.get("recall", 0), 4),
                "F1-Score": round(d.get("f1-score", 0), 4),
                "Support": int(d.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def aspect_rating_consistency(hits: pd.DataFrame) -> pd.DataFrame:
    """Konsistensi sentimen aspek terhadap rating: rata-rata rating per sentimen aspek."""
    if hits.empty:
        return pd.DataFrame()
    g = (
        hits.groupby(["Aspek", "Sentimen"])["Rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Rating Rata-rata", "count": "Jumlah"})
    )
    g["Rating Rata-rata"] = g["Rating Rata-rata"].round(2)
    return g.sort_values(["Aspek", "Sentimen"]).reset_index(drop=True)


def clustering_metrics_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metrik": [
                "Jumlah cluster terpilih (k)",
                "Silhouette Score (final)",
                "Davies-Bouldin Index (final)",
                "k terbaik menurut silhouette",
                "k terbaik menurut elbow",
                "Alasan pemilihan k",
            ],
            "Nilai": [
                result["chosen_k"],
                result["silhouette_final"],
                result["dbi_final"],
                result["best_k_silhouette"],
                result["best_k_elbow"],
                result["reason"],
            ],
        }
    )


def interpret_silhouette(value: float) -> str:
    if value >= 0.50:
        return "Struktur cluster kuat"
    if value >= 0.25:
        return "Struktur cluster cukup, masih dapat diterima"
    if value > 0.0:
        return "Struktur cluster lemah (umum pada teks ulasan pendek)"
    return "Tidak ada struktur cluster yang jelas"