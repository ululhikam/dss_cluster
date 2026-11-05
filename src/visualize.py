"""Fungsi visualisasi: semua figure disimpan ke results/figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config, io_utils

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"

SENT_ORDER = ["Positif", "Netral", "Negatif"]


def _save(fig, name: str):
    path = io_utils.save_figure(fig, name)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Notebook 01
# --------------------------------------------------------------------------
def plot_rating_distribution(dist: pd.DataFrame):
    data = dist[dist["Rating"].astype(str) != "Total"].copy()
    data["Rating"] = data["Rating"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=data, x="Rating", y="Jumlah", hue="Rating", palette="YlOrRd",
                legend=False, ax=axes[0])
    axes[0].set_title("Sebaran Rating Ulasan")
    axes[0].set_xlabel("Rating (bintang)")
    axes[0].set_ylabel("Jumlah ulasan")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%d", fontsize=8)

    axes[1].pie(data["Jumlah"], labels=data["Rating"], autopct="%1.1f%%",
                colors=sns.color_palette("YlOrRd", len(data)), startangle=90)
    axes[1].set_title("Proporsi Rating")
    fig.tight_layout()
    return _save(fig, "01_distribusi_rating")


def plot_completeness(clean_recap: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = clean_recap["Tahap"].tolist()
    values = clean_recap["Jumlah Ulasan"].tolist()
    bars = ax.barh(labels[::-1], values[::-1], color=sns.color_palette("Blues_d", len(values)))
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 80, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax.set_title("Tahapan Pembersihan Data Ulasan")
    ax.set_xlabel("Jumlah ulasan")
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    return _save(fig, "01_tahapan_pembersihan")


def plot_yearly_trend(trend: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(trend["Tahun"], trend["Jumlah"], color="#4c72b0", alpha=0.8, label="Jumlah ulasan")
    ax1.set_xlabel("Tahun")
    ax1.set_ylabel("Jumlah ulasan", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")

    ax2 = ax1.twinx()
    ax2.plot(trend["Tahun"], trend["Rating_Rata2"], color="#c44e52", marker="o", lw=2,
             label="Rating rata-rata")
    ax2.set_ylabel("Rating rata-rata", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.set_ylim(1, 5.5)
    ax1.set_title("Tren Jumlah Ulasan dan Rating Rata-rata per Tahun")
    fig.tight_layout()
    return _save(fig, "01_tren_tahunan")


def plot_review_length(lengths: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(lengths[lengths < lengths.quantile(0.98)], bins=40, ax=ax, color="#55a868")
    ax.set_title("Distribusi Panjang Ulasan (98% data)")
    ax.set_xlabel("Jumlah karakter")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    return _save(fig, "01_panjang_ulasan")


def plot_rating_vs_text(ct: pd.DataFrame):
    data = ct.set_index("Rating")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#5cb85c"])
    ax.set_title("Ketersediaan Teks Ulasan per Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah ulasan")
    ax.legend(title="")
    fig.tight_layout()
    return _save(fig, "01_rating_vs_teks")


# --------------------------------------------------------------------------
# Notebook 02
# --------------------------------------------------------------------------
def plot_preprocessing_stages(stats: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = stats.copy()
    sns.barplot(data=data, y="Tahap", x="Jumlah Token", hue="Tahap", palette="viridis",
                legend=False, ax=axes[0])
    axes[0].set_title("Jumlah Token per Tahap Preprocessing")
    axes[0].set_xlabel("Jumlah token")

    sns.barplot(data=data, y="Tahap", x="Token Unik (Vocab)", hue="Tahap",
                palette="magma", legend=False, ax=axes[1])
    axes[1].set_title("Ukuran Kosakata per Tahap")
    axes[1].set_xlabel("Token unik")
    fig.tight_layout()
    return _save(fig, "02_tahapan_preprocessing")


def plot_top_words(vocab: pd.Series, n: int = 30, title="Kata Paling Sering Muncul"):
    data = vocab.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.5, n * 0.28)))
    sns.barplot(x=data.values, y=data.index, hue=data.index, palette="crest", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frekuensi")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, "02_kata_terbanyak")


def plot_wordcloud(vocab: pd.Series, name: str = "02_wordcloud"):
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    freq = vocab.head(400).to_dict()
    wc = WordCloud(width=1400, height=700, background_color="white",
                   colormap="viridis", max_words=300).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud Ulasan Setelah Preprocessing")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 03
# --------------------------------------------------------------------------
def plot_sentiment_distribution(dist: pd.DataFrame, title: str, name: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [config.PALETTE_SENTIMENT.get(s, "#888") for s in dist["Sentimen"]]
    bars = ax.bar(dist["Sentimen"], dist["Jumlah"], color=colors)
    for bar, v, p in zip(bars, dist["Jumlah"], dist["Persentase"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n({p}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Jumlah ulasan")
    ax.set_ylim(0, dist["Jumlah"].max() * 1.2)
    fig.tight_layout()
    return _save(fig, name)


def plot_cross_rating(ct: pd.DataFrame, name: str, title: str):
    data = ct.set_index("Rating")
    data = data[[c for c in SENT_ORDER if c in data.columns]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data.plot(kind="bar", stacked=True, ax=ax,
              color=[config.PALETTE_SENTIMENT[c] for c in data.columns])
    ax.set_title(title)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah ulasan")
    ax.legend(title="Sentimen teks")
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion(matrix: pd.DataFrame, name: str):
    data = matrix.set_index(matrix.columns[0])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix: Sentimen Teks vs Rating")
    ax.set_ylabel("Acuan (rating)")
    ax.set_xlabel("Prediksi (teks)")
    fig.tight_layout()
    return _save(fig, name)


def plot_score_distribution(scores: pd.Series, name: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(scores, bins=60, ax=ax, color="#4c72b0")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_title("Distribusi Skor Sentimen Lexicon per Ulasan")
    ax.set_xlabel("Skor sentimen")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 04
# --------------------------------------------------------------------------
def plot_aspect_volume(summary: pd.DataFrame, name: str = "04_volume_aspek"):
    data = summary.sort_values("Jumlah Disebut")
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(data) * 0.5)))
    sns.barplot(data=data, y="Aspek", x="Jumlah Disebut", hue="Aspek",
                palette="crest", legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", fontsize=8, padding=2)
    ax.set_title("Jumlah Penyebutan per Aspek Layanan")
    ax.set_xlabel("Jumlah penyebutan")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, name)


def plot_aspect_sentiment(summary: pd.DataFrame, name: str = "04_sentimen_aspek"):
    data = summary.copy().sort_values("% Negatif", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(data) * 0.6)))
    left = np.zeros(len(data))
    for label in SENT_ORDER:
        col = f"% {label}"
        if col not in data.columns:
            continue
        values = data[col].to_numpy()
        ax.barh(data["Aspek"], values, left=left, color=config.PALETTE_SENTIMENT[label], label=label)
        left += values
    ax.set_xlim(0, 100)
    ax.set_title("Komposisi Sentimen per Aspek (%)")
    ax.set_xlabel("Persentase")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    return _save(fig, name)


def plot_priority_scatter(summary: pd.DataFrame, name: str = "04_matriks_prioritas"):
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = summary["Jumlah Disebut"] / summary["Jumlah Disebut"].max() * 900 + 80
    ax.scatter(summary["Jumlah Disebut"], summary["% Negatif"], s=sizes,
               c=summary["% Negatif"], cmap="RdYlGn_r", alpha=0.75, edgecolors="black")
    for _, row in summary.iterrows():
        ax.annotate(row["Aspek"], (row["Jumlah Disebut"], row["% Negatif"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.axhline(30, color="gray", ls="--", lw=1)
    ax.axvline(60, color="gray", ls="--", lw=1)
    ax.set_xlabel("Jumlah penyebutan (volume)")
    ax.set_ylabel("% Negatif")
    ax.set_title("Matriks Prioritas Aspek Layanan\n(gangguan: garis = ambang volume 60 & 30% negatif)")
    fig.tight_layout()
    return _save(fig, name)


def plot_rating_by_aspect(hits: pd.DataFrame, name: str = "04_rating_per_aspek"):
    data = hits.copy()
    order = data.groupby("Aspek")["Rating"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(order) * 0.55)))
    sns.boxplot(data=data, y="Aspek", x="Rating", order=order, hue="Aspek",
                palette="vlag", legend=False, ax=ax)
    ax.set_title("Sebaran Rating Ulasan per Aspek")
    ax.set_xlabel("Rating")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Notebook 05
# --------------------------------------------------------------------------
def plot_elbow(eval_df: pd.DataFrame, knee: int, name: str = "05_elbow"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(eval_df["k"], eval_df["Inertia"], marker="o", color="#4c72b0", lw=2)
    ax.axvline(knee, color="red", ls="--", label=f"Knee (k={knee})")
    ax.set_title("Elbow Method - Inertia per Jumlah Cluster")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Inertia")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_silhouette(eval_df: pd.DataFrame, best_k: int, name: str = "05_silhouette"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(eval_df["k"], eval_df["Silhouette"], marker="s", color="#55a868", lw=2)
    ax.axvline(best_k, color="red", ls="--", label=f"k terbaik = {best_k}")
    ax.set_title("Silhouette Score per Jumlah Cluster")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_dbi(eval_df: pd.DataFrame, name: str = "05_davies_bouldin"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(eval_df["k"], eval_df["Davies-Bouldin"], marker="^", color="#c44e52", lw=2)
    ax.set_title("Davies-Bouldin Index per Jumlah Cluster (semakin kecil semakin baik)")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Davies-Bouldin Index")
    fig.tight_layout()
    return _save(fig, name)


def plot_cluster_sizes(sizes: pd.DataFrame, name: str = "05_ukuran_cluster"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(sizes["Cluster"].astype(str), sizes["Jumlah"],
                  color=config.PALETTE_CLUSTER[: len(sizes)])
    for bar, v, p in zip(bars, sizes["Jumlah"], sizes["Persentase"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({p}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Jumlah Ulasan per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah ulasan")
    ax.set_ylim(0, sizes["Jumlah"].max() * 1.2)
    fig.tight_layout()
    return _save(fig, name)


def plot_cluster_sentiment(profile: pd.DataFrame, name: str = "05_sentimen_cluster"):
    cols = [c for c in profile.columns if c.startswith("Sentimen ") and c.endswith("(%)")]
    if not cols:
        return None
    data = profile.set_index("Cluster")[cols]
    data.columns = [c.replace("Sentimen ", "").replace(" (%)", "") for c in data.columns]
    data = data[[c for c in SENT_ORDER if c in data.columns]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data.plot(kind="bar", stacked=True, ax=ax,
              color=[config.PALETTE_SENTIMENT[c] for c in data.columns])
    ax.set_title("Komposisi Sentimen pada Setiap Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Persentase (%)")
    ax.legend(title="Sentimen", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _save(fig, name)


def plot_cluster_heatmap(terms: pd.DataFrame, name: str = "05_heatmap_term", top_n: int = 12):
    pivot = terms[terms["Peringkat"] <= top_n].pivot_table(
        index="Cluster", columns="Term", values="Bobot TF-IDF", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.55), 5.5))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Rata-rata TF-IDF"})
    ax.set_title("Term Dominan per Cluster (Bobot TF-IDF)")
    ax.set_xlabel("Term")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return _save(fig, name)


def plot_cluster_aspect_heatmap(matrix: pd.DataFrame, name: str = "05_cluster_aspek"):
    if matrix.empty:
        return None
    data = matrix.set_index("Cluster")
    fig, ax = plt.subplots(figsize=(max(9, len(data.columns) * 0.62), 5.5))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Jumlah Penyebutan Aspek per Cluster")
    ax.set_xlabel("Aspek")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return _save(fig, name)


def plot_priority_bar(matrix: pd.DataFrame, name: str = "06_prioritas"):
    if matrix.empty:
        return None
    data = matrix.sort_values("Skor Prioritas")
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(data) * 0.55)))
    colors = sns.color_palette("RdYlGn", len(data))
    bars = ax.barh(data["Aspek"], data["Skor Prioritas"], color=colors)
    for bar, v in zip(bars, data["Skor Prioritas"]):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:.2f}", va="center", fontsize=9)
    ax.set_title("Skor Prioritas Perbaikan per Aspek")
    ax.set_xlabel("Skor prioritas (% negatif x akar volume)")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, name)