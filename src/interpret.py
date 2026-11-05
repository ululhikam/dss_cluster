"""Interpretasi hasil: penamaan cluster, profil, dan rekomendasi."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aspects import _priority_label


def cluster_theme(terms_string: str, keywords_per_cluster: int = 6) -> str:
    """Memberi nama tema cluster dari term dominan.

    Setiap term dominan dipetakan ke aspek layanan memakai kamus domain
    (``ASPECT_KEYWORD_OVERRIDES``). Label yang paling sering muncul dipakai
    sebagai tema; jika tidak ada yang cocok, dua term teratas dipakai apa adanya.
    """
    from .aspects import ASPECT_KEYWORD_OVERRIDES

    terms = [t.strip() for t in str(terms_string).split(",") if t.strip()]
    if not terms:
        return "-"

    # Ciri khas: cluster yang didominasi live music layak diberi nama sendiri.
    head_terms = " ".join(terms[:3])
    if "live" in head_terms or "musik" in head_terms:
        return "Live Music & Suasana"

    votes: dict[str, int] = {}
    counted = 0
    for term in terms[:keywords_per_cluster]:
        label = ASPECT_KEYWORD_OVERRIDES.get(term)
        if label is None:
            for token in term.split():
                if token in ASPECT_KEYWORD_OVERRIDES:
                    label = ASPECT_KEYWORD_OVERRIDES[token]
                    break
        if label:
            votes[label] = votes.get(label, 0) + 1
        counted += 1

    if votes:
        best_label, best_vote = max(votes.items(), key=lambda kv: kv[1])
        share = best_vote / counted if counted else 0.0
        if share >= 0.5:
            return best_label
        # Aspek teratas tetap informatif, tetapi temanya bercampur.
        return f"{best_label} (beragam)"
    return " / ".join(terms[:2])


def cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_terms: dict[int, str],
) -> pd.DataFrame:
    """Profil tiap cluster: ukuran, rating, sentimen, dan term dominan."""
    work = df.copy()
    work["cluster"] = labels

    has_doc_sent = "sentimen_teks" in work.columns
    rows = []
    for cluster, g in work.groupby("cluster"):
        terms = top_terms.get(round(cluster), top_terms.get(cluster, "-"))
        row = {
            "Cluster": int(cluster),
            "Jumlah Ulasan": len(g),
            "Persentase": round(len(g) / len(work) * 100, 2),
            "Rating Rata-rata": round(float(g["Rating"].mean()), 3),
            "Rating 1-2": int((g["Rating"] <= 2).sum()),
            "Rating 3": int((g["Rating"] == 3).sum()),
            "Rating 4-5": int((g["Rating"] >= 4).sum()),
            "Tema (otomatis)": cluster_theme(terms),
            "Term Dominan": terms,
        }
        if has_doc_sent:
            for label in ("Positif", "Negatif", "Netral"):
                row[f"Sentimen {label} (%)"] = round(
                    float((g["sentimen_teks"] == label).mean() * 100), 2
                )
            row["Sentimen Dominan"] = (
                g["sentimen_teks"].value_counts().idxmax()
                if g["sentimen_teks"].notna().any()
                else "-"
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Jumlah Ulasan", ascending=False).reset_index(drop=True)


def cluster_examples(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    n: int = 3,
) -> pd.DataFrame:
    """Contoh ulasan representatif per cluster (dekat centroid tidak diperlukan;
    dipilih ulasan dengan rating ekstrem dan panjang memadai)."""
    work = df.copy()
    work["cluster"] = labels
    rows = []
    for cluster, g in work.groupby("cluster"):
        g = g[g["Review"].astype(str).str.len() >= 60]
        pos = g.nlargest(max(1, n // 2 + n % 2), "Rating")
        neg = g.nsmallest(max(1, n // 2), "Rating")
        for _, r in pos.iterrows():
            rows.append(
                {
                    "Cluster": int(cluster),
                    "Jenis": "Positif",
                    "Rating": int(r["Rating"]),
                    "Review": str(r["Review"])[:280].replace("\n", " "),
                }
            )
        for _, r in neg.iterrows():
            rows.append(
                {
                    "Cluster": int(cluster),
                    "Jenis": "Negatif",
                    "Rating": int(r["Rating"]),
                    "Review": str(r["Review"])[:280].replace("\n", " "),
                }
            )
    return pd.DataFrame(rows)


def cluster_aspect_matrix(
    df: pd.DataFrame,
    labels: np.ndarray,
    hits: pd.DataFrame | None,
) -> pd.DataFrame:
    """Cross-tab cluster x aspek (jumlah penyebutan dan % negatif)."""
    if hits is None or hits.empty or "review_id" not in hits.columns:
        return pd.DataFrame()

    cluster_map = dict(zip(df["review_id"], labels))
    work = hits.copy()
    work["cluster"] = work["review_id"].map(cluster_map)
    work = work.dropna(subset=["cluster"])
    work["cluster"] = work["cluster"].astype(int)
    if work.empty:
        return pd.DataFrame()

    count = work.pivot_table(
        index="cluster", columns="Aspek", values="Skor", aggfunc="count", fill_value=0
    )
    neg = work.assign(is_neg=(work["Sentimen"] == "Negatif").astype(int)).pivot_table(
        index="cluster", columns="Aspek", values="is_neg", aggfunc="mean", fill_value=0
    )
    pct_neg = (neg * 100).round(1)
    pct_neg.columns = [f"{c} (%neg)" for c in pct_neg.columns]
    out = count.join(pct_neg)
    out.index.name = "Cluster"
    return out.reset_index()


def priority_matrix(aspect_summary: pd.DataFrame) -> pd.DataFrame:
    """Matriks keputusan: volume x % negatif, dengan hidden complaint."""
    if aspect_summary.empty:
        return pd.DataFrame()

    work = aspect_summary.copy()
    # Pastikan label prioritas mengikuti peringkat terbaru, bukan nilai lama
    # yang mungkin tersimpan di CSV.
    work["Peringkat Prioritas"] = work["Skor Prioritas"].rank(
        ascending=False, method="min"
    ).astype(int)
    work["Rekomendasi"] = [
        _priority_label(row, len(work)) for _, row in work.iterrows()
    ]
    volume_threshold = float(work["Jumlah Disebut"].median())
    neg_threshold = float(work["% Negatif"].median())
    work["Kuadran"] = work.apply(
        lambda r: _quadrant(r, volume_threshold, neg_threshold), axis=1
    )
    cols = [
        "Aspek", "Jumlah Disebut", "% Negatif", "% Positif",
        "Rating Rata-rata", "Hidden Complaint", "Skor Prioritas",
        "Peringkat Prioritas", "Rekomendasi", "Kuadran",
    ]
    cols = [c for c in cols if c in work.columns]
    return work[cols].sort_values("Skor Prioritas", ascending=False).reset_index(drop=True)


def _quadrant(row, volume_threshold: float, neg_threshold: float) -> str:
    volume_high = row["Jumlah Disebut"] >= volume_threshold
    neg_high = row["% Negatif"] >= neg_threshold
    if volume_high and neg_high:
        return "Prioritas Utama (sering dibahas & banyak negatif)"
    if not volume_high and neg_high:
        return "Temuan Khusus (jarang dibahas tapi negatif)"
    if volume_high and not neg_high:
        return "Kekuatan (sering dipuji)"
    return "Minor (jarang dibahas, sedikit negatif)"


def recommendations(aspect_summary: pd.DataFrame) -> pd.DataFrame:
    """Rekomendasi tindakan per aspek berdasarkan peringkat prioritas."""
    if aspect_summary.empty:
        return pd.DataFrame()

    actions = {
        "Kecepatan": "Evaluasi alur dapur dan koordinasi antar pelayan; terapkan target waktu saji per menu.",
        "Fasilitas": "Periksa kelengkapan dan kondisi fasilitas (ruang, meja, colokan, area VIP) secara berkala.",
        "Parkir": "Tertibkan tarif dan juru parkir; sediakan area parkir tambahan pada jam ramai.",
        "Kebersihan": "Tambahkan jadwal pembersihan toilet dan area makan; audit kebersihan mingguan.",
        "Minuman": "Standarkan resep dan suhu penyajian; lakukan quality check rasa minuman.",
        "Pelayanan": "Pelatihan keramahan dan responsivitas staf; perjelas SOP penanganan keluhan.",
        "Makanan": "Kontrol konsistensi rasa dan porsi; uji rasa berkala pada menu paling sering dipesan.",
        "Harga": "Komunikasikan value (porsi, fasilitas) dan tinjau harga menu yang dianggap mahal.",
        "Suasana": "Pertahankan konsep dan live music; jaga kenyamanan akustik saat ramai.",
    }

    rows = []
    for _, row in aspect_summary.iterrows():
        aspect = row["Aspek"]
        action = actions.get(aspect)
        if action is None:
            action = (
                f"Tinjau ulasan negatif pada aspek '{aspect}' dan tentukan tindakan lanjutan "
                f"berdasarkan contoh ulasan terlampir."
            )
        rows.append(
            {
                "Aspek": aspect,
                "Peringkat": int(row["Peringkat Prioritas"]),
                "Prioritas": row["Rekomendasi"],
                "% Negatif": row["% Negatif"],
                "Jumlah Disebut": row["Jumlah Disebut"],
                "Rekomendasi Tindakan": action,
            }
        )
    return pd.DataFrame(rows).sort_values("Peringkat").reset_index(drop=True)


def executive_summary(
    df: pd.DataFrame,
    aspect_summary: pd.DataFrame,
    cluster_result: dict | None = None,
) -> pd.DataFrame:
    """Ringkasan angka kunci untuk bab hasil."""
    rows = [
        {"Indikator": "Jumlah ulasan dianalisis", "Nilai": len(df)},
        {"Indikator": "Rating rata-rata", "Nilai": round(float(df["Rating"].mean()), 3)},
    ]
    if "sentimen_teks" in df.columns:
        vc = df["sentimen_teks"].value_counts()
        total = len(df)
        for label in ("Positif", "Netral", "Negatif"):
            n = int(vc.get(label, 0))
            rows.append(
                {"Indikator": f"Sentimen {label}", "Nilai": f"{n} ({n / total * 100:.2f}%)"}
            )
    if not aspect_summary.empty:
        top = aspect_summary.sort_values("Peringkat Prioritas").iloc[0]
        best = aspect_summary.sort_values("Skor Prioritas").iloc[0]
        rows += [
            {"Indikator": "Aspek paling perlu diperbaiki", "Nilai": f"{top['Aspek']} ({top['% Negatif']:.1f}% negatif)"},
            {"Indikator": "Aspek terbaik", "Nilai": f"{best['Aspek']} ({best['% Positif']:.1f}% positif)"},
            {"Indikator": "Jumlah aspek terdeteksi", "Nilai": len(aspect_summary)},
        ]
    if cluster_result is not None:
        rows += [
            {"Indikator": "Jumlah cluster (k)", "Nilai": cluster_result["chosen_k"]},
            {"Indikator": "Silhouette Score", "Nilai": cluster_result["silhouette_final"]},
        ]
    return pd.DataFrame(rows)