"""Pemeriksaan dan profil dataset mentah (Tahap 1: Data Understanding)."""

from __future__ import annotations

import pandas as pd

from . import config, io_utils


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan satu baris per atribut."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "Atribut": col,
                "Tipe": str(s.dtype),
                "Non-Null": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "Missing (%)": round(s.isna().mean() * 100, 2),
                "Unique": int(s.nunique(dropna=True)),
                "Contoh Nilai": _first_non_null(s),
            }
        )
    return pd.DataFrame(rows)


def _first_non_null(s: pd.Series, limit: int = 60) -> str:
    val = s.dropna()
    if val.empty:
        return "-"
    text = str(val.iloc[0]).replace("\n", " | ").replace("<br>", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Sebaran rating beserta persentase."""
    vc = df["Rating"].value_counts().sort_index()
    out = pd.DataFrame({"Rating": vc.index, "Jumlah": vc.values})
    out["Persentase"] = (out["Jumlah"] / out["Jumlah"].sum() * 100).round(2)
    out.loc[len(out)] = ["Total", out["Jumlah"].sum(), 100.0]
    return out


def completeness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik kelengkapan data teks ulasan."""
    review = df["Review"].astype("string")
    has_text = review.fillna("").str.strip().ne("")
    non_null = df["Review"].notna()

    metrics = {
        "Total baris": len(df),
        "Ulasan dengan teks": int(has_text.sum()),
        "Ulasan tanpa teks (hanya rating)": int((~has_text).sum()),
        "Nilai kosong (NaN) pada kolom Review": int((~non_null).sum()),
        "Duplikat teks ulasan": int(review[has_text].duplicated().sum()),
        "Duplikat Review ID": int(df["Review ID"].duplicated().sum()) if "Review ID" in df else 0,
        "Ulasan dengan balasan pemilik": int(df["Owner Response"].notna().sum())
        if "Owner Response" in df
        else 0,
    }
    return pd.DataFrame(
        {"Metrik": list(metrics.keys()), "Nilai": list(metrics.values())}
    )


def review_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik panjang karakter ulasan berteks."""
    text = df["Review"].astype("string").fillna("").str.strip()
    lengths = text[text.ne("")].str.len()
    desc = lengths.describe()
    return pd.DataFrame(
        {
            "Statistik": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "Panjang Karakter": [round(float(desc[c]), 2) for c in
                                 ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
        }
    )


def rating_vs_text_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Sebaran rating untuk ulasan berteks vs tanpa teks."""
    has_text = df["Review"].astype("string").fillna("").str.strip().ne("")
    ct = pd.crosstab(df["Rating"], has_text.rename("Punya Teks"))
    ct.index = ct.index.map(lambda x: f"{int(x)} bintang")
    ct.columns = ["Dengan Teks" if bool(c) else "Tanpa Teks" for c in ct.columns]
    return ct.reset_index().rename(columns={"Rating": "Rating"})


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah ulasan per tahun berdasarkan kolom Timestamp."""
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    out = (
        pd.DataFrame({"Tahun": ts.dt.year, "Rating": df["Rating"]})
        .dropna(subset=["Tahun"])
        .groupby("Tahun")
        .agg(Jumlah=("Rating", "size"), Rating_Rata2=("Rating", "mean"))
        .reset_index()
    )
    out["Tahun"] = out["Tahun"].astype(int)
    out["Rating_Rata2"] = out["Rating_Rata2"].round(3)
    return out


def language_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Sebaran bahasa ulasan (kolom Language pada hasil scraping)."""
    vc = df["Language"].fillna("(kosong)").value_counts()
    out = pd.DataFrame({"Bahasa": vc.index, "Jumlah": vc.values})
    out["Persentase"] = (out["Jumlah"] / len(df) * 100).round(2)
    return out


def build_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membuang ulasan tanpa teks dan duplikat, mengembalikan data + rekap."""
    before = len(df)
    work = df.copy()
    work["Review"] = work["Review"].astype("string")
    work["has_text"] = work["Review"].fillna("").str.strip().ne("")

    step1 = work[work["has_text"]].copy()
    after_text = len(step1)

    step1["review_norm_key"] = (
        step1["Review"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    step2 = step1.drop_duplicates(subset="review_norm_key", keep="first").copy()
    after_dedup = len(step2)

    step2 = step2[step2["Review"].str.len() >= config.MIN_REVIEW_CHARS].copy()
    after_short = len(step2)

    recap = pd.DataFrame(
        {
            "Tahap": [
                "Dataset mentah",
                "Setelah membuang ulasan tanpa teks",
                "Setelah membuang duplikat teks",
                "Setelah membuang ulasan terlalu pendek",
            ],
            "Jumlah Ulasan": [before, after_text, after_dedup, after_short],
            "Dibuang": [0, before - after_text, after_text - after_dedup, after_dedup - after_short],
        }
    )

    clean = step2.drop(columns=["has_text", "review_norm_key"]).reset_index(drop=True)
    clean.insert(0, "review_id", range(1, len(clean) + 1))
    clean["review_len"] = clean["Review"].str.len()
    return clean, recap


def run_audit(path=None) -> dict:
    """Menjalankan seluruh pemeriksaan dan menyimpan tabel hasil."""
    df = io_utils.load_raw(path)
    clean, recap = build_clean_dataset(df)

    tables = {
        "01_dataset_overview.csv": dataset_overview(df),
        "01_rating_distribution.csv": rating_distribution(df),
        "01_completeness_summary.csv": completeness_summary(df),
        "01_review_length_stats.csv": review_length_stats(df),
        "01_rating_vs_text.csv": rating_vs_text_crosstab(df),
        "01_yearly_trend.csv": yearly_trend(df),
        "01_language_distribution.csv": language_distribution(df),
        "01_cleaning_recap.csv": recap,
    }
    for name, table in tables.items():
        io_utils.save_csv(table, config.TABLES_DIR / name)

    io_utils.save_csv(clean, config.CLEAN_CSV)

    return {
        "raw": df,
        "clean": clean,
        "tables": tables,
        "recap": recap,
    }