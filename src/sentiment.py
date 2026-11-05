"""Analisis sentimen berbasis lexicon (level dokumen dan level klausa)."""

from __future__ import annotations

import re

import pandas as pd

from . import config

# --------------------------------------------------------------------------
# Splitter klausa
# --------------------------------------------------------------------------
CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;,\n]|\b(?:tapi|tetapi|namun|namunnya|cuma|hanya saja|sayangnya|"
    r"walaupun|walau|meskipun|sedangkan|padahal|selain itu)\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+|\b(?:tetapi|tapi|namun|sedangkan)\b", flags=re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Memecah ulasan menjadi klausa, mempertahankan pemisah 'tapi' sebagai batas."""
    text = str(text).replace("<br>", ". ").replace("<br/>", ". ").replace("<br />", ". ")
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(text)]
    return [p for p in parts if len(p.split()) >= 1]


def split_sentences(text: str) -> list[str]:
    text = str(text).replace("<br>", ". ").replace("<br/>", ". ").replace("<br />", ". ")
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# Skoring
# --------------------------------------------------------------------------
def score_clause(
    clause: str,
    lexicon: dict[str, float],
    *,
    negators: set[str] | None = None,
    boosters: dict[str, float] | None = None,
) -> float:
    """Menghitung skor sentimen satu klausa dengan penanganan negasi & booster."""
    negators = negators or config.NEGATORS
    boosters = boosters or config.BOOSTERS
    words = re.findall(r"[a-z]+", str(clause).lower())
    total = 0.0

    for i, word in enumerate(words):
        if word not in lexicon:
            continue
        value = lexicon[word]

        # Negasi hingga 2 token ke belakang.
        if i >= 1 and words[i - 1] in negators:
            value = -value
        elif i >= 2 and words[i - 2] in negators:
            value = -value

        # Booster (penguat) hingga 2 token ke belakang.
        for back in (1, 2):
            if i >= back and words[i - back] in boosters:
                value *= boosters[words[i - back]]
                break

        total += value
    return total


def label_from_score(score: float) -> str:
    if score > config.POS_THRESHOLD:
        return "Positif"
    if score < config.NEG_THRESHOLD:
        return "Negatif"
    return "Netral"


def score_text(text: str, lexicon: dict[str, float], **kwargs) -> float:
    """Skor dokumen = jumlah skor seluruh klausa."""
    return sum(score_clause(c, lexicon, **kwargs) for c in split_clauses(text))


def label_text(text: str, lexicon: dict[str, float], **kwargs) -> str:
    return label_from_score(score_text(text, lexicon, **kwargs))


# --------------------------------------------------------------------------
# Label dari rating (hanya untuk pembanding, BUKAN label utama)
# --------------------------------------------------------------------------
def label_from_rating(rating: int) -> str:
    rating = int(rating)
    if rating >= 4:
        return "Positif"
    if rating <= 2:
        return "Negatif"
    return "Netral"


# --------------------------------------------------------------------------
# Sentimen tingkat dokumen
# --------------------------------------------------------------------------
def label_documents(
    df: pd.DataFrame,
    lexicon: dict[str, float],
    text_col: str = "Review",
) -> pd.DataFrame:
    out = df.copy()
    scores, labels = [], []
    for text in out[text_col]:
        s = score_text(text, lexicon)
        scores.append(s)
        labels.append(label_from_score(s))
    out["skor_sentimen"] = scores
    out["sentimen_teks"] = labels
    out["sentimen_rating"] = out["Rating"].map(label_from_rating)
    out["sesuai_rating"] = out["sentimen_teks"] == out["sentimen_rating"]
    out["word_count"] = out[text_col].astype(str).str.split().str.len()
    out["skor_per_kata"] = (out["skor_sentimen"] / out["word_count"].clip(lower=1)).round(4)
    return out


def sentiment_distribution(df: pd.DataFrame, col: str = "sentimen_teks") -> pd.DataFrame:
    order = ["Positif", "Netral", "Negatif"]
    vc = df[col].value_counts().reindex(order, fill_value=0)
    out = pd.DataFrame({"Sentimen": vc.index, "Jumlah": vc.values})
    out["Persentase"] = (out["Jumlah"] / out["Jumlah"].sum() * 100).round(2)
    return out


def cross_rating_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Positif", "Netral", "Negatif"]
    ct = pd.crosstab(df["Rating"], df["sentimen_teks"]).reindex(columns=order, fill_value=0)
    ct.index = ct.index.map(lambda x: f"{int(x)} bintang")
    ct = ct.reset_index().rename(columns={"Rating": "Rating"})
    return ct


def gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Ulasan rating tinggi yang isinya negatif, dan sebaliknya."""
    hid = df[(df["Rating"] >= 4) & (df["sentimen_teks"] == "Negatif")]
    mis = df[(df["Rating"] <= 2) & (df["sentimen_teks"] == "Positif")]
    return pd.DataFrame(
        {
            "Kategori": [
                "Hidden complaint (rating 4-5 tapi teks negatif)",
                "Hidden praise (rating 1-2 tapi teks positif)",
                "Total ulasan dibandingkan",
            ],
            "Jumlah": [len(hid), len(mis), len(df)],
            "Persentase": [
                round(len(hid) / len(df) * 100, 2),
                round(len(mis) / len(df) * 100, 2),
                100.0,
            ],
        }
    )


def top_agreement_examples(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Contoh ekstrem: paling positif dan paling negatif menurut lexicon."""
    cols = ["Rating", "Review", "skor_sentimen", "sentimen_teks", "sentimen_rating"]
    pos = df.nlargest(n, "skor_sentimen")[cols].assign(Kategori="Skor tertinggi")
    neg = df.nsmallest(n, "skor_sentimen")[cols].assign(Kategori="Skor terendah")
    return pd.concat([pos, neg], ignore_index=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run_document_sentiment(df: pd.DataFrame, lexicon: dict[str, float],
                           text_col: str = "Review") -> tuple[pd.DataFrame, dict]:
    # Skor dihitung pada teks yang sudah dinormalisasi (bentuk baku seperti
    # "gak enak" -> "tidak enak") sambil mempertahankan tanda baca agar
    # pemecahan klausa tetap bekerja.
    if "text_clause" in df.columns:
        base = df.copy()
        base["_text_for_scoring"] = base["text_clause"].fillna(base[text_col])
    elif "text_normalized" in df.columns:
        base = df.copy()
        base["_text_for_scoring"] = base["text_normalized"].fillna(base[text_col])
    else:
        base = df.copy()
        base["_text_for_scoring"] = base[text_col]

    scored = label_documents(base, lexicon, text_col="_text_for_scoring")
    scored = scored.drop(columns=["_text_for_scoring"])

    tables = {
        "sentiment_distribution": sentiment_distribution(scored),
        "sentiment_distribution_rating": sentiment_distribution(scored, "sentimen_rating"),
        "cross_rating": cross_rating_sentiment(scored),
        "gap": gap_analysis(scored),
        "extremes": top_agreement_examples(scored),
    }
    tables["cross_rating_pct"] = _cross_pct(scored)
    return scored, tables


def _cross_pct(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Positif", "Netral", "Negatif"]
    ct = pd.crosstab(df["Rating"], df["sentimen_teks"], normalize="index")
    ct = ct.reindex(columns=order, fill_value=0) * 100
    ct.index = ct.index.map(lambda x: f"{int(x)} bintang")
    return ct.round(2).reset_index().rename(columns={"Rating": "Rating"})


def agreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    agree = int(df["sesuai_rating"].sum())
    total = len(df)
    return pd.DataFrame(
        {
            "Metrik": [
                "Total ulasan",
                "Sentimen teks sama dengan rating",
                "Sentimen teks berbeda dengan rating",
                "Tingkat kesesuaian (%)",
            ],
            "Nilai": [total, agree, total - agree, round(agree / total * 100, 2)],
        }
    )