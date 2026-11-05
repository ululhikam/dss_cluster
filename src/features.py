"""Representasi teks TF-IDF."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


def build_tfidf(
    docs,
    *,
    min_df: int = config.TFIDF_MIN_DF,
    max_df: float = config.TFIDF_MAX_DF,
    ngram_range: tuple = config.TFIDF_NGRAM_RANGE,
    sublinear_tf: bool = config.TFIDF_SUBLINEAR,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=False,  # teks sudah dinormalisasi
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def tfidf_summary(vectorizer: TfidfVectorizer, matrix) -> pd.DataFrame:
    vocab_size = len(vectorizer.get_feature_names_out())
    nnz = matrix.nnz
    density = nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.size else 0.0
    return pd.DataFrame(
        {
            "Metrik": [
                "Jumlah dokumen", "Jumlah fitur (term)", "Total nilai non-zero",
                "Kepadatan matriks (%)", "Nilai minimum non-zero", "Nilai maksimum",
                "rata-rata fitur per dokumen",
            ],
            "Nilai": [
                matrix.shape[0], vocab_size, nnz, round(density * 100, 3),
                round(float(matrix.data.min()), 4) if nnz else 0.0,
                round(float(matrix.data.max()), 4) if nnz else 0.0,
                round(nnz / matrix.shape[0], 2) if matrix.shape[0] else 0.0,
            ],
        }
    )


def top_terms_per_document(vectorizer: TfidfVectorizer, matrix, doc_index: int, n: int = 8):
    names = vectorizer.get_feature_names_out()
    row = matrix.getrow(doc_index)
    order = row.toarray().ravel().argsort()[::-1][:n]
    return [(names[i], round(float(row[0, i]), 4)) for i in order if row[0, i] > 0]


def term_document_frequency(vectorizer: TfidfVectorizer, matrix) -> pd.Series:
    df = (matrix > 0).sum(axis=0).A1
    return pd.Series(df, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)


def run_tfidf(docs, **kwargs):
    vectorizer, matrix = build_tfidf(docs, **kwargs)
    return vectorizer, matrix, tfidf_summary(vectorizer, matrix)