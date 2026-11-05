"""Smoke test cepat: memastikan seluruh modul src berjalan tanpa error.

Jalankan: python -m tests.smoke
"""

from __future__ import annotations

import sys
import traceback

from src import audit, features, interpret, io_utils, lexicon, preprocessing


def main(sample: int = 300) -> int:
    print("=" * 70)
    print("SMOKE TEST modul src")
    print("=" * 70)

    raw = io_utils.load_raw()
    print(f"[1] raw dataset: {raw.shape}")

    clean, recap = audit.build_clean_dataset(raw)
    print(f"[2] clean dataset: {clean.shape}")
    print(recap.to_string(index=False))

    small = clean.head(sample).copy()
    small["review_id"] = range(1, len(small) + 1)

    proc, stats, extras = preprocessing.preprocess_dataframe(small, persist_cache=True)
    print(f"[3] preprocessing OK, kosakata={len(extras['vocab_counts'])}")
    print(stats[["Tahap", "Jumlah Token", "Token Unik (Vocab)"]].to_string(index=False))

    lex = lexicon.build_domain_lexicon()
    print(f"[4] lexicon: {len(lex)} kata")

    from src import sentiment
    scored, tables = sentiment.run_document_sentiment(proc, lex)
    print("[5] sentimen dokumen OK")
    print(tables["sentiment_distribution"].to_string(index=False))

    from src import aspects
    res = aspects.run_aspect_analysis(proc, lex, min_freq=5)
    print(f"[6] aspek OK, aspek={len(res['aspects'])}, hit={len(res['hits'])}")
    print(res["aspects"].head().to_string(index=False))
    print(res["summary"].head().to_string(index=False))

    vectorizer, matrix, summary = features.run_tfidf(proc["tokens_stemmed"])
    print(f"[7] tfidf OK: {matrix.shape}")

    from src import clustering
    cres = clustering.run_clustering(matrix, vectorizer, k=min(3, matrix.shape[0] - 1))
    print(f"[8] clustering OK: k={cres['chosen_k']} sil={cres['silhouette_final']}")

    profile = interpret.cluster_profile(scored, cres["labels"], cres["top_terms_str"])
    print("[9] profil cluster OK")
    print(profile.iloc[:, :6].to_string(index=False))

    pm = interpret.priority_matrix(res["summary"])
    print("[10] matriks prioritas OK")
    print(pm.to_string(index=False))

    print("\nSEMUA MODUL BERJALAN NORMAL")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)