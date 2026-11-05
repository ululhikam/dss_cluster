"""Pipeline preprocessing teks ulasan (Tahap 2 dokumen metodologi)."""

from __future__ import annotations

import html
import re

import pandas as pd

from . import config, io_utils, normalizer

# --------------------------------------------------------------------------
# Pola regex
# --------------------------------------------------------------------------
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F000-\U0001F02F"
    "\u2600-\u27BF"
    "\uFE0F"
    "\u200D"
    "]+",
    flags=re.UNICODE,
)
RE_NUMBER = re.compile(r"\d+")
RE_NON_ALPHA = re.compile(r"[^a-z\s]")
RE_WHITESPACE = re.compile(r"\s+")

STAGE_ORDER = [
    "1. Ulasan mentah (berteks & unik)",
    "2. Case folding",
    "3. Hapus URL, tag HTML, mention, hashtag",
    "4. Hapus emoji & angka",
    "5. Hapus tanda baca",
    "6. Normalisasi kata tidak baku",
    "7. Tokenizing",
    "8. Stopword removal",
    "9. Stemming (Sastrawi)",
]


# --------------------------------------------------------------------------
# Tahap dasar
# --------------------------------------------------------------------------
def case_folding(text: str) -> str:
    return str(text).lower()


def remove_url_html(text: str) -> str:
    text = html.unescape(str(text))
    text = RE_URL.sub(" ", text)
    text = RE_BR.sub(" ", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    return text


def remove_emoji_number(text: str) -> str:
    text = RE_EMOJI.sub(" ", str(text))
    text = RE_NUMBER.sub(" ", text)
    return text


def remove_punctuation(text: str) -> str:
    text = str(text).replace("-", " ")
    text = RE_NON_ALPHA.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in str(text).split() if t]


def remove_stopwords(tokens: list[str], stopwords: set[str] | None = None) -> list[str]:
    sw = stopwords if stopwords is not None else normalizer.get_stopwords()
    return [t for t in tokens if t not in sw and len(t) >= config.MIN_TOKEN_LENGTH]


# --------------------------------------------------------------------------
# Stemming dengan cache
# --------------------------------------------------------------------------
def _load_stem_cache() -> dict[str, str]:
    return io_utils.load_json(config.STEM_CACHE_JSON, default={}) or {}


def _save_stem_cache(cache: dict[str, str]) -> None:
    io_utils.save_json(cache, config.STEM_CACHE_JSON)


def get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    return StemmerFactory().create_stemmer()


def stem_tokens(
    tokens: list[str],
    stemmer,
    cache: dict[str, str],
    *,
    persist: bool = False,
) -> list[str]:
    """Stemming per token memakai cache agar proses ulang menjadi instan."""
    out: list[str] = []
    for tok in tokens:
        stem = cache.get(tok)
        if stem is None:
            stem = stemmer.stem(tok)
            cache[tok] = stem
        if stem:
            out.append(stem)
    if persist:
        _save_stem_cache(cache)
    return out


# --------------------------------------------------------------------------
# Pipeline utama
# --------------------------------------------------------------------------
def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "Review",
    *,
    use_stemming: bool = True,
    persist_cache: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Menjalankan 9 tahap preprocessing dan merekam statistik tiap tahap."""
    stopwords = normalizer.get_stopwords()
    stemmer = get_stemmer() if use_stemming else None
    cache = _load_stem_cache()

    stage_docs: dict[str, list[str]] = {name: [] for name in STAGE_ORDER}
    clause_docs: list[str] = []
    stage_tokens: dict[str, int] = {name: 0 for name in STAGE_ORDER}
    stage_unique: dict[str, int] = {name: 0 for name in STAGE_ORDER}

    examples_idx = list(range(min(5, len(df))))
    examples: list[dict] = []

    texts = df[text_col].astype(str).tolist()

    for raw in texts:
        s2 = case_folding(raw)
        s3 = remove_url_html(s2)
        s4 = remove_emoji_number(s3)
        s5 = remove_punctuation(s4)
        s6 = normalizer.dedupe_repeated_words(normalizer.normalize_text(s5))
        s7 = tokenize(s6)
        s8 = remove_stopwords(s7, stopwords)
        s9 = stem_tokens(s8, stemmer, cache, persist=False) if use_stemming else s8
        # Versi yang mempertahankan tanda baca, dipakai untuk pemecahan klausa
        # pada analisis sentimen per aspek.
        s_clause = normalizer.normalize_text_keep_punctuation(s4)

        per_stage = {
            STAGE_ORDER[0]: raw,
            STAGE_ORDER[1]: s2,
            STAGE_ORDER[2]: s3,
            STAGE_ORDER[3]: s4,
            STAGE_ORDER[4]: s5,
            STAGE_ORDER[5]: s6,
            STAGE_ORDER[6]: " ".join(s7),
            STAGE_ORDER[7]: " ".join(s8),
            STAGE_ORDER[8]: " ".join(s9),
        }
        clause_docs.append(s_clause)
        for name, value in per_stage.items():
            stage_docs[name].append(value)

        stage_tokens[STAGE_ORDER[0]] += len(str(raw).split())
        stage_tokens[STAGE_ORDER[1]] += len(s2.split())
        stage_tokens[STAGE_ORDER[2]] += len(s3.split())
        stage_tokens[STAGE_ORDER[3]] += len(s4.split())
        stage_tokens[STAGE_ORDER[4]] += len(s5.split())
        stage_tokens[STAGE_ORDER[5]] += len(s6.split())
        stage_tokens[STAGE_ORDER[6]] += len(s7)
        stage_tokens[STAGE_ORDER[7]] += len(s8)
        stage_tokens[STAGE_ORDER[8]] += len(s9)

    if persist_cache and use_stemming:
        _save_stem_cache(cache)

    proc = df.copy()
    proc["review_raw"] = stage_docs[STAGE_ORDER[0]]
    proc["text_clean"] = stage_docs[STAGE_ORDER[4]]
    proc["text_normalized"] = stage_docs[STAGE_ORDER[5]]
    proc["text_clause"] = clause_docs
    proc["tokens"] = [" ".join(t.split()) for t in stage_docs[STAGE_ORDER[7]]]
    proc["tokens_stemmed"] = stage_docs[STAGE_ORDER[8]]
    proc["n_tokens_before"] = proc["tokens"].str.split().str.len()
    proc["n_tokens_after"] = proc["tokens_stemmed"].str.split().str.len()

    for name in STAGE_ORDER:
        docs = stage_docs[name]
        vocab: set[str] = set()
        for d in docs:
            vocab.update(str(d).split())
        stage_unique[name] = len(vocab)

    stats = pd.DataFrame(
        {
            "Tahap": STAGE_ORDER,
            "Jumlah Token": [stage_tokens[n] for n in STAGE_ORDER],
            "Token Unik (Vocab)": [stage_unique[n] for n in STAGE_ORDER],
            "Jumlah Dokumen": [len(df)] * len(STAGE_ORDER),
        }
    )
    stats["Rata-rata Token/Dokumen"] = (stats["Jumlah Token"] / stats["Jumlah Dokumen"]).round(2)
    stats["Token Terbuang"] = stats["Jumlah Token"].shift(1).fillna(stats["Jumlah Token"]) - stats["Jumlah Token"]
    stats["Token Terbuang"] = stats["Token Terbuang"].clip(lower=0).astype(int)

    for idx in examples_idx:
        examples.append(
            {
                "No": int(idx) + 1,
                "Rating": int(df.iloc[idx]["Rating"]),
                "Mentah": str(df.iloc[idx][text_col]).replace("\n", " ")[:220],
                "Setelah Cleaning": stage_docs[STAGE_ORDER[4]][idx][:220],
                "Setelah Normalisasi": stage_docs[STAGE_ORDER[5]][idx][:220],
                "Setelah Stopword": stage_docs[STAGE_ORDER[7]][idx][:220],
                "Setelah Stemming": stage_docs[STAGE_ORDER[8]][idx][:220],
            }
        )

    example_df = pd.DataFrame(examples)
    vocab_counts = count_vocabulary(proc["tokens_stemmed"])

    if verbose:
        print(f"[preprocessing] {len(proc)} dokumen diproses")
        print(f"[preprocessing] kosakata akhir: {len(vocab_counts)} kata")
        print(f"[preprocessing] cache stemming: {len(cache)} entri")
        print(f"[preprocessing] dokumen kosong setelah stemming: {(proc['n_tokens_after'] == 0).sum()}")

    return proc, stats, {"examples": example_df, "vocab_counts": vocab_counts, "cache_size": len(cache)}


def count_vocabulary(series: pd.Series) -> pd.Series:
    """Frekuensi kata pada kolom teks yang sudah dipisah spasi."""
    counts: dict[str, int] = {}
    for doc in series:
        for tok in str(doc).split():
            counts[tok] = counts.get(tok, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def run_preprocessing(
    clean_csv=None,
    *,
    use_stemming: bool = True,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Membaca data bersih, memproses, lalu menyimpan artefak."""
    from_path = clean_csv or config.CLEAN_CSV
    df = io_utils.load_csv(from_path)
    proc, stats, extras = preprocess_dataframe(df, use_stemming=use_stemming)

    if save:
        io_utils.save_csv(proc, config.PREP_CSV)
        io_utils.save_csv(stats, config.STAGE_STATS_CSV)
        io_utils.save_csv(extras["examples"], config.TABLES_DIR / "02_preprocessing_examples.csv")

    return proc, stats, extras