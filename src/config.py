"""Konfigurasi global proyek.

Semua path, konstanta, dan hyperparameter dikumpulkan di sini agar notebook
tidak mengandung path hardcoded.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Path
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LEXICON_DIR = DATA_DIR / "lexicon"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RAW_XLSX = RAW_DIR / "silol_kopi_eatery_reviews.xlsx"

for _d in (RAW_DIR, LEXICON_DIR, CACHE_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Artefak antar-notebook
# --------------------------------------------------------------------------
CLEAN_CSV = PROCESSED_DIR / "01_reviews_clean.csv"
PREP_CSV = PROCESSED_DIR / "02_reviews_preprocessed.csv"
STAGE_STATS_CSV = TABLES_DIR / "02_preprocessing_stages.csv"
DOC_SENTIMENT_CSV = PROCESSED_DIR / "03_reviews_document_sentiment.csv"
ASPECT_SENTIMENT_CSV = PROCESSED_DIR / "04_aspect_sentiment.csv"
ASPECT_KEYWORD_CSV = PROCESSED_DIR / "04_aspect_keywords.csv"
ASPECT_SUMMARY_CSV = TABLES_DIR / "04_aspect_summary.csv"
CLUSTERED_CSV = PROCESSED_DIR / "05_reviews_clustered.csv"
K_EVAL_CSV = TABLES_DIR / "05_k_evaluation.csv"
CLUSTER_SUMMARY_CSV = TABLES_DIR / "05_cluster_summary.csv"

TFIDF_MATRIX_NPZ = PROCESSED_DIR / "05_tfidf_matrix.npz"
TFIDF_VECTORIZER_PKL = PROCESSED_DIR / "05_tfidf_vectorizer.pkl"

STEM_CACHE_JSON = CACHE_DIR / "stem_cache.json"
STOPWORDS_JSON = DATA_DIR / "stopwords.json"
INSET_POS_TSV = LEXICON_DIR / "inset_positive.tsv"
INSET_NEG_TSV = LEXICON_DIR / "inset_negative.tsv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
MIN_REVIEW_CHARS = 3          # ulasan lebih pendek (mis. "ok") dibuang
MIN_TOKEN_LENGTH = 4          # token lebih pendek dibuang setelah stopword

# --------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR = True

# --------------------------------------------------------------------------
# K-Means
# --------------------------------------------------------------------------
K_RANGE = range(2, 11)
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# --------------------------------------------------------------------------
# Ekstraksi aspek otomatis
# --------------------------------------------------------------------------
# Ambang frekuensi kata kunci aspek. Bisa diubah sesuai kebutuhan (50-400).
ASPECT_MIN_FREQ = 100
# Jumlah kata kunci maksimum per aspek saat pengelompokan otomatis.
ASPECT_MAX_KEYWORDS = 6
# Jendela token untuk menghitung skor sentimen di sekitar kata aspek.
SENTIMENT_WINDOW = 5

# --------------------------------------------------------------------------
# Sentimen
# --------------------------------------------------------------------------
NEGATORS = {
    "tidak", "tak", "bukan", "kurang", "belum", "jangan", "gak", "ga",
    "nggak", "enggak", "tdk", "tanpa", "kurangnya",
}
BOOSTERS = {
    "sangat": 1.5, "banget": 1.5, "bgt": 1.5, "sekali": 1.3,
    "paling": 1.4, "amat": 1.4, "sangatlah": 1.5,
}
NEG_THRESHOLD = 0.0           # skor < 0 -> negatif
POS_THRESHOLD = 0.0           # skor > 0 -> positif

# --------------------------------------------------------------------------
# Visual
# --------------------------------------------------------------------------
FIG_DPI = 300
PALETTE_SENTIMENT = {"Positif": "#2e7d32", "Netral": "#9e9e9e", "Negatif": "#c62828"}
PALETTE_CLUSTER = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]