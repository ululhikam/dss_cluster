"""Ekstraksi aspek otomatis dan analisis sentimen per aspek.

Alur:
1. Kandidat kata kunci diambil dari frekuensi kata (ambang dapat diatur).
2. Kandidat disaring: stopword, kata sentimen, kata generik, kata Inggris.
3. Kandidat di-stemming agar varian kata menyatu.
4. Kata kunci dikelompokkan menjadi aspek dengan hierarchical clustering
   berdasarkan kemiripan co-occurrence antar kata.
5. Nama aspek diambil otomatis dari kata kunci dengan frekuensi tertinggi.
6. Sentimen dihitung per aspek pada level klausa dengan jendela token.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from . import config, io_utils, normalizer, sentiment

# --------------------------------------------------------------------------
# Kata yang tidak layak menjadi nama aspek
# --------------------------------------------------------------------------
GENERIC_WORDS: set[str] = set(
    """nya aja sih deh dong ya kan banget bgt sekali sangat paling amat
    pokoknya kalo klo kalau udah dah gitu gini kayak kaya sampe sampai
    kesini kesana kesitu dateng datang mampir bilang orang pengunjung
    situ sini sana jam hari kali minggu bulan tahun malam malem siang
    pagi muda tua satu dua tiga banyak sedikit semua lain pun cuma hanya
    juga masih sudah belum akan bisa harus mau pengen ingin baru sebentar
    terus lalu kemudian karena karna jadi tetapi tapi namun dengan pada
    dari yang ini itu dan atau oleh dalam atas bawah sesuai bangun
    silol jogja yogyakarta kota tempat makan minum datang buka tutup
    rekomendasi rekomendasi recomended recommended rekomen sesuai
    """.split()
)

ENGLISH_WORDS: set[str] = set(
    """the and for with you this that they are was not but have from your
    can all here when also their there just what about would could should
    been being has had its our out too very will more most some any who
    how why where which him her them his hers were did does do done got
    get going gonna ok okay nice good great best bad worse worst place
    time hours open close come came look love like want need know think
    make made take took give gave says said see seen say people guys
    friend friends food drink price service taste order room area spot
    hang smoking non level floor really ambience portion
    cozy cosy comfy comfortable recommended recomended
    delicious tasty yummy awesome amazing perfect lovely wonderful
    crowd crowded quiet noisy
    """.split()
)

ASPECT_KEYWORD_OVERRIDES: dict[str, str] = {
    "makan": "Makanan", "masak": "Makanan", "menu": "Makanan",
    "rasa": "Makanan", "nasi": "Makanan", "ayam": "Makanan",
    "soto": "Makanan", "rawon": "Makanan", "bakso": "Makanan",
    "mie": "Makanan", "goreng": "Makanan", "sambal": "Makanan",
    "kentang": "Makanan", "roti": "Makanan", "kue": "Makanan",
    "sayur": "Makanan", "dessert": "Makanan", "porsi": "Makanan",
    "camil": "Makanan", "snack": "Makanan", "sarapan": "Makanan",
    "minum": "Minuman", "kopi": "Minuman", "matcha": "Minuman",
    "jus": "Minuman", "teh": "Minuman", "latte": "Minuman",
    "americano": "Minuman", "mocktail": "Minuman", "smoothie": "Minuman",
    "coklat": "Minuman", "boba": "Minuman",
    "harga": "Harga", "murah": "Harga", "mahal": "Harga",
    "jangkau": "Harga", "worth": "Harga", "budget": "Harga",
    "biaya": "Harga", "bayar": "Harga", "pajak": "Harga",
    "tagih": "Harga", "kantong": "Harga",
    "layan": "Pelayanan", "pelayan": "Pelayanan", "staf": "Pelayanan",
    "karyawan": "Pelayanan", "kasir": "Pelayanan", "ramah": "Pelayanan",
    "respon": "Pelayanan", "tanggap": "Pelayanan", "sigap": "Pelayanan",
    "parkir": "Parkir", "ojol": "Parkir", "valet": "Parkir",
    "akses": "Parkir", "macet": "Parkir", "lokasi": "Parkir",
    "strategis": "Parkir", "motor": "Parkir", "mobil": "Parkir",
    "bersih": "Kebersihan", "kotor": "Kebersihan", "bau": "Kebersihan",
    "toilet": "Kebersihan", "sampah": "Kebersihan", "jorok": "Kebersihan",
    "debu": "Kebersihan",
    "fasilitas": "Fasilitas", "wifi": "Fasilitas", "musholla": "Fasilitas",
    "ruang": "Fasilitas", "meja": "Fasilitas", "kursi": "Fasilitas",
    "lantai": "Fasilitas", "vip": "Fasilitas", "colok": "Fasilitas",
    "interior": "Fasilitas", "qris": "Fasilitas", "ac": "Fasilitas",
    "suasana": "Suasana", "nyaman": "Suasana", "asik": "Suasana",
    "asyik": "Suasana", "rame": "Suasana", "ramai": "Suasana",
    "sepi": "Suasana", "musik": "Suasana", "live": "Suasana",
    "band": "Suasana", "nongkrong": "Suasana", "nongki": "Suasana",
    "ngobrol": "Suasana", "kumpul": "Suasana", "duduk": "Suasana",
    "instagramable": "Suasana", "view": "Suasana", "taman": "Suasana",
    "konser": "Suasana", "santai": "Suasana", "betah": "Suasana",
    "cozy": "Suasana", "teman": "Suasana", "keluarga": "Suasana",
    "lama": "Kecepatan", "cepat": "Kecepatan", "antri": "Kecepatan",
    "nunggu": "Kecepatan", "pesan": "Kecepatan", "pesanan": "Kecepatan",
    "lambat": "Kecepatan", "lemot": "Kecepatan", "telat": "Kecepatan",
}


@dataclass
class AspectModel:
    """Wadah hasil ekstraksi aspek."""

    min_freq: int
    keyword_freq: pd.Series
    keywords: list[str]
    keyword_to_aspect: dict[str, str] = field(default_factory=dict)
    aspect_keywords: dict[str, list[str]] = field(default_factory=dict)
    n_clusters: int = 0
    grouping_eval: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# 1. Kandidat kata kunci dari frekuensi
# --------------------------------------------------------------------------
def keyword_frequency(
    tokens_series: pd.Series,
    *,
    exclude: set[str] | None = None,
) -> pd.Series:
    """Menghitung frekuensi kata pada kolom token hasil stemming."""
    exclude = exclude or set()
    counts: Counter[str] = Counter()
    for doc in tokens_series:
        for tok in str(doc).split():
            if tok and tok not in exclude:
                counts[tok] += 1
    return pd.Series(counts).sort_values(ascending=False)


def extract_candidates(
    tokens_series: pd.Series,
    lexicon: dict[str, float],
    *,
    min_freq: int = config.ASPECT_MIN_FREQ,
    overrides: dict[str, str] | None = None,
) -> pd.Series:
    """Kandidat aspek = kata frekuensi tinggi yang bukan sentimen/umum/Inggris.

    Kata yang terdaftar di kamus aspek domain selalu dipertahankan, meskipun
    juga muncul di lexicon sentimen atau daftar kata generik (mis. ``makan``,
    ``kopi``, ``harga``). Tanpa pengecualian ini, kata aspek yang paling
    penting justru ikut terbuang.
    """
    from .preprocessing import get_stemmer

    stop = normalizer.get_stopwords()
    override_keys = set(overrides or ASPECT_KEYWORD_OVERRIDES)
    stemmer = get_stemmer()
    protected = set(override_keys)
    protected |= {stemmer.stem(w) for w in override_keys}

    exclude = (set(lexicon) | GENERIC_WORDS | ENGLISH_WORDS | stop) - protected
    freq = keyword_frequency(tokens_series, exclude=exclude)
    freq = freq[freq >= min_freq]
    freq = freq[freq.index.str.len() >= 4]
    return freq


# --------------------------------------------------------------------------
# 2. Pengelompokan otomatis kata kunci -> aspek
# --------------------------------------------------------------------------
def _cooccurrence_matrix(tokens_series: pd.Series, words: list[str]) -> np.ndarray:
    index = {w: i for i, w in enumerate(words)}
    mat = np.zeros((len(words), len(words)), dtype=float)
    for doc in tokens_series:
        present = {index[t] for t in str(doc).split() if t in index}
        for i in present:
            for j in present:
                if i != j:
                    mat[i, j] += 1
    np.fill_diagonal(mat, 0.0)
    return mat


def _similarity_from_cooccurrence(cooc: np.ndarray) -> np.ndarray:
    """Cosine similarity antar kata berdasarkan vektor co-occurrence."""
    norms = np.linalg.norm(cooc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = cooc / norms
    sim = normed @ normed.T
    return np.clip(sim, 0.0, 1.0)


def _aspect_anchors(overrides: dict[str, str], available: set[str]) -> dict[str, list[str]]:
    """Kata kunci acuan per aspek (hanya yang benar-benar muncul di korpus)."""
    anchors: dict[str, list[str]] = defaultdict(list)
    for word, aspect in overrides.items():
        if word in available:
            anchors[aspect].append(word)
    return dict(anchors)


def group_keywords(
    tokens_series: pd.Series,
    words: list[str],
    *,
    max_clusters: int = 12,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, int], int, pd.DataFrame]:
    """Mengelompokkan kata kunci dengan hierarchical clustering (average linkage)."""
    if len(words) < 2:
        return {w: 0 for w in words}, 1, pd.DataFrame()

    cooc = _cooccurrence_matrix(tokens_series, words)
    sim = _similarity_from_cooccurrence(cooc)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")

    best_labels = None
    best_score = -1.0
    best_k = 1
    rows = []
    for k in range(2, min(max_clusters, len(words)) + 1):
        labels = fcluster(z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist, labels, metric="precomputed")
        except ValueError:
            continue
        rows.append({"k": k, "silhouette": round(float(score), 4)})
        if score > best_score:
            best_score, best_labels, best_k = score, labels, k

    if best_labels is None:
        best_labels = np.ones(len(words), dtype=int)
        best_k = 1

    mapping = {w: int(lab) for w, lab in zip(words, best_labels)}
    return mapping, best_k, pd.DataFrame(rows)


def _assign_aspects(
    tokens_series: pd.Series,
    words: list[str],
    freq: pd.Series,
    overrides: dict[str, str],
    *,
    min_affinity: float = 0.35,
    min_cluster_members: int = 2,
) -> tuple[dict[str, str], int, pd.DataFrame]:
    """Memberi nama aspek pada setiap kata kunci.

    Strategi (otomatis, dengan kamus domain sebagai lapisan penamaan):

    1. Kata kunci yang dikenali kamus domain langsung diberi nama aspeknya.
    2. Kata kunci sisanya dikelompokkan dengan hierarchical clustering
       berdasarkan co-occurrence, lalu digabungkan ke aspek acuan dengan
       afinitas tertinggi.
    3. Kelompok yang tidak cukup dekat dengan aspek mana pun menjadi aspek
       baru, dinamai dari kata dengan frekuensi tertinggi di kelompok itu.
    """
    keyword_to_aspect: dict[str, str] = {}
    anchors = _aspect_anchors(overrides, set(words))

    for word in words:
        if word in overrides:
            keyword_to_aspect[word] = overrides[word]

    free_words = [w for w in words if w not in overrides]
    if not free_words:
        return keyword_to_aspect, 0, pd.DataFrame()

    mapping, k, eval_df = group_keywords(tokens_series, free_words)

    cooc = _cooccurrence_matrix(tokens_series, words)
    idx = {w: i for i, w in enumerate(words)}
    anchor_words = {a: ws for a, ws in anchors.items() if ws}

    groups: dict[int, list[str]] = defaultdict(list)
    for word, gid in mapping.items():
        groups[gid].append(word)

    for gid, members in groups.items():
        if len(members) < min_cluster_members:
            # Kelompok terlalu kecil: gabungkan ke aspek acuan terdekat bila ada.
            members_sorted = members
        else:
            members_sorted = sorted(members, key=lambda w: -freq.get(w, 0))

        affinity: dict[str, float] = {}
        for aspect, aws in anchor_words.items():
            score = 0.0
            for w in members_sorted:
                for a in aws:
                    if w != a:
                        score += cooc[idx[w], idx[a]]
            affinity[aspect] = score

        total = sum(affinity.values())
        best_aspect, best_score = (None, 0.0)
        if affinity and total > 0:
            best_aspect, best_score = max(affinity.items(), key=lambda kv: kv[1])

        if best_aspect and (best_score / total) >= min_affinity:
            name = best_aspect
        elif len(members_sorted) >= min_cluster_members:
            name = members_sorted[0].capitalize()
        else:
            name = "Lainnya"

        for w in members_sorted:
            keyword_to_aspect[w] = name

    return keyword_to_aspect, k, eval_df


def _stem_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """Menambahkan bentuk ter-stemming dari kata kunci kamus domain."""
    from .preprocessing import get_stemmer

    stemmer = get_stemmer()
    out: dict[str, str] = {}
    for word, aspect in overrides.items():
        out[word] = aspect
        stemmed = stemmer.stem(word)
        if stemmed:
            out.setdefault(stemmed, aspect)
    return out


def build_aspect_model(
    tokens_series: pd.Series,
    lexicon: dict[str, float],
    *,
    min_freq: int = config.ASPECT_MIN_FREQ,
    max_keywords: int = 200,
    min_keywords: int = 5,
    overrides: dict[str, str] | None = None,
) -> AspectModel:
    """Membangun model aspek otomatis dari frekuensi kata."""
    freq = extract_candidates(tokens_series, lexicon, min_freq=min_freq, overrides=overrides)
    if len(freq) < min_keywords:
        raise ValueError(
            f"hanya {len(freq)} kata kunci pada ambang {min_freq}. "
            f"Turunkan min_freq (mis. {max(20, min_freq // 2)})."
        )
    words = list(freq.index[:max_keywords])
    freq_sel = freq.loc[words]

    override_map = _stem_overrides(overrides or ASPECT_KEYWORD_OVERRIDES)
    keyword_to_aspect, n_groups, eval_df = _assign_aspects(
        tokens_series, words, freq_sel, override_map
    )

    aspect_keywords: dict[str, list[str]] = defaultdict(list)
    for word, aspect in keyword_to_aspect.items():
        aspect_keywords[aspect].append(word)
    aspect_keywords = {
        a: sorted(ws, key=lambda w: -freq_sel.get(w, 0))[: config.ASPECT_MAX_KEYWORDS]
        for a, ws in aspect_keywords.items()
    }

    model = AspectModel(
        min_freq=min_freq,
        keyword_freq=freq_sel,
        keywords=words,
        keyword_to_aspect=keyword_to_aspect,
        aspect_keywords=aspect_keywords,
        n_clusters=n_groups,
    )
    model.grouping_eval = eval_df
    return model


def aspect_model_frames(model: AspectModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dua tabel: daftar aspek + daftar kata kunci dengan frekuensinya."""
    aspect_rows = []
    for aspect, words in sorted(model.aspect_keywords.items()):
        aspect_rows.append(
            {
                "Aspek": aspect,
                "Jumlah Kata Kunci": len(words),
                "Kata Kunci": ", ".join(words),
                "Frekuensi Total": int(sum(model.keyword_freq.get(w, 0) for w in words)),
            }
        )
    aspects_df = pd.DataFrame(aspect_rows).sort_values("Frekuensi Total", ascending=False)

    keyword_rows = [
        {
            "Kata Kunci": w,
            "Frekuensi": int(model.keyword_freq.get(w, 0)),
            "Aspek": model.keyword_to_aspect.get(w, "-"),
        }
        for w in model.keywords
    ]
    keywords_df = pd.DataFrame(keyword_rows).sort_values("Frekuensi", ascending=False)
    return aspects_df.reset_index(drop=True), keywords_df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Sentimen per aspek
# --------------------------------------------------------------------------
def _keyword_pattern(keywords: list[str]) -> re.Pattern | None:
    if not keywords:
        return None
    # word-boundary, urutan terpanjang dulu agar frasa kalah panjang tidak salah tangkap
    alts = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b")


def _build_stem_lookup(words: set[str], stemmer=None) -> dict[str, str]:
    """Peta bentuk permukaan -> bentuk dasar untuk pencocokan kata kunci aspek."""
    from .preprocessing import get_stemmer

    cache = io_utils.load_json(config.STEM_CACHE_JSON, default={}) or {}
    stemmer = stemmer or get_stemmer()
    lookup: dict[str, str] = {}
    for word in words:
        stem = cache.get(word)
        if stem is None:
            stem = stemmer.stem(word)
            cache[word] = stem
        lookup[word] = stem
    return lookup


@dataclass
class AspectHit:
    aspect: str
    clause: str
    keyword: str
    score: float


def extract_aspect_sentences(
    df: pd.DataFrame,
    model: AspectModel,
    lexicon: dict[str, float],
    *,
    window: int = config.SENTIMENT_WINDOW,
    text_col: str | None = None,
) -> pd.DataFrame:
    """Satu baris per (ulasan, aspek): sentimen aspek pada konteks terdekat.

    Skor hanya dihitung dari kata sentimen dalam jendela ``window`` token
    di sekitar kata kunci aspek, sehingga "tempatnya oke tapi makanannya
    hambar" tidak mencampur polaritas dua aspek.

    Kata kunci model berbentuk kata dasar (hasil stemming), sedangkan teks
    klausa masih berbentuk permukaan. Karena itu setiap token dipetakan ke
    bentuk dasarnya lebih dulu sebelum dicocokkan.
    """
    if text_col is None:
        text_col = "text_clause" if "text_clause" in df.columns else "text_normalized"

    aspect_stems = {
        aspect: set(words) for aspect, words in model.aspect_keywords.items() if words
    }
    if not aspect_stems:
        return pd.DataFrame()

    surface_vocab: set[str] = set()
    for text in df[text_col].fillna("").astype(str):
        surface_vocab.update(re.findall(r"[a-z]+", text.lower()))
    stem_lookup = _build_stem_lookup(surface_vocab)

    rows: list[dict] = []
    for idx, row in df.iterrows():
        text = str(row.get(text_col) or row.get("Review") or "")
        review_id = row.get("review_id", idx)

        for clause in sentiment.split_clauses(text):
            tokens = re.findall(r"[a-z]+", clause.lower())
            if len(tokens) < 2:
                continue
            stems = [stem_lookup.get(t, t) for t in tokens]

            matched_any = False
            for aspect, stems_set in aspect_stems.items():
                positions = [i for i, s in enumerate(stems) if s in stems_set]
                if not positions:
                    continue
                matched_any = True

                window_scores = []
                matched_keywords = set()
                for pos in positions:
                    matched_keywords.add(tokens[pos])
                    lo = max(0, pos - window)
                    hi = min(len(tokens), pos + window + 1)
                    fragment = " ".join(tokens[lo:hi])
                    window_scores.append(sentiment.score_clause(fragment, lexicon))

                score = float(np.mean(window_scores)) if window_scores else 0.0
                rows.append(
                    {
                        "review_id": review_id,
                        "Rating": int(row.get("Rating", 0)),
                        "Aspek": aspect,
                        "Klausa": clause.strip()[:300],
                        "Kata Kunci": ", ".join(sorted(matched_keywords)),
                        "Skor": round(score, 3),
                        "Sentimen": sentiment.label_from_score(score),
                    }
                )
            if matched_any:
                continue

    return pd.DataFrame(rows)


def summarize_aspects(hits: pd.DataFrame, total_reviews: int) -> pd.DataFrame:
    """Tabel ringkasan aspek: volume, komposisi sentimen, rating, prioritas."""
    if hits.empty:
        return pd.DataFrame()

    grouped = hits.groupby("Aspek")
    rows = []
    for aspect, g in grouped:
        n = len(g)
        pos = int((g["Sentimen"] == "Positif").sum())
        neg = int((g["Sentimen"] == "Negatif").sum())
        neu = int((g["Sentimen"] == "Netral").sum())
        hidden = int(((g["Rating"] >= 4) & (g["Sentimen"] == "Negatif")).sum())
        rows.append(
            {
                "Aspek": aspect,
                "Jumlah Disebut": n,
                "Ulasan Unik": int(g["review_id"].nunique()),
                "% dari Ulasan": round(g["review_id"].nunique() / total_reviews * 100, 2),
                "Positif": pos,
                "Netral": neu,
                "Negatif": neg,
                "% Positif": round(pos / n * 100, 2),
                "% Negatif": round(neg / n * 100, 2),
                "Rating Rata-rata": round(float(g["Rating"].mean()), 2),
                "Hidden Complaint": hidden,
            }
        )

    out = pd.DataFrame(rows)
    out["Skor Prioritas"] = (
        out["% Negatif"] / 100.0 * np.sqrt(out["Jumlah Disebut"])
    ).round(3)
    out["Peringkat Prioritas"] = out["Skor Prioritas"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("Skor Prioritas", ascending=False).reset_index(drop=True)
    out["Rekomendasi"] = [
        _priority_label(row, len(out)) for _, row in out.iterrows()
    ]
    return out


def _priority_label(row, n_aspects: int) -> str:
    """Menentukan tingkat prioritas.

    Skor prioritas bergantung pada volume sehingga nilainya tidak dapat
    dibandingkan antar-dataset. Karena itu tingkat prioritas ditentukan dari
    **peringkat relatif** antar aspek, dengan satu pengecualian: aspek yang
    persentase negatifnya sangat kecil tetap dianggap kekuatan yang perlu
    dipertahankan, sekalipun volumenya besar.
    """
    if row["% Negatif"] < 8 and row["% Positif"] >= 70:
        return "Pertahankan"

    rank = int(row["Peringkat Prioritas"])
    third = max(1, round(n_aspects / 3))
    two_third = max(third + 1, round(n_aspects * 2 / 3))

    if rank <= third:
        return "Prioritas 1 - Perbaikan mendesak"
    if rank <= two_third:
        return "Prioritas 2 - Perlu ditingkatkan"
    return "Prioritas 3 - Pantau"


def aspect_examples(hits: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Contoh klausa paling positif dan paling negatif tiap aspek."""
    out = []
    for aspect, g in hits.groupby("Aspek"):
        for label, part in (("Negatif", g.nsmallest(n, "Skor")), ("Positif", g.nlargest(n, "Skor"))):
            for _, row in part.iterrows():
                out.append(
                    {
                        "Aspek": aspect,
                        "Jenis": label,
                        "Rating": row["Rating"],
                        "Skor": row["Skor"],
                        "Klausa": row["Klausa"],
                    }
                )
    return pd.DataFrame(out)


def run_aspect_analysis(
    df: pd.DataFrame,
    lexicon: dict[str, float],
    *,
    min_freq: int = config.ASPECT_MIN_FREQ,
    window: int = config.SENTIMENT_WINDOW,
) -> dict:
    """Pipeline lengkap ekstraksi aspek + sentimen per aspek."""
    if "tokens_stemmed" not in df.columns:
        raise KeyError("kolom 'tokens_stemmed' tidak ada. Jalankan notebook 02 dulu.")

    model = build_aspect_model(df["tokens_stemmed"], lexicon, min_freq=min_freq)
    aspects_df, keywords_df = aspect_model_frames(model)
    hits = extract_aspect_sentences(df, model, lexicon, window=window)
    summary = summarize_aspects(hits, total_reviews=len(df))
    examples = aspect_examples(hits)

    return {
        "model": model,
        "aspects": aspects_df,
        "keywords": keywords_df,
        "hits": hits,
        "summary": summary,
        "examples": examples,
    }