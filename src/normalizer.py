"""Normalisasi kata tidak baku, singkatan, dan stopword untuk ulasan Indonesia."""

from __future__ import annotations

import re
from functools import lru_cache

from . import config, io_utils

# --------------------------------------------------------------------------
# Kamus normalisasi (kata tidak baku / singkatan / typo -> bentuk baku)
# --------------------------------------------------------------------------
SLANG_MAP: dict[str, str] = {
    # kata ganti & partikel
    "aq": "aku", "gw": "saya", "gue": "saya", "gua": "saya", "sy": "saya",
    "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "jd": "jadi", "jg": "juga", "sdh": "sudah", "blm": "belum", "tdk": "tidak",
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak",
    "enggak": "tidak", "ngga": "tidak", "gasi": "tidak", "g": "tidak",
    "aja": "saja", "aj": "saja", "doang": "saja", "cuma": "cuma",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau", "kalo'": "kalau",
    "yg": "yang", "sy": "saya", "dapet": "dapat", "dpt": "dapat",
    "bgt": "banget", "bngt": "banget", "bnget": "banget", "banget": "banget",
    "udah": "sudah", "udh": "sudah", "dah": "sudah", "blom": "belum",
    "sm": "sama", "sma": "sama", "dr": "dari", "dri": "dari",
    "kmrn": "kemarin", "td": "tadi", "bs": "bisa", "hrs": "harus",
    "bikin": "buat", "dibikin": "dibuat", "bikinnya": "buatnya",
    "trs": "terus", "trus": "terus", "lgi": "lagi", "lg": "lagi",
    "bsk": "besok", "skrg": "sekarang", "skrng": "sekarang",
    "gimana": "bagaimana", "gmn": "bagaimana", "knp": "kenapa",
    "gitu": "begitu", "gini": "begini", "kayak": "seperti", "kaya": "seperti",
    "kyk": "seperti", "kek": "seperti", "keknya": "sepertinya",
    "sampe": "sampai", "sampey": "sampai", "dapet": "dapat",
    "smoga": "semoga", "mudah2an": "semoga", "insyaallah": "semoga",
    "tp": "tapi", "tpi": "tapi", "tapi": "tapi", "namun": "tapi",
    "utk": "untuk", "buat": "untuk", "bagi": "untuk",
    "orang": "orang", "org": "orang", "temen": "teman", "tmn": "teman",
    "temen2": "teman", "teman2": "teman", "anak2": "anak", "anak-anak": "anak",
    "makan2": "makan", "minum2": "minum", "tempat2": "tempat",
    "enak2": "enak", "enak-enak": "enak", "bagus2": "bagus",
    "lama2": "lama", "lama-lama": "lama", "rame2": "ramai",
    "mahal2": "mahal", "murah2": "murah", "cepat2": "cepat",
    # konteks kuliner
    "makanan": "makanan", "makanannya": "makanan", "makananx": "makanan",
    "minuman": "minuman", "minumannya": "minuman", "minumanx": "minuman",
    "pelayanan": "pelayanan", "pelayanannya": "pelayanan",
    "pelayan": "pelayan", "pelayannya": "pelayan", "playanan": "pelayanan",
    "harga": "harga", "harganya": "harga", "hrg": "harga", "hrga": "harga",
    "parkir": "parkir", "parkirnya": "parkir", "parkiran": "parkir",
    "toilet": "toilet", "toiletnya": "toilet", "wc": "toilet",
    "wifi": "wifi", "wifinya": "wifi", "wify": "wifi", "wi-fi": "wifi",
    "tempat": "tempat", "tempatnya": "tempat", "tmpt": "tempat",
    "suasana": "suasana", "suasananya": "suasana",
    "musik": "musik", "musiknya": "musik", "music": "musik",
    "musicnya": "musik", "live": "live",
    "kopi": "kopi", "kopinya": "kopi", "coffee": "kopi", "cafe": "kafe",
    "kafe": "kafe", "kafee": "kafe", "café": "kafe",
    "menu": "menu", "menunya": "menu", "menunnya": "menu",
    "rasa": "rasa", "rasanya": "rasa", "rassa": "rasa",
    "porsi": "porsi", "porsinya": "porsi", "portion": "porsi",
    "mahal": "mahal", "mahalnya": "mahal", "pricey": "mahal",
    "murah": "murah", "murahnya": "murah", "terjangkau": "terjangkau",
    "worth": "worth", "worthit": "worth", "worthed": "worth",
    "overprice": "mahal", "overpriced": "mahal",
    # sifat / penilaian
    "enak": "enak", "enaak": "enak", "enakk": "enak", "enek": "enak",
    "nyaman": "nyaman", "nyamn": "nyaman", "nyamann": "nyaman",
    "bagus": "bagus", "baguss": "bagus", "bgus": "bagus",
    "jelek": "jelek", "jelek2": "jelek", "jlek": "jelek",
    "kotor": "kotor", "kotornya": "kotor", "kotor2": "kotor",
    "bersih": "bersih", "bersihnya": "bersih", "brsih": "bersih",
    "ramah": "ramah", "ramah2": "ramah", "ramahh": "ramah",
    "kesini": "ke sini", "kesana": "ke sana", "kesitu": "ke situ",
    "kekurangan": "kekurangan", "kurang": "kurang", "krg": "kurang",
    "rekomendasi": "rekomendasi", "recommended": "rekomendasi",
    "recomended": "rekomendasi", "rekomen": "rekomendasi",
    "instagramable": "instagramable", "instagramablee": "instagramable",
    "instagenic": "instagramable", "instagramable": "instagramable",
    "nongkrong": "nongkrong", "nongkrongnya": "nongkrong",
    "nongki": "nongkrong", "ngopi": "ngopi", "ngobrol": "ngobrol",
    "hangout": "nongkrong", "kongkow": "nongkrong", "kopdar": "nongkrong",
    "wfc": "kerja", "wfh": "kerja", "meeting": "rapat", "meetup": "rapat",
    "staff": "staf", "stafnya": "staf", "karyawan": "staf",
    "karyawannya": "staf", "server": "pelayan", "waiters": "pelayan",
    "waiter": "pelayan", "pelayannya": "pelayan",
    "order": "pesan", "orderan": "pesanan", "pesenan": "pesanan",
    "pesen": "pesan", "psn": "pesan", "dipesen": "dipesan",
    "dateng": "datang", "dtg": "datang", "kesini": "ke sini",
    "nunggu": "menunggu", "nunggunya": "menunggu", "ngantri": "antri",
    "antrian": "antri", "lama": "lama", "lamaa": "lama", "lamaaaa": "lama",
    "lemot": "lambat", "lelet": "lambat", "lambat": "lambat",
    "cepat": "cepat", "cepet": "cepat", "cepetan": "cepat",
    "berisik": "berisik", "bising": "berisik", "gaduh": "berisik",
    "rame": "ramai", "ramee": "ramai", "sepi": "sepi",
    "luas": "luas", "gede": "besar", "gde": "besar", "besar": "besar",
    "ac": "ac", "vip": "vip", "vipnya": "vip",
    "lantai": "lantai", "meja": "meja", "kursi": "kursi",
    "ruangan": "ruangan", "ruang": "ruangan", "ruangannya": "ruangan",
    "musholla": "musholla", "musala": "musholla", "mushola": "musholla",
    "colokan": "colokan", "stopkontak": "colokan",
    "ojol": "ojol", "gojek": "ojol", "grab": "ojol", "ojek": "ojek",
    "valet": "valet", "tukang": "tukang",
    "goreng": "goreng", "gorengan": "goreng", "nasi": "nasi",
    "ayam": "ayam", "soto": "soto", "rawon": "rawon", "bakso": "bakso",
    "mie": "mie", "mi": "mie", "sambal": "sambal", "kentang": "kentang",
    "roti": "roti", "kue": "kue", "sayur": "sayur", "dessert": "dessert",
    "matcha": "matcha", "latte": "latte", "jus": "jus", "juice": "jus",
    "teh": "teh", "es": "es", "americano": "americano", "kopi": "kopi",
    "asin": "asin", "hambar": "hambar", "pahit": "pahit", "manis": "manis",
    "pedas": "pedas", "dingin": "dingin", "panas": "panas",
    "keras": "keras", "lembek": "lembek", "basi": "basi",
}

# Kata yang dibuang pada tahap stopword tambahan (di luar NLTK Indonesia).
EXTRA_STOPWORDS: set[str] = {
    "nya", "sih", "deh", "dong", "yaa", "yah", "kan", "lah", "kah", "pun",
    "kok", "nih", "tuh", "loh", "lho", "ee", "eh", "hmm", "wkwk", "wkwkwk",
    "haha", "hehe", "hihi", "gg", "bgtt", "bgt", "banget", "sangat",
    "sekali", "paling", "amat", "pokoknya", "intinya", "jujur", "menurut",
    "kayaknya", "kayanya", "sepertinya", "mungkin", "memang", "sih",
    "silol", "jogja", "yogyakarta", "kota", "tempat", "tempatnya",
    "kesini", "kesana", "kemari", "disini", "disana", "sini", "sana",
    "datang", "kesini", "mampir", "buka", "tutup", "jam", "hari", "kali",
    "tanggal", "minggu", "bulan", "tahun", "malam", "malem", "siang",
    "pagi", "sore", "tadi", "nanti", "besok", "kemarin",
    "orang", "pengunjung", "anak", "teman", "keluarga", "saya", "kami",
    "kita", "anda", "dia", "mereka", "aku", "kamu",
    "satu", "dua", "tiga", "empat", "lima", "banyak", "sedikit", "semua",
    "lain", "lagi", "baru", "lama", "sebentar", "terus", "lalu",
    "jadi", "karena", "tapi", "namun", "dengan", "untuk", "pada",
    "dari", "yang", "ini", "itu", "dan", "atau", "oleh", "dalam",
    "atas", "bawah", "juga", "masih", "sudah", "belum", "akan", "bisa",
    "harus", "mau", "ingin", "pengen", "cuma", "hanya", "saja", "aja",
    "the", "and", "for", "with", "you", "this", "that", "they", "are",
    "was", "not", "but", "have", "from", "your", "can", "all", "here",
    "when", "also", "their", "there", "just", "what", "about", "would",
    "could", "should", "been", "being", "has", "had", "its", "our",
    "out", "too", "very", "will", "more", "most", "some", "any", "who",
    "how", "why", "where", "which", "him", "her", "them", "his", "hers",
    "were", "did", "does", "do", "done", "got", "get", "going", "gonna",
    "ok", "okay", "nice", "good", "great", "best", "bad", "worse", "worst",
    "place", "time", "hours", "open", "close", "come", "came", "look",
    "love", "like", "want", "need", "know", "think", "make", "made",
    "take", "took", "give", "gave", "says", "said", "see", "seen",
    "say", "people", "guys", "friend", "friends", "food", "drink",
    "price", "service", "taste", "order", "room", "area", "spot",
    "hang", "smoking", "non", "level", "floor", "really", "very",
}

_REPEAT_CHAR = re.compile(r"(.)\1{2,}")
_REPEAT_WORD = re.compile(r"\b(\w{3,})(?:\s+\1\b)+")
_WORD_RUN = re.compile(r"[a-z]+")


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Stopword gabungan NLTK Indonesia + daftar tambahan + cache JSON."""
    cached = io_utils.load_json(config.STOPWORDS_JSON, default=None)
    if cached:
        return set(cached)

    words: set[str] = set(EXTRA_STOPWORDS)
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words |= set(stopwords.words("indonesian"))
        except LookupError:  # pragma: no cover - unduh sekali jika belum ada
            nltk.download("stopwords", quiet=True)
            words |= set(stopwords.words("indonesian"))
    except Exception:  # pragma: no cover
        pass

    io_utils.save_json(sorted(words), config.STOPWORDS_JSON)
    return words


def normalize_token(token: str) -> str:
    """Normalisasi satu token: huruf kecil, buang pengulangan, petakan slang."""
    token = token.lower().strip()
    if not token:
        return ""
    if token in SLANG_MAP:
        return SLANG_MAP[token]
    token = _REPEAT_CHAR.sub(r"\1", token)
    if token in SLANG_MAP:
        return SLANG_MAP[token]
    if token.endswith("nya") and len(token) > 6 and token[:-3] in SLANG_MAP:
        return SLANG_MAP[token[:-3]]
    return token


def normalize_text(text: str) -> str:
    """Normalisasi seluruh teks per token."""
    tokens = [normalize_token(t) for t in str(text).split()]
    return " ".join(t for t in tokens if t)


def normalize_text_keep_punctuation(text: str) -> str:
    """Normalisasi kata sambil mempertahankan tanda baca.

    Dipakai untuk analisis sentimen berbasis klausa: tanda baca dan kata
    penghubung seperti "tapi" adalah penanda batas antar-aspek, sehingga
    tidak boleh dihapus.
    """
    lowered = str(text).lower()
    replaced = _WORD_RUN.sub(lambda m: normalize_token(m.group(0)) or m.group(0), lowered)
    return dedupe_repeated_words(replaced)


def dedupe_repeated_words(text: str) -> str:
    """Menghapus pengulangan kata berturut-turut ('enak enak enak' -> 'enak')."""
    return _REPEAT_WORD.sub(r"\1", text)


def export_slang_map() -> dict:
    """Kamus normalisasi sebagai dict (untuk ditampilkan di notebook)."""
    return dict(sorted(SLANG_MAP.items()))


def export_slang_frame() -> "list[dict]":
    return [{"Kata Asli": k, "Bentuk Baku": v} for k, v in sorted(SLANG_MAP.items())]