"""Pemuatan lexicon InSet (positive/negative) dengan resolusi konflik."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from . import config, normalizer

INSET_POS_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
INSET_NEG_URL = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"


def download_inset(force: bool = False) -> tuple[Path, Path]:
    """Mengunduh lexicon InSet sekali ke ``data/lexicon``."""
    targets = [(INSET_POS_URL, config.INSET_POS_TSV), (INSET_NEG_URL, config.INSET_NEG_TSV)]
    for url, path in targets:
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
        print(f"[lexicon] diunduh: {path.name}")
    return config.INSET_POS_TSV, config.INSET_NEG_TSV


def _read_tsv(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word = parts[0].strip().lower()
        try:
            weight = float(parts[1])
        except ValueError:
            continue
        if word:
            result[word] = weight
    return result


def load_raw_lexicon(force_download: bool = False) -> tuple[dict, dict]:
    download_inset(force=force_download)
    pos = _read_tsv(config.INSET_POS_TSV)
    neg = _read_tsv(config.INSET_NEG_TSV)
    return pos, neg


def build_lexicon(
    *,
    include_multiword: bool = False,
    min_length: int = 4,
    drop_stopwords: bool = True,
    resolve: str = "absmax",
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Membangun lexicon bersih.

    Parameters
    ----------
    include_multiword : bool
        Ikutkan frasa (contoh "tidak segan"). Default False karena pencocokan
        dilakukan per token.
    min_length : int
        Panjang kata minimum yang dipertahankan.
    drop_stopwords : bool
        Buang kata fungsi agar tidak memberi polaritas palsu.
    resolve : {"absmax", "net", "skip"}
        Cara menyelesaikan kata yang muncul di kedua lexicon. InSet memuat
        kata seperti "enak" di positif (5) dan negatif (-4); default memakai
        bobot absolut terbesar.
    """
    pos, neg = load_raw_lexicon()
    sw = normalizer.get_stopwords() if drop_stopwords else set()

    lexicon: dict[str, float] = {}
    for word in set(pos) | set(neg):
        if (not include_multiword) and (" " in word or "-" in word):
            continue
        if len(word) < min_length or word in sw:
            continue

        p = abs(pos.get(word, 0.0))
        n = abs(neg.get(word, 0.0))

        if word in pos and word in neg:
            if resolve == "skip":
                continue
            if resolve == "net":
                value = pos[word] + neg[word]
            else:  # absmax
                value = p if p >= n else -n
        elif word in pos:
            value = p
        else:
            value = -n

        if value != 0:
            lexicon[word] = value

    if overrides:
        lexicon.update(overrides)
    return lexicon


# Kata yang sering salah dipolarisasi oleh InSet pada domain ulasan kuliner.
DOMAIN_OVERRIDES: dict[str, float] = {
    "enak": 4, "ena": 3, "nikmat": 4, "lezat": 4, "gurih": 3, "mantap": 4,
    "mahal": -3, "kemahalan": -3, "overprice": -3, "pelit": -3,
    "murah": 2, "terjangkau": 3, "worth": 3, "sesuai": 2,
    "lama": -3, "lambat": -3, "lemot": -3, "antri": -2, "nunggu": -2,
    "bersih": 3, "kotor": -4, "jorok": -4, "bau": -3, "debu": -2,
    "ramah": 4, "kasar": -3, "cuek": -3, "jutek": -3, "sigap": 3,
    "nyaman": 3, "asik": 3, "asik": 3, "betah": 3, "gaduh": -3, "berisik": -3,
    "sepi": -2, "ramai": 1, "luas": 2, "sesak": -3, "panas": -2, "dingin": -1,
    "asin": -2, "hambar": -3, "pahit": -2, "basi": -4, "keras": -2, "lembek": -2,
    "kecewa": -4, "mengecewakan": -4, "puas": 3, "suka": 3,
    "bagus": 3, "oke": 2, "ok": 2, "keren": 3, "seru": 3,
    "komplain": -2, "keluh": -2, "salah": -2, "tipu": -4, "nyebelin": -4,
    "standar": -1, "biasa": -1, "lumayan": 1,
}


def build_domain_lexicon(**kwargs) -> dict[str, float]:
    """Lexicon siap pakai: InSet bersih + override domain kuliner."""
    overrides = dict(DOMAIN_OVERRIDES)
    overrides.update(kwargs.pop("overrides", {}) or {})
    return build_lexicon(overrides=overrides, **kwargs)


def lexicon_frame(lexicon: dict[str, float]) -> pd.DataFrame:
    """Dataframe kata -> skor, diurutkan dari skor tertinggi."""
    df = pd.DataFrame(
        {"Kata": list(lexicon.keys()), "Skor": list(lexicon.values())}
    )
    df["Polaritas"] = df["Skor"].map(lambda v: "Positif" if v > 0 else "Negatif")
    return df.sort_values(["Skor", "Kata"], ascending=[False, True]).reset_index(drop=True)


def lexicon_summary(lexicon: dict[str, float]) -> pd.DataFrame:
    values = pd.Series(list(lexicon.values()))
    return pd.DataFrame(
        {
            "Metrik": [
                "Jumlah kata", "Kata positif", "Kata negatif",
                "Skor maksimum", "Skor minimum", "Rata-rata skor",
            ],
            "Nilai": [
                len(lexicon), int((values > 0).sum()), int((values < 0).sum()),
                float(values.max()), float(values.min()), round(float(values.mean()), 3),
            ],
        }
    )