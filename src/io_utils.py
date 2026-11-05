"""Utilitas baca/tulis artefak antar-notebook."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pandas as pd
from scipy import sparse

from . import config


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"gagal membaca {path}")


# --------------------------------------------------------------------------
# Numpy / pickle
# --------------------------------------------------------------------------
def save_sparse(matrix: sparse.spmatrix, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, matrix)
    return path


def load_sparse(path: Path) -> sparse.spmatrix:
    return sparse.load_npz(path)


def save_pickle(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# JSON (cache stemming)
# --------------------------------------------------------------------------
def load_json(path: Path, default=None):
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)
    return path


# --------------------------------------------------------------------------
# Dataset mentah
# --------------------------------------------------------------------------
def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Membaca dataset mentah dan menormalkan nama kolom."""
    path = Path(path or config.RAW_XLSX)
    if not path.exists():
        raise FileNotFoundError(f"dataset tidak ditemukan: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def xlsx_sheet_rows(path: Path, sheet: str = "xl/worksheets/sheet1.xml") -> int:
    """Menghitung baris worksheet tanpa membuka workbook (opsional)."""
    with zipfile.ZipFile(path) as z:
        return z.getinfo(sheet).file_size


def save_figure(fig, name: str, dpi: int = config.FIG_DPI) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path