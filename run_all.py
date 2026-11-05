"""Menjalankan seluruh notebook secara berurutan (headless).

Contoh:
    python run_all.py              # jalankan semua notebook
    python run_all.py 1 2 3        # hanya notebook 01, 02, 03
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"

NOTEBOOKS = [
    "01_data_understanding.ipynb",
    "02_preprocessing.ipynb",
    "03_sentimen_ulasan.ipynb",
    "04_sentimen_aspek.ipynb",
    "05_clustering_kmeans.ipynb",
    "06_interpretasi_rekomendasi.ipynb",
]


def run(names: list[str]) -> int:
    failures = []
    for name in names:
        path = NB_DIR / name
        if not path.exists():
            print(f"[lewati] {name} tidak ditemukan")
            continue
        print(f"\n{'=' * 70}\n[run] {name}\n{'=' * 70}")
        started = time.time()
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            "--ExecutePreprocessor.kernel_name=python3",
            str(path),
        ]
        result = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.time() - started
        status = "OK" if result.returncode == 0 else "GAGAL"
        print(f"[{status}] {name} ({elapsed:.1f} detik)")
        if result.returncode != 0:
            failures.append(name)

    print("\n" + "=" * 70)
    if failures:
        print("Notebook gagal:", ", ".join(failures))
        return 1
    print("Semua notebook selesai tanpa error.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        selected = [n for n in NOTEBOOKS if any(n.startswith(a) for a in args)]
    else:
        selected = NOTEBOOKS
    sys.exit(run(selected))