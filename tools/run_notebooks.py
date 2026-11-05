"""Menjalankan notebook secara headless dan memeriksa error di dalamnya.

Berbeda dari `run_all.py`, skrip ini tidak bergantung pada exit code nbconvert
(PowerShell menganggap output stderr sebagai error) melainkan memindai sel
notebook untuk output bertipe ``error``.

Contoh:
    python tools/run_notebooks.py 03 04
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

NOTEBOOKS = [
    "01_data_understanding.ipynb",
    "02_preprocessing.ipynb",
    "03_sentimen_ulasan.ipynb",
    "04_sentimen_aspek.ipynb",
    "05_clustering_kmeans.ipynb",
    "06_interpretasi_rekomendasi.ipynb",
]


def scan_errors(path: Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(f"cell {i}: {out.get('ename')}: {out.get('evalue')}")
    return errors


def run_one(name: str, timeout: int) -> tuple[bool, float, list[str]]:
    path = NB_DIR / name
    print(f"\n{'=' * 70}\n[run] {name}\n{'=' * 70}", flush=True)
    started = time.time()
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    errors = scan_errors(path)
    print(f"[{'OK' if not errors else 'GAGAL'}] {name} ({elapsed:.1f} s)", flush=True)
    for e in errors:
        print("   ", e, flush=True)
    return (not errors), elapsed, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="*", help="awalan nama notebook, mis. 03")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    targets = NOTEBOOKS
    if args.prefixes:
        targets = [n for n in NOTEBOOKS if any(n.startswith(p) for p in args.prefixes)]

    failed = []
    for name in targets:
        ok, _, errors = run_one(name, args.timeout)
        if not ok:
            failed.append((name, errors))

    print("\n" + "=" * 70)
    if failed:
        print("NOTEBOOK GAGAL:")
        for name, errors in failed:
            print(f"  - {name}")
            for e in errors:
                print(f"      {e}")
        return 1
    print(f"Semua {len(targets)} notebook berhasil dieksekusi tanpa error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())