# Analisis Sentimen dan Clustering Ulasan Pelanggan
## Silol Kopi & Eatery Yogyakarta

Proyek data mining untuk mengolah ulasan pelanggan Google Maps menjadi informasi
yang dapat dipakai mengambil keputusan: **apa yang sudah baik** dan **apa yang
perlu diperbaiki** pada layanan Silol Kopi & Eatery.

---

## Ringkasan pendekatan

| Lapis | Metode | Keluaran |
|---|---|---|
| 1 | TF-IDF + K-Means | tema ulasan otomatis (cluster) |
| 2 | Lexicon InSet + deteksi aspek otomatis | sentimen per aspek layanan |
| 3 | Matriks prioritas | urutan aspek yang perlu diperbaiki |

**Sentimen diambil dari isi teks ulasan, bukan dari rating.** Rating hanya
dipakai sebagai pembanding dan untuk menemukan *hidden complaint* — ulasan
4–5 bintang yang isinya justru keluhan. Alasannya:

> "Suasana ok, makanan tidak terlalu ok, pelayanan kurang memuaskan" — 5 bintang

Bila label diambil dari rating, dua keluhan pada ulasan di atas hilang. Bila
label diambil per ulasan, keluhan itu masih bisa tertutup. Karena itu sentimen
dihitung **per aspek per klausa**.

---

## Struktur proyek

```
tugas DataMining/
├── data/
│   ├── raw/            dataset asli (xlsx)
│   ├── lexicon/        kamus InSet positif & negatif (diunduh sekali)
│   ├── cache/          cache stemming
│   └── processed/      artefak antar-notebook (csv, npz, pkl)
├── src/                seluruh logika (notebook hanya memanggil)
│   ├── config.py       path, konstanta, hyperparameter
│   ├── io_utils.py     baca/tulis artefak
│   ├── audit.py        profil & pembersihan awal dataset
│   ├── normalizer.py   kamus normalisasi + stopword
│   ├── preprocessing.py 8 tahap preprocessing
│   ├── lexicon.py      pemuatan InSet + override domain
│   ├── sentiment.py    skoring sentimen level klausa & dokumen
│   ├── aspects.py      ekstraksi aspek otomatis + sentimen per aspek
│   ├── features.py     TF-IDF
│   ├── clustering.py   K-Means + penentuan k
│   ├── interpret.py    profil cluster, matriks prioritas, rekomendasi
│   ├── evaluate.py     metrik evaluasi
│   └── visualize.py    seluruh figure
├── notebooks/          01-06 (naratif, siap dilampirkan)
├── results/
│   ├── figures/        grafik 300 dpi
│   └── tables/         tabel csv siap tempel ke laporan
├── tools/
│   ├── build_notebooks.py   generator notebook
│   └── run_notebooks.py     eksekusi notebook + cek error
├── tests/smoke.py      smoke test seluruh modul
├── run_all.py          jalankan seluruh notebook
└── requirements.txt
```

---

## Cara menjalankan

### 1. Instalasi

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

### 2. Jalankan seluruh pipeline

```bash
python run_all.py
```

Atau notebook satu per satu sesuai urutan:

| Notebook | Isi | Perkiraan waktu |
|---|---|---|
| `01_data_understanding.ipynb` | profil dataset, pembersihan awal | ±10 detik |
| `02_preprocessing.ipynb` | 8 tahap preprocessing (run pertama ±5 menit karena stemming) | ±10 detik (run berikutnya) |
| `03_sentimen_ulasan.ipynb` | sentimen teks vs rating | ±10 detik |
| `04_sentimen_aspek.ipynb` | ekstraksi aspek + sentimen per aspek | ±40 detik |
| `05_clustering_kmeans.ipynb` | TF-IDF + K-Means | ±40 detik |
| `06_interpretasi_rekomendasi.ipynb` | kesimpulan & rekomendasi | ±10 detik |

Stemming di-cache ke `data/cache/stem_cache.json` sehingga eksekusi ulang
hampir instan.

### 3. Pemeriksaan cepat

```bash
python -m tests.smoke          # uji seluruh modul src
python tools/run_notebooks.py 04 05   # eksekusi & cek error notebook tertentu
```

---

## Konfigurasi penting

Semua parameter ada di `src/config.py`:

| Parameter | Default | Keterangan |
|---|---|---|
| `ASPECT_MIN_FREQ` | 100 | ambang frekuensi kata kunci aspek. Turunkan (mis. 50) untuk lebih banyak aspek, naikkan (mis. 200) untuk lebih ringkas. |
| `SENTIMENT_WINDOW` | 5 | jumlah token di sekitar kata aspek yang ikut dihitung |
| `TFIDF_MIN_DF` / `MAX_DF` | 2 / 0.95 | penyaringan term TF-IDF |
| `K_RANGE` | 2..10 | rentang k yang diuji |
| `RANDOM_STATE` | 42 | reproducibility |

Ambang aspek juga bisa diubah langsung di notebook 04 (`MIN_FREQ`), dan
notebook tersebut menampilkan perbandingan hasil pada ambang 50/100/150/200.

---

## Catatan metodologis

**1. Jumlah data.** Dataset mentah berisi 15.446 baris, tetapi hanya 7.267 yang
memiliki teks ulasan; setelah membuang duplikat dan ulasan terlalu pendek
diperoleh **6.795 ulasan** yang dianalisis. Sisanya adalah rating tanpa komentar
yang tidak dapat diproses dengan TF-IDF.

**2. Aspek ditentukan otomatis.** Kata kunci aspek diambil dari frekuensi kata
yang lolos penyaringan (bukan stopword, bukan kata sentimen, bukan kata umum,
bukan kata Inggris), lalu dikelompokkan dengan *hierarchical clustering*
berdasarkan co-occurrence. Kamus domain hanya dipakai untuk **memberi nama**
kelompok, bukan untuk menentukan kelompoknya. Verifikasi sensitivitas ambang
ditampilkan di notebook 04.

**3. Silhouette rendah itu wajar.** Nilai Silhouette pada data ini berkisar
0,02 karena ulasan sangat pendek (median 52 karakter) dan berbagi kosakata yang
mirip. Nilai ini dilaporkan apa adanya, dan pemilihan k memakai aturan
tambahan: kandidat dengan Silhouette setara dipilih yang paling dekat dengan
titik siku Elbow, serta cluster yang terlalu kecil (<3% data) dieliminasi.

**4. Keterbatasan.** Label sentimen berbasis lexicon belum memahami sarkasme
dan konteks yang sangat spesifik. Kutipan ulasan asli selalu disertakan sebagai
bukti agar hasil dapat diverifikasi manusia.

---

## Keluaran utama

| File | Isi |
|---|---|
| `results/tables/04_aspect_summary.csv` | volume & polaritas setiap aspek |
| `results/tables/04_matriks_prioritas.csv` | peringkat prioritas perbaikan |
| `results/tables/06_rekomendasi.csv` | rekomendasi tindakan per aspek |
| `results/tables/03_metrik_sentimen.csv` | akurasi sentimen teks vs rating |
| `results/tables/05_metrik_cluster.csv` | Silhouette, Davies-Bouldin, alasan pemilihan k |
| `results/figures/*.png` | seluruh grafik untuk laporan |