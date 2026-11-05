"""Generator notebook: satu sumber kebenaran untuk isi notebook.

Jalankan ulang jika ingin meregenerasi notebook:

    python tools/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

BOOTSTRAP = """\
import sys
from pathlib import Path

# Menemukan root proyek agar `import src` berjalan dari mana pun notebook dibuka.
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import config

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

print("Root proyek:", ROOT)
"""


def nb(cells):
    """Membuat notebook dari daftar (tipe, isi)."""
    notebook = nbf.v4.new_notebook()
    out = []
    for kind, body in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(body.strip("\n")))
        else:
            out.append(nbf.v4.new_code_cell(body.strip("\n")))
    notebook["cells"] = out
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    return notebook


def save(notebook, name: str):
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print(f"[notebook] {path.relative_to(ROOT)}  ({len(notebook['cells'])} sel)")


# ==========================================================================
# 01 - Data Understanding
# ==========================================================================
nb01 = nb([
    ("md", """
# 01 - Identifikasi dan Pemeriksaan Dataset

**Proyek:** Analisis Sentimen dan Clustering Ulasan Pelanggan Silol Kopi & Eatery Yogyakarta

Notebook ini menjalankan **Tahap 1** metodologi: memeriksa dataset hasil scraping
untuk mengetahui jumlah data, kelengkapan atribut, keberadaan ulasan kosong,
duplikat, serta sebaran rating.

> Catatan penting: dokumen metodologi menyebut "sekitar 15.000+ ulasan", tetapi
> tidak semua baris berisi teks ulasan. Google Maps tetap mencatat rating tanpa
> komentar, sehingga baris seperti itu tidak dapat dipakai untuk clustering
> berbasis teks. Notebook ini mengukur selisih tersebut secara eksplisit.
"""),
    ("code", BOOTSTRAP),
    ("md", """
## 1.1 Memuat dataset mentah

Dataset dibaca dari `data/raw/silol_kopi_eatery_reviews.xlsx`.
"""),
    ("code", """
from src import io_utils

raw = io_utils.load_raw()
print(f"Ukuran dataset: {raw.shape[0]:,} baris x {raw.shape[1]} kolom")
raw.head(3)
"""),
    ("md", """
## 1.2 Profil atribut

Tabel berikut merangkum tipe data, jumlah nilai kosong, dan contoh nilai untuk
setiap atribut. Ini menjadi dasar **Tabel 2.1** pada laporan.
"""),
    ("code", """
from src import audit

overview = audit.dataset_overview(raw)
io_utils.save_csv(overview, config.TABLES_DIR / "01_dataset_overview.csv")
overview
"""),
    ("md", """
## 1.3 Kelengkapan data teks ulasan

Pemeriksaan utama: berapa banyak baris yang benar-benar memiliki teks ulasan.
"""),
    ("code", """
completeness = audit.completeness_summary(raw)
io_utils.save_csv(completeness, config.TABLES_DIR / "01_completeness_summary.csv")
completeness
"""),
    ("md", """
## 1.4 Sebaran rating

Rating 1-5 bintang dihitung untuk melihat kecenderungan penilaian pelanggan.
"""),
    ("code", """
rating_dist = audit.rating_distribution(raw)
io_utils.save_csv(rating_dist, config.TABLES_DIR / "01_rating_distribution.csv")
rating_dist
"""),
    ("code", """
from src import visualize

visualize.plot_rating_distribution(rating_dist)
"""),
    ("md", """
## 1.5 Ketersediaan teks per rating

Apakah rating rendah lebih sering disertai komentar? Perbandingan ini penting
karena menentukan seberapa besar data yang hilang bila hanya memakai ulasan berteks.
"""),
    ("code", """
rating_vs_text = audit.rating_vs_text_crosstab(raw)
io_utils.save_csv(rating_vs_text, config.TABLES_DIR / "01_rating_vs_text.csv")
rating_vs_text
"""),
    ("code", """
visualize.plot_rating_vs_text(rating_vs_text)
"""),
    ("md", """
## 1.6 Panjang ulasan

Ulasan yang sangat pendek (mis. "ok", "👍") tidak informatif untuk clustering.
"""),
    ("code", """
length_stats = audit.review_length_stats(raw)
io_utils.save_csv(length_stats, config.TABLES_DIR / "01_review_length_stats.csv")
length_stats
"""),
    ("code", """
review_text = raw["Review"].astype("string").fillna("").str.strip()
visualize.plot_review_length(review_text[review_text.ne("")].str.len())
"""),
    ("md", """
## 1.7 Tren ulasan dari waktu ke waktu

Kolom `Timestamp` menyimpan waktu unggah sehingga bisa dipakai untuk analisis tren.
"""),
    ("code", """
trend = audit.yearly_trend(raw)
io_utils.save_csv(trend, config.TABLES_DIR / "01_yearly_trend.csv")
trend
"""),
    ("code", """
visualize.plot_yearly_trend(trend)
"""),
    ("md", """
## 1.8 Sebaran bahasa

Kolom `Language` pada hasil scraping menandai bahasa ulasan. Sebagian besar
ulasan berbahasa Indonesia, sisanya Inggris dan bahasa lain.
"""),
    ("code", """
lang_dist = audit.language_distribution(raw)
io_utils.save_csv(lang_dist, config.TABLES_DIR / "01_language_distribution.csv")
lang_dist.head(10)
"""),
    ("md", """
## 1.9 Pembersihan awal dataset

Tiga aturan diterapkan berurutan:

1. Buang baris tanpa teks ulasan (hanya rating).
2. Buang duplikat teks (setelah dinormalisasi spasi dan huruf kecil).
3. Buang ulasan yang terlalu pendek (< 3 karakter).
"""),
    ("code", """
clean, recap = audit.build_clean_dataset(raw)
io_utils.save_csv(recap, config.TABLES_DIR / "01_cleaning_recap.csv")
io_utils.save_csv(clean, config.CLEAN_CSV)
recap
"""),
    ("code", """
visualize.plot_completeness(recap)
"""),
    ("md", """
## 1.10 Kesimpulan Tahap 1

Ringkasan otomatis angka kunci:
"""),
    ("code", """
ringkasan = pd.DataFrame(
    {
        "Indikator": [
            "Total baris dataset",
            "Baris dengan teks ulasan",
            "Baris tanpa teks (hanya rating)",
            "Duplikat teks",
            "Ulasan siap analisis",
            "Rating rata-rata (data bersih)",
            "Rentang tanggal",
            "Jumlah pengulas unik",
        ],
        "Nilai": [
            f"{len(raw):,}",
            f"{int(raw['Review'].notna().sum()):,}",
            f"{int(raw['Review'].isna().sum()):,}",
            f"{int(raw['Review'].duplicated().sum()):,}",
            f"{len(clean):,}",
            f"{clean['Rating'].mean():.3f}",
            f"{pd.to_datetime(raw['Timestamp'], errors='coerce', utc=True).min():%Y-%m-%d} s.d. "
            f"{pd.to_datetime(raw['Timestamp'], errors='coerce', utc=True).max():%Y-%m-%d}",
            f"{clean['Author'].nunique():,}",
        ],
    }
)
io_utils.save_csv(ringkasan, config.TABLES_DIR / "01_ringkasan_tahap1.csv")
ringkasan
"""),
    ("md", """
### Catatan untuk laporan

- Dataset mentah berisi ribuan baris, namun **hanya sebagian yang memiliki teks**
  sehingga jumlah ulasan yang benar-benar dianalisis lebih kecil.
- Selisih ini bukan kesalahan scraping, melainkan karakteristik data Google Maps
  yang mengizinkan rating tanpa komentar.
- Data bersih disimpan di `data/processed/01_reviews_clean.csv` dan dipakai
  notebook berikutnya.
"""),
])

# ==========================================================================
# 02 - Preprocessing
# ==========================================================================
nb02 = nb([
    ("md", """
# 02 - Preprocessing Teks

**Tahap 2** metodologi: membersihkan teks ulasan sebelum dianalisis.

Tahapan yang dijalankan berurutan:

1. Case folding (huruf kecil)
2. Penghapusan URL, tag HTML (`<br>`), mention, hashtag
3. Penghapusan emoji dan angka
4. Penghapusan tanda baca
5. Normalisasi kata tidak baku dan singkatan
6. Tokenizing
7. Stopword removal
8. Stemming dengan Sastrawi

Seluruh logika berada di `src/preprocessing.py`; notebook hanya memanggil dan
menampilkan hasil agar mudah diaudit.
"""),
    ("code", BOOTSTRAP),
    ("md", """
## 2.1 Memuat data bersih dari Tahap 1
"""),
    ("code", """
from src import io_utils, preprocessing, normalizer

clean = io_utils.load_csv(config.CLEAN_CSV)
print(f"Ulasan yang akan diproses: {len(clean):,}")
clean[["review_id", "Rating", "Review"]].head(3)
"""),
    ("md", """
## 2.2 Kamus normalisasi kata tidak baku

Ulasan Google Maps dipenuhi singkatan dan typo (`gak`, `bgt`, `pelayanannya`,
`enak2`). Normalisasi menyatukan bentuk-bentuk ini agar tidak terpecah menjadi
fitur berbeda saat TF-IDF.

Kamus lengkap: `src/normalizer.py` (`SLANG_MAP`).
"""),
    ("code", """
slang = pd.DataFrame(normalizer.export_slang_frame())
print(f"Jumlah entri kamus normalisasi: {len(slang):,}")
slang.sample(15, random_state=config.RANDOM_STATE)
"""),
    ("md", """
## 2.3 Menjalankan pipeline preprocessing

Proses stemming di-cache ke `data/cache/stem_cache.json`. Pada eksekusi pertama
proses memakan beberapa menit; eksekusi ulang menjadi jauh lebih cepat karena
setiap kata hanya di-stem sekali.
"""),
    ("code", """
proc, stage_stats, extras = preprocessing.run_preprocessing(use_stemming=True)
"""),
    ("md", """
## 2.4 Statistik per tahap

Tabel ini menjadi bukti kuantitatif proses preprocessing di laporan: berapa token
yang hilang pada setiap tahap.
"""),
    ("code", """
stage_stats
"""),
    ("code", """
from src import visualize

visualize.plot_preprocessing_stages(stage_stats)
"""),
    ("md", """
## 2.5 Contoh hasil tiap tahap

Lima ulasan pertama ditampilkan utuh dari teks mentah sampai teks siap analisis.
"""),
    ("code", """
examples = extras["examples"]
examples[["Rating", "Mentah", "Setelah Cleaning", "Setelah Normalisasi", "Setelah Stemming"]]
"""),
    ("md", """
## 2.6 Kosakata akhir

Kata yang paling sering muncul setelah seluruh tahap preprocessing.
"""),
    ("code", """
vocab = extras["vocab_counts"]
vocab.head(30).to_frame("Frekuensi")
"""),
    ("code", """
visualize.plot_top_words(vocab, n=30, title="30 Kata Paling Sering Muncul Setelah Preprocessing")
"""),
    ("code", """
visualize.plot_wordcloud(vocab)
"""),
    ("md", """
## 2.7 Pemeriksaan kualitas

Beberapa pemeriksaan untuk memastikan preprocessing tidak merusak data:
"""),
    ("code", """
cek = pd.DataFrame(
    {
        "Pemeriksaan": [
            "Dokumen kosong setelah stopword",
            "Dokumen kosong setelah stemming",
            "Rata-rata token sebelum stopword",
            "Rata-rata token setelah stemming",
            "Token unik akhir (kosakata)",
            "Entri cache stemming",
        ],
        "Nilai": [
            int((proc["tokens"].str.split().str.len() == 0).sum()),
            int((proc["n_tokens_after"] == 0).sum()),
            round(float(proc["n_tokens_before"].mean()), 2),
            round(float(proc["n_tokens_after"].mean()), 2),
            len(vocab),
            extras["cache_size"],
        ],
    }
)
io_utils.save_csv(cek, config.TABLES_DIR / "02_pemeriksaan_kualitas.csv")
cek
"""),
    ("md", """
## 2.9 Catatan implementasi

- Cache stemming (`data/cache/stem_cache.json`) menyimpan pemetaan
  kata → kata dasar sehingga eksekusi ulang notebook tidak mengulang proses
  yang paling mahal.
- Kolom `text_clause` menyimpan teks yang sudah dinormalisasi **tetapi masih
  mempertahankan tanda baca**. Kolom ini dipakai pada analisis sentimen per
  aspek, karena tanda baca dan kata seperti *tapi* / *namun* adalah penanda
  batas antar-aspek.
- Kolom `tokens_stemmed` dipakai untuk TF-IDF dan ekstraksi kata kunci aspek.

## 2.10 Kesimpulan Tahap 2

- Teks ulasan telah melalui 8 tahap preprocessing dan tersimpan pada kolom
  `tokens_stemmed`.
- Dokumen tanpa token setelah stemming dibiarkan di dataset (tidak dibuang) agar
  jumlah ulasan tetap konsisten; dokumen ini tidak menyumbang fitur TF-IDF.
- Artefak yang dihasilkan:
  - `data/processed/02_reviews_preprocessed.csv`
  - `results/tables/02_preprocessing_stages.csv`
  - `results/tables/02_preprocessing_examples.csv`
  - `data/cache/stem_cache.json`
"""),
])


# ==========================================================================
# 03 - Sentimen dokumen vs rating
# ==========================================================================
nb03 = nb([
    ("md", """
# 03 - Analisis Sentimen Tingkat Ulasan

**Tahap 5 (bagian pertama).** Setiap ulasan diberi polaritas berdasarkan
**isi teksnya**, lalu hasilnya **dibandingkan dengan rating** untuk melihat
seberapa jauh teks mencerminkan rating.

## Mengapa label diambil dari teks, bukan rating?

Rating tidak selalu menggambarkan isi ulasan. Pada dataset ini ditemukan:

| Rating | Isi ulasan | Keluhan yang tersembunyi |
|---|---|---|
| 4 bintang | "Live musik tiap hari. Bumbu kurang meresap" | cita rasa |
| 5 bintang | "Menu rawon, asin. Mohon diperhatikan" | cita rasa |
| 5 bintang | "Suasana ok, makanan tidak terlalu ok, pelayanan kurang memuaskan" | makanan & pelayanan |
| 5 bintang | "Pesen nachos lebih dari 45 menit baru disajikan" | kecepatan |
| 1 bintang | "Pelayanan cukup baik... soal makanan ndak worthed" | pujian pelayanan hilang |

Bila sentimen diambil dari rating, seluruh keluhan pada ulasan 4-5 bintang
otomatis hilang. Karena itu:

- **Label sentimen = hasil analisis teks** (metode lexicon InSet).
- **Rating = acuan pembanding** untuk mengukur kesesuaian dan menemukan
  ulasan yang isinya bertentangan dengan ratingnya (*hidden complaint*).
"""),
    ("code", BOOTSTRAP),
    ("md", """
## 3.1 Memuat data hasil preprocessing
"""),
    ("code", """
from src import io_utils, lexicon, sentiment

df = io_utils.load_csv(config.PREP_CSV)
print(f"Ulasan: {len(df):,}")
df[["review_id", "Rating", "text_normalized", "tokens_stemmed"]].head(3)
"""),
    ("md", """
## 3.2 Menyiapkan lexicon InSet

Lexicon InSet berisi daftar kata positif dan negatif beserta bobotnya.
Karena banyak kata muncul di **kedua** file (mis. `enak` ada di positif +5 dan
negatif -4), resolusi konflik memakai bobot absolut terbesar. Kata fungsi dan
frasa multi-kata dibuang agar tidak menimbulkan polaritas palsu.
"""),
    ("code", """
lex = lexicon.build_domain_lexicon()
io_utils.save_csv(lexicon.lexicon_frame(lex), config.TABLES_DIR / "03_lexicon_digunakan.csv")
lexicon.lexicon_summary(lex)
"""),
    ("code", """
lexicon.lexicon_frame(lex).head(15)
"""),
    ("code", """
pd.DataFrame(
    [{"Kata": k, "Skor": lex[k]} for k in
     ["enak", "nyaman", "ramah", "bersih", "murah", "mahal", "kotor",
      "lama", "lambat", "kecewa", "bau", "worth"] if k in lex]
)
"""),
    ("md", """
## 3.3 Skoring sentimen

Aturan skoring:

- Ulasan dipecah menjadi klausa (`.` `,` `;` `tapi` `namun` `sayangnya` dst.).
- Skor tiap klausa = jumlah bobot kata sentimen.
- **Negasi** (`tidak`, `kurang`, `gak`, `bukan`, ...) membalik polaritas kata
  hingga dua token di belakangnya.
- **Penguat** (`sangat`, `banget`, `sekali`) mengalikan bobot.
- Skor dokumen = total skor seluruh klausa.

Label: skor > 0 → Positif, skor < 0 → Negatif, skor = 0 → Netral.
"""),
    ("code", """
scored, tables = sentiment.run_document_sentiment(df, lex)
io_utils.save_csv(scored, config.DOC_SENTIMENT_CSV)

tables["sentiment_distribution"]
"""),
    ("code", """
from src import visualize

visualize.plot_sentiment_distribution(
    tables["sentiment_distribution"],
    "Distribusi Sentimen Ulasan (berdasarkan teks)",
    "03_distribusi_sentimen_teks",
)
"""),
    ("code", """
visualize.plot_score_distribution(scored["skor_sentimen"], "03_distribusi_skor")
"""),
    ("md", """
## 3.4 Perbandingan dengan rating

Inilah bagian yang diminta metodologi: menguji apakah sentimen hasil analisis
teks sesuai dengan rating yang diberikan pelanggan.
"""),
    ("code", """
tables["cross_rating"]
"""),
    ("code", """
tables["cross_rating_pct"]
"""),
    ("code", """
visualize.plot_cross_rating(
    tables["cross_rating"],
    "03_sentimen_vs_rating",
    "Sentimen Teks vs Rating Pelanggan",
)
"""),
    ("md", """
## 3.5 Metrik kesesuaian

Rating diperlakukan sebagai acuan (1-2 → Negatif, 3 → Netral, 4-5 → Positif).
"""),
    ("code", """
from src import evaluate

metrics = evaluate.classification_metrics(scored)
io_utils.save_csv(metrics, config.TABLES_DIR / "03_metrik_sentimen.csv")
metrics
"""),
    ("code", """
per_class = evaluate.per_class_frame(scored)
io_utils.save_csv(per_class, config.TABLES_DIR / "03_metrik_per_kelas.csv")
per_class
"""),
    ("code", """
cm = evaluate.confusion_frame(scored)
io_utils.save_csv(cm, config.TABLES_DIR / "03_confusion_matrix.csv")
visualize.plot_confusion(cm, "03_confusion_matrix")
cm
"""),
    ("md", """
## 3.6 Analisis gap: *hidden complaint* dan *hidden praise*

Bagian paling berguna untuk pengambilan keputusan: ulasan dengan rating tinggi
tetapi isinya keluhan, dan sebaliknya.
"""),
    ("code", """
gap = tables["gap"]
io_utils.save_csv(gap, config.TABLES_DIR / "03_analisis_gap.csv")
gap
"""),
    ("md", """
### Contoh *hidden complaint* (rating 4-5, teks negatif)

Ini adalah keluhan yang **tidak akan pernah terlihat** jika hanya memakai rating.
"""),
    ("code", """
hidden = scored[(scored["Rating"] >= 4) & (scored["sentimen_teks"] == "Negatif")]
hidden_cols = ["Rating", "Review", "skor_sentimen"]
hidden.sort_values("skor_sentimen")[hidden_cols].head(10)
"""),
    ("md", """
### Contoh *hidden praise* (rating 1-2, teks positif)
"""),
    ("code", """
praise = scored[(scored["Rating"] <= 2) & (scored["sentimen_teks"] == "Positif")]
praise.sort_values("skor_sentimen", ascending=False)[hidden_cols].head(10)
"""),
    ("md", """
## 3.7 Contoh ulasan paling ekstrem
"""),
    ("code", """
tables["extremes"][["Kategori", "Rating", "sentimen_teks", "skor_sentimen", "Review"]]
"""),
    ("md", """
## 3.8 Kesimpulan Tahap 3

- Sentimen dihitung dari **teks**, rating hanya pembanding.
- Tingkat kesesuaian teks vs rating dihitung pada tabel metrik di atas.
  Nilai yang tidak sempurna bukan kegagalan, melainkan temuan: rating bintang
  tidak selalu mencerminkan isi ulasan.
- Jumlah *hidden complaint* menjadi indikator bahwa analisis teks memberi
  informasi tambahan yang tidak tersedia dari rating.
- Hasil level ulasan ini belum cukup untuk menjawab "aspek mana yang perlu
  diperbaiki" karena satu ulasan bisa memuat pujian dan keluhan sekaligus.
  Tahap berikutnya memecah sentimen **per aspek layanan**.
"""),
])

# ==========================================================================
# 04 - Sentimen per aspek
# ==========================================================================
nb04 = nb([
    ("md", """
# 04 - Analisis Sentimen per Aspek Layanan

Notebook ini menjawab pertanyaan inti proyek:

> **Apa yang sudah baik di Silol Kopi & Eatery, dan apa yang masih kurang?**

Label sentimen tetap berasal dari **teks ulasan** (bukan rating), tetapi satu
ulasan sekarang boleh menghasilkan beberapa baris sentimen — satu baris untuk
setiap aspek yang disebut.

## Mengapa harus per aspek?

Ulasan berikut mendapat 5 bintang:

> "Suasana ok, makanan tidak terlalu ok, pelayanan kurang memuaskan"

- Label per ulasan → Positif (atau Netral) → **dua keluhan hilang**.
- Label per aspek → Suasana **Positif**, Makanan **Negatif**, Pelayanan **Negatif**.

## Bagaimana aspek ditentukan? (otomatis)

1. Hitung frekuensi kata pada seluruh ulasan yang telah dipreprocessing.
2. Ambil kata dengan frekuensi ≥ **ambang** (default 100, dapat diubah).
3. Saring: buang stopword, kata sentimen, kata generik, dan kata Inggris —
   agar yang tersisa benar-benar kata benda/aspek layanan.
4. Gabungkan varian kata dengan stemming (`makanan`, `makanannya` → `makan`).
5. Kelompokkan kata kunci yang tersisa dengan *hierarchical clustering*
   berdasarkan co-occurrence.
6. Beri nama tiap kelompok dari aspek acuan domain; kelompok yang tidak cocok
   menjadi aspek baru yang dinamai dari kata tersering.
"""),
    ("code", BOOTSTRAP),
    ("md", """
## 4.1 Memuat data dan lexicon
"""),
    ("code", """
from src import aspects, io_utils, lexicon, sentiment

df = io_utils.load_csv(config.PREP_CSV)
lex = lexicon.build_domain_lexicon()
print(f"Ulasan: {len(df):,} | lexicon: {len(lex):,} kata")
"""),
    ("md", """
## 4.2 Frekuensi kata: mengapa perlu penyaringan?

Tanpa penyaringan, kata terbanyak didominasi kata sentimen (`enak`, `nyaman`),
kata umum (`nya`, `aja`, `banget`), dan kata Inggris (`the`, `and`). Kata-kata
itu tidak bisa menjadi nama aspek layanan, sehingga harus disaring lebih dulu.
"""),
    ("code", """
raw_freq = aspects.keyword_frequency(df["tokens_stemmed"])
freq_table = pd.DataFrame(
    {
        "Ambang Frekuensi": [50, 80, 100, 150, 200, 300, 400],
        "Jumlah Kata Lolos Ambang": [int((raw_freq >= t).sum()) for t in [50, 80, 100, 150, 200, 300, 400]],
    }
)
io_utils.save_csv(freq_table, config.TABLES_DIR / "04_ambang_frekuensi.csv")
freq_table
"""),
    ("code", """
raw_freq.head(40).to_frame("Frekuensi (sebelum disaring)")
"""),
    ("md", """
## 4.3 Menjalankan ekstraksi aspek otomatis

**Ambang frekuensi dapat diubah** lewat `MIN_FREQ` di bawah. Semakin kecil
ambang, semakin banyak kata kunci dan aspek yang terbentuk; semakin besar
ambang, semakin ringkas tetapi ada aspek yang mungkin tidak terdeteksi.
"""),
    ("code", """
MIN_FREQ = config.ASPECT_MIN_FREQ
print(f"Ambang frekuensi yang dipakai: {MIN_FREQ}")

result = aspects.run_aspect_analysis(df, lex, min_freq=MIN_FREQ)
model = result["model"]

print(f"Kata kunci lolos saringan : {len(model.keywords)}")
print(f"Aspek terbentuk           : {len(result['aspects'])}")
"""),
    ("md", """
## 4.4 Pengaruh ambang frekuensi

Bandingkan jumlah aspek yang terbentuk pada beberapa ambang. Tabel ini bisa
dipakai untuk memilih ambang yang paling masuk akal untuk laporan.
"""),
    ("code", """
sweep = []
for th in [50, 100, 150, 200]:
    try:
        m = aspects.build_aspect_model(df["tokens_stemmed"], lex, min_freq=th)
        a_df, _ = aspects.aspect_model_frames(m)
        sweep.append(
            {
                "Ambang": th,
                "Kata Kunci": len(m.keywords),
                "Aspek Terbentuk": len(a_df),
                "Daftar Aspek": ", ".join(a_df["Aspek"].tolist()),
            }
        )
    except ValueError as exc:
        sweep.append({"Ambang": th, "Kata Kunci": 0, "Aspek Terbentuk": 0, "Daftar Aspek": str(exc)})

sweep_df = pd.DataFrame(sweep)
io_utils.save_csv(sweep_df, config.TABLES_DIR / "04_sweep_ambang.csv")
sweep_df
"""),
    ("md", """
## 4.5 Aspek yang terdeteksi dan kata kuncinya
"""),
    ("code", """
aspects_df, keywords_df = result["aspects"], result["keywords"]
io_utils.save_csv(aspects_df, config.TABLES_DIR / "04_aspek_terdeteksi.csv")
io_utils.save_csv(keywords_df, config.ASPECT_KEYWORD_CSV)
aspects_df
"""),
    ("code", """
keywords_df.head(40)
"""),
    ("md", """
## 4.6 Volume penyebutan per aspek

Menunjukkan aspek mana yang paling sering dibicarakan pelanggan.
"""),
    ("code", """
summary = result["summary"]
io_utils.save_csv(summary, config.ASPECT_SUMMARY_CSV)
io_utils.save_csv(result["hits"], config.ASPECT_SENTIMENT_CSV)

from src import visualize

visualize.plot_aspect_volume(summary)
summary
"""),
    ("md", """
## 4.7 Komposisi sentimen per aspek

Inilah tabel inti untuk pengambilan keputusan: **aspek mana yang banyak
dikeluhkan dan aspek mana yang menjadi kekuatan**.
"""),
    ("code", """
summary[[
    "Aspek", "Jumlah Disebut", "Ulasan Unik", "% dari Ulasan",
    "Positif", "Netral", "Negatif", "% Positif", "% Negatif",
    "Rating Rata-rata", "Hidden Complaint",
]]
"""),
    ("code", """
from src import visualize

visualize.plot_aspect_sentiment(summary)
"""),
    ("code", """
visualize.plot_rating_by_aspect(result["hits"])
"""),
    ("md", """
## 4.8 Matriks prioritas perbaikan

Prioritas dihitung dari **% negatif** dikali **akar volume penyebutan**, sehingga
aspek yang sering dibahas *dan* banyak dikeluhkan menempati peringkat atas.
Kolom *hidden complaint* menunjukkan keluhan dari pelanggan yang tetap memberi
rating 4-5 bintang.
"""),
    ("code", """
from src import interpret

priority = interpret.priority_matrix(summary)
io_utils.save_csv(priority, config.TABLES_DIR / "04_matriks_prioritas.csv")
priority
"""),
    ("code", """
visualize.plot_priority_scatter(summary)
"""),
    ("md", """
## 4.9 Bukti kualitatif: contoh klausa per aspek

Angka saja tidak cukup untuk meyakinkan pengelola. Berikut kutipan asli
pelanggan untuk setiap aspek.
"""),
    ("code", """
examples = result["examples"]
io_utils.save_csv(examples, config.TABLES_DIR / "04_contoh_aspek.csv")

neg_examples = examples[examples["Jenis"] == "Negatif"]
for aspect in priority["Aspek"].head(5):
    rows = neg_examples[neg_examples["Aspek"] == aspect].head(3)
    if rows.empty:
        continue
    print(f"\\n### {aspect}")
    for _, r in rows.iterrows():
        print(f"  [{r['Rating']}★ | skor {r['Skor']}] {r['Klausa']}")
"""),
    ("code", """
pos_examples = examples[examples["Jenis"] == "Positif"]
for aspect in priority["Aspek"].head(5):
    rows = pos_examples[pos_examples["Aspek"] == aspect].head(2)
    if rows.empty:
        continue
    print(f"\\n### {aspect}")
    for _, r in rows.iterrows():
        print(f"  [{r['Rating']}★ | skor {r['Skor']}] {r['Klausa']}")
"""),
    ("md", """
## 4.10 Konsistensi sentimen aspek terhadap rating
"""),
    ("code", """
from src import evaluate

consistency = evaluate.aspect_rating_consistency(result["hits"])
io_utils.save_csv(consistency, config.TABLES_DIR / "04_konsistensi_aspek_rating.csv")
consistency
"""),
    ("md", """
## 4.11 Kesimpulan sementara Tahap 4

- Aspek layanan berhasil diekstraksi **otomatis** dari frekuensi kata, dengan
  ambang yang dapat diubah sesuai kebutuhan analisis.
- Setiap aspek memiliki polaritas sendiri sehingga keluhan dan pujian dalam
  satu ulasan tidak saling menutupi.
- Peringkat prioritas menunjukkan urutan aspek yang perlu ditangani lebih dulu.
"""),
])

# ==========================================================================
# 05 - Clustering K-Means
# ==========================================================================
nb05 = nb([
    ("md", """
# 05 - Clustering Ulasan dengan K-Means

**Tahap 3 dan 4** metodologi: representasi teks dengan TF-IDF, lalu pengelompokan
ulasan menggunakan K-Means. Jumlah cluster optimal ditentukan dengan
**Elbow Method** dan **Silhouette Score**.

Clustering di sini bertujuan menemukan **tema ulasan secara otomatis**, tanpa
mengandalkan daftar aspek buatan manusia. Hasilnya nanti dibandingkan dengan
analisis aspek pada notebook 04 — bila keduanya sejalan, kesimpulan menjadi
lebih kuat.
"""),
    ("code", BOOTSTRAP),
    ("md", """
## 5.1 Memuat data hasil preprocessing
"""),
    ("code", """
from src import io_utils, features

df = io_utils.load_csv(config.PREP_CSV)
docs = df["tokens_stemmed"].fillna("").astype(str)
print(f"Ulasan: {len(df):,}")
print(f"Ulasan tanpa token: {int(docs.str.strip().eq('').sum()):,}")
"""),
    ("md", """
## 5.2 Representasi TF-IDF

Parameter yang dipakai:

| Parameter | Nilai | Alasan |
|---|---|---|
| `ngram_range` | (1, 2) | menangkap frasa seperti "live musik", "harga terjangkau" |
| `min_df` | 2 | membuang kata yang hanya muncul sekali (noise) |
| `max_df` | 0.95 | membuang kata yang muncul di hampir semua ulasan |
| `sublinear_tf` | True | meredam dominasi kata yang sangat sering |
"""),
    ("code", """
vectorizer, matrix, tfidf_summary = features.run_tfidf(docs)
io_utils.save_csv(tfidf_summary, config.TABLES_DIR / "05_ringkasan_tfidf.csv")
io_utils.save_sparse(matrix, config.TFIDF_MATRIX_NPZ)
io_utils.save_pickle(vectorizer, config.TFIDF_VECTORIZER_PKL)
tfidf_summary
"""),
    ("code", """
features.term_document_frequency(vectorizer, matrix).head(20).to_frame("Jumlah dokumen")
"""),
    ("md", """
## 5.3 Penentuan jumlah cluster optimal

Untuk setiap k dari 2 sampai 10 dihitung:

- **Inertia** — untuk Elbow Method (semakin kecil semakin rapat).
- **Silhouette Score** — seberapa baik objek berada di clusternya (semakin
  besar semakin baik, maksimum 1).
- **Davies-Bouldin Index** — rasio sebaran dalam cluster terhadap jarak antar
  cluster (semakin kecil semakin baik).
"""),
    ("code", """
from src import clustering

evaluation, models = clustering.evaluate_k(matrix)
io_utils.save_csv(evaluation, config.K_EVAL_CSV)
evaluation
"""),
    ("code", """
from src import visualize

knee = clustering.elbow_knee(evaluation)
best_k = int(evaluation.loc[evaluation["Silhouette"].idxmax(), "k"])

visualize.plot_elbow(evaluation, knee)
visualize.plot_silhouette(evaluation, best_k)
visualize.plot_dbi(evaluation)
print(f"k terbaik menurut Silhouette : {best_k}")
print(f"k siku menurut Elbow         : {knee}")
"""),
    ("md", """
## 5.4 Memilih k dan menjalankan K-Means final

Aturan pemilihan: **k dengan Silhouette tertinggi**, dengan Elbow sebagai
pendukung. Jika hasil kedua metode berbeda, Silhouette dipakai karena
memberi ukuran kualitas pemisahan yang eksplisit.
"""),
    ("code", """
from src import evaluate

result = clustering.run_clustering(matrix, vectorizer)
chosen_k = result["chosen_k"]

print(f"k dipakai: {chosen_k}")
print(f"Silhouette final: {result['silhouette_final']}  -> {evaluate.interpret_silhouette(result['silhouette_final'])}")
print(f"Davies-Bouldin final: {result['dbi_final']}")
"""),
    ("code", """
metrics = evaluate.clustering_metrics_table(result)
io_utils.save_csv(metrics, config.TABLES_DIR / "05_metrik_cluster.csv")
metrics
"""),
    ("code", """
result["sizes"]
"""),
    ("code", """
visualize.plot_cluster_sizes(result["sizes"])
"""),
    ("code", """
result["per_cluster"]
"""),
    ("md", """
## 5.5 Term dominan per cluster

Term dengan rata-rata bobot TF-IDF tertinggi menunjukkan tema tiap cluster.
"""),
    ("code", """
result["top_terms"].pivot_table(
    index="Term", columns="Cluster", values="Bobot TF-IDF", fill_value=0
).round(3)
"""),
    ("code", """
for cluster, terms in result["top_terms_str"].items():
    print(f"Cluster {cluster}: {terms}")
"""),
    ("code", """
visualize.plot_cluster_heatmap(result["top_terms"])
"""),
    ("md", """
## 5.6 Profil cluster

Profil menggabungkan ukuran cluster, sebaran rating, dan komposisi sentimen.
"""),
    ("code", """
from src import interpret

scored = io_utils.load_csv(config.DOC_SENTIMENT_CSV)
profile = interpret.cluster_profile(scored, result["labels"], result["top_terms_str"])
io_utils.save_csv(profile, config.CLUSTER_SUMMARY_CSV)
io_utils.save_csv(result["sizes"], config.TABLES_DIR / "05_ukuran_cluster.csv")
profile
"""),
    ("code", """
visualize.plot_cluster_sentiment(profile)
"""),
    ("md", """
## 5.7 Contoh ulasan per cluster
"""),
    ("code", """
examples = interpret.cluster_examples(scored, result["labels"], n=3)
io_utils.save_csv(examples, config.TABLES_DIR / "05_contoh_cluster.csv")

for cluster in sorted(examples["Cluster"].unique()):
    print(f"\\n=== Cluster {cluster} ===")
    for _, row in examples[examples["Cluster"] == cluster].iterrows():
        print(f"  [{row['Jenis']} | {row['Rating']}★] {row['Review'][:160]}")
"""),
    ("md", """
## 5.8 Menyimpan hasil clustering
"""),
    ("code", """
clustered = scored.copy()
clustered["cluster"] = result["labels"]
io_utils.save_csv(clustered, config.CLUSTERED_CSV)
print(f"Tersimpan: {config.CLUSTERED_CSV.name} ({len(clustered):,} baris)")
clustered[["review_id", "Rating", "cluster", "sentimen_teks"]].head()
"""),
    ("md", """
## 5.9 Kesimpulan Tahap 5

- Teks ulasan berhasil direpresentasikan sebagai matriks TF-IDF.
- Jumlah cluster dipilih berdasarkan Silhouette Score, didukung Elbow Method.
- Nilai Silhouette pada teks ulasan pendek umumnya rendah (di bawah 0,1)
  karena banyak ulasan hanya berisi 5-15 kata dan berbagi kosakata yang mirip.
  Nilai tersebut tetap bermakna untuk analisis eksploratif, tetapi harus
  disampaikan apa adanya di laporan.
- Hasil clustering dipakai pada notebook 06 bersama analisis aspek untuk
  menyusun rekomendasi.
"""),
])

# ==========================================================================
# 06 - Interpretasi dan rekomendasi
# ==========================================================================
nb06 = nb([
    ("md", """
# 06 - Interpretasi Hasil dan Rekomendasi

**Tahap 6** metodologi: menggabungkan seluruh hasil menjadi kesimpulan dan
rekomendasi yang dapat dipakai untuk pengambilan keputusan.

Notebook ini menjawab dua pertanyaan:

1. **Apa yang sudah baik** di Silol Kopi & Eatery?
2. **Apa yang perlu diperbaiki**, dan mana yang paling mendesak?
"""),
    ("code", BOOTSTRAP),
    ("code", """
from src import interpret, io_utils

clustered = io_utils.load_csv(config.CLUSTERED_CSV)
aspect_summary = io_utils.load_csv(config.ASPECT_SUMMARY_CSV)
cluster_profile = io_utils.load_csv(config.CLUSTER_SUMMARY_CSV)
hits_path = config.ASPECT_SENTIMENT_CSV
hits = io_utils.load_csv(hits_path) if hits_path.exists() else None

print(f"Ulasan: {len(clustered):,} | Aspek: {len(aspect_summary)} | Cluster: {clustered['cluster'].nunique()}")
"""),
    ("md", """
## 6.1 Ringkasan eksekutif
"""),
    ("code", """
summary = interpret.executive_summary(clustered, aspect_summary)
io_utils.save_csv(summary, config.TABLES_DIR / "06_ringkasan_eksekutif.csv")
summary
"""),
    ("md", """
## 6.2 Aspek yang sudah baik (kekuatan)

Aspek dengan persentase positif tinggi dan volume penyebutan besar.
"""),
    ("code", """
strengths = aspect_summary.sort_values(["% Positif", "Jumlah Disebut"], ascending=False)
strengths[["Aspek", "Jumlah Disebut", "% Positif", "% Negatif", "Rating Rata-rata"]].head()
"""),
    ("md", """
## 6.3 Aspek yang perlu diperbaiki

Diurutkan berdasarkan skor prioritas (kombinasi volume keluhan dan persentase
negatif).
"""),
    ("code", """
priority = interpret.priority_matrix(aspect_summary)
io_utils.save_csv(priority, config.TABLES_DIR / "06_matriks_prioritas.csv")
priority
"""),
    ("code", """
from src import visualize

visualize.plot_priority_bar(priority)
"""),
    ("md", """
## 6.4 Tema ulasan hasil clustering

K-Means mengelompokkan ulasan berdasarkan kemiripan teks. Tema diberi nama
otomatis dari term dominan tiap cluster dan dipetakan ke aspek layanan.
"""),
    ("code", """
cluster_profile[[
    "Cluster", "Jumlah Ulasan", "Persentase", "Rating Rata-rata",
    "Tema (otomatis)", "Sentimen Dominan", "Term Dominan",
]]
"""),
    ("md", """
## 6.5 Hubungan cluster dengan aspek layanan

Bila cluster hasil K-Means dan aspek hasil ekstraksi otomatis saling mendukung,
kesimpulan menjadi lebih kuat.
"""),
    ("code", """
if hits is not None and not hits.empty:
    matrix_ca = interpret.cluster_aspect_matrix(
        clustered, clustered["cluster"].to_numpy(), hits
    )
    io_utils.save_csv(matrix_ca, config.TABLES_DIR / "06_cluster_x_aspek.csv")
    visualize.plot_cluster_aspect_heatmap(matrix_ca)
    matrix_ca
else:
    print("Data sentimen aspek belum tersedia.")
"""),
    ("md", """
## 6.6 Rekomendasi tindakan per aspek
"""),
    ("code", """
recs = interpret.recommendations(aspect_summary)
io_utils.save_csv(recs, config.TABLES_DIR / "06_rekomendasi.csv")
recs
"""),
    ("md", """
## 6.7 Bukti pendukung: kutipan pelanggan per aspek prioritas
"""),
    ("code", """
examples = io_utils.load_csv(config.TABLES_DIR / "04_contoh_aspek.csv")
neg = examples[examples["Jenis"] == "Negatif"]

for aspect in priority["Aspek"].head(4):
    rows = neg[neg["Aspek"] == aspect].head(3)
    if rows.empty:
        continue
    print(f"\\n### {aspect}")
    for _, r in rows.iterrows():
        print(f"  • [{r['Rating']}★] {r['Klausa'][:180]}")
"""),
    ("md", """
## 6.8 Kesimpulan akhir

Isi bagian ini secara naratif di laporan, dengan angka dari tabel di atas.
Kerangka yang disarankan:

1. **Karakteristik data** — jumlah ulasan yang dianalisis, sebaran rating,
   dan seberapa banyak ulasan tanpa teks.
2. **Sentimen keseluruhan** — proporsi positif/netral/negatif, serta tingkat
   kesesuaian antara teks dan rating.
3. **Tema ulasan** — hasil clustering beserta term dominan tiap cluster.
4. **Aspek layanan** — aspek yang menjadi kekuatan dan aspek yang menjadi
   keluhan, lengkap dengan persentase.
5. **Prioritas perbaikan** — urutan aspek berdasarkan skor prioritas, disertai
   kutipan pelanggan.
6. **Rekomendasi** — tindakan konkret untuk setiap aspek prioritas.

### Keterbatasan

- Label sentimen berbasis lexicon tidak memahami sarkasme dan konteks yang
  sangat spesifik.
- Sebagian besar ulasan sangat pendek sehingga struktur cluster lemah
  (Silhouette rendah) — hal ini wajar pada data ulasan.
- Penentuan ambang frekuensi aspek memengaruhi jumlah aspek yang terbentuk;
  ambang perlu dilaporkan bersama hasilnya agar dapat direproduksi.
"""),
])

if __name__ == "__main__":
    save(nb01, "01_data_understanding.ipynb")
    save(nb02, "02_preprocessing.ipynb")
    save(nb03, "03_sentimen_ulasan.ipynb")
    save(nb04, "04_sentimen_aspek.ipynb")
    save(nb05, "05_clustering_kmeans.ipynb")
    save(nb06, "06_interpretasi_rekomendasi.ipynb")