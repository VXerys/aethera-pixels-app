# Aethera Pixels— Advanced Digital Image Processing (UTS)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-%E2%89%A51.28-FF4B4B)
![OpenCV](https://img.shields.io/badge/opencv-%E2%89%A54.8-5C3EE8)
![License](https://img.shields.io/badge/license-Academic%2FEducational-green)

> Aplikasi pengolahan citra digital berbasis Streamlit dengan pendekatan modular “Pipeline Formula” untuk transformasi, pemulihan, dan peningkatan kualitas citra secara terstruktur.

---

## 📌 Informasi Akademik

| Kategori | Detail |
|---|---|
| Mata Kuliah | Pengolahan Citra Digital |
| Semester | 5 (Lima) |
| Jenis Tugas | Ujian Tengah Semester (UTS) |
| Institusi | Universitas Nusa Putra/Teknik Informatika] |
| Tahun Akademik | 2024/2025 |
| Tanggal Pengumpulan | 9 November 2025 |

---

## 👥 Tim Pengembang

| Nama | NIM | Peran Utama |
|---|---|---|
| M. Sechan Alfarisi | 20230040094 | Project Lead & Core Developer — Arsitektur aplikasi, UI/UX Streamlit, integrasi pipeline, koordinasi tim |
| s | 20230040031 | sl |
| s | 20230040057 | Dokumen |
| s | 20230040018 | PPT |
| s | 20230040062 | s |

---

## 📖 Daftar Isi

- [Aethera Pixels — Advanced Digital Image Processing (UTS)](#aethera-pixelis--advanced-digital-image-processing-uts)
  - [📌 Informasi Akademik](#-informasi-akademik)
  - [👥 Tim Pengembang](#-tim-pengembang)
  - [📖 Daftar Isi](#-daftar-isi)
  - [🎯 Ringkasan Proyek](#-ringkasan-proyek)
  - [🚀 Fitur Utama](#-fitur-utama)
  - [🏗️ Arsitektur \& Struktur Proyek](#️-arsitektur--struktur-proyek)
  - [🧪 Pipeline Overview](#-pipeline-overview)
  - [🔍 Pipeline Detail](#-pipeline-detail)
    - [Super Resolution 4K Enhance](#super-resolution-4k-enhance)
    - [Creative Filmic Effect](#creative-filmic-effect)
    - [Selective Texture Enhancement](#selective-texture-enhancement)
  - [🛠️ Instalasi \& Setup](#️-instalasi--setup)
  - [🧭 Panduan Penggunaan](#-panduan-penggunaan)
  - [🎚️ Preset \& Parameter](#️-preset--parameter)
    - [Super Resolution 4K](#super-resolution-4k)
    - [Creative Filmic Effect](#creative-filmic-effect-1)
    - [Selective Texture Enhancement](#selective-texture-enhancement-1)
  - [📏 Evaluasi \& Metrik Kualitas](#-evaluasi--metrik-kualitas)
  - [💡 Contoh Penggunaan](#-contoh-penggunaan)
  - [🆘 Troubleshooting](#-troubleshooting)
  - [🛣️ Roadmap](#️-roadmap)
  - [⚠️ Batasan Diketahui](#️-batasan-diketahui)
  - [🤝 Kontribusi](#-kontribusi)
  - [📄 Lisensi](#-lisensi)
  - [🗂️ Mapping ke Slide \& PDF](#️-mapping-ke-slide--pdf)
  - [📝 Changelog](#-changelog)

---

## 🎯 Ringkasan Proyek

**Aethera Pixels** adalah aplikasi web interaktif untuk pengolahan citra digital tingkat lanjut. Pendekatan inti adalah **Pipeline Formula**: pre-processing → core algorithm → post-processing. Aplikasi berorientasi edukasi dan praktis untuk kebutuhan UTS, riset, dan demonstrasi.

Sorotan:
- Antarmuka modern (single-page) dengan kontrol jelas dan log proses.
- Preset cerdas (Conservative, Balanced, Aggressive) + mode fine-tune manual.
- Modular dan extensible: mudah menambah filter/fitur baru.
- Metrik objektif (PSNR, SSIM, dsb.) untuk evaluasi kuantitatif.

---

## 🚀 Fitur Utama

- Super Resolution hingga 4× (target 4K) dengan pra-denoise dan pasca-sharpen.
- Creative Filmic Effect: tonal mapping (3D LUT), color grading, film grain.
- Selective Texture Enhancement: detail-preserving enhancement lokal.
- Info gambar, log proses, dan tombol unduh hasil.

---

## 🏗️ Arsitektur & Struktur Proyek

Struktur folder utama:

```
app.py
presets.json
requirements.txt
app/
  main.py
  components/
  pages/
  utils/
assets/
  sample_images/
  style_presets/
pipeline/
  creative_filmic_effect.py
  enhancement.py
  filters.py
  restoration.py
  selective_texture_enhancement.py
  super_resolution_4k.py
utils/
  io_utils.py
  metrics.py
```

Catatan:
- UI utama ada di `app.py`/`app/main.py` (Streamlit).
- Algoritma ada di `pipeline/`. Utilitas (I/O, metrik) di `utils/`.

---

## 🧪 Pipeline Overview

1) Super Resolution 4K Enhance — Upscaling 2×/3×/4× dengan pra-denoise dan pasca-sharpen. Target: tajam, low halo.

2) Creative Filmic Effect — Efek sinematik/vintage via LUT, histogram/LAB adjustment, dan film grain parametrik.

3) Selective Texture Enhancement — Ekstraksi base/detail (guided filter), boosting detail adaptif, dan fusi kembali.

---

## 🔍 Pipeline Detail

### Super Resolution 4K Enhance

Konsep: membersihkan noise, memperbesar, lalu mempertajam untuk hasil natural.

Tahapan:
1. Pre-Denoising (Bilateral/Non-local Means)
2. Upscaling (ESRGAN/Lightweight 4× atau interpolasi LANCZOS sebagai fallback)
3. Post-Sharpening (Unsharp Mask) + halo guard ringan

Formula utama:
- Bilateral Filter: $I_p = \frac{1}{W_p} \sum_{q \in \Omega} I_q\,G_{\sigma_s}(\lVert p-q \rVert)\,G_{\sigma_r}(|I_p - I_q|)$
- Unsharp Mask: $I_{out} = I + \alpha\,(I - G_\sigma(I))$

File terkait: `pipeline/filters.py`, `pipeline/super_resolution_4k.py`

---

### Creative Filmic Effect

Konsep: grading sinematik/retro dengan tonal mapping, penyesuaian warna, dan simulasi grain.

Tahapan:
1. Tonal Mapping (3D LUT)
2. Color Grading (Histogram Matching/LAB Adjustment)
3. Film Grain Overlay (Gaussian/Perlin/Chromatic)

File terkait: `pipeline/enhancement.py`, `pipeline/creative_filmic_effect.py`

---

### Selective Texture Enhancement

Konsep: meningkatkan detail pada area bertekstur tanpa menambah noise pada area halus.

Tahapan:
1. Detail Extraction (Guided Filter) → Base + Detail
2. Adaptive Sharpening (Boost detail layer)
3. Fusion (Laplacian pyramid / weighted blend)

File terkait: `pipeline/selective_texture_enhancement.py`, `pipeline/filters.py`

---

## 🛠️ Instalasi & Setup

Prerequisites:
- Python 3.10+
- pip
- (Opsional) GPU CUDA 11.8+ untuk akselerasi model

Langkah (Windows PowerShell):

```powershell
cd C:\Users\user\aethera-pixelis

# (Opsional) Buat virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

URL lokal: http://localhost:8501

---

## 🧭 Panduan Penggunaan

1. Upload gambar (JPG/PNG; disarankan ≥1280×720, <200 MB).
2. Pilih Pipeline: Super Resolution / Filmic Effect / Texture Enhancement.
3. Pilih Preset (Conservative, Balanced, Aggressive) atau aktifkan Fine-tune.
4. Tekan “Process Image”; amati log proses dan metrik ringkas.
5. Unduh hasil dan bandingkan dengan input.

Tips presentasi: jadikan tiap tahapan pipeline sebagai satu slide dengan screenshot UI.

---

## 🎚️ Preset & Parameter

Preset intensitas (rekomendasi default; dapat disesuaikan):

### Super Resolution 4K
- Conservative: `denoise_sigma=40`, `scale_factor=2`, `sharpen_amount=0.6`
- Balanced: `denoise_sigma=70`, `scale_factor=3`, `sharpen_amount=1.0`
- Aggressive: `denoise_sigma=100`, `scale_factor=4`, `sharpen_amount=1.4`

### Creative Filmic Effect
- Conservative: `color_preset="Warm"`, `grain_intensity=0.1`, `grain_type="Gaussian"`
- Balanced: `color_preset="Teal & Orange"`, `grain_intensity=0.2`, `grain_type="Perlin"`
- Aggressive: `color_preset="Cinematic"`, `grain_intensity=0.35`, `grain_type="Chromatic"`

### Selective Texture Enhancement
- Conservative: `detail_radius=15`, `enhancement_gain=1.2`, `edge_threshold=0.02`
- Balanced: `detail_radius=30`, `enhancement_gain=1.8`, `edge_threshold=0.01`
- Aggressive: `detail_radius=45`, `enhancement_gain=2.4`, `edge_threshold=0.005`

Catatan parameter:
- `denoise_sigma` (10–150), `scale_factor` (2/3/4×), `sharpen_amount` (0.0–2.0)
- `color_preset` (Cinematic/Teal & Orange/Warm/Cool), `grain_intensity` (0.0–0.5)
- `detail_radius` (5–50), `enhancement_gain` (1.0–3.0), `edge_threshold` (0.001–0.1)

---

## 📏 Evaluasi & Metrik Kualitas

Metrik objektif yang digunakan:

- PSNR (Peak Signal-to-Noise Ratio) — lebih tinggi lebih baik. Implementasi: `utils/io_utils.py` (kelas `ImageMetrics`).
- SSIM (Structural Similarity Index) — 1 = identik. Implementasi: `utils/io_utils.py`.
- Laplacian Variance, Edge Response, Local Contrast, Noise Estimation, Histogram Spread — `utils/metrics.py`.

Kaidah evaluasi:
- Jika ground truth tersedia → gunakan PSNR/SSIM.
- Jika tidak ada ground truth → gunakan panel visual + metrik non-referensi (sharpness, contrast, noise).

Rumus SSIM singkat:
$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C1)(2\sigma_{xy} + C2)}{(\mu_x^2 + \mu_y^2 + C1)(\sigma_x^2 + \sigma_y^2 + C2)}$$

---

## 💡 Contoh Penggunaan

1) Super Resolution pada foto lama 720p → pilih “Super Resolution 4K Enhance”, set `denoise_sigma=80`, `scale_factor=4`, `sharpen_amount=1.2`, proses, unduh hasil 4K.

2) Filmic Effect vintage → pilih “Creative Filmic Effect”, set `color_preset="Teal & Orange"`, `grain_intensity=0.2`, proses, unduh hasil.

3) Texture Enhancement pada foto bertekstur → pilih “Selective Texture Enhancement”, set `detail_radius=35`, `enhancement_gain=1.8`, proses, unduh hasil.

---

## 🆘 Troubleshooting

- ModuleNotFoundError: cv2 → `pip install opencv-python opencv-contrib-python --upgrade`
- ModuleNotFoundError: torch → CPU: `pip install torch torchvision` | GPU (CUDA 11.8): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Streamlit error → `pip install streamlit --upgrade`

---

## 🛣️ Roadmap

- Batch processing untuk banyak gambar sekaligus.
- Before–after slider interaktif.
- Rekomendasi preset otomatis berbasis analisis histogram.
- Penyimpanan preset kustom dan log hasil.

---

## ⚠️ Batasan Diketahui

1) ESRGAN/SwinIR dapat fallback ke interpolasi jika memori GPU terbatas.  
2) Ukuran gambar optimal < 8K untuk kinerja stabil.  
3) Proses di CPU lebih lambat; disarankan GPU untuk super resolution.  
4) Mayoritas operasi berjalan di ruang warna BGR (standar OpenCV).

---

## 🤝 Kontribusi

1. Fork repository  
2. Buat branch fitur: `git checkout -b feature/NamaFitur`  
3. Commit: `git commit -m "Add NamaFitur"`  
4. Push: `git push origin feature/NamaFitur`  
5. Buka Pull Request

Pedoman pengembangan singkat:
- Tambah filter baru di `pipeline/filters.py` lalu hubungkan di UI (`app.py`).
- Tambah pipeline baru sebagai modul di `pipeline/` + dokumentasi singkat.

---

## 📄 Lisensi

Proyek untuk keperluan akademik/educational. Lihat dokumen teknis (jika ada) di `docs/dokumen-teknis.pdf`.

---

## 🗂️ Mapping ke Slide & PDF

Rekomendasi struktur slide (presentasi 10–15 menit):

1) Judul & Tim — badge, tagline singkat, tabel tim.  
2) Latar Belakang & Tujuan — masalah, motivasi, dan tujuan proyek.  
3) Arsitektur — struktur folder dan alur Pipeline Formula.  
4) Pipeline Overview — ketiga pipeline dan use-case ringkas.  
5) Pipeline Detail: Super Resolution — tahapan + parameter kunci + preset.  
6) Pipeline Detail: Filmic Effect — tahapan + preset + contoh visual.  
7) Pipeline Detail: Texture Enhancement — tahapan + preset + contoh visual.  
8) Demo & Hasil — sebelum/sesudah, metrik (PSNR/SSIM/sharpness).  
9) Evaluasi & Diskusi — interpretasi metrik, trade-off kualitas vs waktu.  
10) Batasan & Roadmap — limitasi dan rencana lanjutan.  
11) Kesimpulan & Referensi.

Rekomendasi Bab PDF (laporan): Pendahuluan → Tinjauan Pustaka → Metode (Pipeline) → Implementasi → Eksperimen & Evaluasi → Diskusi → Kesimpulan & Saran → Lampiran.

---

## 📝 Changelog

- 1.2.0 (9 Nov 2025): UI single-page stabil, README terstruktur untuk UTS, penambahan mapping slide/PDF, konsistensi metrik & parameter.

---

Terakhir diperbarui: 9 November 2025  
Versi: 1.2.0
