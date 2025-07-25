<div align="center">

# 📊 Analisis Sektoral & Dashboard Interaktif UMKM Kota Batu

Sebuah proyek analisis data untuk memetakan dan mengelompokkan UMKM di Kota Batu, disajikan dalam sebuah *dashboard* web yang modern dan interaktif.

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

</div>

<br>

<div align="center">
  <img src="[GANTI_DENGAN_PATH_SCREENSHOT_ANDA, MISALNYA: assets/dashboard-dark.png]" alt="Tampilan Dashboard UMKM" width="80%">
</div>

---

## 🌟 Tentang Proyek

Proyek ini bertujuan untuk melakukan analisis mendalam terhadap **27.000+ data UMKM di Kota Batu** guna mendukung perumusan kebijakan ekonomi lokal yang lebih tepat sasaran. Proses analisis meliputi pembersihan data, analisis data eksploratif (EDA), hingga penerapan *machine learning* (Clustering K-Means) untuk mengidentifikasi pengelompokan alami UMKM berdasarkan lokasi dan sektor usaha.

Seluruh hasil analisis dan wawasan disajikan dalam sebuah **dashboard web interaktif** yang dibangun menggunakan Streamlit, memungkinkan para pemangku kepentingan untuk menjelajahi data dengan mudah dan intuitif.

---

## ✨ Fitur Utama Dashboard

-   **📈 Visualisasi Data Komprehensif:** Peta persebaran interaktif, serta grafik batang dinamis untuk analisis sektoral dan kluster.
-   **⚙️ Panel Filter Multi-Lapis:** Filter data secara *real-time* berdasarkan Kecamatan, Sektor Usaha, dan Kluster UMKM.
-   **🎨 Desain Adaptif & Modern:** Tampilan *dashboard* (termasuk kartu metrik dan grafik) secara otomatis menyesuaikan dengan tema **Terang (Light)** dan **Gelap (Dark)**.
-   **🧠 Hasil Clustering Machine Learning:** Lihat pengelompokan UMKM yang unik, hasil dari model K-Means, untuk menemukan pola tersembunyi.
-   **🖱️ Peta Interaktif Profesional:** Didukung oleh Pydeck dengan *tooltip* informatif dan opsi untuk mengubah tema peta.
-   **📥 Ekspor Data ke CSV:** Unduh data yang telah difilter langsung dari *dashboard* untuk analisis lebih lanjut.

---

## 🖼️ Tampilan Aplikasi

| Mode Terang (Light Theme) | Mode Gelap (Dark Theme) |
| :---: | :---: |
| <img src="[GANTI_DENGAN_PATH_SCREENSHOT_ANDA, MISALNYA: assets/dashboard-light.png]" alt="Mode Terang" width="100%"> | <img src="[GANTI_DENGAN_PATH_SCREENSHOT_ANDA, MISALNYA: assets/dashboard-dark.png]" alt="Mode Gelap" width="100%"> |


---

## 🛠️ Teknologi yang Digunakan

-   **Bahasa Pemrograman:** Python 3.8+
-   **Dashboard & UI:** Streamlit
-   **Manipulasi Data:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn
-   **Visualisasi Data:** Plotly Express, Pydeck, Folium (di notebook)

---

## 🚀 Instalasi dan Persiapan

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di mesin lokal Anda.

### 1. Prasyarat
-   Pastikan Anda telah menginstal **Python versi 3.8** atau yang lebih baru.
-   Git (opsional, untuk kloning repositori).

### 2. Langkah-langkah Instalasi

1.  **Kloning Repositori (atau Unduh ZIP)**
    ```bash
    git clone [URL_REPOSITORI_ANDA]
    cd [NAMA_FOLDER_PROYEK]
    ```

2.  **Buat Lingkungan Virtual (Sangat Direkomendasikan)**
    Ini akan mengisolasi dependensi proyek Anda.
    ```bash
    # Membuat virtual environment bernama 'venv'
    python -m venv venv

    # Mengaktifkan di Windows
    .\venv\Scripts\activate

    # Mengaktifkan di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi**
    Buat file `requirements.txt` (jika belum ada) dan instal semua pustaka yang dibutuhkan.
    ```bash
    pip install -r requirements.txt
    ```

---

## 📄 Membuat File `requirements.txt`

Jika Anda belum memiliki file ini, Anda bisa membuatnya dengan konten berikut:

```txt
streamlit
pandas
plotly
scikit-learn
numpy
openpyxl # Diperlukan untuk membaca file .xlsx
folium # Opsional, untuk notebook
```
Atau, jika Anda sudah menginstal semuanya di *virtual environment*, jalankan perintah:
```bash
pip freeze > requirements.txt
```

---

## ⚙️ Menjalankan Proyek

1.  **Pastikan File Data Ada**
    Pastikan file `umkm_batu_clustered.csv` berada di direktori utama proyek, sejajar dengan `app.py`. Jika belum ada, jalankan terlebih dahulu *notebook* analisis data Anda untuk menghasilkan file ini.

2.  **Jalankan Aplikasi Streamlit**
    Buka terminal Anda, pastikan Anda berada di direktori utama proyek dan *virtual environment* sudah aktif, lalu jalankan:
    ```bash
    streamlit run app.py
    ```

3.  **Buka di Browser**
    Aplikasi akan otomatis terbuka di browser Anda, biasanya di alamat `http://localhost:8501`.

---

## 📂 Struktur Folder

Berikut adalah struktur folder yang direkomendasikan untuk proyek ini:

```
.
├── assets/
│   ├── dashboard-light.png
│   └── dashboard-dark.png
├── venv/
├── Analisis_UMKM.ipynb       # Notebook untuk analisis & clustering
├── app.py                    # Script utama dashboard Streamlit
├── umkm_batu_clustered.csv   # Dataset hasil olahan
├── 3579_umkm23.xlsx          # Dataset mentah (opsional)
├── README.md
└── requirements.txt
```

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.

[Bayu Ardiyansyah] - [2025]