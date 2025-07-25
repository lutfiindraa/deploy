# pages/1_📊_Analisis_Penduduk.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
from core_utils import (
    init_firebase, load_data, render_sidebar, create_pie_chart, 
    create_bar_chart, analyze_checkbox_question, download_data
)

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman

# --- Konfigurasi Halaman & Memuat Data ---
st.set_page_config(page_title="Analisis Penduduk", layout="wide")
st.title("📊 Analisis Demografi & Sosial Ekonomi Penduduk")
st.markdown("Analisis dari formulir **DC-Penduduk**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal. Dasbor tidak dapat dimuat.")
    st.stop()

# Memuat data spesifik untuk 'DC-Penduduk'
df_raw = load_data(db, "formSubmissions", "DC-Penduduk")
# Menerapkan filter global dari sidebar
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-Penduduk' yang cocok dengan filter yang Anda pilih.")
    st.stop()

# --- Analisis dalam Tabs ---
tab1, tab2, tab3 = st.tabs(["🧑‍🤝‍🧑 Demografi", "💼 Sosial & Ekonomi", "♻️ Perilaku Sampah"])

with tab1:
    st.header("Analisis Karakteristik Demografi Rumah Tangga")
    
    st.subheader("Distribusi Umur Kepala Rumah Tangga (KRT)")
    if '201' in df.columns:
        # Mengubah kolom tanggal lahir menjadi format datetime
        dob_series = pd.to_datetime(df['201'], errors='coerce')
        valid_dobs = dob_series.dropna()
        
        if not valid_dobs.empty:
            # --- [PERBAIKAN] ---
            # Pastikan kedua sisi operasi pengurangan adalah tz-naive.
            # 1. Buat waktu sekarang menjadi naive
            current_time_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            
            # 2. Hapus timezone dari data tanggal lahir jika ada
            if valid_dobs.dt.tz is not None:
                valid_dobs = valid_dobs.dt.tz_localize(None)
            # --------------------

            ages = (current_time_naive - valid_dobs).dt.days / 365.25
            
            # Filter umur yang wajar (antara 15 dan 100 tahun)
            reasonable_ages = ages[(ages >= 15) & (ages < 100)]
            
            if not reasonable_ages.empty:
                fig = px.histogram(reasonable_ages, nbins=20, title="Distribusi Umur KRT", marginal="box")
                fig.update_layout(xaxis_title="Umur KRT (Tahun)", yaxis_title="Frekuensi")
                st.plotly_chart(fig, use_container_width=True)
                download_data(reasonable_ages.to_frame(name='umur_krt'), "data_umur_krt")
            else:
                st.info("Tidak ada data umur yang wajar untuk ditampilkan.")
        else:
            st.info("Data Tanggal Lahir KRT (201) tidak valid atau kosong.")
    else:
        st.info("Kolom Tanggal Lahir KRT (201) tidak ditemukan.")

    st.markdown("---")
    
    st.subheader("Distribusi Jumlah Anggota Rumah Tangga (ART)")
    if '105' in df.columns:
        df['105_numeric'] = pd.to_numeric(df['105'], errors='coerce')
        art_data = df['105_numeric'].dropna()
        if not art_data.empty:
            # Menentukan jumlah bins agar tidak error jika max value kecil
            nbins = min(int(art_data.max()), 30) if art_data.max() > 0 else 1
            fig_art = px.histogram(art_data, nbins=nbins, title="Distribusi Jumlah ART")
            fig_art.update_layout(xaxis_title="Jumlah Anggota Rumah Tangga", yaxis_title="Frekuensi (Jumlah RT)")
            st.plotly_chart(fig_art, use_container_width=True)
            download_data(art_data.to_frame(name='jumlah_art'), "data_jumlah_art")
        else:
            st.info("Data Jumlah ART (105) tidak valid atau kosong.")
    else:
        st.info("Kolom Jumlah ART (105) tidak ditemukan.")

with tab2:
    st.header("Analisis Karakteristik Sosial & Ekonomi")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '110_1', "Kepemilikan Rekening Bank Aktif")
    with col2:
        create_pie_chart(df, '210_1', "Kepemilikan BPJS Ketenagakerjaan")

    st.markdown("---")
    
    education_order = [
        'Tidak/Belum Sekolah', 'Tidak Tamat SD', 'SD/Sederajat', 
        'SMP/Sederajat', 'SMA/Sederajat', 'Diploma I/II/III', 
        'Diploma IV/S1', 'S2/S3'
    ]
    # Menggunakan fungsi bar chart dengan urutan kategori kustom
    st.subheader("Distribusi Pendidikan Terakhir Kepala Rumah Tangga")
    if '202' in df.columns:
        counts_edu = df['202'].value_counts().reindex(education_order).dropna().reset_index()
        counts_edu.columns = ['kategori', 'jumlah']
        fig_edu = px.bar(counts_edu, x='kategori', y='jumlah', title="Pendidikan Terakhir KRT", text_auto=True)
        st.plotly_chart(fig_edu, use_container_width=True)
        download_data(counts_edu, "distribusi_pendidikan_krt")
    else:
        st.info("Kolom Pendidikan KRT (202) tidak ditemukan.")

    st.markdown("---")
    create_bar_chart(df, '207_1', "Status Pekerjaan Utama", is_horizontal=True)


with tab3:
    st.header("Analisis Perilaku Terkait Pengelolaan Sampah")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '301', "Apakah Melakukan Pemilahan Sampah?")
    with col2:
        create_pie_chart(df, '307', "Kesediaan Mengikuti Pelatihan Sampah")

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        create_pie_chart(df, '310', "Mengetahui Adanya Bank Sampah Desa?")
    with col4:
        create_pie_chart(df, '312', "Opini Terhadap Fasilitas Pengelolaan Sampah Desa")
        
    st.markdown("---")
    analyze_checkbox_question(df, '306', "Jenis Tempat Pembuangan Sampah Keluarga")
    st.markdown("---")
    analyze_checkbox_question(df, '313', "Usulan Peningkatan Pengelolaan Sampah")

# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-Penduduk'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_penduduk")