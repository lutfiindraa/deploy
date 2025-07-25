# pages/4_🏞️_Analisis_Tempat_Wisata.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core_utils import (
    init_firebase, load_data, render_sidebar, 
    create_pie_chart, analyze_checkbox_question, download_data
)

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman

# --- Konfigurasi & Load Data ---
st.set_page_config(page_title="Analisis Tempat Wisata", layout="wide")
st.title("🏞️ Analisis Pengelolaan Sampah di Tempat Wisata")
st.markdown("Analisis dari formulir **DC-Tempat Wisata**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

df_raw = load_data(db, "formSubmissions", "DC-Tempat Wisata")
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-Tempat Wisata' yang cocok dengan filter yang dipilih.")
    st.stop()

# --- Analisis dalam Tabs ---
tab1, tab2 = st.tabs(["📈 Profil & Volume Sampah", "❗ Kendala & Kebutuhan"])

with tab1:
    st.header("Profil, Pengunjung, dan Volume Sampah")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '201', "Distribusi Jenis Tempat Wisata")
    with col2:
        create_pie_chart(df, '202', "Distribusi Status Pengelola")
        
    st.markdown("---")
    st.subheader("Volume Sampah yang Dihasilkan per Hari (Kode: 301)")
    # Custom parsing untuk data grid-numeric volume sampah
    if '301' in df.columns:
        volume_data = []
        for index, row in df.iterrows():
            item = row['301']
            if isinstance(item, dict) and 'default_row' in item:
                hari_biasa = pd.to_numeric(item['default_row'].get('Hari Biasa', {}).get('Angka (dalam m³)'), errors='coerce')
                peak_season = pd.to_numeric(item['default_row'].get('Peak Season', {}).get('Angka (dalam m³)'), errors='coerce')
                if pd.notna(hari_biasa) and pd.notna(peak_season):
                    volume_data.append({'Kondisi': 'Hari Biasa', 'Volume (m³)': hari_biasa})
                    volume_data.append({'Kondisi': 'Peak Season', 'Volume (m³)': peak_season})
        
        if volume_data:
            volume_df = pd.DataFrame(volume_data)
            fig = px.box(volume_df, x='Kondisi', y='Volume (m³)', color='Kondisi',
                         title="Perbandingan Sebaran Volume Sampah", points="all")
            st.plotly_chart(fig, use_container_width=True)
            download_data(volume_df, "volume_sampah_wisata")

with tab2:
    st.header("Analisis Kendala dan Dukungan yang Dibutuhkan")
    analyze_checkbox_question(df, '309', "Kendala yang Dihadapi Tempat Wisata")
    st.markdown("---")
    analyze_checkbox_question(df, '310', "Dukungan yang Paling Dibutuhkan Tempat Wisata")
    
# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-Tempat Wisata'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_tempat_wisata")