# pages/3_🏫_Analisis_Sekolah.py
import streamlit as st
import pandas as pd
import plotly.express as px
from core_utils import (
    init_firebase, load_data, render_sidebar, create_pie_chart, 
    analyze_checkbox_question, download_data
)


if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman

# --- Konfigurasi & Load Data ---
st.set_page_config(page_title="Analisis Sekolah", layout="wide")
st.title("🏫 Analisis Pengelolaan Sampah di Sekolah")
st.markdown("Analisis dari formulir **DC-Sekolah**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

df_raw = load_data(db, "formSubmissions", "DC-Sekolah")
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-Sekolah' yang cocok dengan filter yang dipilih.")
    st.stop()

# --- KPIs dan Statistik Umum ---
st.header("Statistik Umum Sekolah (Berdasarkan Filter)")
numeric_cols = ['203', '204', '304', '305']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Rata-rata Guru & Pegawai", f"{df['203'].mean():.1f}" if '203' in df else "N/A")
kpi2.metric("Rata-rata Murid", f"{df['204'].mean():.1f}" if '204' in df else "N/A")
kpi3.metric("Rata-rata Petugas Kebersihan", f"{df['305'].mean():.1f}" if '305' in df else "N/A")

# --- Analisis dalam Tabs ---
tab1, tab2 = st.tabs(["♻️ Fasilitas & Perilaku", "❗ Kendala & Kebutuhan"])

with tab1:
    st.header("Analisis Fasilitas, Edukasi, dan Komposisi Sampah")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '201', "Distribusi Jenjang Pendidikan")
    with col2:
        create_pie_chart(df, '303', "Ketersediaan Tempat Sampah Terpilah")
        
    st.markdown("---")
    create_pie_chart(df, '308', "Frekuensi Kampanye Pengurangan Sampah ke Siswa")
    
    st.subheader("Komposisi Sampah (Kode: 302)")
    # Custom parsing untuk data grid-numeric komposisi
    if '302' in df.columns:
        composition_data = df['302'].dropna()
        composition_summary = {}
        for item in composition_data:
            if isinstance(item, dict) and 'default_row' in item:
                for waste_type, value_dict in item['default_row'].items():
                    if isinstance(value_dict, dict):
                        percent = pd.to_numeric(value_dict.get('Jumlah Persen (%)'), errors='coerce')
                        if pd.notna(percent):
                            composition_summary[waste_type] = composition_summary.get(waste_type, 0) + percent
        if composition_summary:
            total_sum = sum(composition_summary.values())
            # Normalisasi untuk mendapatkan rata-rata persentase
            avg_composition = {k: (v / len(composition_data)) for k, v in composition_summary.items()}
            comp_df = pd.DataFrame(list(avg_composition.items()), columns=['Jenis Sampah', 'Rata-rata Persentase (%)'])
            fig = px.bar(comp_df, x='Jenis Sampah', y='Rata-rata Persentase (%)', title="Rata-Rata Komposisi Sampah di Sekolah", text_auto='.1f')
            st.plotly_chart(fig, use_container_width=True)
            download_data(comp_df, "komposisi_sampah_sekolah")

with tab2:
    st.header("Analisis Kendala dan Dukungan yang Dibutuhkan")
    analyze_checkbox_question(df, '309', "Kendala yang Dihadapi Sekolah")
    st.markdown("---")
    analyze_checkbox_question(df, '310', "Dukungan yang Paling Dibutuhkan Sekolah")

# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-Sekolah'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_sekolah")