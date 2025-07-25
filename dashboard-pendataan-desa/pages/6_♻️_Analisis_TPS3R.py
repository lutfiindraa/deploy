# pages/6_♻️_Analisis_TPS3R.py
import streamlit as st
import pandas as pd
import plotly.express as px
from core_utils import (
    init_firebase, load_data, render_sidebar, 
    create_pie_chart, analyze_checkbox_question, download_data
)


if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman
    
    
# --- Konfigurasi & Load Data ---
st.set_page_config(page_title="Analisis TPS3R", layout="wide")
st.title("♻️ Analisis Operasional TPS3R")
st.markdown("Analisis dari formulir **DC-TPS3R**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

df_raw = load_data(db, "formSubmissions", "DC-TPS3R")
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-TPS3R' yang cocok dengan filter yang dipilih.")
    st.stop()

# --- KPIs Operasional ---
st.header("Statistik Kunci Operasional TPS3R")
numeric_cols = ['203', '205']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total RT Dilayani", f"{int(df['203'].sum())}" if '203' in df else "N/A")
iuran_count = df[df['204'] == 'Ya'].shape[0] if '204' in df else 0
kpi2.metric("TPS3R dengan Iuran", f"{iuran_count}")
avg_iuran = df['205'].dropna().mean() if '205' in df and not df['205'].dropna().empty else 0
kpi3.metric("Rata-rata Iuran", f"Rp {avg_iuran:,.0f}")

# --- Analisis dalam Tabs ---
tab1, tab2 = st.tabs(["🔄 Proses & Pengolahan", "🛠️ Fasilitas & Kendala"])

with tab1:
    st.header("Analisis Proses Pemilahan dan Pengolahan Sampah")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '209', "Apakah Sampah Sudah Dipilah dari Rumah?")
    with col2:
        create_pie_chart(df, '210', "Apakah Dilakukan Pemilahan Ulang di TPS3R?")
        
    st.markdown("---")
    st.subheader("Rata-Rata Persentase Alur Pengolahan Sampah (Kode: 211)")
    # Custom parsing untuk data grid-numeric persentase
    if '211' in df.columns:
        olah_data = df['211'].dropna()
        olah_summary = {}
        for item in olah_data:
            if isinstance(item, dict):
                inner_dict = next(iter(item.values()), {}) # Ambil dict pertama di dalam
                if isinstance(inner_dict, dict):
                    for method, value_dict in inner_dict.items():
                        if isinstance(value_dict, dict):
                            percent = pd.to_numeric(value_dict.get('Jumlah (%)'), errors='coerce')
                            if pd.notna(percent):
                                olah_summary[method] = olah_summary.get(method, 0) + percent
        if olah_summary:
            avg_olah_df = pd.DataFrame(list(olah_summary.items()), columns=['Metode Pengolahan', 'Total Persen'])
            avg_olah_df['Rata-Rata Persentase'] = avg_olah_df['Total Persen'] / len(olah_data)
            
            fig = px.pie(avg_olah_df, names='Metode Pengolahan', values='Rata-Rata Persentase', 
                         title="Komposisi Rata-Rata Alur Pengolahan Sampah", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            download_data(avg_olah_df, "presentase_pengolahan_tps3r")

with tab2:
    st.header("Analisis Fasilitas dan Kendala Operasional")
    analyze_checkbox_question(df, '212', "Fasilitas yang Dimiliki TPS3R Saat Ini")
    st.markdown("---")
    analyze_checkbox_question(df, '225', "Kendala Utama yang Dihadapi TPS3R")

# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-TPS3R'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_tps3r")