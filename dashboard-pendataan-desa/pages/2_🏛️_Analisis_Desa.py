# pages/2_🏛️_Analisis_Desa.py
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
st.set_page_config(page_title="Analisis Desa", layout="wide")
st.title("🏛️ Analisis Kebijakan & Infrastruktur Desa")
st.markdown("Analisis dari formulir **DC-Desa**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

# Muat data spesifik untuk 'DC-Desa'
df_raw = load_data(db, "formSubmissions", "DC-Desa")
# Terapkan filter dari sidebar
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-Desa' yang cocok dengan filter yang dipilih.")
    st.stop()

# --- Analisis dalam Tabs ---
tab1, tab2, tab3 = st.tabs(["📜 Kebijakan & Regulasi", "💰 Pendanaan & Edukasi", "🏗️ Infrastruktur & Kendala"])

with tab1:
    st.header("Analisis Kebijakan, Regulasi, dan Penanggung Jawab")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '201', "Ketersediaan Regulasi Sampah (Perdes/SK)")
    with col2:
        create_pie_chart(df, '206', "Pihak Penanggung Jawab Pengelolaan Sampah")
    
    st.markdown("---")
    analyze_checkbox_question(df, '202', "Integrasi Pengelolaan Sampah dalam Dokumen Perencanaan")

with tab2:
    st.header("Analisis Pendanaan dan Program Edukasi")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '203', "Apakah Dibiayai dari APBDes?")
    with col2:
        create_pie_chart(df, '208', "Frekuensi Edukasi/Sosialisasi kepada Masyarakat")
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        create_pie_chart(df, '204', "Rata-rata Alokasi Dana per Tahun (Jika 'Ya')")
    with col4:
        analyze_checkbox_question(df, '209', "Pihak yang Pernah Dilibatkan dalam Edukasi")

with tab3:
    st.header("Analisis Ketersediaan Infrastruktur dan Kendala")
    st.subheader("Jumlah Sarana/Prasarana yang Tersedia (Kode: 211)")

    # Custom parsing untuk data grid-numeric infrastruktur
    if '211' in df.columns:
        infra_data = df['211'].dropna()
        infra_counts = {}
        for item in infra_data:
            if isinstance(item, dict) and 'default_row' in item:
                for facility, value_dict in item['default_row'].items():
                    if isinstance(value_dict, dict) and 'Angka' in value_dict:
                        count = pd.to_numeric(value_dict['Angka'], errors='coerce')
                        if pd.notna(count):
                            infra_counts[facility] = infra_counts.get(facility, 0) + count
        
        if infra_counts:
            infra_df = pd.DataFrame(list(infra_counts.items()), columns=['Fasilitas', 'Jumlah Total'])
            fig = px.bar(infra_df, x='Fasilitas', y='Jumlah Total', title="Akumulasi Sarana/Prasarana di Desa", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            download_data(infra_df, "jumlah_infrastruktur_desa")
        else:
            st.info("Data infrastruktur (211) tidak dalam format yang bisa diproses.")
            
    st.markdown("---")
    analyze_checkbox_question(df, '212', "Kendala Utama dalam Pengelolaan Sampah")

# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-Desa'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_desa")