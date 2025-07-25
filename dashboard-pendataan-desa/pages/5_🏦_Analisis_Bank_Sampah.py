# pages/5_🏦_Analisis_Bank_Sampah.py
import streamlit as st
import pandas as pd
import plotly.express as px
from core_utils import init_firebase, load_data, render_sidebar, create_pie_chart, download_data


if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman
    
# --- Fungsi Helper Spesifik Halaman ---
def parse_monthly_grid(df, column_name, value_prefix):
    """Mem-parsing grid bulanan dan mengembalikannya sebagai DataFrame deret waktu."""
    if column_name not in df.columns:
        return pd.DataFrame()
    
    monthly_data = []
    for index, row in df.iterrows():
        item = row[column_name]
        if isinstance(item, dict) and 'default_row' in item:
            for month, value_dict in item['default_row'].items():
                if isinstance(value_dict, dict):
                    value_key = next((key for key in value_dict if value_prefix in key), None)
                    if value_key:
                        value = pd.to_numeric(value_dict[value_key], errors='coerce')
                        if pd.notna(value):
                            monthly_data.append({'Bulan': month, 'Nilai': value})
    if not monthly_data:
        return pd.DataFrame()

    summary_df = pd.DataFrame(monthly_data)
    summary_df['BulanDate'] = pd.to_datetime(summary_df['Bulan'], format='%B %Y', errors='coerce')
    return summary_df.dropna(subset=['BulanDate']).sort_values('BulanDate')

# --- Konfigurasi & Load Data ---
st.set_page_config(page_title="Analisis Bank Sampah", layout="wide")
st.title("🏦 Analisis Kinerja Bank Sampah")
st.markdown("Analisis dari formulir **DC-Bank Sampah**.")

db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

df_raw = load_data(db, "formSubmissions", "DC-Bank Sampah")
df = render_sidebar(df_raw)

if df.empty:
    st.warning("Tidak ada data 'DC-Bank Sampah' yang cocok dengan filter yang dipilih.")
    st.stop()

# --- KPIs Nasabah ---
st.header("Statistik Kinerja Nasabah")
numeric_cols = ['301', '302']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

total_nasabah = df['301'].sum() if '301' in df else 0
nasabah_aktif = df['302'].sum() if '302' in df else 0
persen_aktif = (nasabah_aktif / total_nasabah * 100) if total_nasabah > 0 else 0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Nasabah Tercatat", f"{int(total_nasabah)}")
kpi2.metric("Total Nasabah Aktif", f"{int(nasabah_aktif)}")
kpi3.metric("Tingkat Keaktifan Nasabah", f"{persen_aktif:.1f}%")

# --- Analisis dalam Tabs ---
tab1, tab2 = st.tabs(["📈 Tren Bulanan", "⚙️ Operasional & Sarana"])

with tab1:
    st.header("Tren Volume Sampah Dikelola & Nilai Penjualan per Bulan")
    
    # Tren Volume
    volume_df = parse_monthly_grid(df, '304', 'Angka (dalam kg)')
    if not volume_df.empty:
        st.subheader("Rata-rata Volume Sampah Dikelola (kg)")
        fig_vol = px.line(volume_df, x='BulanDate', y='Nilai', markers=True, title="Tren Volume Sampah Bulanan")
        fig_vol.update_layout(xaxis_title="Bulan", yaxis_title="Volume (kg)")
        st.plotly_chart(fig_vol, use_container_width=True)
        download_data(volume_df[['Bulan', 'Nilai']], "tren_volume_bank_sampah")

    # Tren Penjualan
    penjualan_df = parse_monthly_grid(df, '305', 'Angka (dalam kg)') # Asumsi nama kolom, sesuaikan jika beda
    if not penjualan_df.empty:
        st.subheader("Rata-rata Nilai Penjualan Sampah")
        fig_penj = px.line(penjualan_df, x='BulanDate', y='Nilai', markers=True, title="Tren Nilai Penjualan Bulanan")
        fig_penj.update_layout(xaxis_title="Bulan", yaxis_title="Nilai Penjualan")
        st.plotly_chart(fig_penj, use_container_width=True)
        download_data(penjualan_df[['Bulan', 'Nilai']], "tren_penjualan_bank_sampah")

with tab2:
    st.header("Analisis Operasional dan Sarana Prasarana")
    col1, col2 = st.columns(2)
    with col1:
        create_pie_chart(df, '303', "Frekuensi Operasional Bank Sampah")
    with col2:
        create_pie_chart(df, '307', "Ketersediaan Sarana & Prasarana")
        
    st.markdown("---")
    create_pie_chart(df, '308', "Ketersediaan Pencatatan (Buku Kas/Sampah)")

# Menampilkan data mentah di akhir
st.markdown("---")
st.header("📋 Data Mentah (Sesuai Filter)")
if st.checkbox("Tampilkan Data Mentah 'DC-Bank Sampah'"):
    st.dataframe(df)
    download_data(df, "data_mentah_dc_bank_sampah")