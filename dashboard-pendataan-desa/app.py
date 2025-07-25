# dashboard-pendataan-desa/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from core_utils import init_firebase, load_data, download_data
from auth_utils import login_admin, logout

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dasbor Utama Pendataan",
    page_icon="🏡",
    layout="wide"
)

# --- Inisialisasi Koneksi ---
db_admin = init_firebase()      # Untuk data

# --- Fungsi untuk Menampilkan Halaman Login ---
def render_login_page():
    st.markdown("<h1 style='text-align: center;'>Selamat Datang Admin</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Silakan Login untuk Mengakses Dasbor</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="admin@desa.com")
            password = st.text_input("Password", type="password")
            st.markdown("##") # Spasi
            submit_button = st.form_submit_button(label="🔑 LOG IN", use_container_width=True)

            if submit_button:
                if not email or not password:
                    st.warning("Harap masukkan email dan password.")
                else:
                    with st.spinner("Memverifikasi..."):
                        login_admin(email, password)

# --- Fungsi untuk Menampilkan Dasbor Utama ---
def render_main_dashboard():
    st.sidebar.success(f"Login sebagai: **{st.session_state.get('user_email', 'Admin')}**")
    st.sidebar.button("Logout", on_click=logout, use_container_width=True, type="primary")
    st.sidebar.markdown("---")

    st.title("🏡 Dasbor Utama Analitik Pendataan Desa")
    st.markdown("Gambaran umum dari seluruh data yang terkumpul di semua formulir.")

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache dibersihkan. Memuat ulang data...")
        st.rerun()

    df_all = load_data(db_admin, "formSubmissions")

    if df_all.empty:
        st.warning("Tidak ada data yang ditemukan. Coba refresh atau periksa koneksi ke Firebase.")
        st.stop()
    
    st.header("Metrik Kunci Keseluruhan (KPIs)")
    total_submissions = len(df_all)
    num_forms = df_all['formTitle'].nunique()
    num_users = df_all['userId'].nunique()
    latest_submission = df_all['submittedAt'].max().strftime('%d %b %Y, %H:%M')

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Submisi", f"{total_submissions}")
    kpi2.metric("Jumlah Jenis Formulir", f"{num_forms}")
    kpi3.metric("Jumlah Pendata Aktif", f"{num_users}")
    kpi4.metric("Data Terakhir Masuk", latest_submission)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Submisi per Formulir")
        form_counts = df_all['formTitle'].value_counts().reset_index()
        form_counts.columns = ['Formulir', 'Jumlah']
        fig_pie = px.pie(form_counts, names='Formulir', values='Jumlah', hole=0.4, title="Proporsi Submisi")
        st.plotly_chart(fig_pie, use_container_width=True)
        download_data(form_counts, "distribusi_formulir")

    with col2:
        st.subheader("Aktivitas Submisi per Pendata (Top 10)")
        user_counts = df_all['userName'].value_counts().nlargest(10).reset_index()
        user_counts.columns = ['Nama Pendata', 'Jumlah Submisi']
        fig_bar = px.bar(user_counts, x='Nama Pendata', y='Jumlah Submisi', text_auto=True)
        fig_bar.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_bar, use_container_width=True)
        download_data(user_counts, "top_10_pendata")

    st.subheader("Tren Submisi Harian (Semua Formulir)")
    df_all['Tanggal'] = df_all['submittedAt'].dt.date
    daily_submissions = df_all.groupby('Tanggal').size().reset_index(name='Jumlah')
    fig_line = px.line(daily_submissions, x='Tanggal', y='Jumlah', markers=True, title="Jumlah Submisi Harian")
    st.plotly_chart(fig_line, use_container_width=True)
    download_data(daily_submissions, "tren_harian_global")

    st.markdown("---")
    st.header("📋 Data Mentah Gabungan")
    st.info("Pilih halaman analisis spesifik dari menu di samping untuk melihat detail dan filter yang lebih mendalam.")
    if st.checkbox("Tampilkan Tabel Data Mentah Gabungan"):
        st.dataframe(df_all)
        download_data(df_all, "data_mentah_gabungan")

# --- Logika Utama: Tampilkan Login atau Dasbor ---
if 'logged_in' in st.session_state and st.session_state.logged_in:
    render_main_dashboard()
else:
    render_login_page()