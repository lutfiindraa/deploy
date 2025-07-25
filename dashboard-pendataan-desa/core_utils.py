# File: core_utils.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import os
import traceback
from streamlit.errors import StreamlitSecretNotFoundError

# --- Fungsi Inisialisasi Firebase (Final) ---
@st.cache_resource
def init_firebase():
    """Menginisialisasi koneksi Firebase Admin SDK menggunakan service account."""
    try:
        # Cek untuk menghindari inisialisasi ganda
        if not firebase_admin._apps:
            try:
                # Coba ambil dari st.secrets dan langsung ubah jadi dictionary
                cred_dict = dict(st.secrets["firebase_service_account"])
                
                # Jika berhasil, kita di mode deployed
                st.session_state.is_deployed = True
                cred = credentials.Certificate(cred_dict)
                st.info("Koneksi Firebase menggunakan Streamlit Secrets (Mode Deployed).")
            
            except StreamlitSecretNotFoundError:
                # Jika secrets tidak ditemukan, kita di mode lokal
                st.session_state.is_deployed = False
                base_dir = os.path.dirname(os.path.abspath(__file__))
                local_key_path = os.path.join(base_dir, "serviceAccountKey.json")
                
                if not os.path.exists(local_key_path):
                    st.error(f"File kredensial '{local_key_path}' tidak ditemukan untuk mode lokal.")
                    st.caption("Pastikan file 'serviceAccountKey.json' berada di folder yang sama dengan 'app.py'.")
                    return None
                
                cred = credentials.Certificate(local_key_path)
                st.info(f"Koneksi Firebase menggunakan file lokal '{local_key_path}'.")
            
            firebase_admin.initialize_app(cred)
            
        return firestore.client()
    except Exception as e:
        st.error(f"Gagal menginisialisasi Firebase: {e}")
        st.info(f"Traceback: {traceback.format_exc()}")
        return None

# --- Fungsi Memuat & Memproses Data ---
@st.cache_data(ttl=300) # Cache data selama 5 menit
def load_data(_db_conn, collection_name, form_title_filter=None):
    """
    Memuat data dari Firestore, memprosesnya, dan mengubahnya menjadi DataFrame format 'wide'.
    Jika form_title_filter=None, memuat semua data.
    """
    if not _db_conn:
        return pd.DataFrame()
        
    try:
        query = _db_conn.collection(collection_name)
        if form_title_filter:
            query = query.where("formTitle", "==", form_title_filter)
        
        docs = query.stream()
        
        all_submissions = []
        for doc in docs:
            doc_data = doc.to_dict()
            flat_data = {
                'doc_id': doc.id,
                'formTitle': doc_data.get('formTitle', 'N/A'),
                'userId': doc_data.get('userId', 'N/A'),
                'userName': doc_data.get('userName', 'N/A'),
                'formId': doc_data.get('formId', 'N/A'),
                'submittedAt': doc_data.get('submittedAt'),
                'updatedAt': doc_data.get('updatedAt')
            }
            
            answers = doc_data.get('answers', [])
            for answer_item in answers:
                if isinstance(answer_item, dict):
                    q_code = answer_item.get('questionCode')
                    answer = answer_item.get('answer')
                    if q_code:
                        flat_data[q_code] = answer
            
            all_submissions.append(flat_data)

        if not all_submissions:
            return pd.DataFrame()

        df = pd.DataFrame(all_submissions)

        for col in ['submittedAt', 'updatedAt']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)

        return df.sort_values(by='submittedAt', ascending=False)
        
    except Exception as e:
        st.error(f"Kesalahan saat memuat data: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# --- Fungsi Utilitas untuk Tombol Unduh ---
def download_data(df, title):
    """Menyediakan tombol unduh untuk DataFrame."""
    filename = f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Unduh Data '{title}'",
        data=csv_data,
        file_name=filename,
        mime='text/csv',
        key=f"download_{filename}"
    )

# --- Fungsi untuk Visualisasi Umum ---
def create_pie_chart(df, column, title, hole=0.4):
    """Membuat pie chart dari kolom DataFrame."""
    if column not in df.columns or df[column].isnull().all():
        st.warning(f"Tidak ada data valid di kolom '{column}' untuk chart '{title}'.")
        return
    counts = df[column].value_counts().reset_index()
    counts.columns = ['kategori', 'jumlah']
    fig = px.pie(counts, names='kategori', values='jumlah', title=title, hole=hole)
    st.plotly_chart(fig, use_container_width=True)
    download_data(counts, title)

def create_bar_chart(df, column, title, is_horizontal=False):
    """Membuat bar chart dari kolom DataFrame."""
    if column not in df.columns or df[column].isnull().all():
        st.warning(f"Tidak ada data valid di kolom '{column}' untuk chart '{title}'.")
        return
    counts = df[column].value_counts().reset_index()
    counts.columns = ['kategori', 'jumlah']
    if is_horizontal:
        fig = px.bar(counts, y='kategori', x='jumlah', title=title, orientation='h', text_auto=True)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig = px.bar(counts, x='kategori', y='jumlah', title=title, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    download_data(counts, title)

def analyze_checkbox_question(df, column, title):
    """Menganalisis kolom yang berasal dari pertanyaan checkbox (berisi list)."""
    if column not in df.columns or df[column].isnull().all():
        st.warning(f"Tidak ada data valid di kolom '{column}' untuk analisis '{title}'.")
        return
    
    data_exploded = df[[column]].copy().dropna(subset=[column]).explode(column)
    counts = data_exploded[column].value_counts().reset_index()
    counts.columns = ['kategori', 'jumlah_sebutan']
    
    st.subheader(title)
    fig = px.bar(counts, y='kategori', x='jumlah_sebutan', orientation='h', text_auto=True)
    fig.update_layout(yaxis_title="Opsi Jawaban", xaxis_title="Jumlah Sebutan", yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    download_data(counts, title)

# --- Fungsi untuk Sidebar ---
def render_sidebar(df):
    """Merender sidebar dengan filter-filter umum."""
    st.sidebar.header("⚙️ Filter Global")
    
    if df.empty or 'submittedAt' not in df.columns or df['submittedAt'].isnull().all():
        st.sidebar.warning("Tidak ada data valid untuk difilter.")
        return df

    min_date = df['submittedAt'].min().date()
    max_date = df['submittedAt'].max().date()
    
    date_range = st.sidebar.date_input(
        "Rentang Tanggal Submit",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_filter"
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        df = df[(df['submittedAt'] >= start_datetime) & (df['submittedAt'] <= end_datetime)]
    
    if '101' in df.columns:
        all_dusuns = sorted(df['101'].dropna().unique())
        if all_dusuns:
            selected_dusuns = st.sidebar.multiselect(
                "Pilih Dusun",
                options=all_dusuns,
                default=all_dusuns,
                key="dusun_filter"
            )
            df = df[df['101'].isin(selected_dusuns)]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Menampilkan **{len(df)}** baris data setelah filter.")
    return df