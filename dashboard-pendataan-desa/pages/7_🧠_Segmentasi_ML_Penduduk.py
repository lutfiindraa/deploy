# pages/7_🧠_Segmentasi_ML_Penduduk.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
# [PERBAIKAN] Menggunakan core_utils, bukan data_utils atau auth
from core_utils import init_firebase, load_data, render_sidebar, download_data

if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🔒 Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop() # Menghentikan eksekusi sisa halaman

st.set_page_config(page_title="Segmentasi ML", layout="wide")

st.title("🧠 Machine Learning: Segmentasi Rumah Tangga")
st.markdown("""
Halaman ini menggunakan model **K-Means Clustering** untuk mengelompokkan rumah tangga ke dalam segmen-segmen berdasarkan karakteristik sosial-ekonomi dan perilaku pengelolaan sampah mereka. Tujuannya adalah untuk mengidentifikasi profil rumah tangga yang berbeda, sehingga program intervensi dapat dirancang lebih tepat sasaran.
""")

# --- Konfigurasi & Load Data ---
db = init_firebase()
if not db:
    st.error("Koneksi Firebase gagal.")
    st.stop()

df_raw = load_data(db, "formSubmissions", "DC-Penduduk")
# Tidak menggunakan sidebar filter di halaman ini agar analisisnya konsisten
if df_raw.empty:
    st.warning("Data 'DC-Penduduk' tidak ditemukan atau kosong. Analisis ML tidak dapat dilanjutkan.")
    st.stop()

# --- Feature Engineering & Selection ---
features = {
    '202': 'categorical', # Ijazah KRT
    '204': 'numeric',     # Jumlah ART bekerja
    '301': 'categorical', # Pemilahan sampah
    '307': 'categorical'  # Kesediaan ikut pelatihan
}
# Pastikan semua kolom fitur ada sebelum melanjutkan
required_cols = list(features.keys())
if not all(col in df_raw.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df_raw.columns]
    st.error(f"Kolom yang dibutuhkan untuk analisis ML tidak ditemukan: {', '.join(missing)}")
    st.stop()
    
df = df_raw[required_cols].copy().dropna()
# Konversi kolom numerik
for col, type in features.items():
    if type == 'numeric':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()


if len(df) < 10:
    st.error(f"Data yang valid tidak cukup untuk melakukan analisis clustering (hanya {len(df)} baris).")
    st.stop()

# --- Pipeline & Model Training ---
st.sidebar.header("Kontrol Model")
n_clusters = st.sidebar.slider("Pilih Jumlah Segmen (Cluster)", min_value=2, max_value=6, value=3, step=1)

numeric_features = [k for k, v in features.items() if v == 'numeric']
categorical_features = [k for k, v in features.items() if v == 'categorical']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))])
df['cluster'] = pipeline.fit_predict(df)

# --- Visualisasi Hasil Clustering (dengan PCA) ---
st.header("Visualisasi Segmen Rumah Tangga")

pca_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('pca', PCA(n_components=2))])
df_pca = pca_pipeline.fit_transform(df.drop('cluster', axis=1))
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_pca['cluster'] = df['cluster'].values
df_pca['cluster_str'] = df_pca['cluster'].astype(str)

fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='cluster_str',
                     title=f'Visualisasi {n_clusters} Segmen Rumah Tangga (via PCA)',
                     hover_data={'cluster_str': True})
fig_pca.update_layout(legend_title_text='Segmen')
st.plotly_chart(fig_pca, use_container_width=True)
st.info("Setiap titik mewakili satu rumah tangga. Titik dengan warna yang sama berada dalam satu segmen. Jarak antar titik menunjukkan kemiripan karakteristik mereka.")
download_data(df, "data_segmentasi_ml")

# --- Interpretasi & Insight per Cluster ---
st.header("Profil dan Insight per Segmen")
for i in sorted(df['cluster'].unique()):
    with st.expander(f"**Analisis Segmen {i}**"):
        cluster_df = df[df['cluster'] == i]
        st.write(f"**Jumlah Rumah Tangga:** {len(cluster_df)} ({len(cluster_df)/len(df):.1%})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Karakteristik Utama:**")
            for col in numeric_features:
                st.write(f"- Rata-rata Anggota Bekerja (204): **{cluster_df[col].mean():.2f}**")
            for col in categorical_features:
                mode_val = cluster_df[col].mode()[0]
                st.write(f"- {col} (dominan): **{mode_val}**")
        with col2:
            st.write("**Distribusi Pendidikan KRT (202):**")
            st.dataframe(cluster_df['202'].value_counts(normalize=True).mul(100).round(1))

        # --- Insight yang Dapat Ditindaklanjuti ---
        st.subheader("💡 Actionable Insight")
        kesediaan_dominan = cluster_df['307'].mode()[0]
        if kesediaan_dominan.lower() == 'ya':
            insight_text = f"**Segmen {i}** menunjukkan **kemauan tinggi** untuk berpartisipasi dalam pelatihan. Ini adalah target utama untuk program edukasi pengelolaan sampah lanjutan seperti komposting atau pembuatan produk daur ulang."
        else:
            insight_text = f"**Segmen {i}** cenderung **tidak bersedia** mengikuti pelatihan. Fokus intervensi untuk segmen ini sebaiknya pada penyuluhan dasar dan penyediaan fasilitas (tong sampah terpilah) untuk membangun kesadaran awal, bukan program pelatihan yang kompleks."
        st.success(insight_text)