import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import pydeck as pdk
import base64
import os
import json
import copy
from shapely.geometry import Point, shape

# =============================================================================
# KONFIGURASI HALAMAN DAN JUDUL
# =============================================================================
st.set_page_config(
    page_title="Dashboard UMKM Kota Batu",
    page_icon="🏪",
    layout="wide"
)

# Menambahkan CSS Kustom untuk Tampilan Lebih Interaktif dan Seragam
st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f8f9fa; /* Sangat terang, hampir putih */
        color: #343a3d; /* Abu-abu gelap untuk teks utama */
    }

    /* Header Styling */
    h1 {
        color: #0056b3; /* Biru tua untuk primary header */
        font-size: 2.8em;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 3px solid #007bff; /* Biru cerah sebagai underline */
        margin-bottom: 30px;
        animation: fadeIn 1s ease-out; /* Animasi fade-in */
    }

    h2 {
        color: #007bff; /* Biru cerah untuk subheaders */
        font-size: 2.2em;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
        border-left: 5px solid #007bff; /* Border kiri untuk penekanan */
        padding-left: 10px;
        animation: slideInLeft 0.8s ease-out; /* Animasi slide dari kiri */
    }

    h3 {
        color: #007bff; /* Biru cerah untuk smaller headers */
        font-size: 1.8em;
        font-weight: 500;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }

    /* Markdown Styling */
    .stMarkdown {
        color: #495057; /* Abu-abu sedang */
        line-height: 1.6;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #e9ecef; /* Abu-abu muda */
        color: #0056b3;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        transition: background-color 0.3s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #dee2e6; /* Sedikit lebih gelap saat hover */
    }
    .streamlit-expanderContent {
        background-color: #ffffff; /* Putih */
        border: 1px solid #ced4da; /* Border abu-abu */
        border-radius: 8px;
        padding: 20px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }

    /* Button Styling */
    .stButton > button {
        background-color: #007bff; /* Biru cerah */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        border: none; /* Hilangkan border default */
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Biru tua saat hover */
        transform: translateY(-2px); /* Efek angkat */
    }

    /* Selectbox/Multiselect Styling */
    .stMultiSelect > div > div > div:first-child,
    .stSelectbox > div > div:first-child {
        border-radius: 8px;
        border: 1px solid #ced4da; /* Border abu-abu */
        padding: 5px;
        background-color: white;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); /* Sedikit shadow ke dalam */
    }
    .stRadio > label {
        color: #0056b3; /* Biru tua */
        font-weight: bold;
    }

    /* St.metric styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff; /* Putih */
        border: 1px solid #dee2e6; /* Border abu-abu */
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.08);
        transition: transform 0.3s ease-in-out;
        height: 100%;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
    }
    div[data-testid="stMetric"] > label {
        color: #0056b3; /* Biru tua untuk judul metrik */
        font-weight: bold;
        font-size: 1.1em;
    }
    div[data-testid="stMetric"] > div {
        color: #007bff; /* Biru cerah untuk nilai metrik */
        font-size: 2.0em;
        font-weight: bold;
    }
    div[data-testid="stMetric"] > div > div {
        color: #495057; /* Warna untuk sub-nilai metrik (jika ada) */
        font-size: 0.9em;
    }


    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; /* Jarak antar tab lebih rapat */
        justify-content: center; /* Pusatkan tab */
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #f1f8ff; /* Biru sangat muda */
        border-radius: 8px; /* Sudut membulat penuh */
        padding: 0px 25px;
        font-size: 1.1em;
        color: #0056b3; /* Biru tua */
        font-weight: bold;
        transition: background-color 0.3s, color 0.3s, transform 0.2s;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0f0ff; /* Biru lebih muda saat hover */
        color: #004085; /* Biru lebih tua */
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #007bff; /* Biru cerah untuk tab aktif */
        color: white;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.2); /* Shadow lebih kuat */
    }

    /* Plotly Chart Container */
    .stPlotlyChart {
        border: 1px solid #dee2e6; /* Border abu-abu */
        border-radius: 10px;
        padding: 10px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.08);
        background-color: white;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.08);
        border: 1px solid #dee2e6;
    }
    
    /* Legend Styling */
    .legend-item {
        display:flex; 
        align-items:center; 
        margin-bottom:8px;
        font-size: 0.95em;
        color: #495057;
    }
    .legend-color-box {
        width:22px; 
        height:22px; 
        border-radius:50%; 
        margin-right:12px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 1px 1px 3px rgba(0,0,0,0.05); /* Sedikit shadow pada kotak warna */
    }

    /* Animasi */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    /* Menerapkan animasi ke widget utama Streamlit */
    .stBlock, .stVerticalBlock, .stHorizontalBlock, .stBlock > div, .stVerticalBlock > div, .stHorizontalBlock > div {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Pastikan kolom memiliki tinggi yang sama untuk KPI dan metrik */
    div[data-testid*="stColumn"] {
        display: flex;
        flex-direction: column;
        justify-content: stretch;
    }
    div[data-testid*="stColumn"] > div {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    div[data-testid*="stColumn"] > div > div {
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🏪 Dashboard Analitik UMKM Kota Batu")
st.markdown("Dashboard ini menyajikan analisis sektoral, geografis, dan *clustering* UMKM dengan visualisasi batas wilayah yang interaktif dan informatif.")

# =============================================================================
# PEMETAAN NAMA KLUSTER DAN KONFIGURASI
# =============================================================================
# ALASAN PENAMAAN KLUSTER:
# Penamaan ini adalah hasil interpretasi analitis setelah algoritma K-Means selesai.
# Analis memeriksa komposisi setiap klaster (misal: dominasi sektor usaha, lokasi geografis)
# lalu memberikan nama yang deskriptif untuk mempermudah pemahaman.
# Contoh: Klaster yang banyak berisi hotel dan restoran di dekat tempat wisata diberi nama 'Penunjang Pariwisata'.
cluster_mapping = {
    '0': 'Potensi Berkembang', '1': 'UMKM Mikro', '2': 'Unggulan Lokal',
    '3': 'Penunjang Pariwisata', '4': 'Layanan Jasa', '5': 'Skala Menengah'
}

# =============================================================================
# FUNGSI-FUNGSI BANTUAN
# =============================================================================
@st.cache_data(ttl=3600)  # Cache data selama 1 jam untuk efisiensi
def load_and_correct_data(umkm_file_path, geojson_path_desa, geojson_path_kecamatan):
    """
    Fungsi utama untuk memuat data dan yang terpenting: MENGKOREKSI LOKASI.
    Fungsi ini melakukan 'Spatial Join' dua kali untuk memastikan setiap UMKM
    memiliki nama desa dan kecamatan yang akurat berdasarkan koordinatnya.
    """
    try:
        df_umkm = pd.read_csv(umkm_file_path)
        gdf_desa = gpd.read_file(geojson_path_desa)
        gdf_kecamatan = gpd.read_file(geojson_path_kecamatan)
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}. Pastikan semua file berada di direktori yang benar.")
        return None

    # 1. Bersihkan data UMKM dan siapkan untuk analisis geospasial
    df_umkm['latitude'] = pd.to_numeric(df_umkm['latitude'], errors='coerce')
    df_umkm['longitude'] = pd.to_numeric(df_umkm['longitude'], errors='coerce')
    df_umkm.dropna(subset=['latitude', 'longitude'], inplace=True)

    # 2. Ubah DataFrame UMKM menjadi GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_umkm['longitude'], df_umkm['latitude'])]
    gdf_umkm = gpd.GeoDataFrame(df_umkm, geometry=geometry, crs="EPSG:4326")

    # 3. Pastikan semua data menggunakan sistem proyeksi (CRS) yang sama
    if gdf_umkm.crs != gdf_desa.crs:
        gdf_desa = gdf_desa.to_crs(gdf_umkm.crs)
    if gdf_umkm.crs != gdf_kecamatan.crs:
        gdf_kecamatan = gdf_kecamatan.to_crs(gdf_umkm.crs)

    # 4. SPATIAL JOIN TAHAP 1: Mendapatkan NAMA DESA AKURAT
    gdf_merged_desa = gpd.sjoin(gdf_umkm, gdf_desa[['nm_kelurahan', 'geometry']], how="left", predicate='within')
    
    # 5. PERBAIKAN: Hapus kolom 'index_right' yang dibuat oleh sjoin pertama sebelum sjoin kedua
    if 'index_right' in gdf_merged_desa.columns:
        gdf_merged_desa = gdf_merged_desa.drop(columns=['index_right'])

    # 6. SPATIAL JOIN TAHAP 2: Mendapatkan NAMA KECAMATAN AKURAT
    gdf_merged_final = gpd.sjoin(gdf_merged_desa, gdf_kecamatan[['nm_kecamatan', 'geometry']], how="left", predicate='within')

    # 7. Buat kolom baru yang akurat dan bersihkan formatnya
    gdf_merged_final['nama_desa_akurat'] = gdf_merged_final['nm_kelurahan'].str.title().fillna('Tidak Terpetakan')
    gdf_merged_final['nama_kecamatan_akurat'] = gdf_merged_final['nm_kecamatan'].str.title().fillna('Tidak Terpetakan')
    
    # 8. Terapkan pemetaan nama kluster
    gdf_merged_final['cluster'] = gdf_merged_final['cluster'].astype(str)
    gdf_merged_final['nama_kluster'] = gdf_merged_final['cluster'].map(cluster_mapping).fillna('Lainnya')
    
    # 9. Hapus kolom-kolom bantu yang tidak diperlukan lagi
    cols_to_drop = ['geometry', 'index_right', 'nm_kelurahan', 'nm_kecamatan']
    for col in cols_to_drop:
        if col in gdf_merged_final.columns:
            gdf_merged_final = gdf_merged_final.drop(columns=col)
    
    return pd.DataFrame(gdf_merged_final)


@st.cache_data
def load_local_geojson(file_path):
    """Memuat data GeoJSON (tetap digunakan untuk peta batas wilayah)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Gagal memuat GeoJSON '{os.path.basename(file_path)}': {e}")
        return None

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def hex_to_rgb(hex_color):
    if isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7:
        try:
            return [int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]
        except ValueError:
            return [128, 128, 128]
    return [128, 128, 128]

def get_dominant_value(umkm_in_area, column_name):
    if umkm_in_area.empty or umkm_in_area[column_name].value_counts().empty:
        return "N/A"
    return umkm_in_area[column_name].value_counts().index[0]

def calculate_area_statistics(df_filtered, geojson_data, boundary_level, color_by):
    # Fungsi ini tetap sama seperti kode asli Anda
    if not geojson_data or df_filtered.empty: return geojson_data
    result_geojson = copy.deepcopy(geojson_data)
    area_prop_map = {'Kecamatan': 'nm_kecamatan', 'Kelurahan/Desa': 'nm_kelurahan', 'Kota': 'nm_dati2'}
    geojson_key = area_prop_map.get(boundary_level, 'nm_kelurahan')
    
    # Gunakan nama kolom yang sudah akurat
    df_filtered['geometry'] = [Point(xy) for xy in zip(df_filtered['longitude'], df_filtered['latitude'])]
    gdf_filtered = gpd.GeoDataFrame(df_filtered, geometry='geometry', crs="EPSG:4326")

    for feature in result_geojson['features']:
        polygon = shape(feature['geometry'])
        # Lakukan join spasial kecil di sini
        umkm_in_area_gdf = gdf_filtered[gdf_filtered.within(polygon)]
        
        jumlah_umkm = len(umkm_in_area_gdf)
        cluster_dominan = get_dominant_value(umkm_in_area_gdf, 'nama_kluster')
        sektor_dominan = get_dominant_value(umkm_in_area_gdf, 'nama_sektor')
        
        feature['properties']['jumlah_umkm'] = jumlah_umkm
        feature['properties']['cluster_dominan'] = cluster_dominan
        feature['properties']['sektor_dominan'] = sektor_dominan
        
        nama_wilayah = str(feature['properties'].get(geojson_key, 'N/A')).title()
        tooltip_base = f"""<div style='max-width:300px; background:rgba(0,0,0,0.8); color:white; padding:10px; border-radius:5px;'><div style='font-size:1.1em; font-weight:bold; margin-bottom:8px;'>🗺️ Wilayah: {nama_wilayah}</div><div style='margin-bottom:4px;'><span style='color:#FCD34D;'>📊 Jumlah UMKM:</span> <b>{jumlah_umkm}</b></div>"""
        tooltip_end = f"<div style='margin-bottom:4px;'><span style='color:#34D399;'>🧩 Kluster Dominan:</span> <b>{cluster_dominan}</b></div></div>" if color_by == 'Kluster' else f"<div style='margin-bottom:4px;'><span style='color:#F87171;'>🛍️ Sektor Dominan:</span> <b>{sektor_dominan}</b></div></div>"
        feature['properties']['tooltip_html'] = tooltip_base + tooltip_end
        
    return result_geojson

# =============================================================================
# PEMUATAN DATA UTAMA & PERSIAPAN APLIKASI
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
umkm_file_path = os.path.join(script_dir, "umkm_batu_clustered.csv")
geojson_base_path = os.path.join(script_dir, "35.79_Kota_Batu")
geojson_kelurahan_path = os.path.join(geojson_base_path, "35.79_kelurahan.geojson")
geojson_kecamatan_path = os.path.join(geojson_base_path, "35.79_kecamatan.geojson")

# Panggil fungsi baru untuk memuat dan mengkoreksi data
df = load_and_correct_data(umkm_file_path, geojson_kelurahan_path, geojson_kecamatan_path)


if df is not None:
    # =============================================================================
    # PANEL FILTER DAN PENGATURAN (MENGGUNAKAN DATA AKURAT)
    # =============================================================================
    st.markdown("---")
    with st.expander("⚙️ Atur Filter & Tampilan Dashboard", expanded=True):
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            # Gunakan kolom 'nama_kecamatan_akurat'
            kecamatan_list = sorted(df['nama_kecamatan_akurat'].unique())
            selected_kecamatan = st.multiselect("Pilih Kecamatan:", kecamatan_list, default=kecamatan_list)

        with col_filter2:
            df_desa_filtered_for_list = df[df['nama_kecamatan_akurat'].isin(selected_kecamatan)]
            # Gunakan kolom 'nama_desa_akurat'
            desa_list = sorted(df_desa_filtered_for_list['nama_desa_akurat'].unique())
            selected_desa = st.multiselect("Pilih Desa/Kelurahan:", desa_list, default=[])
        
        col_filter3, col_filter4 = st.columns(2)
        with col_filter3:
            sektor_list = sorted(df['nama_sektor'].unique())
            selected_sektor = st.multiselect("Pilih Sektor Usaha:", sektor_list, default=sektor_list)
        with col_filter4:
            kluster_list = sorted(df['nama_kluster'].unique())
            selected_kluster = st.multiselect("Pilih Kluster:", kluster_list, default=kluster_list)
            
        st.markdown("---")
        col_settings1, col_settings2, col_settings3 = st.columns(3)
        with col_settings1:
            boundary_level = st.radio("Detail Batas Wilayah:", ('Kelurahan/Desa', 'Kecamatan', 'Kota'), horizontal=True, key="boundary_level_main")
        with col_settings2:
            color_by = st.radio("Warnai Titik Berdasarkan:", ('Kluster', 'Sektor Usaha'), horizontal=True)
        with col_settings3:
            map_theme = st.radio("Tema Peta:", ("Terang", "Gelap"), horizontal=True)
            
    st.markdown("---")

    # Memuat GeoJSON yang sesuai untuk tampilan batas wilayah
    file_map = {'Kota': "35.79_Kota_Batu.geojson", 'Kecamatan': "35.79_kecamatan.geojson", 'Kelurahan/Desa': "35.79_kelurahan.geojson"}
    geojson_file_path = os.path.join(geojson_base_path, file_map[boundary_level])
    geojson_data = load_local_geojson(geojson_file_path)
    
    # Proses Filter Utama (menggunakan kolom akurat)
    df_filtered = df[df['nama_kecamatan_akurat'].isin(selected_kecamatan) & 
                     df['nama_sektor'].isin(selected_sektor) & 
                     df['nama_kluster'].isin(selected_kluster)]
    if selected_desa:
        df_filtered = df_filtered[df_filtered['nama_desa_akurat'].isin(selected_desa)]

    # =============================================================================
    # 📈 Ringkasan Analitik (KPI Cards)
    # =============================================================================
    st.header("📈 Ringkasan Analitik")
    total_umkm_keseluruhan, total_umkm_terfilter = len(df), len(df_filtered)
    persentase = (total_umkm_terfilter / total_umkm_keseluruhan * 100) if total_umkm_keseluruhan > 0 else 0
    kpi1_val, kpi1_subval = f"{total_umkm_terfilter:,}", f"({persentase:.1f}% dari total)"
    if not df_filtered.empty:
        sektor_top = df_filtered['nama_sektor'].value_counts().nlargest(1)
        kpi2_val, kpi2_subval = sektor_top.index[0], f"{sektor_top.iloc[0]:,} UMKM"
        cluster_top = df_filtered['nama_kluster'].value_counts().nlargest(1)
        kpi3_val, kpi3_subval = cluster_top.index[0], f"{cluster_top.iloc[0]:,} UMKM"
    else:
        kpi2_val, kpi2_subval, kpi3_val, kpi3_subval = "N/A", "Tidak ada data", "N/A", "Tidak ada data"
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1: st.metric(label="🏪 Total UMKM Terfilter", value=kpi1_val, delta=kpi1_subval, delta_color="off")
    with kpi_col2: st.metric(label="🛍️ Sektor Usaha Teratas", value=kpi2_val, delta=kpi2_subval)
    with kpi_col3: st.metric(label="🧩 Kluster Dominan", value=kpi3_val, delta=kpi3_subval)
    st.markdown("---")
    
    # =============================================================================
    # VISUALISASI UTAMA (MENGGUNAKAN TABS)
    # =============================================================================
    tab1, tab2, tab3 = st.tabs(["🗺️ Peta Interaktif (Titik & Batas)", "🎨 Peta Kepadatan UMKM", "📋 Ringkasan & Data"])

    with tab1:
        st.subheader(f"Peta Persebaran Interaktif (Batas Wilayah: {boundary_level})")
        if not df_filtered.empty:
            if color_by == 'Kluster':
                unique_keys, color_palette, merge_col = sorted(df_filtered['nama_kluster'].dropna().unique()), px.colors.qualitative.Plotly, 'nama_kluster'
            else:
                unique_keys, color_palette, merge_col = sorted(df_filtered['nama_sektor'].dropna().unique()), px.colors.qualitative.Light24, 'nama_sektor'
            
            color_lookup = pd.DataFrame({merge_col: unique_keys, 'color': [color_palette[i % len(color_palette)] for i in range(len(unique_keys))]})
            df_filtered_map = pd.merge(df_filtered, color_lookup, on=merge_col, how='left')
            df_filtered_map['color_rgb'] = df_filtered_map['color'].apply(hex_to_rgb)

            # Pydeck visualization (seperti kode asli Anda)
            umkm_geojson = {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                 "properties": {"color": row["color_rgb"], "radius": 30, "is_umkm_point": True,
                                "tooltip_html": f"""<div style='max-width:300px; background:rgba(0,0,0,0.8); color:white; padding:10px; border-radius:5px;'><div style='font-size:1.1em; font-weight:bold; margin-bottom:8px;'>🏪 Nama Usaha: {row.get('namausaha', 'N/A')}</div><div style='margin-bottom:4px;'><span style='color:#FCD34D;'>📊 Sektor:</span> {row.get('nama_sektor', 'N/A')}</div><div style='margin-bottom:4px;'><span style='color:#34D399;'>🧩 Kluster:</span> {row.get('nama_kluster', 'N/A')}</div></div>"""}}
                for _, row in df_filtered_map.iterrows()
            ]}
            boundary_geojson = calculate_area_statistics(df_filtered_map.copy(), geojson_data, boundary_level, color_by) if geojson_data else None
            view_state = pdk.ViewState(latitude=-7.87, longitude=112.52, zoom=11.5, pitch=50)
            mapbox_style = "mapbox://styles/mapbox/light-v10" if map_theme == "Terang" else "mapbox://styles/mapbox/dark-v10"
            
            layers = []
            if boundary_geojson:
                layers.append(pdk.Layer('GeoJsonLayer', data=boundary_geojson, stroked=True, filled=True, get_fill_color=[180, 180, 180, 20], get_line_color=[85, 85, 85, 255] if map_theme == "Terang" else [220, 220, 220, 255], get_line_width=20, pickable=True, auto_highlight=True))
            if umkm_geojson['features']:
                layers.append(pdk.Layer('GeoJsonLayer', data=umkm_geojson, get_fill_color='properties.color', get_radius='properties.radius', pickable=True, auto_highlight=True))
            
            if layers:
                st.pydeck_chart(pdk.Deck(map_style=mapbox_style, initial_view_state=view_state, layers=layers, tooltip={"html": "{tooltip_html}"}))
            else:
                st.warning("Tidak ada data untuk ditampilkan di peta dengan filter saat ini.")

            # Legenda Warna Dinamis
            if not df_filtered_map.empty:
                st.markdown("---"); st.subheader("🎨 Keterangan Warna Titik UMKM")
                color_map = {k: color_palette[i % len(color_palette)] for i, k in enumerate(unique_keys)}
                num_cols = min(len(unique_keys), 3)
                cols = st.columns(num_cols)
                for i, item in enumerate(unique_keys):
                    with cols[i % num_cols]:
                        st.markdown(f'<div class="legend-item"><div class="legend-color-box" style="background-color:{color_map[item]};"></div><span>{item}</span></div>', unsafe_allow_html=True)
        else:
            st.warning("Tidak ada data untuk ditampilkan di peta dengan filter saat ini.")

    with tab2:
        st.subheader("Peta Kepadatan Jumlah UMKM per Desa/Kelurahan")
        geojson_data_desa = load_local_geojson(geojson_kelurahan_path)
        
        if not df_filtered.empty and geojson_data_desa:
            # Gunakan kolom desa akurat untuk grouping
            umkm_per_desa = df_filtered.groupby('nama_desa_akurat')['namausaha'].count().reset_index()
            umkm_per_desa.columns = ['desa', 'Jumlah UMKM']

            # Peta Choropleth (seperti kode asli Anda)
            fig_choro = px.choropleth(
                umkm_per_desa,
                geojson=geojson_data_desa,
                locations='desa',
                featureidkey="properties.nm_kelurahan",
                color='Jumlah UMKM',
                color_continuous_scale="Blues",
                scope="asia",
                hover_name='desa',
                hover_data={'Jumlah UMKM': True, 'desa': False}
            )
            fig_choro.update_geos(
                center={"lon": 112.53, "lat": -7.87},
                lataxis_range=[-8.0, -7.7],
                lonaxis_range=[112.4, 112.7],
                visible=False
            )
            fig_choro.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_choro, use_container_width=True)
        else:
            st.warning("Tidak ada data atau GeoJSON Kelurahan/Desa untuk membuat peta kepadatan.")
    
    with tab3:
        st.subheader("Ringkasan Statistik untuk Desa/Kelurahan Terpilih")
        if not df_filtered.empty and selected_desa:
            for desa in selected_desa:
                # Filter berdasarkan kolom desa akurat
                df_desa_tab = df_filtered[df_filtered['nama_desa_akurat'] == desa]
                if not df_desa_tab.empty:
                    st.markdown(f"#### {desa.title()}"); col_sum1, col_sum2 = st.columns(2)
                    col_sum1.metric("Total UMKM di Desa Ini", f"{len(df_desa_tab)} usaha")
                    col_sum2.metric("Sektor Terbanyak", get_dominant_value(df_desa_tab, 'nama_sektor'))
                    st.markdown("---")
        else:
            st.info("Pilih satu atau beberapa desa/kelurahan dari panel filter untuk melihat ringkasannya di sini.")
            
        st.subheader("Grafik Analitik Sektor & Kluster")
        col_a, col_b = st.columns(2)
        with col_a:
            if not df_filtered.empty:
                sektor_counts = df_filtered['nama_sektor'].value_counts().nlargest(15).sort_values(ascending=True)
                fig_sektor = px.bar(sektor_counts, y=sektor_counts.index, x=sektor_counts.values, orientation='h', title="Top 15 Sektor Usaha", labels={'x': 'Jumlah UMKM', 'y': ''}, text_auto=True, template="streamlit", color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig_sektor.update_layout(showlegend=False, yaxis_title=None), use_container_width=True)
            else: st.info("Tidak ada data sektor usaha.")
        with col_b:
            if not df_filtered.empty:
                cluster_counts = df_filtered['nama_kluster'].value_counts()
                fig_cluster = px.bar(cluster_counts, y=cluster_counts.values, x=cluster_counts.index, title="Jumlah UMKM per Kluster", labels={'x': '', 'y': 'Jumlah UMKM'}, text_auto=True, template="streamlit", color_discrete_sequence=px.colors.sequential.Greens_r)
                st.plotly_chart(fig_cluster.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'}), use_container_width=True)
            else: st.info("Tidak ada data kluster.")
            
        st.markdown("---")
        st.subheader("📄 Data Tabel Lengkap (Sesuai Filter)")
        if not df_filtered.empty:
            st.markdown(get_table_download_link(df_filtered, "umkm_terfilter.csv", "📥 Unduh CSV Data Terfilter"), unsafe_allow_html=True)
            # Tampilkan kolom-kolom yang paling relevan
            display_cols = ['namausaha', 'nama_sektor', 'nama_kluster', 'nama_desa_akurat', 'nama_kecamatan_akurat', 'kegiatan', 'latitude', 'longitude']
            st.dataframe(df_filtered[display_cols], use_container_width=True, height=500)
        else:
            st.info("Tidak ada data UMKM yang cocok dengan filter yang dipilih.")
else:
    st.error("Gagal memuat dan memproses data UMKM. Pastikan file 'umkm_batu_clustered.csv' dan folder '35.79_Kota_Batu' ada di direktori yang sama dengan skrip ini.")
    
st.markdown("---")
st.info("💡 **Tips Interaktif:** Gunakan panel filter untuk menganalisis data. Arahkan kursor ke titik atau wilayah pada peta untuk melihat detailnya.")