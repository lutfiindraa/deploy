import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import pydeck as pdk
import numpy as np
import base64
import os
import json
from shapely.geometry import Point, shape
from sklearn.ensemble import RandomForestRegressor
from pysal.lib import weights
from pysal.explore import esda
from io import BytesIO

# =============================================================================
# KONFIGURASI HALAMAN DAN JUDUL
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analitik & Prediktif UMKM Kota Batu",
    page_icon="🏢",
    layout="wide"
)

# Menambahkan CSS Kustom
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

    /* === PERBAIKAN AKHIR UNTUK st.metric FONT SIZE === */
    /* Target kontainer utama st.metric */
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

    /* MENINGKATKAN UKURAN FONT UNTUK LABEL METRIK (Judul: "Total UMKM Terfilter", "Sektor Usaha Teratas") */
    div[data-testid="stMetricLabel"] {
        color: #0056b3; /* Biru tua untuk judul metrik */
        font-weight: bold;
        font-size: 2.5em !important; /* UKURAN SANGAT BESAR untuk label, diperbesar lagi */
        line-height: 1.2em !important; /* Sesuaikan tinggi baris */
        padding-bottom: 0.1em !important; /* Sedikit spasi di bawah label */
    }

    /* MENGATUR UKURAN FONT UNTUK NILAI UTAMA METRIK (Angka: "32,649"; Teks: "Perdagangan Besar dan Eceran") */
    div[data-testid="stMetricValue"] {
        color: #007bff; /* Biru cerah untuk nilai metrik */
        font-size: 1.6em !important; /* UKURAN MEDIUM untuk nilai utama, LEBIH KECIL dari label, dengan !important */
        line-height: 1.2em !important; /* Sesuaikan tinggi baris */
        font-weight: bold;
    }

    /* MENGATUR UKURAN FONT UNTUK DELTA / SUB-NILAI METRIK (misal: "13,991 UMKM" pada Sektor Teratas) */
    div[data-testid="stMetricDelta"] {
        color: #495057; /* Warna abu-abu untuk sub-nilai */
        font-size: 1.0em !important; /* UKURAN KECIL untuk sub-nilai, dengan !important */
        margin-top: 0.2em !important; /* Sedikit spasi di atas delta */
    }
    /* Akhir dari perbaikan st.metric */


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
""", unsafe_allow_html=True)

st.title("Dashboard Analitik & Prediktif UMKM Kota Batu")
st.markdown(
    "<h2 style='text-align: center;'>Analisis Deskriptif, Spasial, dan Prediktif Lokasi UMKM</h2>",
    unsafe_allow_html=True
)

# =============================================================================
# DATA SIMULASI & PEMETAAN KODE
# =============================================================================
POI_WISATA = {
    'Jatim Park 2': Point(112.529, -7.888),
    'Museum Angkut': Point(112.522, -7.883),
    'Alun-Alun Kota Batu': Point(112.527, -7.871),
    'BNS (Batu Night Spectacular)': Point(112.536, -7.900),
    'Selecta': Point(112.553, -7.818)
}

KECAMATAN_MAP = {'Batu': '[010] Batu', 'Junrejo': '[020] Junrejo', 'Bumiaji': '[030] Bumiaji'}
DESA_MAP = {
    'Oro-Oro Ombo': '[001] Oro-oro Ombo', 'Temas': '[002] Temas', 'Sisir': '[003] Sisir', 'Ngaglik': '[004] Ngaglik',
    'Pesanggrahan': '[005] Pesanggrahan', 'Songgokerto': '[006] Songgokerto', 'Sumberejo': '[007] Sumberejo', 'Sidomulyo': '[008] Sidomulyo',
    'Tlekung': '[001] Tlekung', 'Junrejo': '[002] Junrejo', 'Mojorejo': '[003] Mojorejo', 'Torongrejo': '[004] Torongrejo',
    'Beji': '[005] Beji', 'Pendem': '[006] Pendem', 'Dadaprejo': '[007] Dadaprejo', 'Pandanrejo': '[001] Pandanrejo',
    'Bumiaji': '[002] Bumiaji', 'Bulukerto': '[003] Bulukerto', 'Gunungsari': '[004] Gunungsari', 'Punten': '[005] Punten',
    'Tulungrejo': '[006] Tulungrejo', 'Sumbergondo': '[007] Sumbergondo', 'Giripurno': '[008] Giripurno', 'Sumberbrantas': '[009] Sumberbrantas'
}

# =============================================================================
# FUNGSI-FUNGSI BANTUAN & ANALISIS
# =============================================================================

@st.cache_data
def load_local_geojson(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        st.error(f"Gagal memuat GeoJSON '{os.path.basename(file_path)}': {e}"); return None

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton" style="display:inline-block; text-align:center; text-decoration:none; color:white; background-color:#007bff; padding:10px 20px; border-radius:8px; font-weight:bold;">{text}</a>'

def to_excel_download_link(df, filename, text):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="stButton" style="display:inline-block; text-align:center; text-decoration:none; color:white; background-color:#28a745; padding:10px 20px; border-radius:8px; font-weight:bold;">{text}</a>'


def hex_to_rgb(hex_color):
    if isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7:
        try:
            return [int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]
        except ValueError:
            return [128, 128, 128]
    return [128, 128, 128]

def get_dominant_value(umkm_in_area, column_name):
    if umkm_in_area.empty or umkm_in_area[column_name].value_counts().empty: return "N/A"
    return umkm_in_area[column_name].value_counts().index[0]

def load_and_process_initial_data(umkm_file_path, geojson_path_desa, geojson_path_kecamatan, _poi_data):
    try:
        with st.spinner("Memuat data UMKM dan GeoJSON..."):
            cols_to_use = ['namausaha', 'kegiatan', 'latitude', 'longitude', 'nama_sektor', 'kec', 'desa'] 
            if umkm_file_path.endswith('.csv'):
                df_umkm = pd.read_csv(umkm_file_path, usecols=lambda c: c in cols_to_use)
            elif umkm_file_path.endswith('.xlsx'):
                df_umkm = pd.read_excel(umkm_file_path, usecols=lambda c: c in cols_to_use)
            else:
                st.error("Format file UMKM default tidak didukung. Harap gunakan .csv atau .xlsx.")
                return None
            
            gdf_desa = gpd.read_file(geojson_path_desa)
            gdf_kecamatan = gpd.read_file(geojson_kecamatan_path)
        
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Error memuat data: {e}. Pastikan file UMKM default dan GeoJSON ada di direktori yang benar.")
        return None

    df_umkm.dropna(subset=['latitude', 'longitude'], inplace=True)
    gdf_umkm = gpd.GeoDataFrame(df_umkm, geometry=gpd.points_from_xy(df_umkm.longitude, df_umkm.latitude), crs="EPSG:4326")

    with st.spinner("Melakukan spatial join (pemetaan desa/kecamatan)..."):
        gdf_umkm = gpd.sjoin(gdf_umkm, gdf_desa[['nm_kelurahan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
        gdf_umkm = gpd.sjoin(gdf_umkm, gdf_kecamatan[['nm_kecamatan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
    
    gdf_umkm['nama_desa_akurat'] = gdf_umkm['nm_kelurahan'].str.title().fillna('Tidak Terpetakan')
    gdf_umkm['nama_kecamatan_akurat'] = gdf_umkm['nm_kecamatan'].str.title().fillna('Tidak Terpetakan')
    
    gdf_umkm['kecamatan_display'] = gdf_umkm['nama_kecamatan_akurat'].map(KECAMATAN_MAP).fillna(gdf_umkm['nama_kecamatan_akurat'])
    gdf_umkm['desa_display'] = gdf_umkm['nama_desa_akurat'].map(DESA_MAP).fillna(gdf_umkm['nama_desa_akurat'])

    with st.spinner("Menghitung jarak UMKM ke POI terdekat... (Ini mungkin membutuhkan waktu)"):
        gdf_poi = gpd.GeoDataFrame(geometry=[v for k, v in _poi_data.items()], crs="EPSG:4326")
        gdf_umkm_proj = gdf_umkm.to_crs("EPSG:32749")
        gdf_poi_proj = gdf_poi.to_crs("EPSG:32749")
        
        gdf_umkm['jarak_ke_poi_terdekat_m'] = gdf_umkm_proj.geometry.apply(
            lambda umkm_point: gdf_poi_proj.distance(umkm_point).min()
        )

    return gdf_umkm.drop(columns=['nm_kelurahan', 'nm_kecamatan', 'kec', 'desa'], errors='ignore')

@st.cache_data
def calculate_hotspots(_gdf_desa, _df_umkm):
    with st.spinner("Memproses Analisis Hotspot dan Coldspot..."):
        gdf_desa_copy = _gdf_desa.copy()
        gdf_desa_copy['nm_kelurahan'] = gdf_desa_copy['nm_kelurahan'].str.title()

        umkm_per_desa = _df_umkm.groupby('nama_desa_akurat').size().reset_index(name='jumlah_umkm')
        
        gdf_desa_stats = gdf_desa_copy.merge(umkm_per_desa, left_on='nm_kelurahan', right_on='nama_desa_akurat', how='left')
        gdf_desa_stats['jumlah_umkm'] = gdf_desa_stats['jumlah_umkm'].astype(float).fillna(0.0) 

        try:
            w = weights.Queen.from_dataframe(gdf_desa_stats)
            if w.islands:
                st.warning("Peringatan: Beberapa desa terputus dari jaringan spasial (tidak memiliki tetangga). Analisis hotspot mungkin tidak akurat untuk area tersebut.")
        except ValueError as e:
            st.error(f"Kesalahan dalam menghitung bobot spasial: {e}. Pastikan file GeoJSON memiliki konektivitas spasial yang memadai antar poligon.")
            return gdf_desa_stats.assign(hotspot_label='Tidak Signifikan')

        w.transform = 'r' 

        if len(gdf_desa_stats) > 1 and gdf_desa_stats['jumlah_umkm'].nunique() > 1:
            g_local = esda.G_Local(gdf_desa_stats['jumlah_umkm'], w)
            
            gdf_desa_stats['hotspot_label'] = 'Tidak Signifikan'
            gdf_desa_stats.loc[(g_local.Zs > 1.96) & (g_local.p_sim < 0.05), 'hotspot_label'] = 'Hotspot (Sentra Bisnis)'
            gdf_desa_stats.loc[(g_local.Zs < -1.96) & (g_local.p_sim < 0.05), 'hotspot_label'] = 'Coldspot (Area Sepi)'
        else:
            st.info("Tidak cukup variasi data UMKM per desa untuk melakukan analisis hotspot/coldspot yang signifikan.")
            gdf_desa_stats['hotspot_label'] = 'Tidak Signifikan'
        
        return gdf_desa_stats

@st.cache_resource
def train_suitability_model(_df, sector):
    df_sector = _df[_df['nama_sektor'] == sector].copy()
    if len(df_sector) < 10: return None

    with st.spinner(f"Melatih model AI untuk sektor '{sector}'..."):
        df_sector['success_score'] = 1 / (df_sector['jarak_ke_poi_terdekat_m'] + 1)
        min_score = df_sector['success_score'].min()
        max_score = df_sector['success_score'].max()
        if max_score > min_score:
            df_sector['success_score'] = (df_sector['success_score'] - min_score) / (max_score - min_score)
        else:
            df_sector['success_score'] = 0.5
        
        X = df_sector[['latitude', 'longitude', 'jarak_ke_poi_terdekat_m']]
        y = df_sector['success_score']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
        model.fit(X, y)
        return model

@st.cache_data
def create_prediction_grid(bounds, _poi_data, grid_size=75):
    with st.spinner("Membuat grid prediksi..."):
        lon_min, lat_min, lon_max, lat_max = bounds
        lons = np.linspace(lon_min, lon_max, grid_size)
        lats = np.linspace(lat_min, lat_max, grid_size)
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        
        grid_df = pd.DataFrame({'longitude': grid_lons.ravel(), 'latitude': grid_lats.ravel()})
        gdf_grid = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.longitude, grid_df.latitude), crs="EPSG:4326")

        gdf_poi = gpd.GeoDataFrame(geometry=[v for k, v in _poi_data.items()], crs="EPSG:4326")
        gdf_grid_proj = gdf_grid.to_crs("EPSG:32749")
        gdf_poi_proj = gdf_poi.to_crs("EPSG:32749")

        gdf_grid['jarak_ke_poi_terdekat_m'] = gdf_grid_proj.geometry.apply(
            lambda point: gdf_poi_proj.distance(point).min()
        )
            
        return gdf_grid

def calculate_area_statistics(df_filtered, geojson_data, boundary_level):
    if not geojson_data or df_filtered.empty: return geojson_data
    result_geojson = json.loads(json.dumps(geojson_data))
    area_prop_map = {'Kecamatan': 'nm_kecamatan', 'Kelurahan/Desa': 'nm_kelurahan', 'Kota': 'nm_dati2'}
    geojson_key = area_prop_map.get(boundary_level, 'nm_kelurahan')
    
    gdf_filtered = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered.longitude, df_filtered.latitude), crs="EPSG:4326")

    for feature in result_geojson['features']:
        polygon = shape(feature['geometry'])
        umkm_in_area_gdf = gdf_filtered[gdf_filtered.within(polygon)]
        
        jumlah_umkm = len(umkm_in_area_gdf)
        sektor_dominan = get_dominant_value(umkm_in_area_gdf, 'nama_sektor')
        
        feature['properties']['jumlah_umkm'] = jumlah_umkm
        feature['properties']['sektor_dominan'] = sektor_dominan
        
        nama_wilayah_raw = str(feature['properties'].get(geojson_key, 'N/A')).title()
        if boundary_level == 'Kecamatan':
            nama_wilayah = KECAMATAN_MAP.get(nama_wilayah_raw, nama_wilayah_raw)
        elif boundary_level == 'Kelurahan/Desa':
            nama_wilayah = DESA_MAP.get(nama_wilayah_raw, nama_wilayah_raw)
        else:
            nama_wilayah = "Kota Batu"

        tooltip_html = f"""<div style='max-width:300px; background:rgba(0,0,0,0.8); color:white; padding:10px; border-radius:5px;'>
                                <div style='font-size:1.1em; font-weight:bold; margin-bottom:8px;'>🗺️ Wilayah: {nama_wilayah}</div>
                                <div style='margin-bottom:4px;'><span style='color:#FCD34D;'>📊 Jumlah UMKM:</span> <b>{jumlah_umkm}</b></div>
                                <div style='margin-bottom:4px;'><span style='color:#F87171;'>🛍️ Sektor Dominan:</span> <b>{sektor_dominan}</b></div>
                                </div>"""
        feature['properties']['tooltip_html'] = tooltip_html
        
    return result_geojson

def get_pydeck_download_script(filename="pydeck_map.png"):
    script = f"""
    <div style="margin: 10px 0;">
        <button id="downloadPydeckBtn" onclick="downloadPydeckMap()" 
                style="display:inline-block; text-align:center; text-decoration:none; 
                        color:white; background-color:#17a2b8; padding:10px 20px; 
                        border-radius:8px; font-weight:bold; border:none; cursor:pointer;">
            📥 Unduh Peta PyDeck
        </button>
        <span id="downloadStatus" style="margin-left: 10px; color: #666;"></span>
    </div>
    
    <script>
    function downloadPydeckMap() {{
        const statusEl = document.getElementById('downloadStatus');
        statusEl.textContent = 'Mencari canvas...';
        
        setTimeout(() => {{
            let deckCanvas = null;
            const allCanvases = document.querySelectorAll('canvas');
            for (let canvas of allCanvases) {{
                const rect = canvas.getBoundingClientRect();
                if (rect.width > 300 && rect.height > 200) {{ // Heuristic to find map canvas
                    deckCanvas = canvas;
                    break;
                }}
            }}
            if (!deckCanvas && allCanvases.length > 0) {{ // Fallback to largest canvas
                let largestCanvas = allCanvases[0];
                let largestArea = 0;
                for (let canvas of allCanvases) {{
                    const rect = canvas.getBoundingClientRect();
                    const area = rect.width * rect.height;
                    if (area > largestArea) {{
                        largestArea = area;
                        largestCanvas = canvas;
                    }}
                }}
                deckCanvas = largestCanvas;
            }}
            
            if (deckCanvas) {{
                try {{
                    statusEl.textContent = 'Mengunduh...';
                    const link = document.createElement('a');
                    link.download = '{filename}';
                    link.href = deckCanvas.toDataURL('image/png', 0.9);
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    statusEl.textContent = '✅ Berhasil diunduh!';
                    setTimeout(() => {{ statusEl.textContent = ''; }}, 3000);
                }} catch (error) {{
                    console.error('Download error:', error);
                    statusEl.textContent = '❌ Gagal mengunduh';
                    setTimeout(() => {{ statusEl.textContent = ''; }}, 3000);
                }}
            }} else {{
                statusEl.textContent = '❌ Canvas tidak ditemukan';
                setTimeout(() => {{ statusEl.textContent = ''; }}, 3000);
            }}
        }}, 1000);
    }}
    </script>
    """
    return script

# =============================================================================
# MAIN APP LAYOUT
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
umkm_file_path = os.path.join(script_dir, "umkm_batu_clustered.csv") 
geojson_base_path = os.path.join(script_dir, "35.79_Kota_Batu")
geojson_kelurahan_path = os.path.join(geojson_base_path, "35.79_kelurahan.geojson")
geojson_kecamatan_path = os.path.join(geojson_base_path, "35.79_kecamatan.geojson")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if 'main_data' not in st.session_state:
    st.session_state['main_data'] = load_and_process_initial_data(umkm_file_path, geojson_kelurahan_path, geojson_kecamatan_path, _poi_data=POI_WISATA)

df = st.session_state['main_data']

if df is not None:
    st.sidebar.header("⚙️ Filter & Tampilan")
    
    kecamatan_display_map = df[['kecamatan_display', 'nama_kecamatan_akurat']].drop_duplicates().sort_values('kecamatan_display')
    selected_kecamatan_display = st.sidebar.multiselect("Pilih Kecamatan:", kecamatan_display_map['kecamatan_display'].unique(), default=kecamatan_display_map['kecamatan_display'].unique())
    selected_kecamatan = kecamatan_display_map[kecamatan_display_map['kecamatan_display'].isin(selected_kecamatan_display)]['nama_kecamatan_akurat'].tolist()
    
    df_desa_filtered_for_list = df[df['nama_kecamatan_akurat'].isin(selected_kecamatan)]
    desa_display_map = df_desa_filtered_for_list[['desa_display', 'nama_desa_akurat']].drop_duplicates().sort_values('desa_display')
    selected_desa_display = st.sidebar.multiselect("Pilih Desa/Kelurahan:", desa_display_map['desa_display'].unique(), default=[])
    selected_desa = desa_display_map[desa_display_map['desa_display'].isin(selected_desa_display)]['nama_desa_akurat'].tolist()

    sektor_list = sorted(df['nama_sektor'].unique())
    selected_sektor = st.sidebar.multiselect("Pilih Sektor Usaha:", sektor_list, default=sektor_list)

    map_theme = st.sidebar.radio("Tema Peta:", ("Terang", "Gelap"), horizontal=True)

    df_filtered = df[df['nama_kecamatan_akurat'].isin(selected_kecamatan) & df['nama_sektor'].isin(selected_sektor)]
    if selected_desa:
        df_filtered = df_filtered[df_filtered['nama_desa_akurat'].isin(selected_desa)]

    st.sidebar.markdown("---")
    st.sidebar.header("⬆️ Unggah Data Baru")
    st.sidebar.info("Untuk panduan format file dan unggah, silakan kunjungi tab 'Panduan Penggunaan' di halaman utama.")
    
    with st.sidebar.form(key='upload_form'):
        uploaded_file = st.file_uploader("Pilih file CSV atau XLSX", type=["csv", "xlsx"])
        submit_button = st.form_submit_button(label='Proses Unggahan')

        if submit_button and uploaded_file is not None:
            try:
                with st.spinner("Membaca file yang diunggah..."):
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_extension == '.csv':
                        new_df = pd.read_csv(uploaded_file)
                    elif file_extension == '.xlsx':
                        new_df = pd.read_excel(uploaded_file)
                    else:
                        st.sidebar.error("Format file tidak didukung. Mohon unggah file .csv atau .xlsx.")
                        st.stop()

                st.sidebar.success(f"File '{uploaded_file.name}' berhasil dibaca.")

                required_cols = ['namausaha', 'kegiatan', 'latitude', 'longitude', 'nama_sektor', 'kec', 'desa'] 
                if not all(col in new_df.columns for col in required_cols):
                    st.sidebar.error(f"File harus memiliki semua kolom yang diperlukan: {', '.join(required_cols)}.")
                    st.stop()
                
                with st.spinner("Memproses data lokasi dan atribut UMKM..."):
                    new_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                    new_gdf = gpd.GeoDataFrame(new_df, geometry=gpd.points_from_xy(new_df.longitude, new_df.latitude), crs="EPSG:4326")
                    
                    gdf_desa_geo = gpd.read_file(geojson_kelurahan_path)
                    gdf_kecamatan_geo = gpd.read_file(geojson_kecamatan_path)

                st.sidebar.success("Data UMKM dasar siap.")

                with st.spinner("Melakukan spatial join (pemetaan desa/kecamatan) untuk data baru..."):
                    new_gdf = gpd.sjoin(new_gdf, gdf_desa_geo[['nm_kelurahan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
                    new_gdf = gpd.sjoin(new_gdf, gdf_kecamatan_geo[['nm_kecamatan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
                
                st.sidebar.success("Spatial join selesai.")

                with st.spinner("Menormalkan nama desa/kecamatan dan menghitung jarak ke POI..."):
                    new_gdf['nama_desa_akurat'] = new_gdf['nm_kelurahan'].str.title().fillna('Tidak Terpetakan')
                    new_gdf['nama_kecamatan_akurat'] = new_gdf['nm_kecamatan'].str.title().fillna('Tidak Terpetakan')
                    new_gdf['kecamatan_display'] = new_gdf['nama_kecamatan_akurat'].map(KECAMATAN_MAP).fillna(new_gdf['nama_kecamatan_akurat'])
                    new_gdf['desa_display'] = new_gdf['nama_desa_akurat'].map(DESA_MAP).fillna(new_gdf['nama_desa_akurat'])

                    new_gdf_proj = new_gdf.to_crs("EPSG:32749")
                    gdf_poi_proj = gpd.GeoDataFrame(geometry=[v for k, v in POI_WISATA.items()], crs="EPSG:4326").to_crs("EPSG:32749")

                    new_gdf['jarak_ke_poi_terdekat_m'] = new_gdf_proj.geometry.apply(
                        lambda umkm_point: gdf_poi_proj.distance(umkm_point).min()
                    )
                
                st.sidebar.success("Jarak ke POI berhasil dihitung.")

                new_gdf = new_gdf.drop(columns=['nm_kelurahan', 'nm_kecamatan', 'kec', 'desa'], errors='ignore')
                st.session_state['main_data'] = pd.concat([st.session_state['main_data'], new_gdf], ignore_index=True)
                
                st.sidebar.success(f"Berhasil mengunggah {len(new_df)} UMKM baru! Data telah diperbarui dan siap dianalisis.")
                st.sidebar.markdown(to_excel_download_link(st.session_state['main_data'], "umkm_batu_gabungan.xlsx", "📥 Unduh Data Gabungan (Excel)"), unsafe_allow_html=True)
                
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Terjadi kesalahan saat memproses file: {e}")

    # --- KARTU UNDUH TEMPLATE DATA (KODE BARU) ---
    with st.sidebar.container(border=True):
        st.markdown("<p style='text-align: center; font-weight: bold;'>Actual Dataset Source</p>", unsafe_allow_html=True)
        try:
            # Cek apakah file default (umkm_file_path) ada
            if os.path.exists(umkm_file_path):
                with open(umkm_file_path, "rb") as file:
                    st.download_button(
                        label="📥 Download (.csv)",
                        data=file,
                        file_name='umkm_batu_clustered.csv', # Nama file asli
                        mime='text/csv', # Tipe file CSV
                        use_container_width=True
                    )
            else:
                 st.info("File template default tidak ditemukan.", icon="ℹ️")
        except Exception as e:
            st.error("Gagal memuat template.", icon="🚨")
    # --- AKHIR KARTU ---

    # Define Tabs
    tab_list = ["📖 Panduan Penggunaan", "📊 Ringkasan Umum", "🗺️ Peta Interaktif", "🔒 Analisis Lanjutan & AI"] 
    tab_guide, tab_summary, tab_map, tab_ai = st.tabs(tab_list)

    # === TAB: Panduan Penggunaan ===
    with tab_guide:
        st.header("📖 Panduan Penggunaan Dashboard")
        st.write("Selamat datang di Dashboard Analitik & Prediktif UMKM Kota Batu! Dashboard ini dirancang untuk membantu Anda memahami persebaran, karakteristik, dan potensi lokasi UMKM di Kota Batu. Berikut adalah panduan singkat untuk menggunakannya:")

        st.markdown("<h4>1. Navigasi Dashboard</h4>", unsafe_allow_html=True)
        st.write("Gunakan tab di bagian atas halaman (`Ringkasan Umum`, `Peta Interaktif`, `Analisis Lanjutan & AI`) untuk beralih antara berbagai bagian analisis. Sidebar di sebelah kiri digunakan untuk filter data dan pengaturan tampilan.")

        st.markdown("<h4>2. Mengunggah Data UMKM Baru</h4>", unsafe_allow_html=True)
        st.write(
            "Anda dapat memperbarui atau menambahkan data UMKM yang dianalisis dengan mengunggah file CSV atau XLSX baru melalui opsi 'Unggah Data Baru' di sidebar. "
            "**Penting:** Pastikan format file Anda sesuai dengan panduan di bawah ini."
        )

        st.markdown("<h5>Format File CSV/XLSX yang Benar:</h5>", unsafe_allow_html=True)
        st.write("File Anda harus memiliki kolom-kolom berikut dengan nama yang persis sama (case-sensitive) di baris pertama (header):")
        
        st.markdown("""
        -   `namausaha`: Nama usaha UMKM.
        -   `kegiatan`: Deskripsi kegiatan usaha.
        -   `latitude`: Koordinat lintang (misal: `-7.871`).
        -   `longitude`: Koordinat bujur (misal: `112.527`).
        -   `nama_sektor`: Sektor usaha UMKM (misal: "Kuliner", "Fesyen", "Kerajinan").
        -   `kec`: Nama Kecamatan (misal: "Batu", "Junrejo", "Bumiaji").
        -   `desa`: Nama Desa/Kelurahan (misal: "Sisir", "Oro-Oro Ombo").
        """)

        st.markdown("<h5>Contoh Struktur File CSV/XLSX:</h5>", unsafe_allow_html=True)
        contoh_data_csv = {
            'namausaha': ['Warung Bu Ani', 'Butik Fashionku', 'Kerajinan Kayu Jaya', 'Café Santai'],
            'kegiatan': ['Jual Nasi Pecel', 'Pakaian Muslim Wanita', 'Ukiran Kayu Jati', 'Kopi dan Snack'],
            'latitude': [-7.8712, -7.8856, -7.8700, -7.8923],
            'longitude': [112.5278, 112.5311, 112.5190, 112.5350],
            'nama_sektor': ['Kuliner', 'Fesyen', 'Kerajinan', 'Kuliner'],
            'kec': ['Batu', 'Batu', 'Junrejo', 'Batu'],
            'desa': ['Sisir', 'Oro-Oro Ombo', 'Tlekung', 'Temas']
        }
        df_contoh_csv = pd.DataFrame(contoh_data_csv)
        st.dataframe(df_contoh_csv, hide_index=True)
        st.markdown("---")
        st.markdown("<h4>3. Filter Data</h4>", unsafe_allow_html=True)
        st.write("Gunakan filter di sidebar untuk menyaring data UMKM berdasarkan Kecamatan, Desa/Kelurahan, dan Sektor Usaha. Peta dan grafik akan diperbarui secara otomatis.")

        st.markdown("<h4>4. Analisis Berbagai Tab</h4>", unsafe_allow_html=True)
        st.markdown("-   **Ringkasan Umum**: Melihat jumlah total UMKM, sektor usaha teratas, dan grafik distribusi sektor.")
        st.markdown("-   **Peta Interaktif**: Menjelajahi persebaran UMKM di peta. Anda bisa mengunduh peta dengan tombol di bawah peta PyDeck atau menggunakan ikon kamera di Plotly.")
        st.markdown("-   **Analisis Lanjutan & AI**: Untuk analisis hotspot/coldspot spasial dan rekomendasi lokasi optimal dengan model AI. Tab ini dilindungi kata sandi.")
        st.info("Kata sandi untuk tab Analisis Lanjutan & AI adalah `password`.")

    # === TAB: Ringkasan Umum ===
    with tab_summary:
        st.header("📈 Ringkasan Analitik")
        col1, col2 = st.columns(2)
        
        total_umkm_terfilter = len(df_filtered)
        
        # Card kiri - Total UMKM dengan font besar
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🏪 Total UMKM Terfilter</div>
                <div style="font-size: 3rem; font-weight: bold; color: #1f77b4; margin: 0;">{total_umkm_terfilter:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Card kanan - Sektor Teratas dengan font normal
        with col2:
            if not df_filtered.empty:
                sektor_top = df_filtered['nama_sektor'].value_counts().nlargest(1)
                sektor_nama = sektor_top.index[0]
                sektor_jumlah = sektor_top.iloc[0]
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🛍️ Sektor Usaha Teratas</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: #333; margin-bottom: 0.3rem;">{sektor_nama}</div>
                    <div style="font-size: 1.5rem; color: #28a745; font-weight: 500;">{sektor_jumlah:,} UMKM</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🛍️ Sektor Usaha Teratas</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #333; margin-bottom: 0.3rem;">N/A</div>
                    <div style="font-size: 1rem; color: #28a745; font-weight: 500;">Tidak ada data</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("Grafik Analitik Sektor Usaha")
        if not df_filtered.empty:
            sektor_counts = df_filtered['nama_sektor'].value_counts().nlargest(15).sort_values(ascending=True)
            fig_sektor = px.bar(sektor_counts, y=sektor_counts.index, x=sektor_counts.values, orientation='h', title="Top 15 Sektor Usaha", labels={'x': 'Jumlah UMKM', 'y': ''}, text_auto=True, template="streamlit", color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_sektor.update_layout(
                showlegend=False, 
                yaxis_title=None,
                yaxis=dict(automargin=True),
                height=min(600, len(sektor_counts) * 40 + 150)
            )
            st.plotly_chart(fig_sektor, use_container_width=True)
        else:
            st.info("Tidak ada data sektor usaha untuk ditampilkan.")
        
        st.markdown("---")
        st.subheader("📄 Data Tabel Lengkap (Sesuai Filter)")
        if not df_filtered.empty:
            st.markdown(get_table_download_link(df_filtered, "umkm_terfilter.csv", "📥 Unduh CSV Data Terfilter"), unsafe_allow_html=True)
            display_cols = ['namausaha', 'nama_sektor', 'desa_display', 'kecamatan_display', 'jarak_ke_poi_terdekat_m']
            st.dataframe(df_filtered[display_cols], use_container_width=True, height=400,
                        column_config={"jarak_ke_poi_terdekat_m": st.column_config.NumberColumn("Jarak ke Wisata (m)", format="%d m")})

    # === TAB: Peta Interaktif ===
    with tab_map:
        st.header("🗺️ Peta Persebaran & Kepadatan UMKM")
        file_map = {'Kota': "35.79_Kota_Batu.geojson", 'Kecamatan': "35.79_kecamatan.geojson", 'Kelurahan/Desa': "35.79_kelurahan.geojson"}
        geojson_file_path = os.path.join(geojson_base_path, file_map.get('Kelurahan/Desa', '35.79_kelurahan.geojson'))
        geojson_data = load_local_geojson(geojson_file_path)
        
        if not df_filtered.empty:
            unique_keys, color_palette, merge_col = sorted(df_filtered['nama_sektor'].dropna().unique()), px.colors.qualitative.Light24, 'nama_sektor'
            
            color_lookup = pd.DataFrame({merge_col: unique_keys, 'color': [color_palette[i % len(color_palette)] for i in range(len(unique_keys))]})
            df_filtered_map = pd.merge(df_filtered, color_lookup, on=merge_col, how='left')
            df_filtered_map['color_rgb'] = df_filtered_map['color'].apply(hex_to_rgb)

            umkm_geojson = {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                "properties": {"color": row["color_rgb"], "radius": 30, "is_umkm_point": True,
                               "tooltip_html": f"""<div style='max-width:300px; background:rgba(0,0,0,0.8); color:white; padding:10px; border-radius:5px;'>
                                        <div style='font-size:1.1em; font-weight:bold; margin-bottom:8px;'>🏪 Nama Usaha: {row.get('namausaha', 'N/A')}</div>
                                        <div style='margin-bottom:4px;'><span style='color:#FCD34D;'>🛍️ Sektor:</span> {row.get('nama_sektor', 'N/A')}</div>
                                        <div style='margin-bottom:4px;'><span style='color:#34D399;'>📍 Desa:</span> {row.get('desa_display', 'N/A')}</div>
                                    </div>"""}}
                for _, row in df_filtered_map.iterrows()
            ]}
            boundary_geojson = calculate_area_statistics(df_filtered_map.copy(), geojson_data, 'Kelurahan/Desa') if geojson_data else None
            view_state = pdk.ViewState(latitude=-7.87, longitude=112.52, zoom=11.5, pitch=50)
            mapbox_style = "mapbox://styles/mapbox/light-v10" if map_theme == "Terang" else "mapbox://styles/mapbox/dark-v10"
            
            layers = []
            if boundary_geojson:
                layers.append(pdk.Layer('GeoJsonLayer', data=boundary_geojson, stroked=True, filled=True, get_fill_color=[180, 180, 180, 20], get_line_color=[85, 85, 85, 255] if map_theme == "Terang" else [220, 220, 220, 255], get_line_width=20, pickable=True, auto_highlight=True))
            if umkm_geojson['features']:
                layers.append(pdk.Layer('GeoJsonLayer', data=umkm_geojson, get_fill_color='properties.color', get_radius='properties.radius', pickable=True, auto_highlight=True))
            
            if layers:
                st.pydeck_chart(pdk.Deck(map_style=mapbox_style, initial_view_state=view_state, layers=layers, tooltip={"html": "{tooltip_html}"}))
                
                # Download options
                st.markdown("### 📥 Unduh Peta")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(get_pydeck_download_script("peta_umkm_batu.png"), unsafe_allow_html=True)
                
                with col2:
                    with st.expander("💡 Panduan Download Manual"):
                        st.markdown("""
                        **Jika tombol download tidak bekerja:**
                        1. Klik kanan pada peta di atas
                        2. Pilih "Save image as..." atau "Simpan gambar sebagai..."
                        3. Atau gunakan screenshot (Ctrl+Shift+S di Chrome/Firefox)
                        
                        **Tips:** Pastikan peta sudah dimuat sempurna sebelum mengunduh.
                        """)
            else:
                st.warning("Tidak ada data untuk ditampilkan di peta dengan filter saat ini.")

            # Dynamic Color Legend
            if not df_filtered_map.empty:
                st.markdown("---")
                st.subheader("🎨 Keterangan Warna Titik UMKM (Sektor Usaha)")
                color_map = {k: color_palette[i % len(color_palette)] for i, k in enumerate(unique_keys)}
                num_cols = min(len(unique_keys), 3)
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    for i, item in enumerate(unique_keys):
                        with cols[i % num_cols]:
                            st.markdown(f'<div class="legend-item"><div class="legend-color-box" style="background-color:{color_map[item]};"></div><span>{item}</span></div>', unsafe_allow_html=True)
        else:
            st.warning("Tidak ada data untuk ditampilkan di peta dengan filter saat ini.")

        # Density Map (Choropleth)
        st.markdown("---")
        st.subheader("📊 Peta Kepadatan Jumlah UMKM per Desa/Kelurahan")
        st.info("💡 Untuk mengunduh peta ini, arahkan kursor ke peta dan klik ikon kamera di pojok kanan atas.")
        
        geojson_data_desa = load_local_geojson(geojson_kelurahan_path)
        if not df_filtered.empty and geojson_data_desa:
            umkm_per_desa = df_filtered.groupby('nama_desa_akurat')['namausaha'].count().reset_index(name='Jumlah UMKM')
            
            geojson_data_desa_modified = json.loads(json.dumps(geojson_data_desa))
            for feature in geojson_data_desa_modified['features']:
                if 'nm_kelurahan' in feature['properties']:
                    feature['properties']['nm_kelurahan'] = str(feature['properties']['nm_kelurahan']).title()

            fig_choro = px.choropleth_mapbox(
                umkm_per_desa, geojson=geojson_data_desa_modified,
                locations='nama_desa_akurat',
                featureidkey="properties.nm_kelurahan",
                color='Jumlah UMKM',
                color_continuous_scale="Plasma",
                mapbox_style="carto-positron" if map_theme == "Terang" else "carto-darkmatter",
                center={"lat": -7.87, "lon": 112.52}, zoom=10.5,
                hover_name='nama_desa_akurat', hover_data={'Jumlah UMKM': True, 'nama_desa_akurat': False}
            )
            fig_choro.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
            st.plotly_chart(fig_choro, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk membuat peta kepadatan.")

    # === TAB: Analisis Lanjutan & AI (Locked) ===
    with tab_ai:
        if not st.session_state['authenticated']:
            st.warning("Tab ini dilindungi kata sandi. Silakan masukkan kata sandi untuk mengakses.")
            password_input = st.text_input("Kata Sandi:", type="password")
            if password_input == "password": # Desired password
                st.session_state['authenticated'] = True
                st.success("Akses diberikan! Silakan refresh halaman atau ubah tab untuk melihat konten.")
                st.rerun() # Refresh to display tab content
            elif password_input != "": # If user entered an incorrect password
                st.error("Kata sandi salah.")
        else:
            st.header("🔬 Analisis Spasial Statistik: Hotspot & Coldspot")
            st.info("""
            Analisis Hotspot dan Coldspot menggunakan metode statistik spasial **Getis-Ord Gi\*** untuk mengidentifikasi area di mana UMKM terkonsentrasi secara signifikan (Hotspot) atau jarang ditemukan (Coldspot). Ini membantu memahami pola spasial yang tidak terlihat hanya dari peta kepadatan biasa.
            
            **Bagaimana Cara Kerjanya?**
            Metode ini membandingkan jumlah UMKM di suatu desa dengan jumlah UMKM di desa-desa tetangganya. Jika suatu desa memiliki banyak UMKM dan dikelilingi oleh desa-desa yang juga memiliki banyak UMKM, maka desa tersebut berpotensi menjadi 'Hotspot'. Sebaliknya, jika suatu desa memiliki sedikit UMKM dan dikelilingi oleh desa-desa dengan sedikit UMKM, desa tersebut berpotensi menjadi 'Coldspot'.
            
            **Metrik Statistik:**
            -   **Z-score**: Mengukur berapa standar deviasi nilai suatu fitur dari rata-rata. Z-score positif yang besar menunjukkan pengelompokan nilai tinggi (hotspot), sedangkan Z-score negatif yang kecil menunjukkan pengelompokan nilai rendah (coldspot).
            -   **Pseudo P-value (simulasi)**: Mengestimasi probabilitas bahwa pola spasial yang diamati terjadi secara acak. Nilai P-value yang kecil (misalnya, kurang dari 0.05) menunjukkan bahwa pengelompokan tersebut signifikan secara statistik, bukan karena kebetulan.
            """)

            with st.expander("❓ Memahami Hasil Getis-Ord Gi*: Kenapa 'Tidak Signifikan'?"):
                st.markdown("""
                Jika sebagian besar atau seluruh area diklasifikasikan sebagai **"Tidak Signifikan"**, ini adalah hasil analisis statistik yang normal dan bukan berarti ada kesalahan dalam perhitungan. Ini dapat terjadi karena beberapa alasan:
                
                1.  **Distribusi UMKM Cenderung Acak atau Merata**: Artinya, jumlah UMKM di setiap desa tidak terkonsentrasi secara ekstrem pada satu area dan juga tidak terlalu jarang di area lain secara signifikan. Pola persebaran UMKM di Kota Batu mungkin memang cenderung merata.
                2.  **Variasi Data yang Kurang Signifikan**: Jika perbedaan jumlah UMKM antar desa tidak terlalu besar atau menonjol secara statistik, maka algoritma tidak akan menemukan pengelompokan yang "signifikan" (yaitu, sangat tidak mungkin terjadi secara kebetulan).
                3.  **Keterbatasan Data**: Dengan data UMKM saat ini, mungkin tidak ada pola spasial yang cukup kuat untuk memenuhi ambang batas signifikansi statistik yang ketat (Z-score dan p-value).
                
                **Implikasi "Tidak Signifikan":**
                Ini berarti tidak ada *bukti statistik yang kuat* untuk menyatakan bahwa ada "kantong" Hotspot atau Coldspot UMKM yang jelas pada tingkat kepercayaan 95%. Dalam konteks kebijakan, ini bisa berarti bahwa upaya pengembangan UMKM mungkin tidak perlu difokuskan pada area-area spesifik berdasarkan kepadatan saja, melainkan pada faktor-faktor lain atau strategi yang lebih merata.
                """)
            
            gdf_desa_geojson = gpd.read_file(geojson_kelurahan_path)
            gdf_desa_hotspot = calculate_hotspots(gdf_desa_geojson, st.session_state['main_data'])
            
            # Hotspot Map (Choropleth)
            color_discrete_map = {
                'Hotspot (Sentra Bisnis)': 'red',
                'Coldspot (Area Sepi)': 'blue',
                'Tidak Signifikan': 'grey'
            }
            fig_hotspot = px.choropleth_mapbox(
                gdf_desa_hotspot, geojson=gdf_desa_hotspot.geometry, locations=gdf_desa_hotspot.index,
                color='hotspot_label', color_discrete_map=color_discrete_map,
                category_orders={'hotspot_label': ['Hotspot (Sentra Bisnis)', 'Coldspot (Area Sepi)', 'Tidak Signifikan']},
                mapbox_style="carto-positron" if map_theme == "Terang" else "carto-darkmatter",
                center={"lat": -7.87, "lon": 112.52}, zoom=10.5,
                hover_name='nm_kelurahan', opacity=0.6,
                hover_data={'jumlah_umkm': True}
            )
            fig_hotspot.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend_title_text='Klasifikasi Area', height=600)
            st.plotly_chart(fig_hotspot, use_container_width=True)

            st.markdown("---")
            
            st.header("🤖 Rekomendasi Lokasi Optimal (AI)")
            st.info("""
            Model AI ini membantu memprediksi seberapa "layak" atau "potensial" suatu lokasi baru untuk UMKM berdasarkan pola lokasi UMKM sejenis yang sudah ada dan jaraknya ke Poin of Interest (POI) wisata.
            
            **Bagaimana Cara Kerjanya?**
            1.  **Pembelajaran dari Data Eksisting**: Model ini (Random Forest Regressor) dilatih menggunakan data UMKM yang sudah ada. Ia mempelajari hubungan antara lokasi (latitude, longitude), jarak ke POI wisata, dengan 'skor kelayakan' simulasi dari UMKM yang sudah ada.
            2.  **Skor Kelayakan Simulasi**: Karena data kinerja bisnis riil (misalnya, omset atau jumlah pelanggan) tidak tersedia, kami membuat 'skor kelayakan' simulasi. Skor ini dihitung berdasarkan seberapa dekat UMKM dengan POI wisata utama (semakin dekat, semakin tinggi skornya, dan kemudian dinormalisasi antara 0-1). Ini adalah **representasi yang disederhanakan** dari potensi sukses, dan dalam aplikasi nyata, data kinerja bisnis akan sangat meningkatkan akurasi model.
            3.  **Prediksi pada Grid**: Setelah dilatih, model memprediksi skor kelayakan untuk ribuan titik lokasi potensial di seluruh Kota Batu (dalam bentuk grid).
            
            **Metrik Data Science:**
            -   **Random Forest Regressor**: Ini adalah algoritma Machine Learning yang robust, sering digunakan untuk prediksi. Ia bekerja dengan membangun banyak "pohon keputusan" dan menggabungkan hasilnya untuk prediksi yang lebih akurat.
            -   **Skor Kelayakan (Suitability Score)**: Output model, menunjukkan potensi lokasi. Semakin tinggi skornya (mendekati 1), semakin tinggi potensi kelayakannya berdasarkan pola yang dipelajari.
            
            **Implikasi:**
            -   Membantu calon pelaku UMKM dalam menentukan lokasi strategis.
            -   Memberikan informasi bagi pemerintah daerah untuk merencanakan pengembangan UMKM di area yang berpotensi tinggi.
            -   **Penting**: Model ini adalah alat pendukung keputusan. Keputusan akhir harus mempertimbangkan faktor-faktor lain seperti demografi, persaingan lokal, peraturan, dan ketersediaan sumber daya.
            """)

            selected_sector_ai = st.selectbox("Pilih Sektor Usaha untuk Analisis AI:", options=sektor_list)

            if selected_sector_ai:
                model = train_suitability_model(st.session_state['main_data'], selected_sector_ai)
                if model:
                    with st.spinner(f"Menganalisis dan membuat peta prediksi untuk '{selected_sector_ai}'..."):
                        bounds = st.session_state['main_data'].total_bounds
                        grid_df = create_prediction_grid(bounds, _poi_data=POI_WISATA)
                        
                        features_for_prediction = grid_df[['latitude', 'longitude', 'jarak_ke_poi_terdekat_m']]
                        grid_df['suitability_score'] = model.predict(features_for_prediction)

                        view_state = pdk.ViewState(latitude=-7.87, longitude=112.52, zoom=11, pitch=50)
                        heatmap_layer = pdk.Layer(
                            'HeatmapLayer',
                            data=grid_df,
                            get_position='[longitude, latitude]',
                            get_weight='suitability_score',
                            opacity=0.5,
                            pickable=True,
                            aggregation='"MEAN"'
                        )
                        st.pydeck_chart(pdk.Deck(
                            map_style="mapbox://styles/mapbox/light-v9" if map_theme == "Terang" else "mapbox://styles/mapbox/dark-v9",
                            initial_view_state=view_state,
                            layers=[heatmap_layer],
                            tooltip={"text": "Skor Potensi: {suitability_score:.2f}"}
                        ))
                        st.success(f"Peta potensi untuk sektor '{selected_sector_ai}' berhasil dibuat. Area dengan warna lebih 'panas' (merah/kuning) menunjukkan potensi yang lebih tinggi.")
                else:
                    st.warning(f"Tidak cukup data untuk sektor '{selected_sector_ai}' untuk membangun model AI yang andal. Diperlukan setidaknya 10 data UMKM untuk sektor ini.")

else:
    st.error("Gagal memuat data utama. Pastikan file CSV dan GeoJSON ada di direktori yang benar.")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini dikembangkan untuk memberikan wawasan mendalam mengenai lanskap UMKM di Kota Batu.")