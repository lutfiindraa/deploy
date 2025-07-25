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
    page_icon="",
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

st.title("Dashboard Analitik & Prediktif UMKM Kota Batu")
st.markdown(
    "<h2 style='text-align: center;'>Analisis Deskriptif, Spasial</h2>",
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

# Fungsi ini tidak lagi di-cache karena data akan diambil dari session_state
def load_and_process_initial_data(umkm_file_path, geojson_path_desa, geojson_path_kecamatan, _poi_data):
    try:
        cols_to_use = ['namausaha', 'kegiatan', 'latitude', 'longitude', 'nama_sektor']
        df_umkm = pd.read_csv(umkm_file_path, usecols=lambda c: c in cols_to_use)
        gdf_desa = gpd.read_file(geojson_path_desa)
        gdf_kecamatan = gpd.read_file(geojson_path_kecamatan)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Error memuat data: {e}"); return None

    df_umkm.dropna(subset=['latitude', 'longitude'], inplace=True)
    gdf_umkm = gpd.GeoDataFrame(df_umkm, geometry=gpd.points_from_xy(df_umkm.longitude, df_umkm.latitude), crs="EPSG:4326")

    # Spatial Joins untuk mendapatkan nama wilayah akurat
    gdf_umkm = gpd.sjoin(gdf_umkm, gdf_desa[['nm_kelurahan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
    gdf_umkm = gpd.sjoin(gdf_umkm, gdf_kecamatan[['nm_kecamatan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
    
    # Pembersihan dan pemetaan nama
    gdf_umkm['nama_desa_akurat'] = gdf_umkm['nm_kelurahan'].str.title().fillna('Tidak Terpetakan')
    gdf_umkm['nama_kecamatan_akurat'] = gdf_umkm['nm_kecamatan'].str.title().fillna('Tidak Terpetakan')
    gdf_umkm['kecamatan_display'] = gdf_umkm['nama_kecamatan_akurat'].map(KECAMATAN_MAP).fillna(gdf_umkm['nama_kecamatan_akurat'])
    gdf_umkm['desa_display'] = gdf_umkm['nama_desa_akurat'].map(DESA_MAP).fillna(gdf_umkm['nama_desa_akurat'])

    # === ANALISIS PROKSIMITAS ===
    # Buat GeoDataFrame untuk POI
    gdf_poi = gpd.GeoDataFrame(geometry=[v for k, v in _poi_data.items()], crs="EPSG:4326")
    # Ubah CRS ke sistem terproyeksi (meter) untuk perhitungan jarak yang akurat
    gdf_umkm_proj = gdf_umkm.to_crs("EPSG:32749")
    gdf_poi_proj = gdf_poi.to_crs("EPSG:32749")
    
    # Hitung jarak ke POI terdekat untuk setiap UMKM
    for i, umkm_point in gdf_umkm_proj.iterrows():
        distances = gdf_poi_proj.distance(umkm_point.geometry)
        gdf_umkm.loc[i, 'jarak_ke_poi_terdekat_m'] = distances.min()

    return gdf_umkm.drop(columns=['nm_kelurahan', 'nm_kecamatan'])

@st.cache_data
def calculate_hotspots(_gdf_desa, _df_umkm):
    gdf_desa_copy = _gdf_desa.copy()
    umkm_per_desa = _df_umkm.groupby('nama_desa_akurat').size().reset_index(name='jumlah_umkm')
    
    # Gabungkan jumlah UMKM ke GeoDataFrame desa
    gdf_desa_stats = gdf_desa_copy.merge(umkm_per_desa, left_on='nm_kelurahan', right_on='nama_desa_akurat', how='left')
    gdf_desa_stats['jumlah_umkm'].fillna(0, inplace=True)
    
    # Hitung bobot spasial (spatial weights)
    w = weights.Queen.from_dataframe(gdf_desa_stats)
    w.transform = 'r' # Standardisasi baris
    
    # Lakukan analisis Getis-Ord Gi*
    g_local = esda.G_Local(gdf_desa_stats['jumlah_umkm'], w)
    
    # Klasifikasi hotspot/coldspot
    gdf_desa_stats['hotspot_label'] = 'Tidak Signifikan'
    gdf_desa_stats.loc[(g_local.Zs > 1.96) & (g_local.p_sim < 0.05), 'hotspot_label'] = 'Hotspot (Sentra Bisnis)'
    gdf_desa_stats.loc[(g_local.Zs < -1.96) & (g_local.p_sim < 0.05), 'hotspot_label'] = 'Coldspot (Area Sepi)'
    
    return gdf_desa_stats

@st.cache_resource
def train_suitability_model(_df, sector):
    df_sector = _df[_df['nama_sektor'] == sector].copy()
    if len(df_sector) < 10: return None # Butuh data yang cukup

    # Membuat 'skor sukses' simulasi
    df_sector['success_score'] = 1 / (df_sector['jarak_ke_poi_terdekat_m'] + 1)
    df_sector['success_score'] = (df_sector['success_score'] - df_sector['success_score'].min()) / \
                                 (df_sector['success_score'].max() - df_sector['success_score'].min())
    
    X = df_sector[['latitude', 'longitude', 'jarak_ke_poi_terdekat_m']]
    y = df_sector['success_score']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(X, y)
    return model

@st.cache_data
def create_prediction_grid(bounds, _poi_data, grid_size=75):
    # Buat grid koordinat
    lon_min, lat_min, lon_max, lat_max = bounds
    lons = np.linspace(lon_min, lon_max, grid_size)
    lats = np.linspace(lat_min, lat_max, grid_size)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    
    grid_df = pd.DataFrame({'longitude': grid_lons.ravel(), 'latitude': grid_lats.ravel()})
    gdf_grid = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.longitude, grid_df.latitude), crs="EPSG:4326")

    # Hitung fitur jarak untuk setiap titik di grid
    gdf_poi = gpd.GeoDataFrame(geometry=[v for k, v in _poi_data.items()], crs="EPSG:4326")
    gdf_grid_proj = gdf_grid.to_crs("EPSG:32749")
    gdf_poi_proj = gdf_poi.to_crs("EPSG:32749")

    for i, point in gdf_grid_proj.iterrows():
        distances = gdf_poi_proj.distance(point.geometry)
        gdf_grid.loc[i, 'jarak_ke_poi_terdekat_m'] = distances.min()
        
    return gdf_grid

def calculate_area_statistics(df_filtered, geojson_data, boundary_level):
    if not geojson_data or df_filtered.empty: return geojson_data
    result_geojson = json.loads(json.dumps(geojson_data))  # Deep copy
    area_prop_map = {'Kecamatan': 'nm_kecamatan', 'Kelurahan/Desa': 'nm_kelurahan', 'Kota': 'nm_dati2'}
    geojson_key = area_prop_map.get(boundary_level, 'nm_kelurahan')
    
    df_filtered['geometry'] = [Point(xy) for xy in zip(df_filtered['longitude'], df_filtered['latitude'])]
    gdf_filtered = gpd.GeoDataFrame(df_filtered, geometry='geometry', crs="EPSG:4326")

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

# Untuk mengizinkan unduhan gambar PyDeck
# Ini adalah hack karena Streamlit tidak memiliki fungsi bawaan untuk unduhan gambar kanvas
# Ini akan menyuntikkan HTML dengan JavaScript yang memicu unduhan di browser klien.
# Ganti fungsi get_pydeck_download_script yang lama dengan yang ini:

def get_pydeck_download_script(filename="pydeck_map.png"):
    script = f"""
    <script>
    // Pastikan fungsi ini hanya didefinisikan sekali
    if (!window.downloadPydeckMap) {{
        window.downloadPydeckMap = function() {{
            // Coba berbagai selector untuk menemukan canvas PyDeck
            let deckCanvas = null;
            
            // Selector 1: Cari berdasarkan class deck-canvas
            deckCanvas = document.querySelector('canvas.deck-canvas');
            
            // Selector 2: Jika tidak ditemukan, cari canvas di dalam container PyDeck
            if (!deckCanvas) {{
                const deckContainer = document.querySelector('[data-testid="stDeckGlJsonChart"]');
                if (deckContainer) {{
                    deckCanvas = deckContainer.querySelector('canvas');
                }}
            }}
            
            // Selector 3: Cari canvas dengan id yang mengandung 'deck'
            if (!deckCanvas) {{
                const allCanvases = document.querySelectorAll('canvas');
                for (let canvas of allCanvases) {{
                    if (canvas.id && canvas.id.toLowerCase().includes('deck')) {{
                        deckCanvas = canvas;
                        break;
                    }}
                }}
            }}
            
            // Selector 4: Cari canvas terakhir yang ditambahkan (biasanya PyDeck)
            if (!deckCanvas) {{
                const allCanvases = document.querySelectorAll('canvas');
                if (allCanvases.length > 0) {{
                    deckCanvas = allCanvases[allCanvases.length - 1];
                }}
            }}
            
            if (deckCanvas) {{
                try {{
                    // Tunggu sebentar untuk memastikan rendering selesai
                    setTimeout(function() {{
                        const link = document.createElement('a');
                        link.download = '{filename}';
                        link.href = deckCanvas.toDataURL('image/png');
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        console.log('PyDeck map downloaded successfully');
                    }}, 500);
                }} catch (error) {{
                    console.error('Error downloading PyDeck map:', error);
                    alert('Gagal mengunduh peta. Error: ' + error.message);
                }}
            }} else {{
                alert('Canvas PyDeck tidak ditemukan! Pastikan peta sudah dimuat sepenuhnya.');
                console.log('Available canvases:', document.querySelectorAll('canvas'));
            }}
        }};
    }}
    </script>
    <button onclick="window.downloadPydeckMap()" style="display:inline-block; text-align:center; text-decoration:none; color:white; background-color:#17a2b8; padding:10px 20px; border-radius:8px; font-weight:bold; border:none; cursor:pointer; margin:10px 0;">📥 Unduh Peta PyDeck</button>
    """
    return script

# ALTERNATIF SOLUSI: Gunakan st.download_button dengan screenshot
# Tambahkan fungsi ini sebagai alternatif jika JavaScript tidak bekerja:

def create_pydeck_download_alternative():
    """
    Alternatif untuk download PyDeck - memberikan instruksi manual
    """
    st.info("""
    **Cara mengunduh peta PyDeck:**
    1. Klik kanan pada peta di atas
    2. Pilih "Save image as..." atau "Simpan gambar sebagai..."
    3. Pilih lokasi dan nama file untuk menyimpan
    
    *Atau gunakan screenshot tool browser Anda (Ctrl+Shift+S di Chrome/Firefox)*
    """)

# SOLUSI TAMBAHAN: Menggunakan komponen HTML yang lebih reliable
def get_improved_pydeck_download_script(filename="pydeck_map.png"):
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
        
        // Tunggu sebentar untuk memastikan peta sudah render
        setTimeout(() => {{
            let deckCanvas = null;
            
            // Metode 1: Cari canvas dengan ukuran yang wajar (biasanya peta)
            const allCanvases = document.querySelectorAll('canvas');
            for (let canvas of allCanvases) {{
                const rect = canvas.getBoundingClientRect();
                // Cari canvas yang cukup besar (kemungkinan peta)
                if (rect.width > 300 && rect.height > 200) {{
                    deckCanvas = canvas;
                    break;
                }}
            }}
            
            // Metode 2: Jika tidak ditemukan, ambil canvas terbesar
            if (!deckCanvas && allCanvases.length > 0) {{
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
                    
                    // Buat link download
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
        }}, 1000); // Tunggu 1 detik untuk memastikan rendering selesai
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

# Inisialisasi session state untuk otentikasi
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Inisialisasi data utama di session_state jika belum ada
if 'main_data' not in st.session_state:
    st.session_state['main_data'] = load_and_process_initial_data(umkm_file_path, geojson_kelurahan_path, geojson_kecamatan_path, _poi_data=POI_WISATA)

# Gunakan data dari session_state
df = st.session_state['main_data']

if df is not None:
    st.sidebar.header("⚙️ Filter & Tampilan")
    
    # Filter di Sidebar
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

    # Proses Filter Utama
    df_filtered = df[df['nama_kecamatan_akurat'].isin(selected_kecamatan) & df['nama_sektor'].isin(selected_sektor)]
    if selected_desa:
        df_filtered = df_filtered[df_filtered['nama_desa_akurat'].isin(selected_desa)]

    # --- Bagian Unggah Data Baru ---
    st.sidebar.markdown("---")
    st.sidebar.header("⬆️ Unggah Data Baru")
    st.sidebar.markdown("Unggah file CSV UMKM baru dengan kolom: `namausaha`, `kegiatan`, `latitude`, `longitude`, `nama_sektor`.")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            required_cols = ['namausaha', 'kegiatan', 'latitude', 'longitude', 'nama_sektor']
            if not all(col in new_df.columns for col in required_cols):
                st.sidebar.error("File CSV harus memiliki semua kolom yang diperlukan: 'namausaha', 'kegiatan', 'latitude', 'longitude', 'nama_sektor'.")
            else:
                # Proses data baru mirip dengan data awal
                new_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                new_gdf = gpd.GeoDataFrame(new_df, geometry=gpd.points_from_xy(new_df.longitude, new_df.latitude), crs="EPSG:4326")
                
                # Load GeoJSON files for spatial join (they are cached, so no performance hit)
                gdf_desa_geo = gpd.read_file(geojson_kelurahan_path)
                gdf_kecamatan_geo = gpd.read_file(geojson_kecamatan_path)

                new_gdf = gpd.sjoin(new_gdf, gdf_desa_geo[['nm_kelurahan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
                new_gdf = gpd.sjoin(new_gdf, gdf_kecamatan_geo[['nm_kecamatan', 'geometry']], how="left", predicate='within').drop(columns=['index_right'])
                
                new_gdf['nama_desa_akurat'] = new_gdf['nm_kelurahan'].str.title().fillna('Tidak Terpetakan')
                new_gdf['nama_kecamatan_akurat'] = new_gdf['nm_kecamatan'].str.title().fillna('Tidak Terpetakan')
                new_gdf['kecamatan_display'] = new_gdf['nama_kecamatan_akurat'].map(KECAMATAN_MAP).fillna(new_gdf['nama_kecamatan_akurat'])
                new_gdf['desa_display'] = new_gdf['nama_desa_akurat'].map(DESA_MAP).fillna(new_gdf['nama_desa_akurat'])

                new_gdf_proj = new_gdf.to_crs("EPSG:32749")
                gdf_poi_proj = gpd.GeoDataFrame(geometry=[v for k, v in POI_WISATA.items()], crs="EPSG:4326").to_crs("EPSG:32749")

                for i, umkm_point in new_gdf_proj.iterrows():
                    distances = gdf_poi_proj.distance(umkm_point.geometry)
                    new_gdf.loc[i, 'jarak_ke_poi_terdekat_m'] = distances.min()
                
                # Hapus kolom temporer
                new_gdf = new_gdf.drop(columns=['nm_kelurahan', 'nm_kecamatan'])

                # Gabungkan data baru dengan data yang sudah ada di session_state
                st.session_state['main_data'] = pd.concat([st.session_state['main_data'], new_gdf], ignore_index=True)
                st.sidebar.success(f"Berhasil mengunggah {len(new_df)} UMKM baru! Data telah diperbarui.")

                # Opsi unduh data gabungan ke Excel
                st.sidebar.markdown(to_excel_download_link(st.session_state['main_data'], "umkm_batu_gabungan.xlsx", "📥 Unduh Data Gabungan (Excel)"), unsafe_allow_html=True)
                # Force rerun to re-apply filters and update charts with new data
                st.rerun()

        except Exception as e:
            st.sidebar.error(f"Terjadi kesalahan saat memproses file: {e}")

    # Definisi Tabs
    tab_list = ["📊 Ringkasan Umum", "🗺️ Peta Interaktif", "🔒 Analisis Lanjutan & AI"] # Nama tab sedikit diubah untuk menunjukkan penguncian
    tab_summary, tab_map, tab_ai = st.tabs(tab_list)

    # === TAB 1: RINGKASAN UMUM ===
    with tab_summary:
        st.header("📈 Ringkasan Analitik")
        col1, col2 = st.columns(2)
        total_umkm_terfilter = len(df_filtered)
        col1.metric("🏪 Total UMKM Terfilter", f"{total_umkm_terfilter:,}")
        if not df_filtered.empty:
            sektor_top = df_filtered['nama_sektor'].value_counts().nlargest(1)
            col2.metric("🛍️ Sektor Usaha Teratas", sektor_top.index[0], f"{sektor_top.iloc[0]:,} UMKM")
        else:
            col2.metric("🛍️ Sektor Usaha Teratas", "N/A", "Tidak ada data")
        
        st.markdown("---")
        st.header("Grafik Analitik Sektor Usaha")
        if not df_filtered.empty:
            sektor_counts = df_filtered['nama_sektor'].value_counts().nlargest(15).sort_values(ascending=True)
            fig_sektor = px.bar(sektor_counts, y=sektor_counts.index, x=sektor_counts.values, orientation='h', title="Top 15 Sektor Usaha", labels={'x': 'Jumlah UMKM', 'y': ''}, text_auto=True, template="streamlit", color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig_sektor.update_layout(showlegend=False, yaxis_title=None), use_container_width=True)
        else:
            st.info("Tidak ada data sektor usaha untuk ditampilkan.")
        
        st.markdown("---")
        st.subheader("📄 Data Tabel Lengkap (Sesuai Filter)")
        if not df_filtered.empty:
            st.markdown(get_table_download_link(df_filtered, "umkm_terfilter.csv", "📥 Unduh CSV Data Terfilter"), unsafe_allow_html=True)
            display_cols = ['namausaha', 'nama_sektor', 'desa_display', 'kecamatan_display', 'jarak_ke_poi_terdekat_m']
            st.dataframe(df_filtered[display_cols], use_container_width=True, height=400,
                         column_config={"jarak_ke_poi_terdekat_m": st.column_config.NumberColumn("Jarak ke Wisata (m)", format="%d m")})

    # === TAB 2: PETA INTERAKTIF ===
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
                # Tampilkan peta PyDeck
                st.pydeck_chart(pdk.Deck(map_style=mapbox_style, initial_view_state=view_state, layers=layers, tooltip={"html": "{tooltip_html}"}))
                
                # Opsi download yang diperbaiki
                st.markdown("### 📥 Unduh Peta")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Tombol download dengan JavaScript yang diperbaiki
                    st.markdown(get_improved_pydeck_download_script("peta_umkm_batu.png"), unsafe_allow_html=True)
                
                with col2:
                    # Alternatif instruksi manual
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

            # Legenda Warna Dinamis
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

        # Peta Kepadatan Desa (Choropleth) - bagian ini tetap sama
        st.markdown("---")
        st.subheader("📊 Peta Kepadatan Jumlah UMKM per Desa/Kelurahan")
        st.info("💡 Untuk mengunduh peta ini, arahkan kursor ke peta dan klik ikon kamera di pojok kanan atas.")
        
        geojson_data_desa = load_local_geojson(geojson_kelurahan_path)
        if not df_filtered.empty and geojson_data_desa:
            umkm_per_desa = df_filtered.groupby('nama_desa_akurat')['namausaha'].count().reset_index(name='Jumlah UMKM')
            fig_choro = px.choropleth_mapbox(
                umkm_per_desa, geojson=geojson_data_desa, locations='nama_desa_akurat',
                featureidkey="properties.nm_kelurahan", color='Jumlah UMKM',
                color_continuous_scale="Plasma",
                mapbox_style="carto-positron" if map_theme == "Terang" else "carto-darkmatter",
                center={"lat": -7.87, "lon": 112.52}, zoom=10.5,
                hover_name='nama_desa_akurat', hover_data={'Jumlah UMKM': True, 'nama_desa_akurat': False}
            )
            fig_choro.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_choro, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk membuat peta kepadatan.")

    # === TAB 3: ANALISIS LANJUTAN & AI (DIKUNCI) ===
    with tab_ai:
        if not st.session_state['authenticated']:
            st.warning("Tab ini dilindungi kata sandi. Silakan masukkan kata sandi untuk mengakses.")
            password_input = st.text_input("Kata Sandi:", type="password")
            if password_input == "password": # Kata sandi yang diinginkan
                st.session_state['authenticated'] = True
                st.success("Akses diberikan! Silakan refresh halaman atau ubah tab untuk melihat konten.")
                st.rerun() # Refresh untuk menampilkan konten tab
            elif password_input != "": # Jika user sudah input tapi salah
                st.error("Kata sandi salah.")
        else:
            st.header("🔬 Analisis Spasial Statistik: Hotspot & Coldspot")
            st.info("Analisis ini mengidentifikasi di mana UMKM terkonsentrasi secara signifikan (Sentra Bisnis) dan di mana mereka jarang ditemukan (Area Sepi) berdasarkan statistik spasial.")
            
            gdf_desa_geojson = gpd.read_file(geojson_kelurahan_path)
            # Pastikan calculate_hotspots menerima df dari session_state
            gdf_desa_hotspot = calculate_hotspots(gdf_desa_geojson, st.session_state['main_data'])
            
            # Peta Hotspot
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
            fig_hotspot.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend_title_text='Klasifikasi Area')
            st.plotly_chart(fig_hotspot, use_container_width=True)

            st.markdown("---")
            
            st.header("🤖 Rekomendasi Lokasi Optimal (AI)")
            st.info("Pilih sektor usaha untuk melihat peta potensi lokasi. Model AI memprediksi skor kelayakan berdasarkan jarak ke tempat wisata dan pola lokasi UMKM sejenis yang sudah ada.")

            selected_sector_ai = st.selectbox("Pilih Sektor Usaha untuk Analisis AI:", options=sektor_list)

            if selected_sector_ai:
                # Pastikan train_suitability_model menerima df dari session_state
                model = train_suitability_model(st.session_state['main_data'], selected_sector_ai)
                if model:
                    with st.spinner(f"Menganalisis dan membuat peta prediksi untuk '{selected_sector_ai}'..."):
                        bounds = st.session_state['main_data'].total_bounds
                        grid_df = create_prediction_grid(bounds, _poi_data=POI_WISATA)
                        
                        # Prediksi menggunakan model
                        features_for_prediction = grid_df[['latitude', 'longitude', 'jarak_ke_poi_terdekat_m']]
                        grid_df['suitability_score'] = model.predict(features_for_prediction)

                        # Visualisasi Heatmap
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
                    st.warning(f"Tidak cukup data untuk sektor '{selected_sector_ai}' untuk membangun model AI yang andal.")

else:
    st.error("Gagal memuat data utama. Pastikan file CSV dan GeoJSON ada di direktori yang benar.")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini dikembangkan untuk memberikan wawasan mendalam mengenai lanskap UMKM di Kota Batu.")