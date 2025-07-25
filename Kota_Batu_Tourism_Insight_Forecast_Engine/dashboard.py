# ==============================================================================
# STREAMLIT DASHBOARD - KODE LENGKAP, FINAL, DAN SIAP PRODUKSI
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import warnings
from pathlib import Path
from datetime import datetime
import re # Diimpor untuk ekstraksi tahun

# Impor library tambahan yang diperlukan untuk multi-model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.stattools import acf

# Pastikan openpyxl terinstall untuk membaca file .xlsx
try:
    import openpyxl
except ImportError:
    st.error("Paket 'openpyxl' tidak ditemukan. Silakan install dengan perintah: pip install openpyxl")
    st.stop()


# --- KONFIGURASI HALAMAN & GLOBAL ---
st.set_page_config(
    page_title="Kota Batu Tourism Insights",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Palet warna konsisten untuk model
MODEL_COLORS = {
    'SARIMA': '#636EFA',   # Biru
    'Prophet': '#EF553B',  # Oranye
    'XGBoost': '#00CC96'   # Hijau
}


# --- CSS KUSTOM (DENGAN KARTU UNGU) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { background: transparent; backdrop-filter: none; box-shadow: none; padding: 2rem; margin-top: 1rem; }
    h1 { font-size: 3rem; font-weight: 700; text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }

    /* --- PERUBAHAN: STYLING KARTU METRIK MENJADI UNGU --- */
    .metric-card {
        background: linear-gradient(135deg, #3b4477, #6a72d9); /* Gradasi Ungu Baru */
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white; /* Mengubah warna teks default menjadi putih */
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(142, 68, 173, 0.5); /* Bayangan ungu saat hover */
    }

    /* Menyesuaikan teks judul dan subjudul di dalam kartu ungu */
    .metric-card .metric-title, .metric-card .metric-subtitle {
        color: white;
        opacity: 0.9;
    }

    /* Menyesuaikan teks nilai utama (angka) di dalam kartu ungu */
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        background: none; /* Menghapus gradasi pada teks */
        -webkit-background-clip: unset;
        -webkit-text-fill-color: unset;
    }
    /* -------------------------------------------------------- */

    /* Styling untuk container lain agar tetap transparan */
    .chart-container, .feature-card, .welcome-section {
        background: transparent;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(128, 128, 128, 0.1);
    }
    
    /* Teks untuk container lain (tidak berubah) */
    .feature-card p, .welcome-section p {
        color: var(--text-color);
        opacity: 0.8;
    }

    /* Nilai metrik umum jika digunakan di luar .metric-card (tidak berubah) */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stButton > button { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    .success-msg, .error-msg { color: white; padding: 1rem; border-radius: 12px; margin: 1rem 0; }
    .success-msg { background: linear-gradient(135deg, #4CAF50, #66bb6a); }
    .error-msg { background: linear-gradient(135deg, #F44336, #e57373); }
    .welcome-section h2 { color: var(--text-color); }
    @media (max-width: 768px) { .main .block-container { padding: 1rem; } h1 { font-size: 2rem; } }
    </style>
""", unsafe_allow_html=True)


# --- FUNGSI PEMROSESAN DATA (VERSI FINAL & STABIL) ---
@st.cache_data
def load_and_process_data(uploaded_file=None):
    """
    Memuat data secara dinamis dari file Excel yang diunggah pengguna atau dari file default.
    Jika ada file yang diunggah, fungsi ini akan memprioritaskannya.
    """
    def find_header_row(excel_file, sheet_name):
        try:
            df_scan = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, engine='openpyxl')
            max_score, header_row = 0, -1
            keywords = ['NAMA', 'USAHA', 'JANUARI', 'FEBRUARI', 'MARET']
            for i, row in df_scan.head(30).iterrows():
                row_str = ' '.join(str(x).upper() for x in row.dropna())
                score = sum(1 for keyword in keywords if keyword in row_str)
                if score >= 2 and score > max_score:
                    max_score, header_row = score, i
            if header_row != -1: return header_row
            return 20
        except Exception:
            return 20

    def clean_sheet(df, year):
        if df.empty: return pd.DataFrame()
        df.columns = [str(col).strip() for col in df.columns]
        df.dropna(how='all', inplace=True)
        nama_usaha_col = next((col for col in df.columns if 'NAMA' in str(col).upper() and 'USAHA' in str(col).upper()), df.columns[1])
        jenis_usaha_col = next((col for col in df.columns if 'JENIS' in str(col).upper()), df.columns[0])
        df = df.rename(columns={nama_usaha_col: 'destinasi', jenis_usaha_col: 'jenis'})
        
        months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
        existing_months = [col for col in months if col in df.columns]
        if not existing_months: return pd.DataFrame()
        
        id_cols_exist = [col for col in ['destinasi', 'jenis'] if col in df.columns]
        df.dropna(subset=id_cols_exist, inplace=True)
        df_filtered = df[pd.to_numeric(df['destinasi'], errors='coerce').isna()]
        keywords_to_exclude = ['JUMLAH', 'KETERANGAN', 'Jumlah', 'TOTAL']
        mask = ~df_filtered['destinasi'].astype(str).str.contains('|'.join(keywords_to_exclude), na=False, case=False)
        df_filtered = df_filtered[mask]
        if df_filtered.empty: return pd.DataFrame()

        df_long = df_filtered.melt(id_vars=id_cols_exist, value_vars=existing_months, var_name='bulan_str', value_name='jumlah_wisatawan')
        df_long['jumlah_wisatawan'] = pd.to_numeric(df_long['jumlah_wisatawan'], errors='coerce').fillna(0)
        month_map = {nama: i+1 for i, nama in enumerate(months)}
        df_long['bulan'] = df_long['bulan_str'].map(month_map)
        df_long['tahun'] = year
        df_long.dropna(subset=['tahun', 'bulan'], inplace=True)
        if df_long.empty: return pd.DataFrame()
        df_long['tanggal'] = pd.to_datetime(df_long['tahun'].astype(str) + '-' + df_long['bulan'].astype(str) + '-01', errors='coerce')
        df_long.dropna(subset=['tanggal'], inplace=True)
        return df_long

    try:
        source = None
        if uploaded_file:
            # Jika file diunggah oleh pengguna, gunakan file tersebut
            source = uploaded_file
            st.success(f"File '{uploaded_file.name}' berhasil dibaca. Menganalisis data baru...")
        else:
            # Jika tidak, gunakan file default
            script_dir = Path(__file__).parent 
            source = script_dir / 'DDA2025_DinasPariwisata.xlsx'
            if not Path(source).exists():
                st.error(f"File default 'DDA2025_DinasPariwisata.xlsx' tidak ditemukan.")
                return None, None, None, None, None, None

        xls = pd.ExcelFile(source, engine='openpyxl')
        
        all_cleaned_dfs = []
        data_sheets = [s for s in xls.sheet_names if s.lower().startswith("data ")]
        
        if not data_sheets:
            st.error("Tidak ditemukan sheet dengan format nama 'Data [Tahun]' di dalam file Excel.")
            return None, None, None, None, None, None

        for sheet_name in data_sheets:
            try:
                match = re.search(r'(\d{4})', sheet_name)
                if not match:
                    st.warning(f"Sheet '{sheet_name}' diabaikan karena tidak mengandung format tahun (YYYY).")
                    continue
                
                year = int(match.group(1))
                
                header_row = find_header_row(xls, sheet_name)
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, skiprows=header_row)
                df_clean = clean_sheet(df_raw, year)
                
                if not df_clean.empty:
                    all_cleaned_dfs.append(df_clean)
                    
            except Exception as e:
                st.warning(f"Gagal memproses sheet '{sheet_name}': {e}. Sheet ini akan diabaikan.")
                continue
            
        if not all_cleaned_dfs:
            st.error("Tidak ada data valid yang dapat diproses dari sheet yang ditemukan.")
            return None, None, None, None, None, None

        df_full = pd.concat(all_cleaned_dfs, ignore_index=True)

        if df_full.empty:
            st.error("Gagal menggabungkan data dari semua sheet yang valid.")
            return None, None, None, None, None, None

        df_agregat = df_full.groupby('tanggal')['jumlah_wisatawan'].sum().reset_index().set_index('tanggal')
        if df_agregat.empty:
            st.error("Data agregat kosong.")
            return None, None, None, None, None, None

        full_date_range = pd.date_range(start=df_agregat.index.min(), end=df_agregat.index.max(), freq='MS')
        df_total = df_agregat.reindex(full_date_range)
        df_total['jumlah_wisatawan'] = df_total['jumlah_wisatawan'].interpolate(method='linear').fillna(0)

        df_feat = df_total.copy()
        df_feat['bulan'] = df_feat.index.month
        df_feat['kuartal'] = df_feat.index.quarter
        df_feat['tahun'] = df_feat.index.year
        df_feat['lag_1'] = df_feat['jumlah_wisatawan'].shift(1)
        df_feat['rolling_mean_3'] = df_feat['jumlah_wisatawan'].shift(1).rolling(window=3).mean()
        df_feat['is_libur_puncak'] = df_feat['bulan'].isin([6, 7, 12]).astype(int)
        df_feat = df_feat.fillna(0)

        features = ['bulan', 'kuartal', 'tahun', 'lag_1', 'rolling_mean_3', 'is_libur_puncak']
        X, y = df_feat[features], df_feat['jumlah_wisatawan']

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X, y, verbose=False)

        yearly_insights = {}
        if 'tahun' in df_full.columns and pd.api.types.is_datetime64_any_dtype(df_full['tanggal']):
            years = sorted(df_full['tahun'].unique())
            for year in years:
                df_year = df_full[df_full['tahun'] == year]
                if not df_year.empty:
                    total_visits = df_year['jumlah_wisatawan'].sum()
                    monthly_visits = df_year.groupby(df_year['tanggal'].dt.month_name())['jumlah_wisatawan'].sum()
                    peak_month = monthly_visits.idxmax() if not monthly_visits.empty else "N/A"
                    yearly_insights[year] = {
                        'total': total_visits,
                        'peak_month': peak_month
                    }
        
        metrics = {
            "yearly_insights": yearly_insights
        }

        return df_full, df_total, model, features, df_feat, metrics

    except Exception as e:
        st.error(f"Error fatal saat memuat data: {str(e)}. Pastikan format file Excel sesuai.")
        return None, None, None, None, None, None


def create_features_for_dest(df):
    df_feat = df.copy()
    df_feat['bulan'] = df_feat.index.month
    df_feat['tahun'] = df_feat.index.year
    df_feat['lag_1'] = df_feat['jumlah_wisatawan'].shift(1)
    return df_feat.fillna(0)

def create_metric_card(title, value, subtitle=""):
    # Fungsi ini sekarang hanya mengembalikan string HTML, styling utama ada di CSS.
    return f"""<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div><div class="metric-subtitle">{subtitle}</div></div>"""

def format_number(num):
    if num > 1_000_000: return f"{num / 1_000_000:.1f}M"
    elif num > 1_000: return f"{num / 1_000:.1f}K"
    return str(int(num))

@st.cache_data
def get_destination_forecast(dest_name, _df_full):
    df_dest = _df_full[_df_full['destinasi'] == dest_name].groupby('tanggal')['jumlah_wisatawan'].sum().reset_index()
    df_dest = df_dest.set_index('tanggal').reindex(pd.date_range(start=_df_full['tanggal'].min(), end=_df_full['tanggal'].max(), freq='MS')).fillna(0)
    
    if len(df_dest[df_dest['jumlah_wisatawan'] > 0]) < 12:
        return None, None, None, None, None, None, None

    total_visitors = df_dest['jumlah_wisatawan'].sum()
    peak_month_data = df_dest.groupby(df_dest.index.month_name())['jumlah_wisatawan'].sum()
    peak_month = peak_month_data.idxmax() if not peak_month_data.empty else "N/A"
    insights = {"total": total_visitors, "peak_month": peak_month}

    train_size = int(len(df_dest) * 0.8)
    train, test = df_dest.iloc[:train_size], df_dest.iloc[train_size:]
    
    if len(test) < 3:
        return df_dest, None, None, insights, None, None, None

    metrics = {}
    models_for_forecast = {}
    hist_predictions = pd.DataFrame(index=df_dest.index)
    dest_features = ['bulan', 'tahun', 'lag_1']
    
    try:
        sarima_fit = SARIMAX(train['jumlah_wisatawan'], order=(1,1,1), seasonal_order=(1,0,0,12)).fit(disp=False)
        sarima_pred = sarima_fit.forecast(len(test))
        metrics['SARIMA'] = {'MAPE': mean_absolute_percentage_error(test['jumlah_wisatawan'], sarima_pred), 'RMSE': np.sqrt(mean_squared_error(test['jumlah_wisatawan'], sarima_pred))}
        full_sarima = SARIMAX(df_dest['jumlah_wisatawan'], order=(1,1,1), seasonal_order=(1,0,0,12)).fit(disp=False)
        models_for_forecast['SARIMA'] = full_sarima
        hist_predictions['SARIMA'] = full_sarima.predict(start=df_dest.index[0], end=df_dest.index[-1])
    except:
        metrics['SARIMA'] = {'MAPE': np.nan, 'RMSE': np.nan}

    try:
        prophet_train_df = train.reset_index().rename(columns={'index':'ds', 'jumlah_wisatawan':'y'})
        prophet_model = Prophet().fit(prophet_train_df)
        future = prophet_model.make_future_dataframe(periods=len(test), freq='MS')
        prophet_pred = prophet_model.predict(future)['yhat'][-len(test):]
        metrics['Prophet'] = {'MAPE': mean_absolute_percentage_error(test['jumlah_wisatawan'], prophet_pred), 'RMSE': np.sqrt(mean_squared_error(test['jumlah_wisatawan'], prophet_pred))}
        full_prophet_df = df_dest.reset_index().rename(columns={'index':'ds', 'jumlah_wisatawan':'y'})
        full_prophet = Prophet().fit(full_prophet_df)
        models_for_forecast['Prophet'] = full_prophet
        hist_predictions['Prophet'] = full_prophet.predict(full_prophet.make_future_dataframe(periods=0, freq='MS'))['yhat'].values
    except:
        metrics['Prophet'] = {'MAPE': np.nan, 'RMSE': np.nan}

    try:
        train_feat, test_feat = create_features_for_dest(train), create_features_for_dest(test)
        xgb_model = xgb.XGBRegressor(n_estimators=100).fit(train_feat[dest_features], train_feat['jumlah_wisatawan'])
        xgb_pred = xgb_model.predict(test_feat[dest_features])
        metrics['XGBoost'] = {'MAPE': mean_absolute_percentage_error(test['jumlah_wisatawan'], xgb_pred), 'RMSE': np.sqrt(mean_squared_error(test['jumlah_wisatawan'], xgb_pred))}
        full_xgb = xgb.XGBRegressor(n_estimators=100).fit(create_features_for_dest(df_dest)[dest_features], df_dest['jumlah_wisatawan'])
        models_for_forecast['XGBoost'] = full_xgb
        hist_predictions['XGBoost'] = full_xgb.predict(create_features_for_dest(df_dest)[dest_features])
    except:
            metrics['XGBoost'] = {'MAPE': np.nan, 'RMSE': np.nan}

    future_dates = pd.date_range(start=df_dest.index.max() + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast = pd.DataFrame(index=future_dates)
    
    if 'SARIMA' in models_for_forecast: forecast['SARIMA'] = models_for_forecast['SARIMA'].forecast(12)
    if 'Prophet' in models_for_forecast:
        prophet_future = models_for_forecast['Prophet'].make_future_dataframe(periods=12, freq='MS')
        forecast['Prophet'] = models_for_forecast['Prophet'].predict(prophet_future)['yhat'].values[-12:]
    if 'XGBoost' in models_for_forecast:
        last_data = create_features_for_dest(df_dest)
        xgb_preds = []
        for date in future_dates:
            feat_pred = pd.DataFrame({'bulan':[date.month], 'tahun':[date.year], 'lag_1':[last_data['jumlah_wisatawan'].iloc[-1]]})
            pred = models_for_forecast['XGBoost'].predict(feat_pred[dest_features])[0]
            xgb_preds.append(pred)
            new_row = pd.DataFrame({'jumlah_wisatawan':[pred]}, index=[date])
            last_data = pd.concat([last_data, create_features_for_dest(new_row)])
        forecast['XGBoost'] = xgb_preds

    forecast[forecast < 0] = 0
    hist_predictions[hist_predictions < 0] = 0
    metrics_df = pd.DataFrame(metrics).T.dropna()
    
    best_model_name = "N/A"
    best_forecast_series = None
    if not metrics_df.empty:
        best_model_name = metrics_df['RMSE'].idxmin()
        best_forecast_series = forecast[best_model_name]

    return df_dest, hist_predictions, forecast, insights, metrics_df, best_model_name, best_forecast_series

@st.cache_data
def get_all_destinations_forecast_wide(_df_full):
    all_destinations = sorted(_df_full['destinasi'].dropna().unique().tolist())
    all_forecasts_list = []

    for dest in all_destinations:
        _, _, _, _, _, _, best_forecast = get_destination_forecast(dest, _df_full)
        if best_forecast is not None:
            df_forecast = best_forecast.reset_index()
            df_forecast.columns = ['tanggal', 'prediksi_pengunjung']
            df_forecast['destinasi'] = dest
            all_forecasts_list.append(df_forecast)

    if not all_forecasts_list:
        return pd.DataFrame()

    long_forecast_df = pd.concat(all_forecasts_list, ignore_index=True)
    wide_forecast = pd.pivot_table(long_forecast_df,
                                   values='prediksi_pengunjung',
                                   index='destinasi',
                                   columns='tanggal',
                                   fill_value=0)
    
    wide_forecast.columns = [d.strftime('%B %Y') for d in wide_forecast.columns]
    
    return wide_forecast.astype(int)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.markdown("### 🏔️ Navigasi")
    page_options = [
        "🏙️ Insight Kota Batu & Peramalan", 
        "🏆 Peramalan per Destinasi", 
        "📈 Tren Historis"
    ]
    page = st.selectbox("Pilih Halaman", page_options, key="navigasi_utama")
    st.markdown("---")
    st.markdown(f"**📅 Tanggal:** {datetime.now().strftime('%d %B %Y')}")
    st.markdown("**🔄 Versi:** 20.2 (Wide Tabel Destinasi)")
    st.markdown("---")
    
    # === FITUR BARU: UNGGAH FILE ===
    st.markdown("### ⚙️ Pengaturan")
    uploaded_file = st.file_uploader(
        label="Unggah file data baru (.xlsx)",
        type="xlsx",
        help="Analisis dataset pariwisata Anda sendiri dengan mengunggah file .xlsx baru. Pastikan format kolom dan sheet sama."
    )
    # ===============================

# --- MEMUAT DATA DI AWAL ---
# Memanggil fungsi dengan file yang diunggah sebagai argumen
data_tuple = load_and_process_data(uploaded_file)
if data_tuple and all(item is not None for item in data_tuple):
    df_full, df_total, model_xgb, features, df_feat, dynamic_metrics = data_tuple
else:
    # Jika data gagal dimuat (baik default maupun unggahan), dashboard tidak akan melanjutkan
    df_full, df_total, model_xgb, features, df_feat, dynamic_metrics = (None,) * 6
    st.stop()


# --- HALAMAN GABUNGAN: INSIGHT KOTA BATU & PERAMALAN ---
if page == "🏙️ Insight Kota Batu & Peramalan":
    st.title("🏔️ Insight Kunjungan Wisatawan Kota Batu")
    
    if dynamic_metrics and 'yearly_insights' in dynamic_metrics:
        yearly_insights = dynamic_metrics.get('yearly_insights', {})
        years = sorted(yearly_insights.keys())
        
        if years:
            tabs = st.tabs([f"📈 Insight Tahun {int(year)}" for year in years])
            
            for i, year in enumerate(years):
                with tabs[i]:
                    insights = yearly_insights[year]
                    total_val = insights.get('total', 0)
                    peak_month_val = insights.get('peak_month', 'N/A')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(create_metric_card("Total Kunjungan Tahunan", format_number(total_val), f"Total wisatawan di tahun {int(year)}"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric_card("Bulan Puncak Kunjungan", peak_month_val, f"Bulan tersibuk di tahun {int(year)}"), unsafe_allow_html=True)
    else:
        st.error("Gagal menghitung metrik tahunan. Pastikan file 'DinasPariwisata.xlsx' tersedia dan datanya valid.")
    
    st.markdown("<hr>", unsafe_allow_html=True)

    if df_feat is not None:
        @st.cache_data
        def train_all_models(data_feat):
            trained_models = {}
            trained_models['SARIMA'] = SARIMAX(data_feat['jumlah_wisatawan'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit(disp=False)
            prophet_df = data_feat.reset_index().rename(columns={'index': 'ds', 'jumlah_wisatawan': 'y'})
            trained_models['Prophet'] = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(prophet_df)
            trained_models['XGBoost'] = model_xgb 
            return trained_models

        models = train_all_models(df_feat)
        
        model_options = ['SARIMA', 'Prophet', 'XGBoost']
        
        st.markdown("### Visualiasi Total Kunjungan Wisatawan dan Peramalan")
        st.markdown("**Pilih model peramalan:**")
        cols = st.columns(len(model_options))
        selected_models = []
        for i, model_name in enumerate(model_options):
            with cols[i]:
                if st.checkbox(model_name, value=True, key=f"model_cb_{model_name}"):
                    selected_models.append(model_name)
        
        placeholder = st.empty()
        
        with st.spinner("Menghitung prediksi dan membuat visualisasi..."):
            all_predictions = pd.DataFrame(index=df_feat.index)
            if 'SARIMA' in selected_models: all_predictions['SARIMA'] = models['SARIMA'].predict(start=df_feat.index[0], end=df_feat.index[-1])
            if 'Prophet' in selected_models:
                future_prophet_hist = models['Prophet'].make_future_dataframe(periods=0, freq='MS')
                prophet_pred_hist = models['Prophet'].predict(future_prophet_hist)
                all_predictions['Prophet'] = prophet_pred_hist['yhat'].values
            if 'XGBoost' in selected_models: all_predictions['XGBoost'] = models['XGBoost'].predict(df_feat[features])
            
            months_to_forecast_multi = 12
            future_dates = pd.date_range(start=df_feat.index.max() + pd.DateOffset(months=1), periods=months_to_forecast_multi, freq='MS')
            forecast_df = pd.DataFrame(index=future_dates)
            if 'SARIMA' in selected_models: forecast_df['SARIMA'] = models['SARIMA'].forecast(steps=months_to_forecast_multi)
            if 'Prophet' in selected_models:
                future_prophet_fc = models['Prophet'].make_future_dataframe(periods=months_to_forecast_multi, freq='MS')
                prophet_fc_vals = models['Prophet'].predict(future_prophet_fc)['yhat'].values[-months_to_forecast_multi:]
                forecast_df['Prophet'] = prophet_fc_vals
            if 'XGBoost' in selected_models:
                last_data = df_feat.copy()
                xgb_forecasts = []
                for date in future_dates:
                    features_pred = pd.DataFrame({'bulan': [date.month], 'kuartal': [date.quarter], 'tahun': [date.year], 'lag_1': [last_data['jumlah_wisatawan'].iloc[-1]], 'rolling_mean_3': [last_data['jumlah_wisatawan'].tail(3).mean()], 'is_libur_puncak': [1 if date.month in [6, 7, 12] else 0]})
                    prediction = models['XGBoost'].predict(features_pred[features])[0]
                    xgb_forecasts.append(prediction)
                    new_row = pd.DataFrame({'jumlah_wisatawan': [prediction]}, index=[date])
                    last_data = pd.concat([last_data, new_row])
                forecast_df['XGBoost'] = xgb_forecasts
            
            all_predictions[all_predictions < 0] = 0
            forecast_df[forecast_df < 0] = 0

            with placeholder.container():
                fig_compare = go.Figure()
                last_hist_date_total = df_total.index.max()
                fig_compare.add_trace(go.Scatter(x=df_total.index, y=df_total['jumlah_wisatawan'], mode='lines', name='Data Aktual', line=dict(color='orange', width=4, dash='solid')))
                
                for model_name in selected_models:
                    color = MODEL_COLORS.get(model_name, '#808080')
                    fig_compare.add_trace(go.Scatter(x=all_predictions.index, y=all_predictions[model_name], mode='lines', name=f'Prediksi Historis ({model_name})', line=dict(color=color, width=2, dash='dash')))
                    if model_name in forecast_df.columns:
                        fig_compare.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[model_name], mode='lines+markers', name=f'Peramalan ({model_name})', line=dict(color=color, width=2, dash='dot'), marker=dict(size=5)))

                fig_compare.add_vline(x=last_hist_date_total, line_width=2, line_dash="dash", line_color="grey")
                fig_compare.add_annotation(x=last_hist_date_total, y=0.98, yref="paper", text="Mulai Peramalan", showarrow=False, font=dict(color="grey"), xanchor="left", xshift=5)
                fig_compare.update_layout(title='Perbandingan Data Aktual dan Prediksi Model', xaxis_title='Bulan', yaxis_title='Jumlah Wisatawan', legend_title='Keterangan', hovermode='x unified')
                fig_compare.update_xaxes(dtick="M1", tickformat="%b\n%Y") 
                st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")
        st.markdown("### Data Asli per Tahun")
        st.markdown("Tabel di bawah ini menampilkan data per destinasi, dipisahkan per tahun.")

        if df_full is not None and not df_full.empty:
            years_asli = sorted(df_full['tahun'].unique())
            tabs_asli = st.tabs([f"📄 Data Tahun {int(year)}" for year in years_asli])
            
            months_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
            
            for i, year in enumerate(years_asli):
                with tabs_asli[i]:
                    df_year = df_full[df_full['tahun'] == year]
                    
                    if not df_year.empty:
                        wide_table = pd.pivot_table(df_year,
                                                    values='jumlah_wisatawan',
                                                    index='destinasi',
                                                    columns='bulan_str',
                                                    aggfunc='sum',
                                                    fill_value=0)
                        
                        existing_months_in_order = [m for m in months_order if m in wide_table.columns]
                        wide_table = wide_table[existing_months_in_order]

                        st.dataframe(wide_table.astype(int), use_container_width=True)

                        csv_wide = convert_df_to_csv(wide_table.reset_index())
                        st.download_button(
                            label=f"📥 Unduh Data {int(year)} (CSV)",
                            data=csv_wide,
                            file_name=f"data_pariwisata_{int(year)}.csv",
                            mime="text/csv",
                            key=f"download_btn_{year}"
                        )
                    else:
                        st.info(f"Tidak ada data untuk tahun {int(year)}.")
        else:
            st.warning("Data asli tidak dapat dimuat untuk ditampilkan.")
        st.markdown("---")

        # --- BAGIAN KODE YANG DIKEMBALIKAN ---
        st.markdown("### Analisis Kinerja Model & Faktor Pendorong")
        st.markdown("##### Metrik Kinerja Model (pada Seluruh Data Historis)")
        st.markdown("Tabel ini membandingkan performa setiap model dalam memprediksi data historis. Metrik yang lebih rendah (kecuali R²) menunjukkan performa yang lebih baik.")

        
        comparison_metrics = {}
        y_true = df_feat['jumlah_wisatawan']
        for model_name in selected_models:
            if model_name in all_predictions.columns:
                y_pred = all_predictions[model_name]
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                comparison_metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}
        
        best_total_model_name = ""
        if comparison_metrics:
            metrics_df = pd.DataFrame.from_dict(comparison_metrics, orient='index').dropna()
            if not metrics_df.empty:
                best_total_model_name = metrics_df['RMSE'].idxmin()
            st.dataframe(metrics_df.style.format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'MAPE': '{:.2%}', 'R²': '{:.2%}'}), use_container_width=True)
            st.markdown("""
            <div style="margin-top: 1.5rem; padding: 1rem; border: 1px solid rgba(128,128,128,0.1); border-radius: 8px;">
            <h6 style="margin-bottom: 1rem;">Penjelasan Indikator</h6>
            <ul>
                <li><strong>MAE (Mean Absolute Error):</strong> Memberi gambaran seberapa besar kesalahan prediksi dalam satuan asli (misal: jumlah wisatawan). <strong>Semakin mendekati 0, semakin baik.</strong></li>
                <li><strong>RMSE (Root Mean Squared Error):</strong> Mirip dengan MAE, Berguna untuk melihat adanya prediksi yang sangat melenceng. <strong>Semakin mendekati 0, semakin baik.</strong></li>
                <li><strong>MAPE (Mean Absolute Percentage Error):</strong> Rata-rata persentase kesalahan. Contoh: MAPE 15% berarti rata-rata prediksi meleset 15% dari nilai aktual. <strong>Semakin mendekati 0%, semakin baik.</strong></li>
                <li><strong>R² (R-squared):</strong> Mengukur seberapa besar persentase variasi data yang dapat dijelaskan oleh model. Contoh: R² 0.85 berarti model mampu menjelaskan 85% pola data. <strong>Semakin mendekati 1, semakin baik.</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        if 'XGBoost' in selected_models:
            st.markdown("---")
            st.markdown("##### Faktor Pendorong Utama Prediksi (Khusus XGBoost)")
            feature_importance = pd.DataFrame({'feature': features, 'importance': models['XGBoost'].feature_importances_}).sort_values('importance', ascending=True)
            feature_names_map = {'lag_1': 'Jml Wisatawan Bulan Lalu', 'rolling_mean_3': 'Rata-rata 3 Bulan Terakhir', 'bulan': 'Bulan', 'is_libur_puncak': 'Status Musim Liburan', 'tahun': 'Tahun', 'kuartal': 'Kuartal'}
            feature_importance['feature_display'] = feature_importance['feature'].map(feature_names_map)
            fig_feat = px.bar(feature_importance, x='importance', y='feature_display', orientation='h', title='Tingkat Kepentingan Faktor dalam Model XGBoost', labels={'feature_display': 'Faktor (Fitur)', 'importance': 'Nilai Kepentingan'}, color='importance', color_continuous_scale=px.colors.sequential.Viridis)
            fig_feat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)'), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_feat, use_container_width=True)

    else:
        st.markdown('<div class="error-msg">❌ Gagal memuat data untuk perbandingan model.</div>', unsafe_allow_html=True)
# ... sisa kode tidak berubah dan tetap sama


# --- HALAMAN ANALISIS DESTINASI ---
elif page == "🏆 Peramalan per Destinasi":
    st.title("🏆 Peramalan per Destinasi")
    st.markdown("Pilih satu atau beberapa destinasi untuk melihat analisis mendalam, termasuk bulan puncak, akurasi model, dan peramalan kunjungan.")

    if df_full is not None:
        all_destinations = sorted(df_full['destinasi'].dropna().unique().tolist())
        top_5_default = df_full.groupby('destinasi')['jumlah_wisatawan'].sum().nlargest(5).index.tolist()
        
        selected_destinations = st.multiselect(
            "Pilih Destinasi untuk Dianalisis:",
            options=all_destinations,
            default=top_5_default
        )
        
        forecasts_to_download = []

        if not selected_destinations:
            st.info("Silakan pilih minimal satu destinasi untuk memulai analisis.")
        else:
            for dest in selected_destinations:
                with st.spinner(f"Menganalisis dan meramal untuk {dest}..."):
                    hist_data, hist_preds, forecast_data, insights, metrics_df, best_model, best_forecast = get_destination_forecast(dest, df_full)

                if hist_data is None:
                    st.warning(f"Data untuk **{dest}** tidak cukup untuk membuat analisis & peramalan.")
                    continue

                if best_forecast is not None:
                    df_to_add = best_forecast.reset_index()
                    df_to_add.columns = ['tanggal', 'prediksi_pengunjung']
                    df_to_add['destinasi'] = dest
                    df_to_add['model_terbaik'] = best_model
                    forecasts_to_download.append(df_to_add)

                st.markdown(f"---")
                st.markdown(f"<h3 style='text-align: center; color: #667eea;'>Analisis untuk: {dest}</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1: st.metric(label="Total Pengunjung Tercatat", value=f"{int(insights['total']):,}")
                with col2: st.metric(label="Bulan Puncak Kunjungan", value=insights['peak_month'])

                if metrics_df is not None and not metrics_df.empty:
                    st.markdown("##### Matriks Akurasi Model (pada Data Uji)")
                    st.caption("MAPE: Rata-rata persentase kesalahan. RMSE: Rata-rata kesalahan dalam satuan pengunjung. Nilai lebih kecil lebih baik.")
                    st.dataframe(metrics_df.style.format({'MAPE': '{:.2%}', 'RMSE': '{:,.0f}'}), use_container_width=True)
                
                fig = go.Figure()
                last_hist_date = hist_data.index.max()
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['jumlah_wisatawan'], mode='lines', name='Data Aktual', line=dict(color='orange', width=3, dash='solid')))
                
                if hist_preds is not None and forecast_data is not None:
                    for model_name in forecast_data.columns:
                        color = MODEL_COLORS.get(model_name, '#808080')
                        fig.add_trace(go.Scatter(x=hist_preds.index, y=hist_preds[model_name], mode='lines', name=f'Prediksi Historis ({model_name})', line=dict(color=color, width=2, dash='dash')))
                        fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data[model_name], mode='lines+markers', name=f'Peramalan ({model_name})', line=dict(color=color, width=2, dash='dot'), marker=dict(size=5)))

                fig.add_vline(x=last_hist_date, line_width=2, line_dash="dash", line_color="grey")
                fig.add_annotation(x=last_hist_date, y=0.98, yref="paper", text="Mulai Peramalan", showarrow=False, font=dict(color="grey"), xanchor="left", xshift=5)
                fig.update_layout(
                    title=f'Data Historis vs Peramalan untuk {dest}', 
                    xaxis_title='Bulan', 
                    yaxis_title='Jumlah Pengunjung', 
                    legend_title='Keterangan',
                    hovermode='x unified'
                )
                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
                
                st.plotly_chart(fig, use_container_width=True)
            
            if forecasts_to_download:
                st.markdown("---")
                st.markdown("### 📥 Unduh Data Peramalan per Destinasi")
                st.markdown("Data di bawah ini berisi peramalan 12 bulan ke depan untuk destinasi yang Anda pilih, menggunakan model terbaik untuk masing-masing destinasi, disajikan dalam format tabel.")

                combined_forecast_df = pd.concat(forecasts_to_download, ignore_index=True)
                combined_forecast_df['prediksi_pengunjung'] = combined_forecast_df['prediksi_pengunjung'].astype(int)
                combined_forecast_df['bulan_tahun'] = combined_forecast_df['tanggal'].dt.strftime('%B %Y')
                
                forecast_wide_table = pd.pivot_table(
                    combined_forecast_df,
                    values='prediksi_pengunjung',
                    index='destinasi',
                    columns='bulan_tahun',
                    aggfunc='sum',
                    fill_value=0
                )
                
                unique_months = pd.to_datetime(combined_forecast_df['bulan_tahun'].unique(), format='%B %Y')
                sorted_months_dt = sorted(unique_months)
                sorted_month_columns = [dt.strftime('%B %Y') for dt in sorted_months_dt]
                forecast_wide_table = forecast_wide_table[sorted_month_columns]

                st.dataframe(forecast_wide_table, use_container_width=True)

                csv_data_wide = convert_df_to_csv(forecast_wide_table.reset_index())
                st.download_button(
                    label="📥 Unduh Peramalan Destinasi (Format Tabel)",
                    data=csv_data_wide,
                    file_name="peramalan_per_destinasi_tabel.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    else:
        st.error("Gagal memuat data destinasi. Tidak dapat menampilkan analisis.")


# --- HALAMAN TREN HISTORIS ---
elif page == "📈 Tren Historis":
    st.title("📈 Tren Historis Kunjungan Wisatawan")
    if df_total is not None:
        tab1, tab2, tab3 = st.tabs(["📊 Tren Utama", "📅 Analisis Musiman", "🔗 Analisis Dependensi Temporal"])
        with tab1:
            st.markdown("#### Perkembangan Kunjungan Wisatawan (2022 - 2024)")
            st.markdown("Garis ini menunjukkan total kunjungan bulanan. Data yang kosong telah diisi menggunakan metode interpolasi linear untuk menjaga kontinuitas tren.")
            
            fig_line = px.line(df_total.reset_index(), x='index', y='jumlah_wisatawan', markers=True, color_discrete_sequence=['#667eea'], labels={'index': 'Tanggal', 'jumlah_wisatawan': 'Jumlah Wisatawan'})
            
            fig_line.update_layout(xaxis_title="Periode", yaxis_title="Jumlah Wisatawan", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)'), hovermode='x unified')
            fig_line.update_traces(line=dict(width=3), marker=dict(size=8), hovertemplate='<b>%{x|%B %Y}</b><br>Wisatawan: %{y:,.0f}<extra></extra>')
            fig_line.update_xaxes(dtick="M1", tickformat="%b\n%Y")
            st.plotly_chart(fig_line, use_container_width=True)
        with tab2:
            st.markdown("#### Validasi Hipotesis 1: Pola Musiman Kunjungan")
            st.markdown("""*Box plot* bulanan ini memvalidasi **Hipotesis 1 (Musiman)** dengan menunjukkan distribusi jumlah wisatawan setiap bulan. Dari plot, kita dapat melihat adanya *high season* (puncak kunjungan) di pertengahan (Juni-Juli) dan akhir tahun (Desember), serta *low season* pada bulan-bulan lainnya.""")
            df_monthly = df_total.copy()
            df_monthly['bulan_nama'] = df_monthly.index.month_name()
            fig_box = px.box(df_monthly, x='bulan_nama', y='jumlah_wisatawan', color_discrete_sequence=['#764ba2'], category_orders={"bulan_nama": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]})
            fig_box.update_layout(title='Distribusi Kunjungan Wisatawan per Bulan', xaxis_title='Bulan', yaxis_title='Jumlah Wisatawan', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)'))
            st.plotly_chart(fig_box, use_container_width=True)
        with tab3:
            st.markdown("#### Validasi Hipotesis 2: Dependensi Temporal (Hubungan Antar Waktu)")
            st.markdown("""Plot *Autocorrelation Function* (ACF) ini menguji **Hipotesis 2 (Dependensi Temporal)**. Plot mengukur korelasi data suatu bulan dengan bulan-bulan sebelumnya (*lags*). Tingginya korelasi pada beberapa *lag* pertama dan pada *lag* ke-12 (batang yang melewati area biru) menunjukkan bahwa jumlah wisatawan bulan ini dipengaruhi oleh bulan lalu dan bulan yang sama di tahun sebelumnya. Ini membenarkan penggunaan fitur *lag* dalam model peramalan.""")
            acf_values, confint = acf(df_total['jumlah_wisatawan'].dropna(), nlags=12, alpha=0.05)
            ci = 1.96 / np.sqrt(len(df_total))
            lags = np.arange(len(acf_values))
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=lags, y=acf_values, name='Autocorrelation', marker_color='#667eea'))
            fig_acf.add_shape(type='line', x0=-0.5, y0=ci, x1=12.5, y1=ci, line=dict(color='rgba(128,128,128,0.5)', dash='dash'))
            fig_acf.add_shape(type='line', x0=-0.5, y0=-ci, x1=12.5, y1=-ci, line=dict(color='rgba(128,128,128,0.5)', dash='dash'))
            fig_acf.update_layout(title='Autocorrelation (ACF) Kunjungan Wisatawan Bulanan', xaxis_title='Lag (Selisih Bulan)', yaxis_title='Nilai Korelasi', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='var(--text-color)'))
            st.plotly_chart(fig_acf, use_container_width=True)
    else:
        st.markdown('<div class="error-msg">❌ Gagal memuat data historis.</div>', unsafe_allow_html=True)


# --- FOOTER ---
st.markdown("""
    <div style="background: transparent; border-top: 1px solid rgba(128, 128, 128, 0.2); color: var(--text-color); padding: 2rem; margin-top: 2rem; text-align: center;">
        <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🏔️ Dinas Pariwisata Kota Batu</h4>
        <p style="margin: 0; opacity: 0.8;">© 2025 - Semua hak cipta dilindungi</p>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.6;">
            Dikembangkan dengan ❤️ untuk kemajuan pariwisata Kota Batu
        </div>
    </div>
""", unsafe_allow_html=True)