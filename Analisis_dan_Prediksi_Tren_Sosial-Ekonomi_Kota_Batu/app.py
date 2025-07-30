# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - VERSI FINAL DENGAN PANDUAN (DINAMIS V2)

Deskripsi:
Aplikasi web interaktif yang menampilkan analisis dan prediksi data sosial-ekonomi
untuk Kota Batu, dengan fitur unggah file dan panduan penggunaan terintegrasi.
Versi ini dirancang untuk secara dinamis menyesuaikan tampilan berdasarkan rentang tahun data yang dimuat.

Fitur Utama:
1.  Fitur unggah file Excel untuk analisis data tahun-tahun selanjutnya.
2.  Panduan penggunaan interaktif di halaman Beranda.
3.  Navigasi berbasis halaman (Beranda, PDRB, TPT, IPM, Kemiskinan).
4.  Dropdown interaktif untuk memilih indikator spesifik.
5.  Melatih 6 model prediksi dengan Prophet sebagai default.
6.  Menampilkan tabel data historis dan prediksi di bawah setiap grafik.
7.  Prediksi 3 tahun ke depan.
8.  Teks dan contoh data yang sepenuhnya dinamis menyesuaikan data yang diunggah.

Cara Menjalankan:
1. Pastikan pustaka terinstal: pip install streamlit pandas numpy plotly openpyxl scikit-learn statsmodels prophet
2. Simpan skrip ini sebagai `app.py`.
3. Jalankan dari terminal: `streamlit run app.py`
"""

# ==============================================================================
# 1. SETUP & IMPORT PUSTAKA
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
import datetime

# Pustaka untuk pemodelan
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Import pustaka untuk metrik evaluasi
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# ======================================================================================
# 2. KONFIGURASI HALAMAN & GAYA (STYLING)
# ======================================================================================
st.set_page_config(page_title="Dashboard Analisis Kota Batu", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background-color: #FFFFFF; color: black; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #60A5FA; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: scale(1.05); }
    .metric-value { font-size: 2.5em; font-weight: 700; margin-bottom: 10px; }
    .metric-label { font-size: 1.1em; opacity: 0.9; }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #60A5FA, #667eea);
        margin: 40px 0;
        border-radius: 2px;
    }
    .st-emotion-cache-1clstc5 { /* Target the specific tab style */
        padding-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================================================
# 3. PEMUATAN DATA & CACHING
# ======================================================================================
# Mendefinisikan DEFAULT_FILE_PATH di sini agar bisa diakses load_data
DEFAULT_FILE_PATH = 'Analisis_dan_Prediksi_Tren_Sosial-Ekonomi_Kota_Batu/Data Strategis Kota Batu 2010-2024.xlsx'

@st.cache_data
def load_data(source):
    """Memuat dan membersihkan data dari file Excel/CSV, baik dari path lokal maupun file yang diunggah."""
    data = None
    
    if hasattr(source, 'name'):
        file_name = source.name
    else:
        file_name = source

    file_extension = os.path.splitext(file_name)[1].lower()

    try:
        if file_extension == '.xlsx':
            data = pd.read_excel(source, sheet_name='Sheet1', skiprows=3, engine='openpyxl')
        elif file_extension == '.csv':
            # Asumsi header ada di baris 4 (indeks 3), jadi kita lewati 3 baris pertama
            data = pd.read_csv(source, skiprows=4)
        else:
            st.error(f"Format file '{file_extension}' tidak didukung. Mohon unggah file .xlsx atau .csv.")
            return None

    except Exception as e:
        st.error(f"Gagal membaca file: {e}. Pastikan format file sesuai dengan template.")
        return None

    # --- Common processing steps ---
    data.columns = [str(col).strip().replace('\n', '') for col in data.columns]
    
    # Mencari kolom 'Rincian' atau 'Indikator'
    indicator_col_name = None
    for col in data.columns:
        if 'RINCIAN' in col.upper() or 'INDIKATOR' in col.upper():
            indicator_col_name = col
            break
            
    if not indicator_col_name:
        st.error("Gagal menemukan kolom 'Rincian' atau 'Indikator'. Pastikan file Anda memiliki kolom ini.")
        return None

    data = data.rename(columns={indicator_col_name: 'Indikator'})
    
    # Hapus kolom 'No' dan 'Unnamed' yang tidak perlu
    cols_to_drop = [col for col in data.columns if 'NO' in col.upper() or 'UNNAMED' in col.upper()]
    data = data.drop(columns=cols_to_drop)

    data.dropna(subset=['Indikator'], inplace=True)
    data['Indikator'] = data['Indikator'].str.strip()

    year_columns = [col for col in data.columns if col.isdigit()]
    
    if not year_columns:
        st.error("Tidak ada kolom tahun yang terdeteksi. Pastikan nama kolom tahun berupa angka (misal: 2020, 2021).")
        return None

    for col in year_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data_transposed = data.set_index('Indikator')[year_columns].transpose()
    data_transposed.index = data_transposed.index.astype(str)
    
    # Interpolasi untuk mengisi data kosong
    data_transposed = data_transposed.interpolate(method='linear', limit_direction='both')

    return data_transposed.transpose()


# ======================================================================================
# 4. SIDEBAR & LOGIKA UNGGAH FILE
# ======================================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Navigasi</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Pilih Halaman Analisis:",
        ['🏠 Beranda', '💰 PDRB', '👥 Ketenagakerjaan (TPT)', '📚 IPM', '🏠 Tingkat Kemiskinan'],
        label_visibility="collapsed"
    )
    st.markdown("<hr class='section-divider' style='margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>⚙️ Pengaturan</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Unggah file data baru (.xlsx)",
        type="xlsx",
        help="Gunakan ini untuk mengunggah file data yang sudah Anda perbarui."
    )

data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif os.path.exists(DEFAULT_FILE_PATH):
    data_source = DEFAULT_FILE_PATH
else:
    st.error(f"File default tidak ditemukan di '{DEFAULT_FILE_PATH}'. Silakan unggah file data untuk memulai.")
    st.stop()

df_main = load_data(data_source)

if df_main is None:
    st.warning("Data tidak berhasil dimuat. Dashboard tidak dapat ditampilkan.")
    st.stop()

# --- Mendapatkan tahun awal dan akhir secara dinamis ---
year_cols_int = [int(c) for c in df_main.columns if str(c).isdigit()]
start_year = min(year_cols_int)
latest_year = max(year_cols_int)

if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' berhasil dimuat!")

def find_indicator_by_keywords(keywords, index):
    for idx in index:
        if all(keyword.lower() in idx.lower() for keyword in keywords):
            return idx
    return None

PDRB_ADHB = find_indicator_by_keywords(['pdrb', 'berlaku'], df_main.index)
PDRB_ADHK = find_indicator_by_keywords(['pdrb', 'konstan'], df_main.index)
ANGKATAN_KERJA = find_indicator_by_keywords(['jumlah', 'angkatan', 'kerja'], df_main.index)
PENGANGGURAN = find_indicator_by_keywords(['jumlah', 'pengangguran'], df_main.index)
TPT = find_indicator_by_keywords(['tingkat', 'pengangguran', 'terbuka'], df_main.index)
IPM = find_indicator_by_keywords(['indeks', 'pembangunan', 'manusia'], df_main.index)
KEMISKINAN = find_indicator_by_keywords(['persentase', 'penduduk', 'miskin'], df_main.index)

pdrb_options = sorted([opt for opt in [PDRB_ADHB, PDRB_ADHK] if opt])
tpt_options = sorted([opt for opt in [ANGKATAN_KERJA, PENGANGGURAN, TPT] if opt])

with st.sidebar:
    with st.container(border=True):
        st.markdown("<p style='text-align: center; font-weight: bold;'>Actual Dataset Source</p>", unsafe_allow_html=True)
        try:
            # Gunakan file yang diunggah jika ada, jika tidak, gunakan file default
            file_to_download = uploaded_file.getvalue() if uploaded_file else open(DEFAULT_FILE_PATH, "rb").read()
            download_filename = uploaded_file.name if uploaded_file else os.path.basename(DEFAULT_FILE_PATH)
            
            st.download_button(
                label="📥 Download (.xlsx)",
                data=file_to_download,
                file_name=download_filename,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        except Exception as e:
            st.error("Gagal memuat file sumber.", icon="🚨")


    available_models = ['Prophet', 'Gradient Boosting', 'Random Forest', 'Linear Regression', 'Holt-Winters', 'ARIMA']
    selected_models = st.multiselect(
        "Pilih Model untuk Ditampilkan:",
        options=available_models,
        default=['Prophet']
    )
    
    st.markdown("<hr class='section-divider' style='margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3>ℹ️ Tentang Dashboard</h3>", unsafe_allow_html=True)
    st.info(f"""
    Dashboard ini menampilkan analisis dan perbandingan prediksi untuk indikator kunci di Kota Batu.
    - **Data Historis**: {start_year}–{latest_year}
    - **Prediksi**: {latest_year + 1}–{latest_year + 3}
    - **Sumber**: BPS Kota Batu (diolah)
    """)

# ======================================================================================
# 5. FUNGSI PEMODELAN
# ======================================================================================
@st.cache_data
def generate_all_predictions(indicator_name, _df_data):
    series = _df_data.loc[indicator_name].dropna()
    years = [int(year) for year in series.index]
    series_values = series.values

    historical_fit, future_forecast, evaluation_metrics = {}, {}, {}
    n_forecast = 3
    last_year = max(years)
    future_years = list(range(last_year + 1, last_year + n_forecast + 1))

    def calculate_metrics(model_name, y_true, y_pred):
        if len(y_true) != len(y_pred) or len(y_true) < 2:
            evaluation_metrics[model_name] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE (%)': np.nan, 'R-squared': np.nan}
            return
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        evaluation_metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R-squared': r2}

    X_hist, X_future = np.array(years).reshape(-1, 1), np.array(future_years).reshape(-1, 1)
    ml_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    for name, model in ml_models.items():
        try:
            m = model.fit(X_hist, series_values)
            historical_fit[name] = pd.Series(m.predict(X_hist), index=years)
            future_forecast[name] = pd.Series(m.predict(X_future), index=future_years)
            calculate_metrics(name, series_values, historical_fit[name])
        except Exception:
            historical_fit[name], future_forecast[name] = pd.Series(np.nan, index=years), pd.Series(np.nan, index=future_years)
            calculate_metrics(name, [], [])

    ts_series = pd.Series(series_values, index=pd.to_datetime([f'{y}-12-31' for y in years]))
    try:
        arima_model = ARIMA(ts_series, order=(2, 1, 1)).fit()
        fit_vals = arima_model.fittedvalues
        fit_vals.index = fit_vals.index.year
        historical_fit['ARIMA'] = fit_vals
        future_forecast['ARIMA'] = arima_model.forecast(steps=n_forecast).set_axis(future_years)
        calculate_metrics('ARIMA', series_values[1:], fit_vals.values[1:])
    except Exception:
        historical_fit['ARIMA'], future_forecast['ARIMA'] = pd.Series(np.nan, index=years), pd.Series(np.nan, index=future_years)
        calculate_metrics('ARIMA', [], [])
    try:
        hw_model = ExponentialSmoothing(ts_series, trend='add', seasonal=None, initialization_method='estimated').fit()
        fit_vals = hw_model.fittedvalues
        fit_vals.index = fit_vals.index.year
        historical_fit['Holt-Winters'] = fit_vals
        future_forecast['Holt-Winters'] = hw_model.forecast(steps=n_forecast).set_axis(future_years)
        calculate_metrics('Holt-Winters', series_values, fit_vals.values)
    except Exception:
        historical_fit['Holt-Winters'], future_forecast['Holt-Winters'] = pd.Series(np.nan, index=years), pd.Series(np.nan, index=future_years)
        calculate_metrics('Holt-Winters', [], [])
    try:
        df_prophet = pd.DataFrame({'ds': pd.to_datetime([f'{y}-01-01' for y in years]), 'y': series_values})
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df_prophet)
        future_df = prophet_model.make_future_dataframe(periods=n_forecast, freq='AS')
        full_pred_df = prophet_model.predict(future_df)
        historical_fit['Prophet'] = pd.Series(full_pred_df.iloc[:len(years)]['yhat'].values, index=years)
        future_forecast['Prophet'] = pd.Series(full_pred_df.iloc[len(years):]['yhat'].values, index=future_years)
        calculate_metrics('Prophet', series_values, historical_fit['Prophet'])
    except Exception:
        historical_fit['Prophet'], future_forecast['Prophet'] = pd.Series(np.nan, index=years), pd.Series(np.nan, index=future_years)
        calculate_metrics('Prophet', [], [])
    return historical_fit, future_forecast, evaluation_metrics, years, future_years

# ======================================================================================
# 6. FUNGSI UNTUK MENAMPILKAN KONTEN
# ======================================================================================
def display_analysis_view(indicator_name, y_label):
    if not indicator_name:
        st.warning("Indikator tidak ditemukan dalam dataset. Mohon periksa file sumber.")
        return

    st.markdown(f"### Analisis untuk: {indicator_name}")
    st.info(f"""
    **Cara Membaca Grafik:**
    - **Area Abu-abu Terang:** Area prediksi masa depan ({latest_year + 1}–{latest_year + 3})
    - **Garis Hijau Tebal:** Data historis aktual
    - **Garis Solid Berwarna:** Kecocokan model dengan data historis (Fit).
    - **Garis Putus-Putus Berwarna:** Prediksi model untuk masa depan (Forecast).
    """)

    historical_fit, future_forecast, evaluation_metrics, hist_years, future_years = generate_all_predictions(indicator_name, df_main)
    historical_data = df_main.loc[indicator_name].dropna()

    fig = go.Figure()
    color_palette = {'Prophet': '#F59E0B', 'Gradient Boosting': '#10B981', 'Random Forest': '#3B82F6', 'Linear Regression': '#EF4444', 'Holt-Winters': '#8B5CF6', 'ARIMA': '#6366F1'}

    fig.add_vrect(x0=latest_year + 0.5, x1=latest_year + 3.5, fillcolor="rgba(230, 230, 230, 0.4)", layer="below", line_width=0, annotation_text="Area Peramalan", annotation_position="top left")
    fig.add_vline(x=latest_year + 0.5, line_width=2, line_dash="dash", line_color="grey")

    fig.add_trace(go.Scatter(x=[int(y) for y in historical_data.index], y=historical_data.values, mode='lines+markers', name='Data Aktual', line=dict(color='#059669', width=4), marker=dict(size=8)))

    for model_name in selected_models:
        if model_name not in color_palette or model_name not in historical_fit: continue
        color = color_palette[model_name]
        fit_series, forecast_series = historical_fit.get(model_name), future_forecast.get(model_name)
        if fit_series is not None and not fit_series.empty:
            fig.add_trace(go.Scatter(x=fit_series.index, y=fit_series.values, mode='lines', name=f'{model_name} (Fit)', line=dict(color=color, width=2, dash='solid'), legendgroup=model_name))
            if forecast_series is not None and not forecast_series.empty:
                connect_x, connect_y = [fit_series.index[-1]] + list(forecast_series.index), [fit_series.values[-1]] + list(forecast_series.values)
                fig.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', name=f'{model_name} (Forecast)', line=dict(color=color, width=2.5, dash='dash'), legendgroup=model_name, showlegend=False))

    fig.update_layout(height=500, xaxis_title="Tahun", yaxis_title=y_label, template="plotly_white", hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h4>🔢 Tabel Data Historis & Peramalan</h4>", unsafe_allow_html=True)
    table_df = pd.DataFrame(index=hist_years + future_years)
    table_df.index.name = "Tahun"
    table_df['Data Aktual'] = historical_data.reindex([str(y) for y in table_df.index]).values
    for model_name in selected_models:
        fit, forecast = historical_fit.get(model_name), future_forecast.get(model_name)
        if fit is not None and forecast is not None:
            table_df[model_name] = pd.concat([fit, forecast]).reindex(table_df.index).values
    st.dataframe(table_df.T.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    with st.expander("📊 Lihat Detail Metrik Evaluasi Model (Kecocokan Historis)"):
        st.info("Metrik ini mengukur seberapa baik prediksi model cocok dengan data historis. Nilai yang lebih baik ditandai dengan latar belakang ungu.", icon="💡")
        if selected_models:
            metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
            if not metrics_df.empty:
                metrics_df_filtered = metrics_df.loc[metrics_df.index.intersection(selected_models)]
                if not metrics_df_filtered.empty:
                    st.dataframe(metrics_df_filtered.style.format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'MAPE (%)': '{:.2f}%', 'R-squared': '{:.4f}'}, na_rep="-")
                                .highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color="#C9C5DD", axis=0)
                                .highlight_max(subset=['R-squared'], color="#C9C5DD", axis=0), use_container_width=True)
        else:
            st.warning("Pilih minimal satu model untuk melihat metrik evaluasinya.")


# ======================================================================================
# 7. KONTEN UTAMA (BERDASARKAN NAVIGASI)
# ======================================================================================
st.markdown("<h1 style='text-align: center;'>🏛️ Dashboard Analisis & Prediksi Ekonomi-Sosial Kota Batu</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #A0A0A0; font-size: 1.2em;'>Analisis Komparatif Multi-Model (Data Historis: {start_year}–{latest_year}, Prediksi: {latest_year + 1}–{latest_year + 3})</p>", unsafe_allow_html=True)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if page == '🏠 Beranda':
    tab1, tab2 = st.tabs(["**📈 Indikator Utama**", "**📖 Panduan Penggunaan**"])

    with tab1:
        st.markdown(f"<h2>Ringkasan Kondisi Tahun {latest_year}</h2>", unsafe_allow_html=True)
        st.write(f"Halaman ini menampilkan ringkasan kondisi sosial-ekonomi Kota Batu pada tahun terakhir data tersedia ({latest_year}). Gunakan navigasi di sidebar untuk melihat analisis prediksi mendalam per indikator.")
        def get_latest_value(keywords):
            try:
                full_name = find_indicator_by_keywords(keywords, df_main.index)
                return df_main.loc[full_name, str(latest_year)] if full_name else 0
            except: return 0
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{get_latest_value(['pdrb', 'berlaku']):,.0f}</div><div class='metric-label'>PDRB (Miliar Rp)</div></div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{get_latest_value(['tingkat', 'pengangguran']):.2f}%</div><div class='metric-label'>Tingkat Pengangguran</div></div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{get_latest_value(['pembangunan', 'manusia']):.2f}</div><div class='metric-label'>Indeks Pembangunan Manusia</div></div>", unsafe_allow_html=True)
        with col4: st.markdown(f"<div class='metric-card'><div class='metric-value'>{get_latest_value(['penduduk', 'miskin']):.1f}%</div><div class='metric-label'>Tingkat Kemiskinan</div></div>", unsafe_allow_html=True)

    with tab2:
        st.markdown(f"<h2>Selamat Datang di Dashboard Analisis!</h2>", unsafe_allow_html=True)
        st.write("Dashboard ini dirancang untuk membantu Anda menganalisis dan memprediksi tren data sosial-ekonomi di Kota Batu. Berikut adalah panduan singkat untuk menggunakannya:")

        st.markdown("<h4>Langkah 1: Navigasi Halaman</h4>", unsafe_allow_html=True)
        st.write("Gunakan menu **Navigasi** di sidebar (bilah sisi kiri) untuk berpindah antara halaman analisis yang berbeda, seperti PDRB, Ketenagakerjaan, IPM, dan Tingkat Kemiskinan.")

        # --- REVISI PANDUAN PENGGUNA ---
        st.markdown("<h4>Langkah 2: Memperbarui Data (Opsional)</h4>", unsafe_allow_html=True)
        st.write(
            "Untuk menganalisis data dengan tahun terbaru, Anda dapat mengikuti alur kerja berikut:"
        )
        st.markdown("""
        1.  **Unduh Data Sumber**: Di sidebar, pada bagian **'Actual Dataset Source'**, klik tombol **'Download (.xlsx)'**. Ini akan mengunduh file Excel yang saat ini digunakan oleh dashboard.
        2.  **Perbarui File**: Buka file yang telah diunduh. Tambahkan kolom untuk tahun baru (misalnya, `2025`) dan isikan datanya sesuai dengan format yang ada. Pastikan semua indikator terisi.
        3.  **Unggah Kembali**: Setelah menyimpan perubahan, kembali ke dashboard dan gunakan tombol **'Unggah file data baru (.xlsx)'** di sidebar untuk mengunggah file yang telah Anda perbarui.
        
        Dashboard akan secara otomatis memproses file baru dan memperbarui semua visualisasi dan prediksi.
        """)
        # --- AKHIR REVISI ---
        
        st.info(
            "**Penting:** Pastikan file yang diunggah memiliki format yang sama dengan data sumber, di mana header tabel (Rincian, 2010, 2011, dst.) berada di **baris ke-4**.",
            icon="💡"
        )
        st.markdown("<h5>Contoh Struktur di Excel (berdasarkan 4 tahun terakhir dari data yang dimuat):</h5>", unsafe_allow_html=True)
        
        example_years = sorted([str(y) for y in year_cols_int])[-4:]
        
        example_indicators = {
            "PDRB": find_indicator_by_keywords(['pdrb', 'berlaku'], df_main.index),
            "IPM": IPM,
            "Kemiskinan": KEMISKINAN
        }

        contoh_data = {
            'No': [1, 2, 3, 4],
            'Rincian': [
                example_indicators["PDRB"] or 'PDRB Atas Dasar Harga Berlaku (Miliar Rp)',
                example_indicators["IPM"] or 'Indeks Pembangunan Manusia (IPM)',
                example_indicators["Kemiskinan"] or 'Persentase Penduduk Miskin (%)',
                '...'
            ],
        }

        for year in example_years:
            row_data = []
            for ind_key in ["PDRB", "IPM", "Kemiskinan"]:
                indicator_name = example_indicators[ind_key]
                if indicator_name and str(year) in df_main.columns:
                    row_data.append(f"{df_main.loc[indicator_name, str(year)]:,.2f}")
                else:
                    row_data.append("...")
            row_data.append("...")
            contoh_data[year] = row_data

        df_contoh = pd.DataFrame(contoh_data)
        st.dataframe(df_contoh, hide_index=True, use_container_width=True)

        st.markdown("<h4>Langkah 3: Interaksi dengan Grafik</h4>", unsafe_allow_html=True)
        st.write(
            "Pada halaman analisis (misalnya PDRB), Anda dapat memilih sub-indikator yang berbeda melalui **menu dropdown** atau memilih model prediksi yang ingin dibandingkan di sidebar. "
            "Grafik dan tabel akan otomatis diperbarui."
        )

elif page == '💰 PDRB':
    st.markdown("<h2>💰 Analisis & Prediksi PDRB</h2>", unsafe_allow_html=True)
    if pdrb_options:
        selected_indicator = st.selectbox("Pilih Indikator PDRB untuk Dianalisis:", options=pdrb_options)
        display_analysis_view(selected_indicator, "PDRB (Miliar Rupiah)")
    else: st.error("Data PDRB tidak ditemukan pada file Excel.")
elif page == '👥 Ketenagakerjaan (TPT)':
    st.markdown("<h2>👥 Analisis & Prediksi Ketenagakerjaan</h2>", unsafe_allow_html=True)
    if tpt_options:
        selected_indicator = st.selectbox("Pilih Indikator Ketenagakerjaan untuk Dianalisis:", options=tpt_options)
        y_label = "Persentase (%)" if "Tingkat" in selected_indicator else "Jumlah Jiwa"
        display_analysis_view(selected_indicator, y_label)
    else: st.error("Data Ketenagakerjaan (TPT) tidak ditemukan pada file Excel.")
elif page == '📚 IPM':
    st.markdown("<h2>📚 Analisis & Prediksi Indeks Pembangunan Manusia (IPM)</h2>", unsafe_allow_html=True)
    display_analysis_view(IPM, "Poin IPM")
elif page == '🏠 Tingkat Kemiskinan':
    st.markdown("<h2>🏠 Analisis & Prediksi Tingkat Kemiskinan</h2>", unsafe_allow_html=True)
    display_analysis_view(KEMISKINAN, "Persentase (%)")

# ======================================================================================
# 8. FOOTER
# ======================================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
current_month_year = datetime.datetime.now().strftime("%B %Y")
st.markdown(f"<div style='text-align: center; color: #A0A0A0; margin-top: 40px;'>Dikembangkan dengan Streamlit & Python | {current_month_year}</div>", unsafe_allow_html=True)