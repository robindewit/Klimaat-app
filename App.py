import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
import cdsapi
from geopy.geocoders import Nominatim

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="ERA5 Klimaatstatistieken",
    page_icon="🌤️",
    layout="wide"
)

# --- INSTELLINGEN & HELPERS ---
def get_cds_client():
    url = None
    key = None

    if "cds" in st.secrets:
        url = st.secrets["cds"].get("url")
        key = st.secrets["cds"].get("key")
    elif "url" in st.secrets and "key" in st.secrets:
        url = st.secrets["url"]
        key = st.secrets["key"]

    if not url or not key:
        st.error("❌ CDS URL of Key niet gevonden in Secrets.")
        st.stop()

    home_dir = os.path.expanduser("~")
    cdsapirc_path = os.path.join(home_dir, ".cdsapirc")

    try:
        with open(cdsapirc_path, "w", encoding="utf-8") as f:
            f.write(f"url: {url}\nkey: {key}\n")
    except Exception as e:
        st.warning(f"Kon .cdsapirc niet wegschrijven: {e}")

    try:
        return cdsapi.Client(url=url, key=key)
    except Exception as e:
        st.error(f"❌ Fout bij verbinden met Copernicus CDS API: {e}")
        st.stop()

def ms_to_beaufort(ms):
    if pd.isna(ms): return 0
    if ms < 0.3: return 0
    elif ms < 1.6: return 1
    elif ms < 3.4: return 2
    elif ms < 5.5: return 3
    elif ms < 8.0: return 4
    elif ms < 10.8: return 5
    elif ms < 13.9: return 6
    elif ms < 17.2: return 7
    elif ms < 20.8: return 8
    elif ms < 24.5: return 9
    elif ms < 28.5: return 10
    elif ms < 32.7: return 11
    else: return 12

def degrees_to_cardinal(deg):
    if pd.isna(deg): return "N/A"
    dirs = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
            'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((deg + 11.25) / 22.5)
    return dirs[ix % 16]

MAAND_NAMEN = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]

# --- OPHALEN EN VERWERKEN ERA5 DATA ---
@st.cache_data(show_spinner="Klimaatdata ophalen bij Copernicus CDS...")
def download_era5_point_data(lat, lon, start_jaar, eind_jaar):
    c = get_cds_client()
    jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]

    temp_dir = tempfile.gettempdir()
    download_path = os.path.join(temp_dir, f"era5_klimaat_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc")

    # Uitgebreide variabelenlijst
    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "10m_wind_gust_since_previous_post_processing",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "surface_solar_radiation_downwards",
        "sunshine_duration",
        "snow_depth",
        "snowfall",
        "volumetric_soil_water_layer_1",
        "evaporation",
        "total_precipitation"
    ]

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variables,
        "year": jaren,
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": [
            round(lat + 0.25, 2),
            round(lon - 0.25, 2),
            round(lat - 0.25, 2),
            round(lon + 0.25, 2),
        ],
        "format": "netcdf",
    }

    if os.path.exists(download_path):
        os.remove(download_path)

    c.retrieve("reanalysis-era5-single-levels-monthly-means", request, download_path)

    file_to_open = download_path
    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            target_file = [f for f in extracted_files if f.endswith(('.nc', '.nc4', '.grib'))][0]
            file_to_open = zip_ref.extract(target_file, path=temp_dir)

    ds = xr.open_dataset(file_to_open)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    df = ds_point.to_dataframe().reset_index()

    time_dim = next((t for t in ["valid_time", "time", "date"] if t in df.columns), None)
    df["time_clean"] = pd.to_datetime(df[time_dim])
    df["maand"] = df["time_clean"].dt.month
    df["days_in_month"] = df["time_clean"].dt.days_in_month

    cols_lower = {str(c).lower(): c for c in df.columns}

    # 1. Temperatuur & Dp
    t_var = next((cols_lower[c] for c in ["t2m", "2m_temperature"] if c in cols_lower), None)
    d_var = next((cols_lower[c] for c in ["d2m", "2m_dewpoint_temperature"] if c in cols_lower), None)

    if t_var:
        df["temp_c"] = df[t_var] - 273.15 if df[t_var].mean() > 200 else df[t_var]
    if d_var:
        df["dew_c"] = df[d_var] - 273.15 if df[d_var].mean() > 200 else df[d_var]
        # Relatieve Vochtigheid (Magnus formule benadering)
        df["rh_pct"] = 100 * (np.exp((17.625 * df["dew_c"]) / (243.04 + df["dew_c"])) / 
                              np.exp((17.625 * df["temp_c"]) / (243.04 + df["temp_c"])))
        df["rh_pct"] = df["rh_pct"].clip(0, 100)

    # 2. Wind & Windstoten
    u_var = next((cols_lower[c] for c in ["u10", "10m_u_component_of_wind"] if c in cols_lower), None)
    v_var = next((cols_lower[c] for c in ["v10", "10m_v_component_of_wind"] if c in cols_lower), None)
    fg_var = next((cols_lower[c] for c in ["i10fg", "fg10", "10m_wind_gust_since_previous_post_processing"] if c in cols_lower), None)

    if u_var and v_var:
        df["wind_speed_ms"] = np.sqrt(df[u_var] ** 2 + df[v_var] ** 2)
        df["wind_dir_deg"] = (270 - np.arctan2(df[v_var], df[u_var]) * (180 / np.pi)) % 360
        df["wind_dir_cardinal"] = df["wind_dir_deg"].apply(degrees_to_cardinal)

    if fg_var:
        df["wind_gust_ms"] = df[fg_var]

    # 3. MSLP (Luchtdruk)
    msl_var = next((cols_lower[c] for c in ["msl", "mean_sea_level_pressure"] if c in cols_lower), None)
    if msl_var:
        df["mslp_hpa"] = df[msl_var] / 100.0

    # 4. Bewolking (TCC)
    tcc_var = next((cols_lower[c] for c in ["tcc", "total_cloud_cover"] if c in cols_lower), None)
    if tcc_var:
        df["tcc_pct"] = df[tcc_var] * 100.0

    # 5. Zonnestraling & Zonneduur
    ssrd_var = next((cols_lower[c] for c in ["ssrd", "surface_solar_radiation_downwards"] if c in cols_lower), None)
    sund_var = next((cols_lower[c] for c in ["sund", "sunshine_duration"] if c in cols_lower), None)
    if ssrd_var:
        df["ssrd_mj"] = df[ssrd_var] / 1e6 # J/m² naar MJ/m²
    if sund_var:
        df["sunshine_hrs"] = df[sund_var] / 3600.0 # seconden naar uur

    # 6. Sneeuw & Bodemvocht
    sd_var = next((cols_lower[c] for c in ["sd", "snow_depth"] if c in cols_lower), None)
    sf_var = next((cols_lower[c] for c in ["sf", "snowfall"] if c in cols_lower), None)
    swv_var = next((cols_lower[c] for c in ["swvl1", "volumetric_soil_water_layer_1"] if c in cols_lower), None)

    if sd_var:
        df["snow_depth_cm"] = df[sd_var] * 100.0
    if sf_var:
        df["snowfall_mm"] = df[sf_var] * 1000.0 * df["days_in_month"]
    if swv_var:
        df["soil_water_pct"] = df[swv_var] * 100.0

    # 7. Verdamping / Neerslagtekort
    e_var = next((cols_lower[c] for c in ["e", "evaporation"] if c in cols_lower), None)
    if e_var:
        df["evap_mm"] = np.abs(df[e_var]) * 1000.0 * df["days_in_month"]

    ds.close()
    return df

# --- MAIN INTERFACE ---
st.title("📊 ERA5 Klimaatstatistieken (Maandgemiddelden)")
st.markdown("Analyseer de klimaatverdeling (P10, P50/Mediaan, P90) per kalendermaand over meerdere jaren.")

if "lat" not in st.session_state: st.session_state.lat = 52.10
if "lon" not in st.session_state: st.session_state.lon = 5.18
if "run_fetch" not in st.session_state: st.session_state.run_fetch = False

st.sidebar.header("📍 1. Locatie")
zoek_plaats = st.sidebar.text_input("Zoek plaatsnaam:", placeholder="bijv. De Bilt, Berlijn")
if st.sidebar.button("Zoek"):
    if zoek_plaats:
        try:
            geolocator = Nominatim(user_agent="era5_climate_stats_app", timeout=10)
            location = geolocator.geocode(zoek_plaats)
            if location:
                st.session_state.lat = round(location.latitude, 2)
                st.session_state.lon = round(location.longitude, 2)
                st.sidebar.success(f"Gevonden: ({st.session_state.lat}, {st.session_state.lon})")
        except Exception as e:
            st.sidebar.error(f"Fout: {e}")

st.sidebar.markdown("---")
st.sidebar.header("📅 2. Periode")
lat = st.sidebar.number_input("Lat", value=st.session_state.lat, format="%.2f")
lon = st.sidebar.number_input("Lon", value=st.session_state.lon, format="%.2f")
start_jaar, eind_jaar = st.sidebar.slider("Jaren", 1950, 2025, (1991, 2020))

if st.sidebar.button("🚀 Data Berekenen", type="primary"):
    st.session_state.run_fetch = True

if st.session_state.run_fetch:
    try:
        df = download_era5_point_data(lat, lon, start_jaar, eind_jaar)
        st.success(f"Klimaatprofiel berekend over periode {start_jaar}-{eind_jaar} ({lat}, {lon})")

        # TABBLADEN
        tab_temp, tab_wind, tab_atm, tab_rad, tab_soil = st.tabs([
            "🌡️ Temperatuur & Vocht", 
            "💨 Wind & Windstoten", 
            "🌀 Luchtdruk & Bewolking", 
            "☀️ Zonnestraling & Zonduur", 
            "❄️ Sneeuw & Bodem"
        ])

        # HELPER VOOR BOXPLOT / PERCENTIEL GRAFIEK
        def plot_percentile_chart(df_data, var_col, title, y_label, color_hex="#1f77b4"):
            stats = df_data.groupby("maand")[var_col].agg(
                p10=lambda x: np.percentile(x.dropna(), 10) if len(x.dropna())>0 else np.nan,
                p50='median',
                p90=lambda x: np.percentile(x.dropna(), 90) if len(x.dropna())>0 else np.nan
            ).reset_index()
            stats["maand_naam"] = stats["maand"].apply(lambda m: MAAND_NAMEN[m-1])

            fig = go.Figure()
            # P10-P90 Band
            fig.add_trace(go.Scatter(
                x=stats["maand_naam"], y=stats["p90"],
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=stats["maand_naam"], y=stats["p10"],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=f'rgba(31, 119, 180, 0.2)',
                name='10% - 90% Percentiel Band'
            ))
            # P50 / Mediaan
            fig.add_trace(go.Scatter(
                x=stats["maand_naam"], y=stats["p50"],
                mode='lines+markers', line=dict(color=color_hex, width=3),
                name='50% Percentiel (Mediaan)'
            ))
            fig.update_layout(title=title, yaxis_title=y_label, hovermode="x unified")
            return fig

        # 1. TEMPERATUUR & VOCHT
        with tab_temp:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_percentile_chart(df, "temp_c", "2m Temperatuur per Maand (P10 - P50 - P90)", "Temperatuur (°C)", "#EF553B"), use_container_width=True)
            with col2:
                if "rh_pct" in df.columns:
                    st.plotly_chart(plot_percentile_chart(df, "rh_pct", "Relatieve Vochtigheid per Maand", "Relatieve Vochtigheid (%)", "#00CC96"), use_container_width=True)

        # 2. WIND & WINDROOS
        with tab_wind:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_percentile_chart(df, "wind_speed_ms", "10m Windsnelheid per Maand (P10 - P50 - P90)", "Windsnelheid (m/s)", "#2CA02C"), use_container_width=True)
                if "wind_gust_ms" in df.columns and not df["wind_gust_ms"].isnull().all():
                    st.plotly_chart(plot_percentile_chart(df, "wind_gust_ms", "10m Windstoten per Maand (P10 - P50 - P90)", "Windstoot (m/s)", "#D62728"), use_container_width=True)
            
            with col2:
                st.subheader("Visuele Windroos (Windrichting Verdeling)")
                if "wind_dir_cardinal" in df.columns:
                    wind_counts = df["wind_dir_cardinal"].value_counts().reset_index()
                    wind_counts.columns = ["richting", "frequentie"]
                    
                    fig_rose = px.bar_polar(
                        wind_counts, r="frequentie", theta="richting",
                        template="plotly_dark", title="Windrichting Frequentie",
                        color_discrete_sequence=px.colors.sequential.Plasma
                    )
                    st.plotly_chart(fig_rose, use_container_width=True)

        # 3. LUCHTDRUK & BEWOLKING
        with tab_atm:
            col1, col2 = st.columns(2)
            with col1:
                if "mslp_hpa" in df.columns:
                    st.plotly_chart(plot_percentile_chart(df, "mslp_hpa", "Zeenniveau Luchtdruk (MSLP)", "Luchtdruk (hPa)", "#AB63FA"), use_container_width=True)
            with col2:
                if "tcc_pct" in df.columns:
                    st.plotly_chart(plot_percentile_chart(df, "tcc_pct", "Totale Bewolkingsgraad (TCC)", "Bewolking (%)", "#FFA15A"), use_container_width=True)

        # 4. STRALING & ZON
        with tab_rad:
            col1, col2 = st.columns(2)
            with col1:
                if "ssrd_mj" in df.columns:
                    st.plotly_chart(plot_percentile_chart(df, "ssrd_mj", "Inkomende Zonnestraling (SSRD)", "Straling (MJ/m²)", "#FFD700"), use_container_width=True)
            with col2:
                if "sunshine_hrs" in df.columns:
                    st.plotly_chart(plot_percentile_chart(df, "sunshine_hrs", "Zonneschijnduur per Maand", "Zonduur (Uur)", "#FFA500"), use_container_width=True)

        # 5. SNEEUW & BODEM
        with tab_soil:
            col1, col2 = st.columns(2)
            with col1:
                if "snow_depth_cm" in df.columns and not df["snow_depth_cm"].isnull().all():
                    st.plotly_chart(plot_percentile_chart(df, "snow_depth_cm", "Sneeuwhoogte (P50 & P90)", "Sneeuwhoogte (cm)", "#17BECF"), use_container_width=True)
                if "snowfall_mm" in df.columns and not df["snowfall_mm"].isnull().all():
                    st.plotly_chart(plot_percentile_chart(df, "snowfall_mm", "Sneeuwval per Maand", "Sneeuwval (mm)", "#7F7F7F"), use_container_width=True)
            with col2:
                if "soil_water_pct" in df.columns and not df["soil_water_pct"].isnull().all():
                    st.plotly_chart(plot_percentile_chart(df, "soil_water_pct", "Bodemvocht Bovenlaag (0-7cm)", "Volumetrisch Vocht (%)", "#8C564B"), use_container_width=True)

    except Exception as e:
        st.error(f"Fout bij verwerken klimaatdata: {e}")
