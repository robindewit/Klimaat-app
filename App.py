import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import cdsapi
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="ERA5 Klimaatstatistieken (Locatie)",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ ERA5 Klimaatstatistieken per Locatie")
st.markdown("""
Bekijk klimaatdata en statistieken voor een **specifieke locatie** (plaatsnaam of coördinaten) op basis van ERA5 reanalyse.
""")

# --- SESSIE STATUS EN LOCATIE ZOEKEN ---
if "lat" not in st.session_state: 
    st.session_state.lat = 52.10  # Standaard De Bilt
if "lon" not in st.session_state: 
    st.session_state.lon = 5.18

st.sidebar.header("📍 1. Kies Locatie")

zoek_plaats = st.sidebar.text_input("Zoek plaatsnaam:", placeholder="bijv. De Bilt, Amsterdam, Berlijn")
if st.sidebar.button("Zoek Plaats"):
    if zoek_plaats:
        try:
            geolocator = Nominatim(user_agent="era5_location_app", timeout=10)
            location = geolocator.geocode(zoek_plaats)
            if location:
                st.session_state.lat = round(location.latitude, 2)
                st.session_state.lon = round(location.longitude, 2)
                st.sidebar.success(f"Gevonden: {location.address}")
            else:
                st.sidebar.error("Plaatsnaam niet gevonden.")
        except Exception as e:
            st.sidebar.error(f"Fout bij zoeken: {e}")

st.sidebar.markdown("---")
lat = st.sidebar.number_input("Breedtegraad (Lat °N)", value=st.session_state.lat, format="%.2f")
lon = st.sidebar.number_input("Lengtegraad (Lon °E)", value=st.session_state.lon, format="%.2f")

st.sidebar.header("📅 2. Kies Periode")
start_jaar = st.sidebar.number_input("Startjaar", min_value=1950, max_value=2025, value=1991)
eind_jaar = st.sidebar.number_input("Eindjaar", min_value=1950, max_value=2025, value=2020)

# --- HELPERS EN CDS CLIENT ---
def get_cds_client():
    url = None
    key = None

    if "cds" in st.secrets:
        url = st.secrets["cds"].get("url")
        key = st.secrets["cds"].get("key")
    elif "url" in st.secrets and "key" in st.secrets:
        url = st.secrets["url"]
        key = st.secrets["key"]

    if url and key:
        home_dir = os.path.expanduser("~")
        cdsapirc_path = os.path.join(home_dir, ".cdsapirc")
        with open(cdsapirc_path, "w", encoding="utf-8") as f:
            f.write(f"url: {url}\nkey: {key}\n")
        return cdsapi.Client(url=url, key=key)
    else:
        return cdsapi.Client()

def degrees_to_cardinal(deg):
    if pd.isna(deg): return "N/A"
    dirs = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
            'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((deg + 11.25) / 22.5)
    return dirs[ix % 16]

MAAND_NAMEN = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]

# --- OPHALEN DATA VANUIT COPERNICUS CDS ---
def fetch_era5_point_data(lat, lon, start_yr, end_yr):
    c = get_cds_client()
    temp_dir = tempfile.gettempdir()
    download_path = os.path.join(temp_dir, f"era5_point_{lat}_{lon}_{start_yr}_{end_yr}.nc")

    # Bepaal een kleine box rond de puntlocatie (+/- 0.25 graden) om het rasterpunt op te halen
    area_box = [
        round(lat + 0.25, 2),
        round(lon - 0.25, 2),
        round(lat - 0.25, 2),
        round(lon + 0.25, 2)
    ]

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "total_cloud_cover"
        ],
        "year": [str(y) for y in range(start_yr, end_yr + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": area_box,
        "data_format": "netcdf",
    }

    if not os.path.exists(download_path):
        # JUISTE DATASET NAAM VOOR CDS API v2 / NEW CDS:
        c.retrieve("reanalysis-era5-single-levels-monthly-means", request, download_path)

    # Openen en snijden naar het exacte dichtstbijzijnde punt
    ds = xr.open_dataset(download_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    df = ds_point.to_dataframe().reset_index()
    ds.close()

    # Tijdskolom opschonen
    time_col = next((t for t in ["valid_time", "time", "date"] if t in df.columns), None)
    df["time_clean"] = pd.to_datetime(df[time_col])
    df["maand"] = df["time_clean"].dt.month

    # Variabelen herleiden en berekenen
    cols_lower = {str(c).lower(): c for c in df.columns}

    # Temperatuur (°C)
    t_var = next((cols_lower[c] for c in ["t2m", "2m_temperature"] if c in cols_lower), None)
    if t_var:
        df["temp_c"] = df[t_var] - 273.15 if df[t_var].mean() > 200 else df[t_var]

    # Vochtigheid (%) & Dp
    d_var = next((cols_lower[c] for c in ["d2m", "2m_dewpoint_temperature"] if c in cols_lower), None)
    if d_var and t_var:
        df["dew_c"] = df[d_var] - 273.15 if df[d_var].mean() > 200 else df[d_var]
        df["rh_pct"] = 100 * (np.exp((17.625 * df["dew_c"]) / (243.04 + df["dew_c"])) / 
                              np.exp((17.625 * df["temp_c"]) / (243.04 + df["temp_c"])))
        df["rh_pct"] = df["rh_pct"].clip(0, 100)

    # Wind
    u_var = next((cols_lower[c] for c in ["u10", "10m_u_component_of_wind"] if c in cols_lower), None)
    v_var = next((cols_lower[c] for c in ["v10", "10m_v_component_of_wind"] if c in cols_lower), None)
    if u_var and v_var:
        df["wind_speed_ms"] = np.sqrt(df[u_var] ** 2 + df[v_var] ** 2)
        df["wind_dir_deg"] = (270 - np.arctan2(df[v_var], df[u_var]) * (180 / np.pi)) % 360
        df["wind_dir_cardinal"] = df["wind_dir_deg"].apply(degrees_to_cardinal)

    # Luchtdruk (hPa)
    msl_var = next((cols_lower[c] for c in ["msl", "mean_sea_level_pressure"] if c in cols_lower), None)
    if msl_var:
        df["mslp_hpa"] = df[msl_var] / 100.0

    # Bewolking (%)
    tcc_var = next((cols_lower[c] for c in ["tcc", "total_cloud_cover"] if c in cols_lower), None)
    if tcc_var:
        df["tcc_pct"] = df[tcc_var] * 100.0

    return df

# --- HOOFDPROGRAMMA ---
if st.sidebar.button("🚀 Data Ophalen & Berekenen", type="primary"):
    if start_jaar > eind_jaar:
        st.error("Startjaar mag niet groter zijn dan eindjaar!")
    else:
        with st.spinner(f"Klimaatdata ophalen voor locatie ({lat}, {lon})..."):
            try:
                df = fetch_era5_point_data(lat, lon, start_jaar, eind_jaar)
                st.success(f"Data succesvol geladen voor locatie Lat: {lat}, Lon: {lon} ({start_jaar}-{eind_jaar})")

                # HELPER VOOR PERCENTIEL GRAFIEK
                def plot_percentile_chart(df_data, var_col, title, y_label, color_hex="#1f77b4"):
                    stats = df_data.groupby("maand")[var_col].agg(
                        p10=lambda x: np.percentile(x.dropna(), 10) if len(x.dropna())>0 else np.nan,
                        p50='median',
                        p90=lambda x: np.percentile(x.dropna(), 90) if len(x.dropna())>0 else np.nan
                    ).reset_index()
                    stats["maand_naam"] = stats["maand"].apply(lambda m: MAAND_NAMEN[m-1])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stats["maand_naam"], y=stats["p90"],
                        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=stats["maand_naam"], y=stats["p10"],
                        mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)',
                        name='10% - 90% Percentiel Band'
                    ))
                    fig.add_trace(go.Scatter(
                        x=stats["maand_naam"], y=stats["p50"],
                        mode='lines+markers', line=dict(color=color_hex, width=3),
                        name='Mediaan (P50)'
                    ))
                    fig.update_layout(title=title, yaxis_title=y_label, hovermode="x unified")
                    return fig

                # TABBLADEN VOOR RESULTATEN
                tab1, tab2, tab3 = st.tabs(["🌡️ Temperatuur & Vocht", "💨 Wind", "🌀 Luchtdruk & Bewolking"])

                with tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plot_percentile_chart(df, "temp_c", "2m Temperatuur (°C)", "Temperatuur (°C)", "#EF553B"), use_container_width=True)
                    with c2:
                        if "rh_pct" in df.columns:
                            st.plotly_chart(plot_percentile_chart(df, "rh_pct", "Relatieve Vochtigheid (%)", "Vochtigheid (%)", "#00CC96"), use_container_width=True)

                with tab2:
                    c1, c2 = st.columns(2)
                    with c1:
                        if "wind_speed_ms" in df.columns:
                            st.plotly_chart(plot_percentile_chart(df, "wind_speed_ms", "Windsnelheid (m/s)", "Windsnelheid (m/s)", "#2CA02C"), use_container_width=True)
                    with c2:
                        st.subheader("Windrichting Verdeling")
                        if "wind_dir_cardinal" in df.columns:
                            wind_counts = df["wind_dir_cardinal"].value_counts().reset_index()
                            wind_counts.columns = ["richting", "frequentie"]
                            fig_rose = px.bar_polar(
                                wind_counts, r="frequentie", theta="richting",
                                template="plotly_dark", title="Windrichting Frequentie",
                                color_discrete_sequence=px.colors.sequential.Plasma
                            )
                            st.plotly_chart(fig_rose, use_container_width=True)

                with tab3:
                    c1, c2 = st.columns(2)
                    with c1:
                        if "mslp_hpa" in df.columns:
                            st.plotly_chart(plot_percentile_chart(df, "mslp_hpa", "Luchtdruk op Zeeniveau (hPa)", "Luchtdruk (hPa)", "#AB63FA"), use_container_width=True)
                    with c2:
                        if "tcc_pct" in df.columns:
                            st.plotly_chart(plot_percentile_chart(df, "tcc_pct", "Totale Bewolkingsgraad (%)", "Bewolking (%)", "#FFA15A"), use_container_width=True)

                st.subheader("📋 Ruwe Data (Eerste 24 maanden)")
                st.dataframe(df[["time_clean", "temp_c", "wind_speed_ms", "wind_dir_cardinal", "mslp_hpa", "tcc_pct"]].head(24))

            except Exception as e:
                st.error(f"Er is een fout opgetreden: {e}")
