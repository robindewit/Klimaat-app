import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import cdsapi
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

st.set_page_config(
    page_title="ERA5 Klimaatstatistieken",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ ERA5 Klimaatstatistieken per Locatie")
st.caption("Analyse van interjaarlijkse variatie op basis van maandgemiddelde ERA5 reanalyse data.")

# --- SESSION STATE INITIALISATIE ---
if "lat" not in st.session_state:
    st.session_state.lat = 52.10
if "lon" not in st.session_state:
    st.session_state.lon = 5.18
if "df" not in st.session_state:
    st.session_state.df = None

# --- SIDEBAR ---
st.sidebar.header("📍 1. Kies Locatie")
zoek_plaats = st.sidebar.text_input("Zoek plaatsnaam:", placeholder="bijv. De Bilt, Amsterdam")

if st.sidebar.button("Zoek Plaats"):
    if zoek_plaats:
        try:
            geolocator = Nominatim(user_agent="era5_climate_app_v2", timeout=10)
            location = geolocator.geocode(zoek_plaats)
            if location:
                st.session_state.lat = round(location.latitude, 2)
                st.session_state.lon = round(location.longitude, 2)
                st.sidebar.success(f"Gevonden: {location.address}")
            else:
                st.sidebar.error("Plaatsnaam niet gevonden.")
        except Exception as e:
            st.sidebar.error(f"Fout bij zoeken: {e}")

lat = st.sidebar.number_input("Breedtegraad (Lat °N)", value=st.session_state.lat, format="%.2f")
lon = st.sidebar.number_input("Lengtegraad (Lon °E)", value=st.session_state.lon, format="%.2f")

st.sidebar.header("📅 2. Kies Periode")
start_jaar = st.sidebar.number_input("Startjaar", min_value=1950, max_value=2025, value=1991)
eind_jaar = st.sidebar.number_input("Eindjaar", min_value=1950, max_value=2025, value=2020)

st.sidebar.header("⚙️ 3. Instellingen")
wind_eenheid = st.sidebar.selectbox("Windsnelheid Eenheid:", ["m/s", "kt", "km/h"])

MAAND_NAMEN = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]

# --- DATA OPHALEN MET STREAMLIT CACHING ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_era5_data(lat, lon, start_yr, end_yr):
    url = st.secrets.get("cds", {}).get("url") or st.secrets.get("url")
    key = st.secrets.get("cds", {}).get("key") or st.secrets.get("key")
    
    c = cdsapi.Client(url=url, key=key) if (url and key) else cdsapi.Client()
    
    temp_dir = tempfile.gettempdir()
    download_path = os.path.join(temp_dir, f"era5_{lat}_{lon}_{start_yr}_{end_yr}.nc")

    area_box = [round(lat + 0.25, 2), round(lon - 0.25, 2), round(lat - 0.25, 2), round(lon + 0.25, 2)]

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_wind_speed",
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
        c.retrieve("reanalysis-era5-single-levels-monthly-means", request, download_path)

    ds = xr.open_dataset(download_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    # Bilineaire interpolatie naar exacte locatie
    ds_point = ds.interp({lat_name: lat, lon_name: lon})
    df = ds_point.to_dataframe().reset_index()
    ds.close()

    time_col = next((t for t in ["valid_time", "time", "date"] if t in df.columns), None)
    df["time_clean"] = pd.to_datetime(df[time_col])
    df["maand_nr"] = df["time_clean"].dt.month
    df["maand_naam"] = df["maand_nr"].apply(lambda m: MAAND_NAMEN[m-1])

    cols_lower = {str(c).lower(): c for c in df.columns}

    # Temperatuur
    t_var = next((cols_lower[c] for c in ["t2m", "2m_temperature"] if c in cols_lower), None)
    if t_var:
        df["temp_c"] = df[t_var] - 273.15 if df[t_var].mean() > 200 else df[t_var]

    # Dewpoint & RH
    d_var = next((cols_lower[c] for c in ["d2m", "2m_dewpoint_temperature"] if c in cols_lower), None)
    if d_var and t_var:
        df["dew_c"] = df[d_var] - 273.15 if df[d_var].mean() > 200 else df[d_var]
        df["rh_pct"] = 100 * (np.exp((17.625 * df["dew_c"]) / (243.04 + df["dew_c"])) / 
                              np.exp((17.625 * df["temp_c"]) / (243.04 + df["temp_c"])))
        df["rh_pct"] = df["rh_pct"].clip(0, 100)

    # Wind
    si10_var = next((cols_lower[c] for c in ["si10", "10m_wind_speed", "ws10"] if c in cols_lower), None)
    if si10_var:
        df["wind_speed_ms"] = df[si10_var]
        df["wind_speed_kt"] = df["wind_speed_ms"] * 1.943844
        df["wind_speed_kmh"] = df["wind_speed_ms"] * 3.6

    # Luchtdruk
    msl_var = next((cols_lower[c] for c in ["msl", "mean_sea_level_pressure"] if c in cols_lower), None)
    if msl_var:
        df["mslp_hpa"] = df[msl_var] / 100.0

    # Bewolking
    tcc_var = next((cols_lower[c] for c in ["tcc", "total_cloud_cover"] if c in cols_lower), None)
    if tcc_var:
        df["tcc_pct"] = df[tcc_var] * 100.0

    return df

# --- PLOTTING FUNCTIE ---
def create_monthly_boxplot(df, var_col, title, y_label, color_hex="#1f77b4"):
    df_plot = df.copy()
    df_plot["maand_naam"] = pd.Categorical(df_plot["maand_naam"], categories=MAAND_NAMEN, ordered=True)
    df_avg = df_plot.groupby("maand_naam", observed=False)[var_col].mean().reset_index()

    fig = go.Figure()
    for month_name in MAAND_NAMEN:
        month_data = df_plot[df_plot["maand_naam"] == month_name][var_col].dropna()
        if len(month_data) == 0:
            continue
            
        p10, p25, p50, p75, p90 = np.percentile(month_data, [10, 25, 50, 75, 90])
        
        fig.add_trace(go.Box(
            x=[month_name], q1=[p25], median=[p50], q3=[p75],
            lowerfence=[p10], upperfence=[p90],
            marker_color=color_hex, name=month_name, showlegend=False,
            hovertemplate=f"<b>{month_name}</b><br>P90: %{{upperfence:.2f}}<br>P75: %{{q3:.2f}}<br><b>P50 (Mediaan): %{{median:.2f}}</b><br>P25: %{{q1:.2f}}<br>P10: %{{lowerfence:.2f}}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=df_avg["maand_naam"], y=df_avg[var_col], mode="lines+markers",
        name="Gemiddelde", line=dict(color="red", width=2, dash="dash"),
        hovertemplate="Gemiddelde %{x}: <b>%{y:.2f}</b><extra></extra>"
    ))

    fig.update_layout(
        title=title, xaxis=dict(title="Maand"), yaxis_title=y_label,
        height=400, margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# --- TRIGGER BEREKENING ---
if st.sidebar.button("🚀 Data Ophalen & Berekenen", type="primary"):
    if start_jaar > eind_jaar:
        st.error("Startjaar mag niet groter zijn dan eindjaar!")
    else:
        with st.spinner("Klimaatdata ophalen uit CDS..."):
            try:
                st.session_state.df = fetch_era5_data(lat, lon, start_jaar, eind_jaar)
                st.success("Data succesvol geladen!")
            except Exception as e:
                st.error(f"Fout bij ophalen data: {e}")

# --- DASHBOARD WEERGAVE ---
if st.session_state.df is not None:
    df = st.session_state.df

    tab_grafieken, tab_data = st.tabs(["📊 Klimaatstatistieken", "📋 Data & Export"])

    wind_col = "wind_speed_kt" if wind_eenheid == "kt" else ("wind_speed_kmh" if wind_eenheid == "km/h" else "wind_speed_ms")

    with tab_grafieken:
        st.warning("⚠️ **Statistische toelichting**: Deze grafieken tonen de **interjaarlijkse spreiding van maandgemiddelden** over de gekozen periode (1 datapunten per jaar per maand). Dit geeft de variatie tussen verschillende jaren weer, niet de dagelijkse extremen binnen een maand.")

        if "temp_c" in df.columns:
            st.plotly_chart(create_monthly_boxplot(df, "temp_c", "2m Temperatuur (°C) - Maandgemiddelde spreiding", "Temperatuur (°C)", "#EF553B"), use_container_width=True)

        if "rh_pct" in df.columns:
            st.plotly_chart(create_monthly_boxplot(df, "rh_pct", "Relatieve Vochtigheid (%) - Maandgemiddelde spreiding", "Vochtigheid (%)", "#00CC96"), use_container_width=True)

        if wind_col in df.columns:
            st.plotly_chart(create_monthly_boxplot(df, wind_col, f"Windsnelheid ({wind_eenheid}) - Maandgemiddelde spreiding", f"Snelheid ({wind_eenheid})", "#2CA02C"), use_container_width=True)

        if "mslp_hpa" in df.columns:
            st.plotly_chart(create_monthly_boxplot(df, "mslp_hpa", "Luchtdruk (hPa) - Maandgemiddelde spreiding", "Luchtdruk (hPa)", "#AB63FA"), use_container_width=True)

    with tab_data:
        display_cols = [c for c in ["time_clean", "maand_naam", "temp_c", "rh_pct", wind_col, "mslp_hpa", "tcc_pct"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
        st.download_button(
            label="💾 Download CSV",
            data=df[display_cols].to_csv(index=False),
            file_name=f"era5_klimaat_{lat}_{lon}_{start_jaar}_{eind_jaar}.csv",
            mime="text/csv"
        )
