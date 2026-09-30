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
Bekijk klimaatdata en maandelijkse verdelingen (met maandgemiddelden, 50% en 90% percentielen) voor een **specifieke locatie** op basis van ERA5 reanalyse.
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
        c.retrieve("reanalysis-era5-single-levels-monthly-means", request, download_path)

    ds = xr.open_dataset(download_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    df = ds_point.to_dataframe().reset_index()
    ds.close()

    time_col = next((t for t in ["valid_time", "time", "date"] if t in df.columns), None)
    df["time_clean"] = pd.to_datetime(df[time_col])
    df["maand_nr"] = df["time_clean"].dt.month
    df["maand_naam"] = df["maand_nr"].apply(lambda m: MAAND_NAMEN[m-1])

    cols_lower = {str(c).lower(): c for c in df.columns}

    # Temperatuur (°C)
    t_var = next((cols_lower[c] for c in ["t2m", "2m_temperature"] if c in cols_lower), None)
    if t_var:
        df["temp_c"] = df[t_var] - 273.15 if df[t_var].mean() > 200 else df[t_var]

    # Vochtigheid (%) & Dewpoint
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

# --- FUNCTIE VOOR PER-MAAND PERCENTIEL BOXPLOT MET GEMIDDELDE ---
def create_monthly_percentile_boxplot(df, var_col, title, y_label, color_hex="#1f77b4"):
    df_plot = df.copy()
    df_plot["maand_naam"] = pd.Categorical(df_plot["maand_naam"], categories=MAAND_NAMEN, ordered=True)
    
    # Gemiddelde per maand berekenen voor de rode lijn
    df_avg = df_plot.groupby("maand_naam", observed=False)[var_col].mean().reset_index()

    fig = go.Figure()

    # Boxplot per maand toevoegen op de X-as
    for month_name in MAAND_NAMEN:
        month_data = df_plot[df_plot["maand_naam"] == month_name][var_col].dropna()
        if len(month_data) == 0:
            continue
            
        p10 = np.percentile(month_data, 10)
        p25 = np.percentile(month_data, 25)
        p50 = np.percentile(month_data, 50)
        p75 = np.percentile(month_data, 75)
        p90 = np.percentile(month_data, 90)
        
        fig.add_trace(go.Box(
            x=[month_name],
            q1=[p25],
            median=[p50],
            q3=[p75],
            lowerfence=[p10],
            upperfence=[p90],
            marker_color=color_hex,
            name=month_name,
            showlegend=False,
            hovertemplate=(
                f"<b>{month_name}</b><br>" +
                "90% Percentiel (P90): %{upperfence:.2f}<br>" +
                "75% Percentiel (P75): %{q3:.2f}<br>" +
                "<b>50% Percentiel (Mediaan): %{median:.2f}</b><br>" +
                "25% Percentiel (P25): %{q1:.2f}<br>" +
                "10% Percentiel (P10): %{lowerfence:.2f}<extra></extra>"
            )
        ))

    # Rode stippellijn voor het Maandgemiddelde
    fig.add_trace(go.Scatter(
        x=df_avg["maand_naam"],
        y=df_avg[var_col],
        mode="lines+markers",
        name="Gemiddelde",
        line=dict(color="red", width=2, dash="dash"),
        hovertemplate="Gemiddelde %{x}: <b>%{y:.2f}</b><extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Maand", type="category"),
        yaxis_title=y_label,
        height=450,
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# --- HOOFDPROGRAMMA ---
if st.sidebar.button("🚀 Data Ophalen & Berekenen", type="primary"):
    if start_jaar > eind_jaar:
        st.error("Startjaar mag niet groter zijn dan eindjaar!")
    else:
        with st.spinner(f"Klimaatdata ophalen voor locatie ({lat}, {lon})..."):
            try:
                df = fetch_era5_point_data(lat, lon, start_jaar, eind_jaar)
                st.success(f"Data succesvol geladen voor locatie Lat: {lat}, Lon: {lon} ({start_jaar}-{eind_jaar})")

                # TABBLADEN MAKEN
                tab_grafieken, tab_data = st.tabs([
                    "📊 Klimaatstatistieken & Boxplots", 
                    "📋 Ruwe Data & Exporteren"
                ])

                # --- TAB 1: GRAFIEKEN ONDER ELKAAR ---
                with tab_grafieken:
                    st.info("💡 **Uitleg Grafieken**: Op de X-as staan de 12 afzonderlijke maanden. De gekleurde balken tonen de verdeling per maand (**10%**, **25%**, **50%/mediaan**, **75%** en **90% percentielen** over alle jaren). De **rode onderbroken lijn** geeft het gemiddelde aan per maand.")

                    # 1. Temperatuur
                    if "temp_c" in df.columns:
                        fig_temp = create_monthly_percentile_boxplot(
                            df, "temp_c", 
                            "2m Temperatuur (°C) - Maandelijkse Verdeling & Gemiddelde", 
                            "Temperatuur (°C)", 
                            "#EF553B"
                        )
                        st.plotly_chart(fig_temp, use_container_width=True)
                        st.divider()

                    # 2. Relatieve Vochtigheid
                    if "rh_pct" in df.columns:
                        fig_rh = create_monthly_percentile_boxplot(
                            df, "rh_pct", 
                            "Relatieve Vochtigheid (%) - Maandelijkse Verdeling & Gemiddelde", 
                            "Vochtigheid (%)", 
                            "#00CC96"
                        )
                        st.plotly_chart(fig_rh, use_container_width=True)
                        st.divider()

                    # 3. Windsnelheid
                    if "wind_speed_ms" in df.columns:
                        fig_wind = create_monthly_percentile_boxplot(
                            df, "wind_speed_ms", 
                            "Windsnelheid (m/s) - Maandelijkse Verdeling & Gemiddelde", 
                            "Windsnelheid (m/s)", 
                            "#2CA02C"
                        )
                        st.plotly_chart(fig_wind, use_container_width=True)
                        st.divider()

                    # 4. Windrichting frequentie
                    if "wind_dir_cardinal" in df.columns:
                        st.subheader("💨 Windrichting Verdeling")
                        wind_counts = df["wind_dir_cardinal"].value_counts().reset_index()
                        wind_counts.columns = ["richting", "frequentie"]
                        fig_rose = px.bar_polar(
                            wind_counts, r="frequentie", theta="richting",
                            template="plotly_dark", title="Windrichting Frequentie (Totaal over gehele periode)",
                            color_discrete_sequence=px.colors.sequential.Plasma
                        )
                        st.plotly_chart(fig_rose, use_container_width=True)
                        st.divider()

                    # 5. Luchtdruk
                    if "mslp_hpa" in df.columns:
                        fig_msl = create_monthly_percentile_boxplot(
                            df, "mslp_hpa", 
                            "Luchtdruk op Zeeniveau (hPa) - Maandelijkse Verdeling & Gemiddelde", 
                            "Luchtdruk (hPa)", 
                            "#AB63FA"
                        )
                        st.plotly_chart(fig_msl, use_container_width=True)
                        st.divider()

                    # 6. Bewolkingsgraad
                    if "tcc_pct" in df.columns:
                        fig_tcc = create_monthly_percentile_boxplot(
                            df, "tcc_pct", 
                            "Totale Bewolkingsgraad (%) - Maandelijkse Verdeling & Gemiddelde", 
                            "Bewolking (%)", 
                            "#FFA15A"
                        )
                        st.plotly_chart(fig_tcc, use_container_width=True)

                # --- TAB 2: RUWE DATA ---
                with tab_data:
                    st.subheader("📋 Ruwe Data Overzicht")
                    
                    display_cols = [c for c in ["time_clean", "maand_naam", "temp_c", "rh_pct", "wind_speed_ms", "wind_dir_cardinal", "mslp_hpa", "tcc_pct"] if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)
                    
                    # CSV Download Knop
                    csv_data = df[display_cols].to_csv(index=False)
                    st.download_button(
                        label="💾 Download Alle Data als CSV",
                        data=csv_data,
                        file_name=f"era5_klimaatdata_{lat}_{lon}_{start_jaar}_{eind_jaar}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Er is een fout opgetreden bij het ophalen/verwerken: {e}")
