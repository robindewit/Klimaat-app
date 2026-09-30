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

st.sidebar.header("⚙️ 3. Instellingen")
wind_eenheid = st.sidebar.selectbox(
    "Windsnelheid Eenheid:",
    ["m/s", "kt", "km/h", "Bft"]
)

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

WIND_RICHTINGEN = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
                   'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']

def degrees_to_cardinal(deg):
    if pd.isna(deg): return "N/A"
    ix = int((deg + 11.25) / 22.5)
    return WIND_RICHTINGEN[ix % 16]

def ms_to_beaufort(ms):
    """
    Officiële WMO/KNMI schaal van Beaufort gebaseerd op m/s.
    """
    if pd.isna(ms): return np.nan
    boundaries = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
    for bft, boundary in enumerate(boundaries):
        if ms < boundary:
            return bft
    return 12

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
            "10m_wind_speed",           # Scalaire windsnelheid (voorkomt uitmiddelen u/v vector)
            "10m_u_component_of_wind",  # U-vector voor windrichting
            "10m_v_component_of_wind",  # V-vector voor windrichting
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

    # Wind (Scalaire windsnelheid & richting)
    si10_var = next((cols_lower[c] for c in ["si10", "10m_wind_speed", "ws10"] if c in cols_lower), None)
    u_var = next((cols_lower[c] for c in ["u10", "10m_u_component_of_wind"] if c in cols_lower), None)
    v_var = next((cols_lower[c] for c in ["v10", "10m_v_component_of_wind"] if c in cols_lower), None)

    if si10_var:
        df["wind_speed_ms"] = df[si10_var]
    elif u_var and v_var:
        # Fallback als si10 niet in de dataset zit
        df["wind_speed_ms"] = np.sqrt(df[u_var] ** 2 + df[v_var] ** 2)

    if "wind_speed_ms" in df.columns:
        df["wind_speed_kt"] = df["wind_speed_ms"] * 1.943844
        df["wind_speed_kmh"] = df["wind_speed_ms"] * 3.6
        df["wind_speed_bft"] = df["wind_speed_ms"].apply(ms_to_beaufort)

    if u_var and v_var:
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
    
    df_avg = df_plot.groupby("maand_naam", observed=False)[var_col].mean().reset_index()

    fig = go.Figure()

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

# --- FUNCTIE VOOR WINDROOS MET WINDSNELHEIDSKLASSEN ---
def create_wind_rose(df, wind_col, wind_unit):
    df_rose = df.dropna(subset=[wind_col, "wind_dir_cardinal"]).copy()
    
    # Bepaal categorieën/bins op basis van geselecteerde eenheid
    if wind_unit == "kt":
        bins = [0, 4, 10, 16, 22, 28, np.inf]
        labels = ["< 4 kt", "4-10 kt", "10-16 kt", "16-22 kt", "22-28 kt", "> 28 kt"]
    elif wind_unit == "km/h":
        bins = [0, 10, 20, 30, 40, 50, np.inf]
        labels = ["< 10 km/h", "10-20 km/h", "20-30 km/h", "30-40 km/h", "40-50 km/h", "> 50 km/h"]
    elif wind_unit == "Bft":
        bins = [-0.5, 1.5, 3.5, 5.5, 7.5, 9.5, 12.5]
        labels = ["0-1 Bft", "2-3 Bft", "4-5 Bft", "6-7 Bft", "8-9 Bft", ">= 10 Bft"]
    else:  # m/s
        bins = [0, 2, 4, 6, 8, 11, np.inf]
        labels = ["< 2 m/s", "2-4 m/s", "4-6 m/s", "6-8 m/s", "8-11 m/s", "> 11 m/s"]

    df_rose["wind_cat"] = pd.cut(df_rose[wind_col], bins=bins, labels=labels, right=False)

    # Telling per richting en snelheidsklasse
    counts = df_rose.groupby(["wind_dir_cardinal", "wind_cat"], observed=False).size().reset_index(name="frequentie")
    counts["wind_dir_cardinal"] = pd.Categorical(counts["wind_dir_cardinal"], categories=WIND_RICHTINGEN, ordered=True)
    counts = counts.sort_values(["wind_dir_cardinal", "wind_cat"])

    fig = px.bar_polar(
        counts,
        r="frequentie",
        theta="wind_dir_cardinal",
        color="wind_cat",
        template="plotly_dark",
        title=f"Windroos met Windsnelheidsverdeling ({wind_unit})",
        color_discrete_sequence=px.colors.sequential.Turbo,
        category_orders={"wind_dir_cardinal": WIND_RICHTINGEN, "wind_cat": labels}
    )

    fig.update_polars(
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
            categoryorder="array",
            categoryarray=WIND_RICHTINGEN
        )
    )
    fig.update_layout(
        legend_title_text=f"Snelheid ({wind_unit})",
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

                tab_grafieken, tab_data = st.tabs([
                    "📊 Klimaatstatistieken & Boxplots", 
                    "📋 Ruwe Data & Exporteren"
                ])

                # Bepalen van de juiste kolom en labels op basis van geselecteerde wind-eenheid
                if wind_eenheid == "kt":
                    wind_col = "wind_speed_kt"
                    wind_label = "Windsnelheid (kt)"
                elif wind_eenheid == "km/h":
                    wind_col = "wind_speed_kmh"
                    wind_label = "Windsnelheid (km/h)"
                elif wind_eenheid == "Bft":
                    wind_col = "wind_speed_bft"
                    wind_label = "Windsnelheid (Bft)"
                else:
                    wind_col = "wind_speed_ms"
                    wind_label = "Windsnelheid (m/s)"

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
                    if wind_col in df.columns:
                        fig_wind = create_monthly_percentile_boxplot(
                            df, wind_col, 
                            f"{wind_label} - Maandelijkse Verdeling & Gemiddelde", 
                            wind_label, 
                            "#2CA02C"
                        )
                        st.plotly_chart(fig_wind, use_container_width=True)
                        st.divider()

                    # 4. Windroos met windsnelheden per richting
                    if "wind_dir_cardinal" in df.columns and wind_col in df.columns:
                        st.subheader("💨 Windroos met Windsnelheidsverdeling")
                        fig_rose = create_wind_rose(df, wind_col, wind_eenheid)
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

                with tab_data:
                    st.subheader("📋 Ruwe Data Overzicht")
                    
                    display_cols = [c for c in ["time_clean", "maand_naam", "temp_c", "rh_pct", wind_col, "wind_dir_cardinal", "mslp_hpa", "tcc_pct"] if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)
                    
                    csv_data = df[display_cols].to_csv(index=False)
                    st.download_button(
                        label="💾 Download Alle Data als CSV",
                        data=csv_data,
                        file_name=f"era5_klimaatdata_{lat}_{lon}_{start_jaar}_{eind_jaar}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Er is een fout opgetreden bij het ophalen/verwerken: {e}")
