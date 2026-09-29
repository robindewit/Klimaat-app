import os
import tempfile
import calendar
import cdsapi
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import xarray as xr

# 1. Pagina instellingen
st.set_page_config(
    page_title="ERA5 Klimaat Explorer", page_icon="🌍", layout="wide"
)

st.title("🌍 ERA5 Klimaat Explorer")
st.markdown(
    "Analyseer live **ECMWF ERA5** heranalysedata via de Copernicus Climate Data Store (CDS)."
)


# 2. CDS API Client initialiseren
def get_cds_client():
    url = st.secrets.get("CDS_URL", "https://cds.climate.copernicus.eu/api")
    key = st.secrets.get("CDS_KEY", None)
    if not key:
        st.error(
            "⚠️ Geen CDS_KEY gevonden in Streamlit Secrets! Voeg de sleutel toe via"
            " de Instellingen van de app."
        )
        st.stop()
    return cdsapi.Client(url=url, key=key)


# 3. Geocoding via Open-Meteo API
def geocode_location(location_name):
    preset_cities = {
        "middelburg": (51.4988, 3.6109, "Middelburg, Zeeland, Nederland"),
        "rotterdam": (51.9244, 4.4777, "Rotterdam, Zuid-Holland, Nederland"),
        "amsterdam": (52.3676, 4.9041, "Amsterdam, Noord-Holland, Nederland"),
        "breda": (51.5866, 4.7759, "Breda, Noord-Brabant, Nederland"),
        "bergen op zoom": (
            51.4946,
            4.2872,
            "Bergen op Zoom, Noord-Brabant, Nederland",
        ),
        "dresden": (51.0504, 13.7373, "Dresden, Saksen, Duitsland"),
        "utrecht": (52.0907, 5.1214, "Utrecht, Nederland"),
        "eindhoven": (51.4416, 5.4697, "Eindhoven, Noord-Brabant, Nederland"),
    }

    clean_query = location_name.strip().lower()
    if clean_query in preset_cities:
        return preset_cities[clean_query]

    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location_name)}&count=1&language=nl&format=json"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                lat = round(result["latitude"], 4)
                lon = round(result["longitude"], 4)

                name = result.get("name", location_name)
                country = result.get("country", "")
                admin1 = result.get("admin1", "")

                address_parts = [p for p in [name, admin1, country] if p]
                full_address = ", ".join(address_parts)

                return lat, lon, full_address
    except Exception:
        pass

    return None, None, None


# Helper functie voor Beaufort conversie
def ms_to_beaufort(ms):
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


# Helper functie om graden om te zetten naar windrichting sector (16 windstreken)
def degrees_to_cardinal(deg):
    dirs = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
            'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((deg + 11.25) / 22.5) % 16
    return dirs[ix]


# 4. Functie om live ERA5 data op te halen (Temperatuur + Wind + Neerslag)
@st.cache_data(show_spinner="Live ERA5-data ophalen bij Copernicus CDS...")
def download_era5_point_data(lat, lon, start_jaar, eind_jaar):
    c = get_cds_client()
    jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(
        temp_dir, f"era5_full_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc"
    )

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
            "snowfall"
        ],
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

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means", request, output_path
    )

    ds = xr.open_dataset(output_path)

    # Tijdsdimensie detecteren
    time_dim = None
    for possible_time in ["valid_time", "time", "date", "valid_month"]:
        if possible_time in ds.dims or possible_time in ds.coords:
            time_dim = possible_time
            break

    if not time_dim:
        st.error(
            "Kon de tijdsdimensie niet vinden in het CDS bestand. Gevonden"
            f" variabelen: {list(ds.coords.keys())}"
        )
        st.stop()

    ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = ds_point.to_dataframe().reset_index()

    df["time_clean"] = pd.to_datetime(df[time_dim])
    df["jaar"] = df["time_clean"].dt.year
    df["maand"] = df["time_clean"].dt.month

    # Temperatuur verwerken
    t_var = next((v for v in ["t2m", "2m_temperature", "var167"] if v in df.columns), None)
    if t_var:
        df["temperatuur_c"] = df[t_var] - 273.15

    # Wind verwerken
    u_var = next((v for v in ["u10", "10m_u_component_of_wind", "var165"] if v in df.columns), None)
    v_var = next((v for v in ["v10", "10m_v_component_of_wind", "var166"] if v in df.columns), None)

    if u_var and v_var:
        df["wind_speed_ms"] = np.sqrt(df[u_var]**2 + df[v_var]**2)
        df["wind_speed_kt"] = df["wind_speed_ms"] * 1.94384
        df["wind_speed_bft"] = df["wind_speed_ms"].apply(ms_to_beaufort)
        df["wind_dir_deg"] = (270 - np.arctan2(df[v_var], df[u_var]) * (180 / np.pi)) % 360
        df["wind_dir_cardinal"] = df["wind_dir_deg"].apply(degrees_to_cardinal)

    # Neerslag verwerken (Omrekening m/dag naar mm/maand)
    tp_var = next((v for v in ["tp", "total_precipitation", "var228"] if v in df.columns), None)
    sf_var = next((v for v in ["sf", "snowfall", "var144"] if v in df.columns), None)

    if tp_var:
        # Aantal dagen per maand berekenen voor nauwkeurige conversie
        df["days_in_month"] = df["time_clean"].dt.days_in_month
        df["neerslag_mm"] = df[tp_var] * 1000 * df["days_in_month"]
        
        if sf_var:
            df["sneeuw_mm"] = df[sf_var] * 1000 * df["days_in_month"]
            # Zorgen dat sneeuw niet groter kan zijn dan totale neerslag (data ruis opvangen)
            df["sneeuw_mm"] = np.minimum(df["sneeuw_mm"], df["neerslag_mm"])
            df["regen_mm"] = np.maximum(0, df["neerslag_mm"] - df["sneeuw_mm"])
        else:
            df["regen_mm"] = df["neerslag_mm"]
            df["sneeuw_mm"] = 0.0

    ds.close()
    return df


# 5. Session State initialiseren
if "lat" not in st.session_state:
    st.session_state["lat"] = 51.4988
if "lon" not in st.session_state:
    st.session_state["lon"] = 3.6109
if "location_name" not in st.session_state:
    st.session_state["location_name"] = "Middelburg, Zeeland, Nederland"


def update_coords_from_search():
    query = st.session_state.get("city_input_key", "")
    if query.strip():
        found_lat, found_lon, full_address = geocode_location(query)
        if found_lat is not None and found_lon is not None:
            st.session_state["lat"] = found_lat
            st.session_state["lon"] = found_lon
            st.session_state["location_name"] = full_address
            st.session_state["input_lat"] = found_lat
            st.session_state["input_lon"] = found_lon
            st.toast(f"📍 Gevonden: {full_address}", icon="✅")
        else:
            st.toast("❌ Locatie niet gevonden. Controleer de spelling.", icon="⚠️")


def on_manual_coord_change():
    st.session_state["lat"] = st.session_state["input_lat"]
    st.session_state["lon"] = st.session_state["input_lon"]
    st.session_state["location_name"] = "Aangepaste coördinaten"


# 6. Zijbalk instellingen
st.sidebar.header("⚙️ Instellingen")

st.sidebar.subheader("1. Locatie zoeken")
st.sidebar.text_input(
    "Voer een stad of locatie in:",
    value="Middelburg",
    key="city_input_key",
    help="Bijv. Middelburg, Rotterdam, Breda, Dresden of Parijs",
)

st.sidebar.button(
    "🔍 Zoek & Kopieer Coördinaten",
    on_click=update_coords_from_search,
    use_container_width=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Geselecteerde Coördinaten:**")

latitude = st.sidebar.number_input(
    "Breedtegraad (Lat)",
    min_value=-90.0,
    max_value=90.0,
    step=0.01,
    format="%.4f",
    key="input_lat",
    value=st.session_state["lat"],
    on_change=on_manual_coord_change,
)

longitude = st.sidebar.number_input(
    "Lengtegraad (Lon)",
    min_value=-180.0,
    max_value=180.0,
    step=0.01,
    format="%.4f",
    key="input_lon",
    value=st.session_state["lon"],
    on_change=on_manual_coord_change,
)

st.sidebar.subheader("2. Periode")
jaar_bereik = st.sidebar.slider("Jaarbereik", 1950, 2025, (1990, 2024))

st.sidebar.subheader("3. Eenheden")
wind_unit = st.sidebar.selectbox(
    "Windsnelheid Eenheid:",
    options=["m/s", "kt", "Bft"],
    index=0
)

fetch_data = st.sidebar.button(
    "🚀 Haal ERA5 Data Op", type="primary", use_container_width=True
)


# 7. Hoofdscherm logica
if fetch_data or "era5_df" in st.session_state:
    if fetch_data:
        current_lat = st.session_state["input_lat"]
        current_lon = st.session_state["input_lon"]
        current_label = st.session_state.get(
            "location_name", "Aangepaste coördinaten"
        )

        try:
            with st.spinner("Verbinding maken met Copernicus CDS..."):
                st.session_state["era5_df"] = download_era5_point_data(
                    current_lat, current_lon, jaar_bereik[0], jaar_bereik[1]
                )
                st.session_state["loc_info"] = (
                    f"{current_label} (Lat: {current_lat:.4f}°N, Lon:"
                    f" {current_lon:.4f}°E)"
                )
        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het ophalen van de data: {e}")
            st.stop()

    df = st.session_state["era5_df"]

    st.caption(f"📍 **Locatie:** {st.session_state['loc_info']}")

    maand_namen = {
        1: "Jan", 2: "Feb", 3: "Mrt", 4: "Apr", 5: "Mei", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec"
    }
    df["maand_naam"] = df["maand"].map(maand_namen)

    # Tabs voor de Klimaatstudie
    tab_temp, tab_wind, tab_precip = st.tabs([
        "🌡️ Temperatuur Studie", "💨 Wind Studie", "🌧️ Neerslag Studie"
    ])

    # --- TAB 1: TEMPERATUUR ---
    with tab_temp:
        st.subheader("📊 ERA5 2m Temperatuur")
        
        df_stats_t = (
            df.groupby(["maand", "maand_naam"])["temperatuur_c"]
            .agg(
                gemiddelde="mean",
                p50="median",
                p90=lambda x: np.percentile(x, 90),
                min_temp="min",
                max_temp="max",
            )
            .reset_index()
        )

        jaargemiddelde_t = df["temperatuur_c"].mean()
        warmste_maand_row = df_stats_t.loc[df_stats_t["gemiddelde"].idxmax()]
        koudste_maand_row = df_stats_t.loc[df_stats_t["gemiddelde"].idxmin()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Klimaatgemiddelde (Totaal)", f"{jaargemiddelde_t:.2f} °C")
        col2.metric(
            f"Warmste Maand ({warmste_maand_row['maand_naam']})",
            f"{warmste_maand_row['gemiddelde']:.2f} °C",
            f"P90: {warmste_maand_row['p90']:.2f} °C",
        )
        col3.metric(
            f"Koudste Maand ({koudste_maand_row['maand_naam']})",
            f"{koudste_maand_row['gemiddelde']:.2f} °C",
            f"P90: {koudste_maand_row['p90']:.2f} °C",
        )

        fig_t = go.Figure()
        fig_t.add_trace(
            go.Box(
                x=df["maand_naam"],
                y=df["temperatuur_c"],
                name="Verdeling (P50 & Range)",
                boxpoints=False,
                fillcolor="rgba(100, 149, 237, 0.4)",
                line=dict(color="#1f77b4", width=2),
                whiskerwidth=0.8,
                boxmean=True,
            )
        )

        fig_t.add_trace(
            go.Scatter(
                x=df_stats_t["maand_naam"],
                y=df_stats_t["gemiddelde"],
                mode="markers+lines",
                name="Gemiddelde (◆)",
                marker=dict(size=10, color="red", symbol="diamond"),
                line=dict(color="red", width=1.5, dash="dot"),
            )
        )

        fig_t.update_layout(
            title=f"Maandelijkse Temperatuurverdeling & Extremen ({jaar_bereik[0]}-{jaar_bereik[1]})",
            yaxis_title="Temperatuur (°C)",
            xaxis_title="Maand",
            xaxis=dict(categoryorder="array", categoryarray=list(maand_namen.values())),
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # --- TAB 2: WIND ---
    with tab_wind:
        st.subheader("💨 ERA5 10m Wind Klimatologie")

        if wind_unit == "m/s":
            wind_col = "wind_speed_ms"
            unit_label = "m/s"
        elif wind_unit == "kt":
            wind_col = "wind_speed_kt"
            unit_label = "kt"
        else:
            wind_col = "wind_speed_bft"
            unit_label = "Bft"

        df_stats_w = (
            df.groupby(["maand", "maand_naam"])[wind_col]
            .agg(
                gemiddelde="mean",
                p50="median",
                p90=lambda x: np.percentile(x, 90),
                max_wind="max"
            )
            .reset_index()
        )

        gem_wind_totaal = df[wind_col].mean()
        windigste_maand = df_stats_w.loc[df_stats_w["gemiddelde"].idxmax()]
        p90_totaal = np.percentile(df[wind_col], 90)

        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Klimaatgemiddelde Windsnelheid", f"{gem_wind_totaal:.2f} {unit_label}")
        w_col2.metric(
            f"Windrijkste Maand ({windigste_maand['maand_naam']})",
            f"{windigste_maand['gemiddelde']:.2f} {unit_label}",
            f"P50: {windigste_maand['p50']:.2f} {unit_label}"
        )
        w_col3.metric(
            f"90e Percentiel (P90 Totaal)",
            f"{p90_totaal:.2f} {unit_label}"
        )

        fig_w = go.Figure()

        fig_w.add_trace(
            go.Box(
                x=df["maand_naam"],
                y=df[wind_col],
                name="Spreiding (P50 Mediaan)",
                boxpoints=False,
                fillcolor="rgba(46, 204, 113, 0.3)",
                line=dict(color="#27ae60", width=2),
                whiskerwidth=0.8,
            )
        )

        fig_w.add_trace(
            go.Scatter(
                x=df_stats_w["maand_naam"],
                y=df_stats_w["gemiddelde"],
                mode="markers+lines",
                name="Gemiddelde (◆)",
                marker=dict(size=10, color="darkgreen", symbol="diamond"),
                line=dict(color="darkgreen", width=1.5, dash="dot"),
            )
        )

        fig_w.update_layout(
            title=f"Windsnelheid Klimatologie per Maand ({jaar_bereik[0]}-{jaar_bereik[1]})",
            yaxis_title=f"Windsnelheid ({unit_label})",
            xaxis_title="Maand",
            xaxis=dict(categoryorder="array", categoryarray=list(maand_namen.values())),
            hovermode="x unified",
        )
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("---")
        st.subheader("🧭 Windroos (Frequentie van Windrichtingen)")

        selected_month = st.selectbox(
            "Filter Windroos op Maand:",
            options=["Hele Jaar"] + list(maand_namen.values()),
            index=0
        )

        if selected_month == "Hele Jaar":
            df_rose = df.copy()
        else:
            df_rose = df[df["maand_naam"] == selected_month]

        if wind_unit == "m/s":
            bins = [0, 2, 4, 6, 8, 10, 100]
            labels = ['< 2', '2 - 4', '4 - 6', '6 - 8', '8 - 10', '> 10']
        elif wind_unit == "kt":
            bins = [0, 4, 8, 12, 16, 22, 100]
            labels = ['< 4', '4 - 8', '8 - 12', '12 - 16', '16 - 22', '> 22']
        else:
            bins = [-0.5, 1.5, 3.5, 5.5, 7.5, 12.5]
            labels = ['0-1 Bft', '2-3 Bft', '4-5 Bft', '6-7 Bft', '> 8 Bft']

        df_rose['speed_class'] = pd.cut(df_rose[wind_col], bins=bins, labels=labels, include_lowest=True)

        fig_rose = px.bar_polar(
            df_rose,
            r="wind_speed_ms",
            theta="wind_dir_cardinal",
            color="speed_class",
            template="plotly_white",
            color_discrete_sequence=px.colors.sequential.Viridis,
            title=f"Windroos - {selected_month} ({jaar_bereik[0]}-{jaar_bereik[1]})",
            category_orders={
                "wind_dir_cardinal": ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
                                      'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
            }
        )

        fig_rose.update_layout(
            polar=dict(
                radialaxis=dict(ticksuffix=" %", angle=45),
                angularaxis=dict(direction="clockwise")
            ),
            legend_title=f"Snelheid ({unit_label})"
        )

        st.plotly_chart(fig_rose, use_container_width=True)

    # --- TAB 3: NEERSLAG ---
    with tab_precip:
        st.subheader("🌧️ ERA5 Totale Neerslag Klimatologie")

        # Statistieken berekenen per maand
        df_stats_p = (
            df.groupby(["maand", "maand_naam"])["neerslag_mm"]
            .agg(
                gemiddelde="mean",
                p50="median",
                p90=lambda x: np.percentile(x, 90),
                max_precip="max"
            )
            .reset_index()
        )

        # Neerslag types gemiddeld per maand
        df_type_p = (
            df.groupby(["maand", "maand_naam"])[["regen_mm", "sneeuw_mm"]]
            .mean()
            .reset_index()
        )

        # Jaartotalen berekenen voor overzicht
        jaartotaal_gem = df.groupby("jaar")["neerslag_mm"].sum().mean()
        natste_maand_row = df_stats_p.loc[df_stats_p["gemiddelde"].idxmax()]
        droogste_maand_row = df_stats_p.loc[df_stats_p["gemiddelde"].idxmin()]

        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Gemiddelde Jaarlijkse Neerslag", f"{jaartotaal_gem:.1f} mm")
        p_col2.metric(
            f"Natste Maand ({natste_maand_row['maand_naam']})",
            f"{natste_maand_row['gemiddelde']:.1f} mm",
            f"P90: {natste_maand_row['p90']:.1f} mm"
        )
        p_col3.metric(
            f"Droogste Maand ({droogste_maand_row['maand_naam']})",
            f"{droogste_maand_row['gemiddelde']:.1f} mm",
            f"P50: {droogste_maand_row['p50']:.1f} mm"
        )

        # GRAFIEK 1: Neerslagsommen Boxplot met Gemiddelde en P90
        fig_p = go.Figure()

        # Boxplot (toont P50 mediaan en spreiding)
        fig_p.add_trace(
            go.Box(
                x=df["maand_naam"],
                y=df["neerslag_mm"],
                name="Spreiding (P50 Mediaan)",
                boxpoints=False,
                fillcolor="rgba(52, 152, 219, 0.3)",
                line=dict(color="#2980b9", width=2),
                whiskerwidth=0.8,
            )
        )

        # P90 Lijn
        fig_p.add_trace(
            go.Scatter(
                x=df_stats_p["maand_naam"],
                y=df_stats_p["p90"],
                mode="markers+lines",
                name="90e Percentiel (P90)",
                marker=dict(size=8, color="#e67e22", symbol="triangle-up"),
                line=dict(color="#e67e22", width=2, dash="dash"),
            )
        )

        # Gemiddelde Lijn
        fig_p.add_trace(
            go.Scatter(
                x=df_stats_p["maand_naam"],
                y=df_stats_p["gemiddelde"],
                mode="markers+lines",
                name="Gemiddelde (◆)",
                marker=dict(size=10, color="navy", symbol="diamond"),
                line=dict(color="navy", width=1.5, dash="dot"),
            )
        )

        fig_p.update_layout(
            title=f"Maandelijkse Neerslagsommen (mm) & Extremen ({jaar_bereik[0]}-{jaar_bereik[1]})",
            yaxis_title="Neerslag (mm/maand)",
            xaxis_title="Maand",
            xaxis=dict(categoryorder="array", categoryarray=list(maand_namen.values())),
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("---")
        st.subheader("❄️ Neerslagsoort: Regen vs. Sneeuw (Waterequivalent)")

        # GRAFIEK 2: Neerslagsoort (Stacked Bar Chart)
        fig_type = go.Figure()

        fig_type.add_trace(
            go.Bar(
                x=df_type_p["maand_naam"],
                y=df_type_p["regen_mm"],
                name="Regen (Vloeibaar)",
                marker_color="#3498db"
            )
        )

        fig_type.add_trace(
            go.Bar(
                x=df_type_p["maand_naam"],
                y=df_type_p["sneeuw_mm"],
                name="Sneeuw (Waterequivalent)",
                marker_color="#95a5a6"
            )
        )

        fig_type.update_layout(
            barmode="stack",
            title=f"Gemiddelde Neerslagopbouw per Maand ({jaar_bereik[0]}-{jaar_bereik[1]})",
            yaxis_title="Neerslag (mm/maand)",
            xaxis_title="Maand",
            xaxis=dict(categoryorder="array", categoryarray=list(maand_namen.values())),
            hovermode="x unified"
        )
        st.plotly_chart(fig_type, use_container_width=True)

else:
    st.info(
        "👈 Zoek een locatie of stel coördinaten in de zijbalk in en klik op **'🚀"
        " Haal ERA5 Data Op'**."
    )
