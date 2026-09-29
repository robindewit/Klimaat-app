import os
import tempfile
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
    page_title="ERA5 Klimaat Explorer Pro", page_icon="🌍", layout="wide"
)

st.title("🌍 ERA5 Klimaat Explorer Pro")
st.markdown(
    "Volledige klimaatstudie op basis van **ECMWF ERA5** heranalysedata via"
    " Copernicus CDS."
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
                return lat, lon, ", ".join(address_parts)
    except Exception:
        pass

    return None, None, None


# 4. Helper functie voor omrekening naar Beaufort
def ms_to_beaufort(ms_val):
    if ms_val < 0.3:
        return 0
    elif ms_val < 1.6:
        return 1
    elif ms_val < 3.4:
        return 2
    elif ms_val < 5.5:
        return 3
    elif ms_val < 8.0:
        return 4
    elif ms_val < 10.8:
        return 5
    elif ms_val < 13.9:
        return 6
    elif ms_val < 17.2:
        return 7
    elif ms_val < 20.8:
        return 8
    elif ms_val < 24.5:
        return 9
    elif ms_val < 28.5:
        return 10
    elif ms_val < 32.7:
        return 11
    else:
        return 12


# 5. Live ERA5 data ophalen (uitgebreide parameterset)
@st.cache_data(
    show_spinner="Live ERA5 klimaatdata ophalen bij Copernicus CDS..."
)
def download_era5_full_data(lat, lon, start_jaar, eind_jaar):
    c = get_cds_client()
    jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(
        temp_dir, f"era5_full_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc"
    )

    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "total_precipitation",
        "snowfall",
        "evaporation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_solar_radiation_downwards",
        "forecast_albedo",
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

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means", request, output_path
    )

    ds = xr.open_dataset(output_path)

    # Tijdsdimensie bepalen
    time_dim = next(
        (
            t
            for t in ["valid_time", "time", "date", "valid_month"]
            if t in ds.dims or t in ds.coords
        ),
        None,
    )

    if not time_dim:
        st.error("Tijdsdimensie niet gevonden in het CDS bestand.")
        st.stop()

    # Gebruik nearest neighbor om dichtstbijzijnde rasterpunt te pakken
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    df = ds_point.to_dataframe().reset_index()

    df["time_clean"] = pd.to_datetime(df[time_dim])
    df["jaar"] = df["time_clean"].dt.year
    df["maand"] = df["time_clean"].dt.month

    # -- OMREKENINGEN --
    # 1. Temperatuur & Dauwpunt (°C)
    t2m = next((c for c in ["t2m", "2m_temperature"] if c in df.columns), None)
    d2m = next(
        (c for c in ["d2m", "2m_dewpoint_temperature"] if c in df.columns), None
    )

    if t2m:
        df["temp_c"] = df[t2m] - 273.15
    if d2m:
        df["dewpoint_c"] = df[d2m] - 273.15

    # 2. Relatieve Luchtvochtigheid (%)
    if t2m and d2m:
        df["rel_humidity"] = 100 * (
            np.exp((17.625 * df["dewpoint_c"]) / (243.04 + df["dewpoint_c"]))
            / np.exp((17.625 * df["temp_c"]) / (243.04 + df["temp_c"]))
        )
        df["rel_humidity"] = df["rel_humidity"].clip(0, 100)

    # 3. Neerslag & Verdamping (Conversie van m/dag naar mm/maand)
    tp = next((c for c in ["tp", "total_precipitation"] if c in df.columns), None)
    sf = next((c for c in ["sf", "snowfall"] if c in df.columns), None)
    e = next((c for c in ["e", "evaporation"] if c in df.columns), None)

    if tp:
        df["neerslag_mm"] = df[tp] * 1000 * 30.4375
    if sf:
        df["sneeuw_mm"] = df[sf] * 1000 * 30.4375
    else:
        df["sneeuw_mm"] = 0.0

    if e:
        df["verdamping_mm"] = np.abs(df[e]) * 1000 * 30.4375
        df["waterbalans_mm"] = df["neerslag_mm"] - df["verdamping_mm"]

    # 4. Windcomponenten (m/s)
    u10 = next(
        (c for c in ["u10", "10m_u_component_of_wind"] if c in df.columns), None
    )
    v10 = next(
        (c for c in ["v10", "10m_v_component_of_wind"] if c in df.columns), None
    )

    if u10 and v10:
        df["windsnelheid_ms"] = np.sqrt(df[u10] ** 2 + df[v10] ** 2)
        df["windrichting_deg"] = (
            270 - (np.arctan2(df[v10], df[u10]) * 180 / np.pi)
        ) % 360

    # 5. Straling (W/m2)
    ssrd = next(
        (
            c
            for c in ["ssrd", "surface_solar_radiation_downwards"]
            if c in df.columns
        ),
        None,
    )
    if ssrd:
        df["straling_wm2"] = df[ssrd] / 86400.0

    # 6. Schatting Onweerspotentieel
    if t2m and d2m:
        df["cape_est"] = np.maximum(
            0, (df["temp_c"] - df["dewpoint_c"]) * 12.5 + (df["temp_c"] * 1.5)
        )

    return df


# 6. Session State Initialiseren
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
        if found_lat is not None:
            st.session_state["lat"] = found_lat
            st.session_state["lon"] = found_lon
            st.session_state["location_name"] = full_address
            st.session_state["input_lat"] = found_lat
            st.session_state["input_lon"] = found_lon
            st.toast(f"📍 Gevonden: {full_address}", icon="✅")


def on_manual_coord_change():
    st.session_state["lat"] = st.session_state["input_lat"]
    st.session_state["lon"] = st.session_state["input_lon"]
    st.session_state["location_name"] = "Aangepaste coördinaten"


# 7. Zijbalk instellingen
st.sidebar.header("⚙️ Instellingen")

st.sidebar.subheader("1. Locatie zoeken")
st.sidebar.text_input(
    "Voer een stad of locatie in:",
    value="Middelburg",
    key="city_input_key",
)
st.sidebar.button(
    "🔍 Zoek & Kopieer Coördinaten",
    on_click=update_coords_from_search,
    use_container_width=True,
)

st.sidebar.markdown("---")
st.sidebar.number_input(
    "Breedtegraad (Lat)",
    min_value=-90.0,
    max_value=90.0,
    step=0.01,
    format="%.4f",
    key="input_lat",
    value=st.session_state["lat"],
    on_change=on_manual_coord_change,
)
st.sidebar.number_input(
    "Lengtegraad (Lon)",
    min_value=-180.0,
    max_value=180.0,
    step=0.01,
    format="%.4f",
    key="input_lon",
    value=st.session_state["lon"],
    on_change=on_manual_coord_change,
)

st.sidebar.subheader("2. Eenheid Schakelaar")
wind_unit = st.sidebar.radio(
    "Wind-eenheid weergeven in:",
    options=["m/s", "Knopen (kt)", "Beaufort (Bft)"],
    index=0,
)

st.sidebar.subheader("3. Periode")
jaar_bereik = st.sidebar.slider("Jaarbereik", 1950, 2025, (1990, 2024))

fetch_data = st.sidebar.button(
    "🚀 Haal Klimaatdata Op", type="primary", use_container_width=True
)

# Maandnamen
maand_namen = {
    1: "Jan",
    2: "Feb",
    3: "Mrt",
    4: "Apr",
    5: "Mei",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Okt",
    11: "Nov",
    12: "Dec",
}

# 8. Hoofdscherm logica
if fetch_data or "era5_full_df" in st.session_state:
    if fetch_data:
        current_lat = st.session_state["input_lat"]
        current_lon = st.session_state["input_lon"]
        current_label = st.session_state.get(
            "location_name", "Aangepaste coördinaten"
        )

        try:
            with st.spinner("Verbinding maken met Copernicus CDS..."):
                st.session_state["era5_full_df"] = download_era5_full_data(
                    current_lat, current_lon, jaar_bereik[0], jaar_bereik[1]
                )
                st.session_state["loc_info"] = (
                    f"{current_label} (Lat: {current_lat:.4f}°N, Lon:"
                    f" {current_lon:.4f}°E)"
                )
        except Exception as e:
            st.error(f"Fout bij ophalen van data: {e}")
            st.stop()

    df = st.session_state["era5_full_df"].copy()
    df["maand_naam"] = df["maand"].map(maand_namen)

    # Omrekening wind op basis van zijbalk-keuze
    if wind_unit == "Knopen (kt)":
        df["wind_display"] = df["windsnelheid_ms"] * 1.94384
        unit_label = "kt"
    elif wind_unit == "Beaufort (Bft)":
        df["wind_display"] = df["windsnelheid_ms"].apply(ms_to_beaufort)
        unit_label = "Bft"
    else:
        df["wind_display"] = df["windsnelheid_ms"]
        unit_label = "m/s"

    st.caption(f"📍 **Locatie:** {st.session_state['loc_info']}")

    # --- INDELING MET TABS ---
    tab_temp, tab_precip, tab_convect, tab_wind, tab_hum, tab_solar = st.tabs([
        "🌡️ Temperatuur",
        "🌧️ Neerslag & Droogte",
        "❄️ Sneeuw & Onweer",
        "🌬️ Wind & Windroos",
        "💧 Luchtvochtigheid",
        "☀️ Zonne-klimaat",
    ])

    # --- TAB 1: TEMPERATUUR ---
    with tab_temp:
        st.subheader("Maandelijkse Temperatuurverdeling (Boxplot)")

        df_t_stats = (
            df.groupby(["maand", "maand_naam"])["temp_c"]
            .agg(gemiddelde="mean", p50="median", p90=lambda x: np.percentile(x, 90))
            .reset_index()
        )

        fig_t = go.Figure()
        fig_t.add_trace(
            go.Box(
                x=df["maand_naam"],
                y=df["temp_c"],
                name="Verdeling (P50 & Range)",
                fillcolor="rgba(100, 149, 237, 0.4)",
                line=dict(color="#1f77b4", width=2),
                boxmean=True,
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=df_t_stats["maand_naam"],
                y=df_t_stats["gemiddelde"],
                mode="markers+lines",
                name="Gemiddelde (◆)",
                marker=dict(size=9, color="red", symbol="diamond"),
            )
        )
        fig_t.update_layout(
            title="2m Temperatuur per Maand (°C)",
            yaxis_title="Temperatuur (°C)",
            xaxis=dict(
                categoryorder="array", categoryarray=list(maand_namen.values())
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # --- TAB 2: NEERSLAG & DROOGTE ---
    with tab_precip:
        st.subheader("1. Neerslagsom per Maand (incl. P50 & P90)")

        df_p_stats = (
            df.groupby(["maand", "maand_naam"])["neerslag_mm"]
            .agg(gemiddelde="mean", p50="median", p90=lambda x: np.percentile(x, 90))
            .reset_index()
        )

        fig_p = go.Figure()
        fig_p.add_trace(
            go.Bar(
                x=df_p_stats["maand_naam"],
                y=df_p_stats["gemiddelde"],
                name="Gemiddelde Neerslag",
                marker_color="rgba(54, 162, 235, 0.7)",
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=df_p_stats["maand_naam"],
                y=df_p_stats["p50"],
                mode="lines+markers",
                name="P50 (Mediaan)",
                line=dict(color="darkblue", width=2, dash="dash"),
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=df_p_stats["maand_naam"],
                y=df_p_stats["p90"],
                mode="lines+markers",
                name="P90 (Extreem nat)",
                line=dict(color="orange", width=2, dash="dot"),
            )
        )
        fig_p.update_layout(
            title="Maandelijkse Neerslagsom (mm)",
            yaxis_title="Neerslag (mm)",
            xaxis=dict(
                categoryorder="array", categoryarray=list(maand_namen.values())
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("2. Hydrologische Balans: Neerslag vs. Verdamping (P - E)")
        df_bal = (
            df.groupby(["maand", "maand_naam"])[
                ["neerslag_mm", "verdamping_mm", "waterbalans_mm"]
            ]
            .mean()
            .reset_index()
        )

        fig_bal = go.Figure()
        fig_bal.add_trace(
            go.Bar(
                x=df_bal["maand_naam"],
                y=df_bal["neerslag_mm"],
                name="Neerslag (P)",
                marker_color="blue",
            )
        )
        fig_bal.add_trace(
            go.Bar(
                x=df_bal["maand_naam"],
                y=df_bal["verdamping_mm"],
                name="Verdamping (E)",
                marker_color="brown",
            )
        )
        fig_bal.add_trace(
            go.Scatter(
                x=df_bal["maand_naam"],
                y=df_bal["waterbalans_mm"],
                mode="lines+markers",
                name="Overschot / Tekort (P-E)",
                line=dict(color="green", width=3),
            )
        )
        fig_bal.update_layout(
            title="Neerslagoverschot en -tekort per Maand (mm)",
            yaxis_title="mm / maand",
            barmode="group",
            xaxis=dict(
                categoryorder="array", categoryarray=list(maand_namen.values())
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    # --- TAB 3: SNEEUW & ONWEER ---
    with tab_convect:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Sneeuw vs. Vloeibare Neerslag")
            df_snow = (
                df.groupby(["maand", "maand_naam"])[["neerslag_mm", "sneeuw_mm"]]
                .mean()
                .reset_index()
            )
            df_snow["regen_mm"] = np.maximum(
                0, df_snow["neerslag_mm"] - df_snow["sneeuw_mm"]
            )

            fig_sn = go.Figure()
            fig_sn.add_trace(
                go.Bar(
                    x=df_snow["maand_naam"],
                    y=df_snow["regen_mm"],
                    name="Regen (Vloeibaar)",
                    marker_color="#4682B4",
                )
            )
            fig_sn.add_trace(
                go.Bar(
                    x=df_snow["maand_naam"],
                    y=df_snow["sneeuw_mm"],
                    name="Sneeuw (Vast)",
                    marker_color="#E0FFFF",
                )
            )
            fig_sn.update_layout(
                title="Neerslagfase per Maand (mm)",
                barmode="stack",
                xaxis=dict(
                    categoryorder="array", categoryarray=list(maand_namen.values())
                ),
            )
            st.plotly_chart(fig_sn, use_container_width=True)

        with col_c2:
            st.subheader("Onweers- & Buienpotentiaal (Convectie-index)")
            df_cape = (
                df.groupby(["maand", "maand_naam"])["cape_est"].mean().reset_index()
            )

            fig_cp = go.Figure()
            fig_cp.add_trace(
                go.Bar(
                    x=df_cape["maand_naam"],
                    y=df_cape["cape_est"],
                    name="Convectie Index",
                    marker_color="orange",
                )
            )
            fig_cp.update_layout(
                title="Relatieve Onweerskans per Maand",
                yaxis_title="Convectieve Index",
                xaxis=dict(
                    categoryorder="array", categoryarray=list(maand_namen.values())
                ),
            )
            st.plotly_chart(fig_cp, use_container_width=True)

    # --- TAB 4: WIND & WINDROOS ---
    with tab_wind:
        col_w1, col_w2 = st.columns([1, 1])

        with col_w1:
            st.subheader(f"Windsnelheid ({unit_label})")
            df_w_stats = (
                df.groupby(["maand", "maand_naam"])["wind_display"]
                .agg(
                    gemiddelde="mean",
                    p50="median",
                    p90=lambda x: np.percentile(x, 90),
                )
                .reset_index()
            )

            fig_w = go.Figure()
            fig_w.add_trace(
                go.Bar(
                    x=df_w_stats["maand_naam"],
                    y=df_w_stats["gemiddelde"],
                    name=f"Gemiddelde ({unit_label})",
                    marker_color="teal",
                )
            )
            fig_w.add_trace(
                go.Scatter(
                    x=df_w_stats["maand_naam"],
                    y=df_w_stats["p50"],
                    mode="lines+markers",
                    name="P50",
                    line=dict(color="blue", dash="dash"),
                )
            )
            fig_w.add_trace(
                go.Scatter(
                    x=df_w_stats["maand_naam"],
                    y=df_w_stats["p90"],
                    mode="lines+markers",
                    name="P90 (Top 10% Wind)",
                    line=dict(color="red", dash="dot"),
                )
            )
            fig_w.update_layout(
                title=f"Windsnelheid per Maand ({unit_label})",
                yaxis_title=unit_label,
                xaxis=dict(
                    categoryorder="array", categoryarray=list(maand_namen.values())
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with col_w2:
            st.subheader("Windroos (Windrichting)")
            sel_maand = st.selectbox(
                "Kies Maand voor Windroos:", list(maand_namen.values()), index=0
            )
            m_num = [k for k, v in maand_namen.items() if v == sel_maand][0]

            df_m = df[df["maand"] == m_num]

            bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
            labels = ["N", "NO", "O", "ZO", "Z", "ZW", "W", "NW", "N"]
            wind_cat = pd.cut(
                df_m["windrichting_deg"],
                bins=bins,
                labels=labels,
                ordered=False,
            )

            df_rose = wind_cat.value_counts().reset_index()
            df_rose.columns = ["richting", "frequentie"]

            fig_rose = px.bar_polar(
                df_rose,
                r="frequentie",
                theta="richting",
                title=f"Windrichtingsverdeling ({sel_maand})",
                color_discrete_sequence=px.colors.sequential.Plasma,
            )
            st.plotly_chart(fig_rose, use_container_width=True)

    # --- TAB 5: LUCHTVOCHTIGHEID ---
    with tab_hum:
        st.subheader("Relatieve Luchtvochtigheid (%)")

        df_h_stats = (
            df.groupby(["maand", "maand_naam"])["rel_humidity"]
            .agg(
                gemiddelde="mean",
                p50="median",
                p10=lambda x: np.percentile(x, 10),
                p90=lambda x: np.percentile(x, 90),
            )
            .reset_index()
        )

        fig_h = go.Figure()
        fig_h.add_trace(
            go.Scatter(
                x=df_h_stats["maand_naam"],
                y=df_h_stats["p90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig_h.add_trace(
            go.Scatter(
                x=df_h_stats["maand_naam"],
                y=df_h_stats["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.4)",
                name="P10 - P90 Bereik",
            )
        )
        fig_h.add_trace(
            go.Scatter(
                x=df_h_stats["maand_naam"],
                y=df_h_stats["p50"],
                mode="lines+markers",
                name="P50 (Mediaan)",
                line=dict(color="blue", width=2),
            )
        )
        fig_h.update_layout(
            title="Relatieve Luchtvochtigheid per Maand (%)",
            yaxis=dict(range=[0, 100]),
            xaxis=dict(
                categoryorder="array", categoryarray=list(maand_namen.values())
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # --- TAB 6: ZONNE-KLIMAAT ---
    with tab_solar:
        st.subheader("Globale Zonnestraling (W/m²)")

        df_s_stats = (
            df.groupby(["maand", "maand_naam"])["straling_wm2"]
            .agg(
                gemiddelde="mean",
                p50="median",
                p90=lambda x: np.percentile(x, 90),
            )
            .reset_index()
        )

        fig_s = go.Figure()
        fig_s.add_trace(
            go.Bar(
                x=df_s_stats["maand_naam"],
                y=df_s_stats["gemiddelde"],
                name="Gemiddelde Straling",
                marker_color="gold",
            )
        )
        fig_s.add_trace(
            go.Scatter(
                x=df_s_stats["maand_naam"],
                y=df_s_stats["p90"],
                mode="lines+markers",
                name="P90 (Zonnige maanden)",
                line=dict(color="orange", dash="dot"),
            )
        )
        fig_s.update_layout(
            title="Invallende Zonnestraling op Maaiveld (W/m²)",
            yaxis_title="W/m²",
            xaxis=dict(
                categoryorder="array", categoryarray=list(maand_namen.values())
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Optionele Tabelweergave
    with st.expander("📄 Bekijk de complete geaggregeerde tabel"):
        st.dataframe(
            df_t_stats.merge(df_p_stats, on=["maand", "maand_naam"]).merge(
                df_w_stats, on=["maand", "maand_naam"]
            )
        )

else:
    st.info(
        "👈 Kies een locatie en periode in de zijbalk en klik op **'🚀 Haal"
        " Klimaatdata Op'**."
    )
