import os
import tempfile
import cdsapi
import numpy as np
import pandas as pd
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


# 4. Functie om live ERA5 data op te halen
@st.cache_data(show_spinner="Live ERA5-data ophalen bij Copernicus CDS...")
def download_era5_point_data(lat, lon, start_jaar, eind_jaar):
  c = get_cds_client()
  jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]

  temp_dir = tempfile.gettempdir()
  output_path = os.path.join(
      temp_dir, f"era5_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc"
  )

  request = {
      "product_type": "monthly_averaged_reanalysis",
      "variable": "2m_temperature",
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

  # Variabele detecteren
  var_name = None
  for possible_var in ["t2m", "2m_temperature", "var167"]:
    if possible_var in ds.data_vars:
      var_name = possible_var
      break

  if not var_name:
    var_name = list(ds.data_vars.keys())[0]

  ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
  df = ds_point[[var_name]].to_dataframe().reset_index()

  df["time_clean"] = pd.to_datetime(df[time_dim])
  df["temperatuur_c"] = df[var_name] - 273.15
  df["jaar"] = df["time_clean"].dt.year
  df["maand"] = df["time_clean"].dt.month

  return df


# 5. Session State initialiseren
if "lat" not in st.session_state:
  st.session_state["lat"] = 51.4988
if "lon" not in st.session_state:
  st.session_state["lon"] = 3.6109
if "location_name" not in st.session_state:
  st.session_state["location_name"] = "Middelburg, Zeeland, Nederland"


# Callbacks voor synchronisatie van invoervelden
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
      st.toast(
          "❌ Locatie niet gevonden. Controleer de spelling.", icon="⚠️"
      )


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

  st.subheader("📊 Live ERA5 2m Temperatuur")
  st.caption(f"📍 **Locatie:** {st.session_state['loc_info']}")

  # Maandnamen toevoegen
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
  df["maand_naam"] = df["maand"].map(maand_namen)

  # --- STATISTIEKEN BEREKENEN PER MAAND ---
  df_stats = (
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

  jaargemiddelde = df["temperatuur_c"].mean()

  # Metrics
  warmste_maand_row = df_stats.loc[df_stats["gemiddelde"].idxmax()]
  koudste_maand_row = df_stats.loc[df_stats["gemiddelde"].idxmin()]

  col1, col2, col3 = st.columns(3)
  col1.metric("Klimaatgemiddelde (Totaal)", f"{jaargemiddelde:.2f} °C")
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

  # --- BOXPLOT INSTELLEN MET PLOTLY GRAPH OBJECTS ---
  fig = go.Figure()

  # 1. Toevoegen van de Boxplot (Toont P50/Mediaan, Min, Max, Kwadranten en Whiskers tot P90)
  fig.add_trace(
      go.Box(
          x=df["maand_naam"],
          y=df["temperatuur_c"],
          name="Verdeling (P50 & P90)",
          boxpoints=False,  # Geen losse stippen voor een strak beeld
          fillcolor="rgba(100, 149, 237, 0.4)",  # Zachtblauw
          line=dict(color="#1f77b4", width=2),
          whiskerwidth=0.8,
          boxmean=True,  # Toont automatisch het GEMIDDELDE als een gestreepte/stippellijn in de box
      )
  )

  # 2. Toevoegen van een opvallende marker voor het GEMIDDELDE per maand
  fig.add_trace(
      go.Scatter(
          x=df_stats["maand_naam"],
          y=df_stats["gemiddelde"],
          mode="markers+lines",
          name="Gemiddelde (◆)",
          marker=dict(size=10, color="red", symbol="diamond"),
          line=dict(color="red", width=1.5, dash="dot"),
          hovertemplate=(
              "Maand: %{x}<br>Gemiddelde: %{y:.2f} °C<extra></extra>"
          ),
      )
  )

  # 3. Toevoegen van een lijn voor P90 (Top 10% grens)
  fig.add_trace(
      go.Scatter(
          x=df_stats["maand_naam"],
          y=df_stats["p90"],
          mode="lines+markers",
          name="90% Percentiel (P90)",
          line=dict(color="#ff7f0e", width=2, dash="dash"),
          marker=dict(size=6, color="#ff7f0e"),
          hovertemplate="Maand: %{x}<br>P90: %{y:.2f} °C<extra></extra>",
      )
  )

  # Layout verfijnen
  fig.update_layout(
      title=(
          "Maandelijkse Temperatuurverdeling & Extremen"
          f" ({jaar_bereik[0]}-{jaar_bereik[1]})"
      ),
      yaxis_title="Temperatuur (°C)",
      xaxis_title="Maand",
      xaxis=dict(
          categoryorder="array", categoryarray=list(maand_namen.values())
      ),
      hovermode="x unified",
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
  )

  # Horizontale stippellijn voor het klimaatgemiddelde over alle jaren
  fig.add_hline(
      y=jaargemiddelde,
      line_dash="dot",
      line_color="gray",
      annotation_text=f"Norm: {jaargemiddelde:.2f} °C",
      annotation_position="bottom right",
  )

  st.plotly_chart(fig, use_container_width=True)

  # Ruwe statistieken overzichtstabel
  with st.expander("📄 Bekijk de berekende statistieken per maand"):
    st.dataframe(
        df_stats[[
            "maand_naam",
            "gemiddelde",
            "p50",
            "p90",
            "min_temp",
            "max_temp",
        ]].rename(columns={
            "maand_naam": "Maand",
            "gemiddelde": "Gemiddelde (°C)",
            "p50": "P50 / Mediaan (°C)",
            "p90": "P90 (°C)",
            "min_temp": "Minimum (°C)",
            "max_temp": "Maximum (°C)",
        })
    )

else:
  st.info(
      "👈 Zoek een locatie of stel coördinaten in de zijbalk in en klik op **'🚀"
      " Haal ERA5 Data Op'**."
  )
