import os
import tempfile
import cdsapi
import numpy as np
import pandas as pd
import plotly.express as px
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


# 3. Geocoding via Open-Meteo API (Gratis, zonder API key, zeer betrouwbaar)
def geocode_location(location_name):
  # Bekende fallback steden voor direct resultaat bij netwerkproblemen
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
      ],  # [North, West, South, East]
      "format": "netcdf",
  }

  c.retrieve(
      "reanalysis-era5-single-levels-monthly-means", request, output_path
  )

  ds = xr.open_dataset(output_path)

  # 🔍 Tijdsdimensie detecteren
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

  # 🔍 Variabele detecteren
  var_name = None
  for possible_var in ["t2m", "2m_temperature", "var167"]:
    if possible_var in ds.data_vars:
      var_name = possible_var
      break

  if not var_name:
    var_name = list(ds.data_vars.keys())[0]

  # Dichtstbijzijnde punt selecteren
  ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")
  df = ds_point[[var_name]].to_dataframe().reset_index()

  df["time_clean"] = pd.to_datetime(df[time_dim])
  df["temperatuur_c"] = df[var_name] - 273.15
  df["jaar"] = df["time_clean"].dt.year
  df["maand"] = df["time_clean"].dt.month

  return df


# 5. Session State voor coördinaten & adres initialiseren
if "lat" not in st.session_state:
  st.session_state["lat"] = 51.5
if "lon" not in st.session_state:
  st.session_state["lon"] = 4.3
if "location_name" not in st.session_state:
  st.session_state["location_name"] = "Middelburg, Zeeland, Nederland"


# 6. Zijbalk instellingen
st.sidebar.header("⚙️ Instellingen")

st.sidebar.subheader("1. Locatie zoeken")

search_mode = st.sidebar.radio(
    "Zoekmethode:", ["Stad / Adres", "Coördinaten (Lat / Lon)"], horizontal=True
)

if search_mode == "Stad / Adres":
  city_input = st.sidebar.text_input(
      "Voer een stad of locatie in:",
      value="Middelburg",
      help="Bijv. Amsterdam, Parijs, Bergen op Zoom of Dresden",
  )
  if st.sidebar.button("🔍 Zoek locatie", use_container_width=True):
    if city_input.strip():
      with st.spinner("Locatie zoeken..."):
        found_lat, found_lon, full_address = geocode_location(city_input)
        if found_lat is not None and found_lon is not None:
          st.session_state["lat"] = found_lat
          st.session_state["lon"] = found_lon
          st.session_state["location_name"] = full_address
          st.sidebar.success(f"📍 Gevonden: {full_address}")
        else:
          st.sidebar.error(
              "❌ Locatie niet gevonden. Controleer de spelling of voer direct"
              " coördinaten in."
          )

st.sidebar.markdown("**Geselecteerde Coördinaten:**")
latitude = st.sidebar.number_input(
    "Breedtegraad (Lat)",
    value=st.session_state["lat"],
    min_value=-90.0,
    max_value=90.0,
    step=0.01,
    format="%.4f",
    key="input_lat",
)
longitude = st.sidebar.number_input(
    "Lengtegraad (Lon)",
    value=st.session_state["lon"],
    min_value=-180.0,
    max_value=180.0,
    step=0.01,
    format="%.4f",
    key="input_lon",
)

st.session_state["lat"] = latitude
st.session_state["lon"] = longitude

st.sidebar.subheader("2. Periode")
jaar_bereik = st.sidebar.slider("Jaarbereik", 1950, 2025, (1990, 2024))

fetch_data = st.sidebar.button(
    "🚀 Haal ERA5 Data Op", type="primary", use_container_width=True
)


# 7. Hoofdscherm logica
if fetch_data or "era5_df" in st.session_state:
  if fetch_data:
    try:
      with st.spinner("Verbinding maken met Copernicus CDS..."):
        st.session_state["era5_df"] = download_era5_point_data(
            latitude, longitude, jaar_bereik[0], jaar_bereik[1]
        )

        loc_label = st.session_state.get("location_name", "Aangepaste locatie")
        st.session_state["loc_info"] = (
            f"{loc_label} (Lat: {latitude:.4f}°N, Lon: {longitude:.4f}°E)"
        )
    except Exception as e:
      st.error(f"Er is een fout opgetreden bij het ophalen van de data: {e}")
      st.stop()

  df = st.session_state["era5_df"]

  st.subheader("📊 Live ERA5 2m Temperatuur")
  st.caption(f"📍 **Locatie:** {st.session_state['loc_info']}")

  # Jaargemiddelden berekenen
  df_jaar = df.groupby("jaar")["temperatuur_c"].mean().reset_index()
  klimaat_norm = df_jaar["temperatuur_c"].mean()

  # Metrics
  col1, col2, col3 = st.columns(3)
  col1.metric("Klimaatgemiddelde", f"{klimaat_norm:.2f} °C")
  col2.metric(
      "Warmste Jaar",
      f"{df_jaar['temperatuur_c'].max():.2f} °C",
      delta=f"{df_jaar['temperatuur_c'].max() - klimaat_norm:.2f} °C",
  )
  col3.metric(
      "Koudste Jaar",
      f"{df_jaar['temperatuur_c'].min():.2f} °C",
      delta=f"{df_jaar['temperatuur_c'].min() - klimaat_norm:.2f} °C",
  )

  # Grafiek
  fig = px.line(
      df_jaar,
      x="jaar",
      y="temperatuur_c",
      title=(
          "Jaarlijkse Gemiddelde Temperatuur"
          f" ({jaar_bereik[0]}-{jaar_bereik[1]})"
      ),
      markers=True,
      labels={"temperatuur_c": "Temperatuur (°C)", "jaar": "Jaar"},
  )

  fig.add_hline(
      y=klimaat_norm,
      line_dash="dash",
      line_color="red",
      annotation_text=f"Norm: {klimaat_norm:.2f} °C",
  )

  st.plotly_chart(fig, use_container_width=True)

  # Ruwe dataset inzien
  with st.expander("📄 Bekijk de ruwe dataset"):
    st.dataframe(df[["time_clean", "jaar", "maand", "temperatuur_c"]])

else:
  st.info(
      "👈 Kies een locatie (via stadsnaam of coördinaten) in de zijbalk en klik"
      " op **'🚀 Haal ERA5 Data Op'**."
  )
