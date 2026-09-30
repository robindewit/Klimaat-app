import os
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
import cdsapi

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title="ERA5 Klimaat & Weer Viewer",
    page_icon="🌤️",
    layout="wide"
)

# --- INSTELLINGEN & HELPERS ---
def get_cds_client():
    """Initialiseert de CDS API client via Streamlit Secrets of lokaal .cdsapirc bestand."""
    try:
        # Check eerst of de secrets aanwezig zijn op Streamlit Cloud / lokaal secrets.toml
        if "cds" in st.secrets:
            return cdsapi.Client(
                url=st.secrets["cds"]["url"],
                key=st.secrets["cds"]["key"]
            )
        # Indien geen secrets, valt cdsapi terug op het lokale ~/.cdsapirc bestand
        return cdsapi.Client()
    except Exception as e:
        st.error(
            "❌ Kan geen verbinding maken met de CDS API. "
            "Controleer of je Streamlit Secrets ([cds] url/key) of je `.cdsapirc` bestand juist ingesteld zijn."
        )
        st.stop()

def ms_to_beaufort(ms):
    """Converteert m/s naar Beaufort schaal."""
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
    """Converteert graden naar windrichting (N, NO, O, etc.)."""
    dirs = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO', 
            'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((deg + 11.25) / 22.5)
    return dirs[ix % 16]

# --- ROBUUSTE DOWNLOAD & VERWERKINGSFUNCTIE ---
@st.cache_data(show_spinner="Live ERA5-data ophalen bij Copernicus CDS...")
def download_era5_point_data(lat, lon, start_jaar, eind_jaar):
    c = get_cds_client()
    jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(
        temp_dir, f"era5_full_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc"
    )

    # CDS Request opbouwen
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
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

    # Zorg dat eventuele oude/corrupte bestanden eerst worden verwijderd
    if os.path.exists(output_path):
        os.remove(output_path)

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means", request, output_path
    )

    # --- VALIDATIE VAN HET GEDOWNLOADDE BESTAND ---
    with open(output_path, "rb") as f:
        header = f.read(100)

    # Een geldig NetCDF-3 bestand begint met 'CDF', NetCDF-4/HDF5 met '\x89HDF'
    if not (header.startswith(b"CDF") or header.startswith(b"\x89HDF")):
        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            error_content = f.read()
        st.error("⚠️ CDS heeft geen geldig NetCDF-bestand teruggegeven. Foutmelding van CDS:")
        st.code(error_content)
        st.stop()

    # Probeer het NetCDF-bestand te openen met xarray
    try:
        ds = xr.open_dataset(output_path, engine="netcdf4")
    except Exception as e:
        try:
            ds = xr.open_dataset(output_path, engine="h5netcdf")
        except Exception as inner_e:
            st.error(f"Kan het NetCDF-bestand niet lezen: {inner_e}")
            st.stop()

    # Tijdsdimensie detecteren
    time_dim = None
    for possible_time in ["valid_time", "time", "date", "valid_month"]:
        if possible_time in ds.dims or possible_time in ds.coords:
            time_dim = possible_time
            break

    if not time_dim:
        st.error(f"Geen tijdsdimensie gevonden. Dimensies in bestand: {list(ds.dims.keys())}")
        st.stop()

    # Detecteer geografische namen
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    # Selecteer dichtstbijzijnde roosterpunt
    ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    df = ds_point.to_dataframe().reset_index()

    df["time_clean"] = pd.to_datetime(df[time_dim])
    df["jaar"] = df["time_clean"].dt.year
    df["maand"] = df["time_clean"].dt.month
    df["days_in_month"] = df["time_clean"].dt.days_in_month

    # 1. Temperatuur verwerken (K -> °C)
    t_var = next((v for v in ["t2m", "2m_temperature", "var167"] if v in df.columns), None)
    if t_var:
        df["temperatuur_c"] = df[t_var] - 273.15

    # 2. Windcomponenten verwerken (u & v -> snelheid + richting)
    u_var = next((v for v in ["u10", "10m_u_component_of_wind", "var165"] if v in df.columns), None)
    v_var = next((v for v in ["v10", "10m_v_component_of_wind", "var166"] if v in df.columns), None)

    if u_var and v_var:
        df["wind_speed_ms"] = np.sqrt(df[u_var] ** 2 + df[v_var] ** 2)
        df["wind_speed_kt"] = df["wind_speed_ms"] * 1.94384
        df["wind_speed_bft"] = df["wind_speed_ms"].apply(ms_to_beaufort)
        df["wind_dir_deg"] = (270 - np.arctan2(df[v_var], df[u_var]) * (180 / np.pi)) % 360
        df["wind_dir_cardinal"] = df["wind_dir_deg"].apply(degrees_to_cardinal)

    # 3. Neerslag verwerken (m s^-1 of m -> mm/maand)
    tp_var = next((v for v in ["tp", "total_precipitation", "var228"] if v in df.columns), None)
    if tp_var:
        units = ds[tp_var].attrs.get("units", "m s**-1")
        if units in ["m s**-1", "m/s", "m s-1"]:
            df["neerslag_mm"] = df[tp_var] * 86400.0 * df["days_in_month"] * 1000.0
        elif units in ["m", "meters"]:
            df["neerslag_mm"] = df[tp_var] * 1000.0
        else:
            df["neerslag_mm"] = df[tp_var] * 86400.0 * df["days_in_month"] * 1000.0

    ds.close()
    return df

# --- INTERFACE & SIDEBAR ---
st.title("🌤️ ERA5 Klimaat & Weer Analyse Tool")
st.markdown("Download en analyseer maandelijkse ERA5 reanalyse-data direct vanaf Copernicus (CDS).")

st.sidebar.header("Locatie & Periode")
lat = st.sidebar.number_input("Breedtegraad (Latitude)", value=52.10, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Lengtegraad (Longitude)", value=5.18, step=0.01, format="%.2f")

start_jaar, eind_jaar = st.sidebar.slider(
    "Selecteer periode",
    min_value=1950,
    max_value=2025,
    value=(2015, 2024)
)

if st.sidebar.button("Data ophalen", type="primary"):
    st.cache_data.clear()

# --- MAIN APP LOGICA ---
try:
    df = download_era5_point_data(lat, lon, start_jaar, eind_jaar)

    st.success(f"Data succesvol ingeladen voor coördinaten ({lat:.2f}, {lon:.2f}) voor {start_jaar}-{eind_jaar}!")

    # KPI Overzicht
    col1, col2, col3 = st.columns(3)
    col1.metric("Gemiddelde Temp.", f"{df['temperatuur_c'].mean():.1f} °C")
    col2.metric("Gemiddelde Wind", f"{df['wind_speed_ms'].mean():.1f} m/s ({df['wind_speed_bft'].mean():.0f} Bft)")
    col3.metric("Jaarlijkse Neerslag (~avg)", f"{(df['neerslag_mm'].sum() / (eind_jaar - start_jaar + 1)):.0f} mm")

    st.divider()

    # Tabs voor visualisaties
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperatuur", "🌧️ Neerslag", "💨 Wind", "📄 Ruwe Data"])

    with tab1:
        st.subheader("Temperatuurverloop")
        fig_temp = px.line(
            df, x="time_clean", y="temperatuur_c",
            labels={"time_clean": "Datum", "temperatuur_c": "Temperatuur (°C)"},
            title="2m Temperatuur (Maandgemiddelden)"
        )
        fig_temp.update_traces(line_color="#EF553B")
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        st.subheader("Maandelijkse Neerslag")
        fig_precip = px.bar(
            df, x="time_clean", y="neerslag_mm",
            labels={"time_clean": "Datum", "neerslag_mm": "Neerslag (mm)"},
            title="Totale Maandelijkse Neerslag"
        )
        fig_precip.update_traces(marker_color="#636EFA")
        st.plotly_chart(fig_precip, use_container_width=True)

    with tab3:
        st.subheader("Windsnelheid & Richting")
        fig_wind = px.line(
            df, x="time_clean", y="wind_speed_ms",
            labels={"time_clean": "Datum", "wind_speed_ms": "Windsnelheid (m/s)"},
            title="10m Windsnelheid (Maandgemiddelden)"
        )
        fig_wind.update_traces(line_color="#00CC96")
        st.plotly_chart(fig_wind, use_container_width=True)

    with tab4:
        st.subheader("Tabelweergave")
        st.dataframe(
            df[["time_clean", "temperatuur_c", "neerslag_mm", "wind_speed_ms", "wind_speed_bft", "wind_dir_cardinal"]], 
            use_container_width=True
        )

except Exception as e:
    st.info("⚡ Klik in de sidebar op **Data ophalen** of controleer je parameters om te starten.")
