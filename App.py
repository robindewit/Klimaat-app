import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cdsapi
import xarray as xr
import os
import tempfile

# 1. Pagina instellingen
st.set_page_config(
    page_title="ERA5 Klimaat Explorer",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 ERA5 Klimaat Explorer")
st.markdown("Analyseer live **ECMWF ERA5** heranalysedata via de Copernicus Climate Data Store (CDS).")

# 2. CDS API Client initialiseren
def get_cds_client():
    url = st.secrets.get("CDS_URL", "https://cds.climate.copernicus.eu/api")
    key = st.secrets.get("CDS_KEY", None)
    if not key:
        st.error("⚠️ Geen CDS_KEY gevonden in Streamlit Secrets! Voeg de sleutel toe via de Instellingen van de app.")
        st.stop()
    return cdsapi.Client(url=url, key=key)

# 3. Functie om live ERA5 data op te halen
@st.cache_data(show_spinner="Live ERA5-data ophalen bij Copernicus CDS...")
def download_era5_point_data(lat, lon, start_jaar, eind_jaar):
    c = get_cds_client()
    jaren = [str(y) for y in range(start_jaar, eind_jaar + 1)]
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"era5_{lat}_{lon}_{start_jaar}_{eind_jaar}.nc")
    
    request = {
        'dataset_short_name': 'reanalysis-era5-single-levels-monthly-means',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': jaren,
        'month': [f"{m:02d}" for m in range(1, 13)],
        'time': '00:00',
        'area': [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
        'data_format': 'netcdf',
        'download_format': 'unarchived'
    }
    
    c.retrieve('reanalysis-era5-single-levels-monthly-means', request, output_path)
    
    # Dataset openen
    ds = xr.open_dataset(output_path)
    
    # 🔍 Slimme detectie van de Tijdsdimensie (time, valid_time, date)
    time_dim = None
    for possible_time in ['valid_time', 'time', 'date', 'valid_month']:
        if possible_time in ds.dims or possible_time in ds.coords:
            time_dim = possible_time
            break
            
    if not time_dim:
        st.error(f"Kon de tijdsdimensie niet vinden in het CDS bestand. Gevonden variabelen: {list(ds.coords.keys())}")
        st.stop()

    # 🔍 Slimme detectie van de Variabele (t2m, 2m_temperature, var167)
    var_name = None
    for possible_var in ['t2m', '2m_temperature', 'var167']:
        if possible_var in ds.data_vars:
            var_name = possible_var
            break
            
    if not var_name:
        # Kies de eerste beschikbare datavariabele als terugvaloptie
        var_name = list(ds.data_vars.keys())[0]
        
    # Selecteer dichtstbijzijnde punt
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    
    # DataFrame maken
    df = ds_point[[var_name]].to_dataframe().reset_index()
    
    # Kolommen uniform maken
    df['time_clean'] = pd.to_datetime(df[time_dim])
    df['temperatuur_c'] = df[var_name] - 273.15  # Kelvin naar Celsius
    df['jaar'] = df['time_clean'].dt.year
    df['maand'] = df['time_clean'].dt.month
    
    return df

# 4. Zijbalk instellingen
st.sidebar.header("⚙️ Instellingen")

st.sidebar.subheader("1. Locatie")
latitude = st.sidebar.number_input("Breedtegraad (Lat)", value=51.5, min_value=-90.0, max_value=90.0, step=0.1)
longitude = st.sidebar.number_input("Lengtegraad (Lon)", value=4.3, min_value=-180.0, max_value=180.0, step=0.1)

st.sidebar.subheader("2. Periode")
jaar_bereik = st.sidebar.slider("Jaarbereik", 1950, 2025, (1990, 2024))

fetch_data = st.sidebar.button("🚀 Haal ERA5 Data Op", type="primary")

# 5. Hoofdscherm logica
if fetch_data or "era5_df" in st.session_state:
    if fetch_data:
        try:
            with st.spinner("Verbinding maken met Copernicus CDS..."):
                st.session_state["era5_df"] = download_era5_point_data(latitude, longitude, jaar_bereik[0], jaar_bereik[1])
                st.session_state["loc_info"] = f"Lat: {latitude}°N, Lon: {longitude}°E"
        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het ophalen van de data: {e}")
            st.stop()
            
    df = st.session_state["era5_df"]
    
    st.subheader(f"📊 Live ERA5 2m Temperatuur voor {st.session_state['loc_info']}")
    
    # Jaargemiddelden berekenen
    df_jaar = df.groupby("jaar")["temperatuur_c"].mean().reset_index()
    klimaat_norm = df_jaar["temperatuur_c"].mean()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Klimaatgemiddelde", f"{klimaat_norm:.2f} °C")
    col2.metric("Warmste Jaar", f"{df_jaar['temperatuur_c'].max():.2f} °C", delta=f"{df_jaar['temperatuur_c'].max() - klimaat_norm:.2f} °C")
    col3.metric("Koudste Jaar", f"{df_jaar['temperatuur_c'].min():.2f} °C", delta=f"{df_jaar['temperatuur_c'].min() - klimaat_norm:.2f} °C")
    
    # Grafiek
    fig = px.line(
        df_jaar, 
        x="jaar", 
        y="temperatuur_c", 
        title=f"Jaarlijkse Gemiddelde Temperatuur ({jaar_bereik[0]}-{jaar_bereik[1]})",
        markers=True,
        labels={"temperatuur_c": "Temperatuur (°C)", "jaar": "Jaar"}
    )
    
    fig.add_hline(
        y=klimaat_norm, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Norm: {klimaat_norm:.2f} °C"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Ruwe dataset inzien
    with st.expander("📄 Bekijk de ruwe dataset"):
        st.dataframe(df[['time_clean', 'jaar', 'maand', 'temperatuur_c']])

else:
    st.info("👈 Stel de coördinaten en periode in de zijbalk in en klik op **'🚀 Haal ERA5 Data Op'**.")
