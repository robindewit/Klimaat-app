import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import cdsapi

st.set_page_config(page_title="ERA5 Klimaatdata Verwerker", layout="wide")

st.title("🌤️ ERA5 Klimaatdata Verwerker & Visualisatie")
st.markdown("""
Deze applicatie haalt maandgemiddelde klimaatdata op via de Copernicus CDS API 
en berekend automatisch de afgeleide windsnelheid en -richting.
""")

# --- Zijbalk Instellingen ---
st.sidebar.header("Parameters instellen")

start_year = st.sidebar.number_input("Startjaar", min_value=1950, max_value=2026, value=1991)
end_year = st.sidebar.number_input("Eindjaar", min_value=1950, max_value=2026, value=1999)

st.sidebar.subheader("Gebied (Bounding Box)")
north = st.sidebar.number_input("Noord (°N)", value=52.35, format="%.2f")
west = st.sidebar.number_input("West (°E)", value=4.93, format="%.2f")
south = st.sidebar.number_input("Zuid (°N)", value=51.85, format="%.2f")
east = st.sidebar.number_input("Oost (°E)", value=5.43, format="%.2f")

# --- CDS Ophalen Functie ---
def fetch_cds_data(years, bbox):
    """
    Haalt ERA5 maandgemiddelden op via CDS API.
    """
    client = cdsapi.Client()
    
    # Gebruik de juiste variabelen voor maandgemiddelden
    request = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
        ],
        'year': [str(y) for y in years],
        'month': [f"{m:02d}" for m in range(1, 13)],
        'time': '00:00',
        'area': bbox,  # [North, West, South, East]
        'data_format': 'netcdf',  # Vernieuwde parameter t.o.v. 'format'
    }
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "era5_data_download.nc")
    
    client.retrieve('reanalysis-monthly-means-of-daily-means', request, output_path)
    return output_path

# --- Data Verwerkingsfunctie ---
def process_netcdf(file_path):
    """
    Leest het NetCDF bestand in en berekend afgeleide waarden (windsnelheid, windrichting).
    """
    ds = xr.open_dataset(file_path)
    
    # 10m Windsnelheid berekenen uit u- en v-componenten
    if 'u10' in ds and 'v10' in ds:
        ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        # Windrichting in graden (meteoconventie: richting waar de wind VANDAAN komt)
        ds['wind_dir'] = (270 - np.arctan2(ds['v10'], ds['u10']) * (180 / np.pi)) % 360
        
    # Temperatuur van Kelvin naar Celsius omzetten indien aanwezig
    if 't2m' in ds:
        ds['t2m_degC'] = ds['t2m'] - 273.15
        
    return ds

# --- Hoofdinterface ---
if st.sidebar.button("🚀 Klimaatdata Ophalen & Verwerken", type="primary"):
    if start_year > end_year:
        st.error("Startjaar mag niet groter zijn dan eindjaar!")
    else:
        years_list = list(range(start_year, end_year + 1))
        area_bbox = [north, west, south, east]
        
        with st.spinner("Data wordt opgevraagd bij Copernicus CDS (dit kan enkele minuten duren)..."):
            try:
                nc_file = fetch_cds_data(years_list, area_bbox)
                st.success("Data succesvol opgehaald!")
                
                ds = process_netcdf(nc_file)
                
                # --- Visualisaties ---
                st.subheader("📊 Resultaten en Visualisatie")
                
                # Zet om naar pandas DataFrame voor grafieken (gemiddelde over het geselecteerde gebied)
                df_spatial_mean = ds.mean(dim=['latitude', 'longitude']).to_dataframe().reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Gemiddelde Windsnelheid (10m)")
                    st.line_chart(df_spatial_mean.set_index('valid_time' if 'valid_time' in df_spatial_mean else 'time')['wind_speed'])
                    
                with col2:
                    if 't2m_degC' in df_spatial_mean:
                        st.markdown("### Gemiddelde Temperatuur (°C)")
                        st.line_chart(df_spatial_mean.set_index('valid_time' if 'valid_time' in df_spatial_mean else 'time')['t2m_degC'])

                # Datatabel tonen
                st.subheader("📋 Dataset Overzicht")
                st.dataframe(df_spatial_mean.head(24))
                
                # Download knop voor verwerkte CSV
                csv_data = df_spatial_mean.to_csv(index=False)
                st.download_button(
                    label="💾 Download verwerkte data als CSV",
                    data=csv_data,
                    file_name=f"era5_verwerkt_{start_year}_{end_year}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Er is een fout opgetreden bij het ophalen/verwerken: {e}")
