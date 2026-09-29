import streamlit as st
import cdsapi
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ERA5 Klimaat Dashboard",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 ERA5 Klimaat Visualisatie Tool")
st.markdown("""
Visualiseer klimaat trends op basis van ECMWF ERA5 reanalyse data.
_Let op: Voor het ophalen van data is een actieve CDS API key nodig in de omgeving of `.cdsapirc` file._
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Parameters & Locatie")

lat = st.sidebar.number_input("Breedtegraad (Latitude)", min_value=-90.0, max_value=90.0, value=52.1, step=0.1)
lon = st.sidebar.number_input("Lengtegraad (Longitude)", min_value=-180.0, max_value=180.0, value=5.1, step=0.1)

start_year = st.sidebar.number_input("Startjaar", min_value=1940, max_value=2025, value=1990)
end_year = st.sidebar.number_input("Eindjaar", min_value=1940, max_value=2025, value=2023)

if start_year > end_year:
    st.sidebar.error("Startjaar moet kleiner of gelijk zijn aan het eindjaar.")

# --- DATA FETCHING FUNCTION ---
@st.cache_data(show_spinner=False)
def fetch_era5_data(lat, lon, start_yr, end_yr):
    c = cdsapi.Client()
    
    years = [str(y) for y in range(start_yr, end_yr + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    
    # Bounding box rondom de geselecteerde coördinaten
    area = [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25]
    
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        temp_filename = tmp_file.name

    try:
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': [
                    '2m_temperature',
                    'total_precipitation',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'surface_solar_radiation_downwards',
                    'forecast_albedo'
                ],
                'year': years,
                'month': months,
                'time': '00:00',
                'area': area,
                'format': 'netcdf',
            },
            temp_filename
        )
        
        # Open dataset met xarray
        ds = xr.open_dataset(temp_filename)
        
        # Selecteer de dichtstbijzijnde gridcel
        ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
        df = ds_point.to_dataframe().reset_index()
        
        ds.close()
        return df

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- MAIN CONTROLS & FETCHING ---
if st.sidebar.button("Data Ophalen & Analyseren", type="primary"):
    with st.spinner("Data wordt opgehaald bij Copernicus CDS... Dit kan even duren."):
        try:
            df = fetch_era5_data(lat, lon, start_year, end_year)
            st.session_state['era5_df'] = df
            st.success("Data succesvol geladen!")
        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het ophalen van de data: {e}")

# --- DASHBOARD CONTENT ---
if 'era5_df' in st.session_state:
    df = st.session_state['era5_df']
    
    # Preprocessing
    if 't2m' in df.columns:
        df['t2m_celsius'] = df['t2m'] - 273.15
    if 'tp' in df.columns:
        df['tp_mm_day'] = df['tp'] * 1000
    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed'] = (df['u10']**2 + df['v10']**2)**0.5
    if 'ssrd' in df.columns:
        df['ssrd_wm2'] = df['ssrd'] / 86400  # J/m² per dag naar W/m²

    tabs = st.tabs([
        "🌡️ Temperatuur", 
        "🌧️ Neerslag", 
        "💨 Wind", 
        "☀️ Zonnestraling", 
        "📋 Data Tabel"
    ])

    # Tab 1: Temperatuur
    with tabs[0]:
        st.subheader("2m Temperatuur Trend (°C)")
        if 't2m_celsius' in df.columns:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=df['valid_time'] if 'valid_time' in df.columns else df['time'], 
                                       y=df['t2m_celsius'], 
                                       mode='lines', 
                                       name='Temperatuur (°C)',
                                       line=dict(color='firebrick')))
            fig_t.update_layout(xaxis_title="Tijd", yaxis_title="Temperatuur (°C)", hovermode="x unified")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.warning("Temperatuurdata (t2m) niet gevonden in de dataset.")

    # Tab 2: Neerslag
    with tabs[1]:
        st.subheader("Totale Neerslag (mm/dag equivalent)")
        if 'tp_mm_day' in df.columns:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=df['valid_time'] if 'valid_time' in df.columns else df['time'], 
                                   y=df['tp_mm_day'], 
                                   name='Neerslag (mm/dag)',
                                   marker_color='royalblue'))
            fig_p.update_layout(xaxis_title="Tijd", yaxis_title="Neerslag (mm/dag)", hovermode="x unified")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("Neerslagdata (tp) niet gevonden in de dataset.")

    # Tab 3: Wind
    with tabs[2]:
        st.subheader("10m Windsnelheid (m/s)")
        if 'wind_speed' in df.columns:
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=df['valid_time'] if 'valid_time' in df.columns else df['time'], 
                                       y=df['wind_speed'], 
                                       mode='lines', 
                                       name='Windsnelheid (m/s)',
                                       line=dict(color='seagreen')))
            fig_w.update_layout(xaxis_title="Tijd", yaxis_title="Windsnelheid (m/s)", hovermode="x unified")
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.warning("Winddata (u10/v10) niet gevonden in de dataset.")

    # Tab 4: Zonnestraling
    with tabs[3]:
        st.subheader("Oppervlakte Zonnestraling (W/m²)")
        if 'ssrd_wm2' in df.columns:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=df['valid_time'] if 'valid_time' in df.columns else df['time'], 
                                       y=df['ssrd_wm2'], 
                                       mode='lines', 
                                       name='Zonnestraling (W/m²)',
                                       line=dict(color='orange')))
            fig_s.update_layout(xaxis_title="Tijd", yaxis_title="Zonnestraling (W/m²)", hovermode="x unified")
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.warning("Zonnestralingsdata (ssrd) niet gevonden in de dataset.")

    # Tab 5: Data Tabel
    with tabs[4]:
        st.subheader("Ruwe Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download knop voor de verwerkte CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data als CSV",
            data=csv,
            file_name=f"era5_data_{lat}_{lon}_{start_year}_{end_year}.csv",
            mime="text/csv",
        )
else:
    st.info("Kies je gewenste locatie en periode in de zijbalk en klik op 'Data Ophalen & Analyseren'.")
