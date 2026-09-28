import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Pagina instellingen
st.set_page_config(
    page_title="ERA5 Klimaat Explorer",
    page_icon="🌍",
    layout="wide"
)

# Titel
st.title("🌍 ERA5 Klimaat Explorer")
st.markdown("Welkom bij je eigen Klimaat Studies App! Dit is de allereerste live cloud-versie.")

# Zijbalk met opties
st.sidebar.header("⚙️ Instellingen")

st.sidebar.subheader("1. Locatie selectie")
latitude = st.sidebar.number_input("Breedtegraad (Lat)", value=51.5, step=0.1)
longitude = st.sidebar.number_input("Lengtegraad (Lon)", value=4.3, step=0.1)

st.sidebar.subheader("2. Parameters")
variabele = st.sidebar.selectbox(
    "Kies variabele",
    ["2m Temperatuur (°C)", "Neerslag (mm/dag)", "Windsnelheid 10m (m/s)"]
)

jaar_bereik = st.sidebar.slider("Periode", 1950, 2026, (1991, 2025))

# Demo visualisatie
st.subheader(f"📊 Tijdreeks preview voor {variabele}")
st.caption(f"Coördinaten: {latitude}°N, {longitude}°E | Periode: {jaar_bereik[0]} - {jaar_bereik[1]}")

jaren = np.arange(jaar_bereik[0], jaar_bereik[1] + 1)
trend = (jaren - jaar_bereik[0]) * 0.03
basis_temp = 10.5 + trend + np.random.normal(0, 0.5, len(jaren))

df = pd.DataFrame({"Jaar": jaren, "Waarde": np.round(basis_temp, 2)})

col1, col2, col3 = st.columns(3)
col1.metric("Gemiddelde", f"{df['Waarde'].mean():.2f}")
col2.metric("Maximum", f"{df['Waarde'].max():.2f}")
col3.metric("Minimum", f"{df['Waarde'].min():.2f}")

fig = px.line(df, x="Jaar", y="Waarde", title=f"Demonstratie verloop van {variabele}", markers=True)
fig.add_hline(y=df["Waarde"].mean(), line_dash="dash", line_color="red", annotation_text="Gemiddelde")
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Gefeliciteerd! De app draait live via jouw GitHub repository.")
