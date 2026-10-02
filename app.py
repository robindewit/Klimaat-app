"""Module 5: hoofdinterface van de Copernicus ERA5 Klimaatapp (Streamlit)."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
import streamlit as st

from components.charts import create_climate_fan_chart, create_monthly_quantile_chart
from utils.cds_client import get_cds_client
from utils.data_fetcher import PARAMETERS, fetch_era5_monthly_data
from utils.geocoding import search_location
from utils.pdf_generator import generate_climate_pdf
from utils.stats import (
    COL_MONTH_NR, VALUE_COLUMNS, calculate_monthly_climatology, number_decimals,
)

st.set_page_config(page_title="Copernicus Klimaatapp", page_icon="🌍", layout="wide")

MIN_YEAR, MAX_YEAR = 1979, 2025
DEFAULT_PERIOD = (1991, 2020)
FIG_BOX = "Boxplot (50% Percentiel, bereik P10-P90)"
FIG_FAN = "Klimaat-bandgrafiek (P10-P90)"


# --------------------------------------------------------------------------
# Hulpfuncties
# --------------------------------------------------------------------------
def show_chart(fig) -> None:
    """Toon een Plotly-figuur op volle breedte (oude en nieuwe Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # noqa: BLE001
        st.plotly_chart(fig, use_container_width=True)


def short_name(display_name: str) -> str:
    """Korte, bestandsnaam-veilige locatienaam (eerste deel van het adres)."""
    first = display_name.split(",")[0]
    ascii_name = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_") or "locatie"


def build_sidebar() -> tuple[dict | None, tuple[int, int], bool]:
    """Teken de sidebar. Geeft (locatie, (start, eind), knop_ingedrukt) terug."""
    with st.sidebar:
        st.header("⚙️ Instellingen")

        st.subheader("📍 Locatie")
        query = st.text_input("Zoek locatie (bijv. stad, adres of regio)", "Utrecht")
        manual = st.checkbox("Handmatige coördinaten gebruiken")

        location: dict | None
        if manual:
            lat = st.number_input("Breedtegraad (°)", -90.0, 90.0, 52.37, 0.01, format="%.2f")
            lon = st.number_input("Lengtegraad (°)", -180.0, 180.0, 4.90, 0.01, format="%.2f")
            location = {"display_name": f"Handmatig ({lat:.2f}°, {lon:.2f}°)", "lat": lat, "lon": lon}
        else:
            location = search_location(query)
            if location:
                st.success(f"📍 {location['display_name']}")
                st.caption(f"Breedtegraad {location['lat']:.4f}° · lengtegraad {location['lon']:.4f}°")

        st.subheader("📅 Tijdsperiode")
        period = st.slider("Jaren (van t/m)", MIN_YEAR, MAX_YEAR, DEFAULT_PERIOD)
        if period[1] - period[0] + 1 < 10:
            st.warning("Minder dan 10 jaar: percentielen zijn dan weinig betrouwbaar (gebruikelijk: 30 jaar).")

        st.divider()
        clicked = st.button("Haal Klimaatdata Op", type="primary", disabled=location is None)
    return location, (int(period[0]), int(period[1])), clicked


def run_analysis(location: dict, period: tuple[int, int]) -> None:
    """Haal alle parameters in één call op, bereken statistiek, grafieken en PDF."""
    lat, lon = location["lat"], location["lon"]
    years = tuple(range(period[0], period[1] + 1))

    get_cds_client()  # toont zelf een duidelijke fout + stopt bij ontbrekende secrets

    try:
        with st.spinner(
            "Alle klimaatparameters ophalen uit Copernicus... Dit kan bij de eerste keer "
            "enkele minuten duren (wachtrij van de CDS)."
        ):
            ds = fetch_era5_monthly_data(lat, lon, years)
            available = [p for p in PARAMETERS if p.key in ds.data_vars]
            if not available:
                raise KeyError(f"geen bekende parameters in data ({list(ds.data_vars)})")

            stats: dict[str, dict] = {}
            figures: dict[str, dict] = {}
            for p in available:
                df = calculate_monthly_climatology(ds, p.key)
                stats[p.title] = {"df": df, "unit": p.unit, "note": p.note}
                figures[p.title] = {
                    FIG_BOX: create_monthly_quantile_chart(df, p.title, p.unit),
                    FIG_FAN: create_climate_fan_chart(df, p.title, p.unit),
                }
    except KeyError as exc:
        st.error(f"De verwachte variabelen ontbreken in de gedownloade data: {exc}.")
        return
    except ValueError as exc:
        st.error(f"Ongeldige invoer: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Het ophalen van de data is mislukt.\n\n"
            f"**Technische melding:** `{exc}`\n\n"
            "**Mogelijke oplossingen:**\n"
            "- Controleer `url` en `key` in de Streamlit *Secrets*.\n"
            "- Heb je de ERA5-licentie geaccepteerd op de CDS-website?\n"
            "- Kies een kortere periode of probeer het over enkele minuten opnieuw."
        )
        st.session_state.pop("analysis", None)
        return

    # PDF-rapport (mag falen zonder de rest van de app te blokkeren)
    pdf_bytes, pdf_error = None, None
    try:
        with st.spinner("Klimaatrapport (PDF) samenstellen..."):
            pdf_bytes = generate_climate_pdf(
                location["display_name"], lat, lon, f"{period[0]}-{period[1]}", stats, figures
            )
    except Exception as exc:  # noqa: BLE001
        pdf_error = f"Het PDF-rapport kon niet worden gemaakt: {exc}"

    st.session_state["analysis"] = {
        "location": location,
        "grid": (float(ds["latitude"]), float(ds["longitude"])),
        "period": period,
        "parameters": available,
        "stats": stats,
        "figures": figures,
        "pdf_bytes": pdf_bytes,
        "pdf_error": pdf_error,
        "pdf_name": f"Klimaatstudie_{short_name(location['display_name'])}.pdf",
    }


def pdf_button(result: dict, key: str) -> None:
    """Download-knop voor het PDF-rapport (of een waarschuwing als dat mislukte)."""
    if result["pdf_bytes"]:
        st.download_button(
            "📄 Download Volledig Klimaatrapport (PDF)",
            data=result["pdf_bytes"],
            file_name=result["pdf_name"],
            mime="application/pdf",
            type="primary",
            key=key,
        )
    elif key == "pdf_top":
        st.warning(result["pdf_error"] or "PDF-rapport niet beschikbaar.")


def render_results(result: dict) -> None:
    """Toon locatie, PDF-knoppen en één tab per parameter."""
    loc = result["location"]
    start, end = result["period"]

    st.subheader(f"📍 {loc['display_name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coördinaten", f"{loc['lat']:.3f}°, {loc['lon']:.3f}°")
    c2.metric("Gridpunt (ERA5)", f"{result['grid'][0]:.2f}°, {result['grid'][1]:.2f}°")
    c3.metric("Periode", f"{start} – {end}")
    st.caption("ERA5 heeft een resolutie van circa 0,25° (±28 km); het dichtstbijzijnde gridpunt wordt gebruikt.")

    pdf_button(result, "pdf_top")

    tabs = st.tabs([p.tab_label for p in result["parameters"]])
    for tab, p in zip(tabs, result["parameters"]):
        with tab:
            if p.note:
                st.info(p.note)
            figs = result["figures"][p.title]
            show_chart(figs[FIG_BOX])
            show_chart(figs[FIG_FAN])

            df: pd.DataFrame = result["stats"][p.title]["df"]
            table = df.drop(columns=[COL_MONTH_NR]).copy()
            table[VALUE_COLUMNS] = table[VALUE_COLUMNS].round(number_decimals(df))
            st.markdown(f"**Statistiek per maand** ({p.unit})")
            st.dataframe(table, hide_index=True)
            st.download_button(
                "⬇️ Download tabel als CSV (Excel NL)",
                data=table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                file_name=f"klimaat_{p.title.lower()}_{start}-{end}.csv",
                mime="text/csv",
                key=f"csv_{p.key}",
            )

    st.divider()
    pdf_button(result, "pdf_bottom")


# --------------------------------------------------------------------------
def main() -> None:
    st.title("🌍 Copernicus Klimaatapp")
    st.subheader("Maandelijkse klimaatstatistieken op basis van ERA5-reanalysedata")
    st.write(
        "Zoek een locatie, kies een periode en haal alle klimaatparameters tegelijk op. "
        "Per maand zie je het gemiddelde, het 50% Percentiel (P50) en het "
        "percentielbereik P10–P90."
    )

    location, period, clicked = build_sidebar()
    if clicked and location:
        run_analysis(location, period)

    if "analysis" in st.session_state:
        render_results(st.session_state["analysis"])
        st.caption("Wijzig je invoer en klik opnieuw op *Haal Klimaatdata Op* om te verversen.")
    else:
        st.info(
            "👈 Nog geen analyse gestart. Zoek in de zijbalk een locatie (stad, adres of regio), "
            "kies de periode en klik op **Haal Klimaatdata Op**."
        )


main()
