"""Module 5: hoofdinterface van de Copernicus ERA5 Klimaatapp (Gebruikers- en Expert Studie)."""
from __future__ import annotations

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    create_climate_fan_chart, create_grouped_month_bars, create_idf_chart,
    create_monthly_box_chart, create_multi_series_chart, create_skewt,
    create_threshold_calendar, create_wind_rose, create_yearly_bar_chart,
)
from utils.cds_client import get_cds_credentials
from utils.data_fetcher import fetch_bundle
from utils.geocoding import search_location
from utils.pdf_generator import generate_climate_pdf
from utils.stats import (
    COL_MEAN, COL_MONTH, COL_MONTH_NR, COL_P50, MONTH_NAMES, UNIT_CHOICES,
    build_monthly_stats, compute_hourly_products, convert_stats_df, number_decimals,
    profile_tables, rose_class_labels, traffic_levels,
)

st.set_page_config(page_title="Copernicus Klimaatapp", page_icon="🌍", layout="wide")

MODE_USER = "Gebruikers Studie (Operationeel)"
MODE_EXPERT = "Expert Studie (Fysisch & Profielen)"
MIN_YEAR, MAX_YEAR, DEFAULT_PERIOD = 1979, 2025, (1991, 2020)
LAYERS = ["Laag 1 (0–7 cm)", "Laag 2 (7–28 cm)", "Laag 3 (28–100 cm)", "Laag 4 (100–289 cm)"]

ERROR_TEXT = {
    "hourly": "Uurdata (timeseries) niet beschikbaar: Tx/Tn, warmte-/vorstdagen, WBGT, windroos, windkans, windstoten (uurmax) en IDF ontbreken.",
    "monthly:core": "Basisparameters (temperatuur, dauwpunt, wind, neerslag, verdamping, zonnestraling) niet beschikbaar.",
    "monthly:aviation": "Windstoten, zicht en wolkenhoogte niet beschikbaar.",
    "monthly:surface_expert": "Neerslagopsplitsing, warmtefluxen, grenslaaghoogte, CAPE en CIN niet beschikbaar.",
    "monthly:soil": "Bodemvocht en bodemtemperatuur niet beschikbaar.",
    "pressure": "Verticale profielen (drukvlakken) niet beschikbaar.",
}


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
    first = display_name.split(",")[0]
    ascii_name = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_") or "locatie"


def _f(value: float, decimals: int = 1) -> str:
    return "-" if not np.isfinite(value) else f"{value:.{decimals}f}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def build_sidebar() -> dict:
    """Teken de sidebar en geef alle instellingen terug."""
    with st.sidebar:
        st.header("⚙️ Instellingen")
        mode = st.radio("Selecteer Studietype", [MODE_USER, MODE_EXPERT])

        st.subheader("📍 Locatie")
        query = st.text_input("Zoek locatie (bijv. stad, adres of regio)", "Utrecht")
        manual = st.checkbox("Handmatige coördinaten gebruiken")
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

        st.subheader("📐 Eenheden")
        units = {
            "wind": st.radio("Wind", UNIT_CHOICES["wind"], horizontal=True),
            "solar": st.radio("Zonnestraling", UNIT_CHOICES["solar"], horizontal=True),
            "cloud": st.radio("Wolkenhoogte", UNIT_CHOICES["cloud"], horizontal=True),
        }
        st.subheader("🎯 Drempels")
        hot = st.number_input("Warmtedag: Tx boven (°C)", 20.0, 45.0, 30.0, 1.0)
        bft = st.slider("Windkans vanaf (Bft)", 4, 10, 6)
        skew_month = st.selectbox("Maand voor Skew-T (Expert)", MONTH_NAMES, index=6)

        st.divider()
        clicked = st.button("Haal Klimaatdata Op", type="primary", disabled=location is None)
    return {"mode": mode, "location": location, "period": (int(period[0]), int(period[1])),
            "units": units, "thr": {"hot": float(hot), "bft": int(bft)},
            "skew_month": skew_month, "clicked": clicked}


# --------------------------------------------------------------------------
# Analyse (downloaden + statistiek)
# --------------------------------------------------------------------------
def run_analysis(mode: str, location: dict, period: tuple[int, int]) -> None:
    """Haal alle data op en bewaar de statistiek in session_state."""
    lat, lon = location["lat"], location["lon"]
    years = tuple(range(period[0], period[1] + 1))
    get_cds_credentials()  # toont zelf een duidelijke fout + stopt bij ontbrekende secrets

    try:
        bundle = fetch_bundle(lat, lon, years, mode == MODE_EXPERT)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Het ophalen van de data is mislukt.\n\n"
            f"**Technische melding:** `{exc}`\n\n"
            "**Mogelijke oplossingen:**\n"
            "- Controleer `url` en `key` in de Streamlit *Secrets*.\n"
            "- Heb je de licenties van de ERA5-datasets geaccepteerd op de CDS-website?\n"
            "- Kies een kortere periode of probeer het over enkele minuten opnieuw."
        )
        st.session_state.pop("analysis", None)
        return

    errors = dict(bundle.errors)
    if errors:
        fetch_bundle.clear()  # mislukte onderdelen niet 24 uur onthouden

    stats = build_monthly_stats(bundle.monthly) if bundle.monthly is not None else {}
    profiles = profile_tables(bundle.pressure) if bundle.pressure is not None else None
    grid = (lat, lon)
    if bundle.monthly is not None:
        grid = (float(bundle.monthly["latitude"]), float(bundle.monthly["longitude"]))

    st.session_state["analysis"] = {
        "run_id": time.time(), "mode": mode, "location": location, "period": period,
        "grid": grid, "stats": stats, "profiles": profiles, "hourly": bundle.hourly,
        "errors": errors, "hp_cache": {},
    }
    st.session_state.pop("pdf", None)


def get_hourly_products(res: dict, thr: dict) -> dict:
    """Afgeleide uurindicatoren (gememoriseerd per drempelcombinatie)."""
    key = (thr["hot"], thr["bft"])
    if key not in res["hp_cache"]:
        res["hp_cache"][key] = compute_hourly_products(res["hourly"], thr["hot"], thr["bft"])
    return res["hp_cache"][key]


# --------------------------------------------------------------------------
# Items & secties (dezelfde structuur voor app en PDF)
# --------------------------------------------------------------------------
def _fig(fig, size: tuple[int, int] = (1000, 450)) -> dict:
    return {"kind": "figure", "fig": fig, "size": size}


def _tbl(title: str, df: pd.DataFrame, decimals: int = 2) -> dict:
    return {"kind": "table", "title": title, "df": df, "decimals": decimals}


def _note(text: str) -> dict:
    return {"kind": "note", "text": text}


def _stat_items(df: pd.DataFrame | None, title: str, unit: str) -> list[dict]:
    """Boxplot + statistiektabel voor één parameter."""
    if df is None:
        return []
    return [_fig(create_monthly_box_chart(df, title, unit)),
            _tbl(f"{title} ({unit}): statistiek per maand", df.drop(columns=[COL_MONTH_NR]), number_decimals(df))]


def _metrics(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"kind": "metrics", "metrics": pairs}] if pairs else []


def section_thermal(S: dict, H: dict, thr: dict) -> list[dict]:
    items: list[dict] = []
    pairs = []
    if "t2m" in S:
        pairs.append(("Gem. jaartemperatuur", f"{_f(S['t2m'][COL_MEAN].mean())} °C"))
    if "hot_year" in H:
        pairs.append((f"Warmtedagen (Tx > {thr['hot']:.0f} °C) / jaar", _f(H["hot_year"].mean())))
        pairs.append(("Vorstdagen (Tn < 0 °C) / jaar", _f(H["frost_year"].mean())))
    if "wbgt_days" in H:
        pairs.append(("Dagen WBGT ≥ 28 °C / jaar", _f(H["wbgt_days"][COL_MEAN].sum())))
    items += _metrics(pairs)
    items += _stat_items(S.get("t2m"), "2m Temperatuur", "°C")
    items += _stat_items(S.get("d2m"), "Dauwpunt (2 m)", "°C")
    if "tx" in H:
        items.append(_fig(create_multi_series_chart({"Tx (dagmaximum)": H["tx"], "Tn (dagminimum)": H["tn"]},
                                                    "Dagelijkse Tx en Tn", "°C")))
        items.append(_tbl("Tx: statistiek per maand (°C)", H["tx"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_tbl("Tn: statistiek per maand (°C)", H["tn"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_fig(create_yearly_bar_chart(
            {f"Warmtedagen (Tx > {thr['hot']:.0f} °C)": H["hot_year"], "Vorstdagen (Tn < 0 °C)": H["frost_year"]},
            "Warmte- en vorstdagen per jaar", "dagen")))
        items.append(_fig(create_grouped_month_bars(
            {"Warmtedagen": H["hot_month"], "Vorstdagen": H["frost_month"]},
            "Warmte- en vorstdagen per maand (gemiddeld aantal)", "dagen")))
    if "wbgt" in H:
        items += _stat_items(H["wbgt"], "WBGT dagmaximum (indicatief)", "°C")
        items += _stat_items(H["wbgt_days"], "Dagen met WBGT ≥ 28 °C", "dagen/maand")
        items.append(_note("WBGT is een schaduwbenadering (0,567·T + 0,393·e + 3,94) uit temperatuur en dauwpunt; "
                           "zon- en windeffecten zijn niet meegenomen. Tx/Tn en dagen zijn UTC-dagen."))
    if "hot_year" not in H:
        items.append(_note(ERROR_TEXT["hourly"]))
    return items


def section_hydro(S: dict) -> list[dict]:
    items: list[dict] = []
    if "tp" not in S or "e" not in S:
        return [_note("Neerslag en/of verdamping niet beschikbaar.")]
    items += _metrics([
        ("Jaarneerslag (P)", f"{_f(S['tp'][COL_MEAN].sum(), 0)} mm"),
        ("Jaarverdamping (E)", f"{_f(S['e'][COL_MEAN].sum(), 0)} mm"),
        ("Jaarlijks overschot (P − E)", f"{_f(S['pe'][COL_MEAN].sum(), 0)} mm"),
    ])
    items.append(_fig(create_grouped_month_bars({"Neerslag (P)": S["tp"], "Verdamping (E)": S["e"]},
                                                "Neerslag vs. verdamping", "mm/maand")))
    items += _stat_items(S["tp"], "Neerslag (P)", "mm/maand")
    items += _stat_items(S["e"], "Verdamping (E)", "mm/maand")
    items += _stat_items(S["pe"], "Neerslagoverschot (P − E)", "mm/maand")
    items.append(_fig(create_climate_fan_chart(S["cum_pe"], "Cumulatieve waterbalans (P − E)", "mm")))
    items.append(_tbl("Cumulatieve waterbalans (mm): statistiek per maand", S["cum_pe"].drop(columns=[COL_MONTH_NR]), 0))
    items.append(_note("Per jaar gecumuleerd van januari tot december; positief = neerslagoverschot, negatief = tekort."))
    return items


def section_wind(S: dict, H: dict, units: dict, thr: dict, period: tuple[int, int]) -> list[dict]:
    items: list[dict] = []
    wu = units["wind"]
    if "wind_speed_h" in H:
        speed, s_title = H["wind_speed_h"], "Windsnelheid (uurgemiddelde)"
    elif "wind_speed" in S:
        speed, s_title = S["wind_speed"], "Windsnelheid (vectorgemiddelde)"
        items.append(_note("Zonder uurdata is windsnelheid berekend uit de gemiddelde windvector en ligt die te laag."))
    else:
        speed, s_title = None, ""
    if speed is not None:
        items += _stat_items(convert_stats_df(speed, "wind", wu), s_title, wu)
    if "gust_max" in H:
        items += _stat_items(convert_stats_df(H["gust_max"], "wind", wu), "Hoogste windstoot per maand", wu)
    elif "i10fg" in S:
        items += _stat_items(convert_stats_df(S["i10fg"], "wind", wu), "Windstoten (gem. momentane stoot)", wu)
    if "rose" in H:
        items.append(_fig(create_wind_rose(H["rose"], rose_class_labels(wu),
                                           f"Windroos {period[0]}–{period[1]} (richting waar de wind vandaan komt)"), (700, 600)))
    if "wind_exceed" in H:
        items += _stat_items(H["wind_exceed"], f"Kans op wind ≥ {thr['bft']} Bft", "% van de uren")
    if "wind_exceed" not in H:
        items.append(_note(ERROR_TEXT["hourly"]))
    if wu == "Bft":
        items.append(_note("Beaufort is hier de continue schaal v = 0,836·B^1,5; percentielen blijven geldig."))
    return items


def _calendar_rows(S: dict, H: dict, thr: dict) -> list[dict]:
    rows: list[dict] = []

    def add(label, df, fmt, orange, red, worse):
        if df is None:
            return
        v = df[COL_P50].to_numpy(float)
        rows.append({"label": label, "levels": traffic_levels(v, orange, red, worse).tolist(),
                     "text": ["–" if not np.isfinite(x) else format(x, fmt) for x in v]})

    add("Zicht P50 (km)", S.get("vis"), ".0f", 20, 10, False)
    if "cbh" in S:
        add("Wolkenbasis P50 (ft)", convert_stats_df(S["cbh"], "cloud", "ft"), ".0f", 3000, 1500, False)
    if "ssrd" in S:
        add("Zonnestraling P50 (W/m²)", convert_stats_df(S["ssrd"], "solar", "W/m²"), ".0f", 150, 75, False)
    add("Windstoot max P50 (m/s)", H.get("gust_max"), ".0f", 20, 25, True)
    add(f"Kans wind ≥ {thr['bft']} Bft P50 (%)", H.get("wind_exceed"), ".0f", 5, 15, True)
    add("WBGT dagmax P50 (°C)", H.get("wbgt"), ".0f", 25, 28, True)
    return rows


def section_solar(S: dict, H: dict, units: dict, thr: dict) -> list[dict]:
    items: list[dict] = []
    if "ssrd" in S:
        su = units["solar"]
        items += _stat_items(convert_stats_df(S["ssrd"], "solar", su), "Inkomende zonnestraling", su)
    if "cbh" in S:
        cu = units["cloud"]
        items += _stat_items(convert_stats_df(S["cbh"], "cloud", cu), "Wolkenhoogte (basis)", cu)
        items.append(_note("Wolkenhoogte is een maandgemiddelde van de wolkenbasis; bij onbewolkte uren is de waarde niet gedefinieerd."))
    items += _stat_items(S.get("vis"), "Zicht", "km")
    rows = _calendar_rows(S, H, thr)
    if rows:
        items.append(_fig(create_threshold_calendar(rows, "Drempelkalender (groen / oranje / rood op basis van P50)"),
                          (1000, 70 * len(rows) + 120)))
        items.append(_note("Drempels (oranje/rood): zicht ≤ 20/10 km; wolkenbasis ≤ 3000/1500 ft; zonnestraling ≤ 150/75 W/m²; "
                           "windstoot ≥ 20/25 m/s; windkans ≥ 5/15 %; WBGT ≥ 25/28 °C. Grijs = geen data."))
    if "ssrd" not in S and "vis" not in S and "cbh" not in S:
        items.append(_note(ERROR_TEXT["monthly:aviation"]))
    return items


def section_thermo(S: dict, P: dict | None, skew_month: str) -> list[dict]:
    items: list[dict] = []
    m = MONTH_NAMES.index(skew_month) + 1
    if P and m in P:
        prof = P[m]
        items.append(_fig(create_skewt(prof, skew_month), (800, 700)))
        names = {"p": "Druk (hPa)", "t": "T (°C)", "td": "Td (°C)", "q": "q (g/kg)", "u": "u (m/s)",
                 "v": "v (m/s)", "wind_speed": "Wind (m/s)", "wind_dir": "Richting (°)", "z": "Geopot. (gpm)"}
        table = prof[[c for c in names if c in prof]].rename(columns=names)
        items.append(_tbl(f"Gemiddeld profiel {skew_month} (1000–200 hPa)", table, 1))
    else:
        items.append(_note(ERROR_TEXT["pressure"]))
    if "cape" in S and "cin" in S:
        items.append(_fig(create_multi_series_chart({"CAPE": S["cape"], "CIN": S["cin"]}, "CAPE en CIN: verloop per maand", "J/kg")))
        items.append(_tbl("CAPE (J/kg): statistiek per maand", S["cape"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_tbl("CIN (J/kg): statistiek per maand", S["cin"].drop(columns=[COL_MONTH_NR]), 1))
    items += _stat_items(S.get("blh"), "Grenslaaghoogte", "m")
    if "cape" not in S and "blh" not in S:
        items.append(_note(ERROR_TEXT["monthly:surface_expert"]))
    return items


def section_convective(S: dict, H: dict) -> list[dict]:
    items: list[dict] = []
    if "lsp" in S and "cp" in S:
        items.append(_fig(create_grouped_month_bars({"Stratiform (large-scale)": S["lsp"], "Convectief": S["cp"]},
                                                    "Stratiforme vs. convectieve neerslag", "mm/maand", stacked=True)))
        items.append(_tbl("Stratiforme neerslag (mm/maand)", S["lsp"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_tbl("Convectieve neerslag (mm/maand)", S["cp"].drop(columns=[COL_MONTH_NR]), 1))
    idf = H.get("idf")
    if idf is not None and len(idf):
        items.append(_fig(create_idf_chart(idf), (1000, 500)))
        piv = idf.pivot(index="duur_h", columns="T", values="intensiteit_mmh")
        piv.columns = [f"T = {int(c)} jr" for c in piv.columns]
        items.append(_tbl("IDF: neerslagintensiteit (mm/uur)", piv.reset_index().rename(columns={"duur_h": "Duur (u)"}), 2))
        items.append(_note("IDF is gefit (Gumbel) op jaarmaxima van uurlijkse ERA5-neerslag. ERA5 middelt over ±28 km en "
                           "onderschat korte, lokale piekbuien; herhalingstijden boven de lengte van de periode zijn extrapolaties."))
    elif "idf" not in H:
        items.append(_note(ERROR_TEXT["hourly"]))
    if "slhf" in S and "sshf" in S:
        items.append(_fig(create_multi_series_chart({"Latente warmteflux (slhf)": S["slhf"], "Voelbare warmteflux (sshf)": S["sshf"]},
                                                    "Warmtefluxen aan het oppervlak", "W/m²")))
        items.append(_tbl("Latente warmteflux (W/m²)", S["slhf"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_tbl("Voelbare warmteflux (W/m²)", S["sshf"].drop(columns=[COL_MONTH_NR]), 1))
        items.append(_note("Fluxen zijn positief wanneer ze van het oppervlak naar de atmosfeer gaan."))
    return items


def section_soil(S: dict) -> list[dict]:
    items: list[dict] = []
    for prefix, title, unit, dec in (("swvl", "Bodemvocht", "m³/m³", 3), ("stl", "Bodemtemperatuur", "°C", 1)):
        layers = {LAYERS[i]: S[f"{prefix}{i + 1}"] for i in range(4) if f"{prefix}{i + 1}" in S}
        if not layers:
            continue
        items.append(_fig(create_multi_series_chart(layers, f"{title} per dieptelaag", unit)))
        table = pd.DataFrame({COL_MONTH: MONTH_NAMES, **{k: v[COL_MEAN].to_numpy(float) for k, v in layers.items()}})
        items.append(_tbl(f"{title}: gemiddelde per maand ({unit})", table, dec))
    return items or [_note(ERROR_TEXT["monthly:soil"])]


def build_sections(res: dict, units: dict, thr: dict, skew_month: str) -> list[dict]:
    """Bouw alle secties (tabs) voor de gekozen studie-modus."""
    S, P = res["stats"], res["profiles"]
    H = get_hourly_products(res, thr)
    sections = [
        {"title": "Thermisch & Comfort", "items": section_thermal(S, H, thr)},
        {"title": "Hydrologie & Waterbalans", "items": section_hydro(S)},
        {"title": "Wind & Storm", "items": section_wind(S, H, units, thr, res["period"])},
        {"title": "Zonne-energie & Luchtvaart", "items": section_solar(S, H, units, thr)},
    ]
    if res["mode"] == MODE_EXPERT:
        sections += [
            {"title": "Thermodynamica & Sfeeropbouw", "items": section_thermo(S, P, skew_month)},
            {"title": "Convectie & Intensiteit", "items": section_convective(S, H)},
            {"title": "Bodemprofielen", "items": section_soil(S)},
        ]
    return sections


# --------------------------------------------------------------------------
# Weergave
# --------------------------------------------------------------------------
def render_items(items: list[dict], prefix: str) -> None:
    for i, it in enumerate(items):
        kind = it["kind"]
        if kind == "metrics":
            for col, (label, value) in zip(st.columns(len(it["metrics"])), it["metrics"]):
                col.metric(label, value)
        elif kind == "figure":
            show_chart(it["fig"])
        elif kind == "note":
            st.info(it["text"])
        elif kind == "table":
            with st.expander(f"📋 {it['title']}"):
                st.dataframe(it["df"].round(it["decimals"]), hide_index=True)
                st.download_button("⬇️ CSV (Excel NL)",
                                   it["df"].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                                   file_name=f"{re.sub(r'[^A-Za-z0-9]+', '_', it['title'])[:60]}.csv",
                                   mime="text/csv", key=f"csv_{prefix}_{i}")


def pdf_controls(res: dict, sections: list[dict], sig: tuple, key: str, allow_generate: bool) -> None:
    """Knoppen voor het genereren/downloaden van het PDF-rapport."""
    pdf = st.session_state.get("pdf")
    valid = pdf is not None and pdf["sig"] == sig
    c1, c2 = st.columns(2)
    if allow_generate and c1.button("📄 Genereer PDF-rapport", key=f"gen_{key}"):
        loc = res["location"]
        with st.spinner("PDF-rapport samenstellen (grafieken exporteren, dit kan even duren)..."):
            try:
                data = generate_climate_pdf(loc["display_name"], loc["lat"], loc["lon"],
                                            f"{res['period'][0]}-{res['period'][1]}", res["mode"], sections)
                st.session_state["pdf"] = {"sig": sig, "bytes": data, "error": None}
            except Exception as exc:  # noqa: BLE001
                st.session_state["pdf"] = {"sig": sig, "bytes": None, "error": str(exc)}
        st.rerun()
    if valid and pdf["bytes"]:
        name = f"Klimaatstudie_{short_name(res['location']['display_name'])}_{'Expert' if res['mode'] == MODE_EXPERT else 'Gebruiker'}.pdf"
        c2.download_button("📄 Download Volledig Klimaatrapport (PDF)", data=pdf["bytes"], file_name=name,
                           mime="application/pdf", type="primary", key=f"dl_{key}")
    elif valid and pdf["error"]:
        st.warning(f"Het PDF-rapport kon niet worden gemaakt: {pdf['error']}")
    elif pdf is not None:
        c2.caption("Instellingen zijn gewijzigd: genereer het rapport opnieuw.")


def render_results(res: dict, cfg: dict) -> None:
    loc, (start, end) = res["location"], res["period"]
    if cfg["mode"] != res["mode"]:
        st.warning(f"De getoonde resultaten horen bij de **{res['mode']}**. Klik op *Haal Klimaatdata Op* voor de gekozen studie.")

    sections = build_sections(res, cfg["units"], cfg["thr"], cfg["skew_month"])
    sig = (res["run_id"], tuple(cfg["units"].items()), cfg["thr"]["hot"], cfg["thr"]["bft"], cfg["skew_month"])

    st.subheader(f"📍 {loc['display_name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coördinaten", f"{loc['lat']:.3f}°, {loc['lon']:.3f}°")
    c2.metric("Gridpunt (ERA5)", f"{res['grid'][0]:.2f}°, {res['grid'][1]:.2f}°")
    c3.metric("Periode", f"{start} – {end}")
    c4.metric("Studie", "Expert" if res["mode"] == MODE_EXPERT else "Gebruiker")

    if res["errors"]:
        st.warning("Niet alle onderdelen zijn opgehaald; de rest van het rapport is wel beschikbaar:\n\n"
                   + "\n".join(f"- {ERROR_TEXT.get(k, k)}" for k in res["errors"]))
        with st.expander("Technische details"):
            for k, v in res["errors"].items():
                st.code(f"{k}: {v}")

    pdf_controls(res, sections, sig, "top", allow_generate=True)
    tabs = st.tabs([s["title"] for s in sections])
    for tab, sec in zip(tabs, sections):
        with tab:
            render_items(sec["items"], re.sub(r"\W+", "", sec["title"]))
    st.divider()
    pdf_controls(res, sections, sig, "bottom", allow_generate=False)


# --------------------------------------------------------------------------
def main() -> None:
    st.title("🌍 Copernicus Klimaatapp")
    st.subheader("Klimaatstatistieken op locatieniveau (ERA5-reanalyse)")
    st.write(
        "Kies een studietype, zoek een locatie en haal de data op. Per maand zie je P10, P25, "
        "P50, P75, P90 en het gemiddelde. **Gebruikers Studie**: operationele en impactgerichte "
        "indicatoren. **Expert Studie**: aanvullend thermodynamica, convectie en bodemprofielen."
    )
    cfg = build_sidebar()
    if cfg["clicked"] and cfg["location"]:
        run_analysis(cfg["mode"], cfg["location"], cfg["period"])

    if "analysis" in st.session_state:
        render_results(st.session_state["analysis"], cfg)
    else:
        st.info("👈 Nog geen analyse gestart. Kies een studietype, zoek een locatie en klik op "
                "**Haal Klimaatdata Op**. De eerste keer kan het enkele minuten duren (wachtrij van de CDS).")


if __name__ == "__main__":
    main()
