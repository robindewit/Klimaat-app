"""Module 6: locaties zoeken (geocoding) via OpenStreetMap Nominatim."""
from __future__ import annotations

import re

import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

# Nominatim vraagt om een herkenbare user-agent. Tip: voeg je e-mailadres toe,
# bijv. "klimaatapp (jouwnaam@voorbeeld.nl)", om blokkades te voorkomen.
USER_AGENT = "klimaatapp"

_NL_POSTCODE = re.compile(r"^\d{4}\s?[A-Za-z]{2}$")


@st.cache_data(ttl=86400, show_spinner=False)
def _geocode(query: str) -> dict | None:
    """Voer de eigenlijke zoekopdracht uit (gecachet; fouten worden NIET gecachet)."""
    geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
    # Nederlandse postcodes ("3000 WB") expliciet naar Nederland sturen.
    search = f"{query}, Nederland" if _NL_POSTCODE.match(query) else query
    result = geolocator.geocode(search, language="nl")
    if result is None:
        return None
    return {
        "display_name": str(result.address),
        "lat": float(result.latitude),
        "lon": float(result.longitude),
    }


def search_location(query: str) -> dict | None:
    """Zet een adres, plaatsnaam of postcode om naar coördinaten.

    Args:
        query: Zoektekst, bv. ``"Groningen"``, ``"Kaapstad"`` of ``"3000 WB"``.

    Returns:
        ``{'display_name': str, 'lat': float, 'lon': float}`` of ``None`` als
        niets is gevonden of de zoekdienst niet bereikbaar is (er wordt dan een
        melding in de UI getoond).
    """
    query = (query or "").strip()
    if not query:
        st.info("Vul een locatie in om te zoeken.")
        return None

    try:
        location = _geocode(query)
    except GeocoderTimedOut:
        st.warning("De zoekdienst reageerde niet op tijd. Probeer het zo opnieuw.")
        return None
    except (GeocoderUnavailable, GeocoderServiceError) as exc:
        st.error(
            "De zoekdienst (OpenStreetMap) is niet bereikbaar of weigert het verzoek. "
            "Probeer het later opnieuw of gebruik handmatige coördinaten. "
            f"(Detail: {exc})"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - UI mag nooit crashen op zoeken
        st.error(f"Onverwachte fout bij het zoeken naar de locatie: {exc}")
        return None

    if location is None:
        st.warning(f"Geen locatie gevonden voor '{query}'. Probeer een andere schrijfwijze.")
    return location
