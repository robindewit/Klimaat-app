"""Module 1: authenticatie voor de Copernicus Climate Data Store (CDS) API."""
from __future__ import annotations

from typing import NoReturn

import cdsapi
import streamlit as st

_SETUP_HELP = """
**CDS-credentials ontbreken of zijn niet correct geconfigureerd.**

Maak het bestand `.streamlit/secrets.toml` aan (lokaal) of vul de *Secrets* in
bij je app-instellingen op Streamlit Cloud (*Settings → Secrets*):

```toml
[cds]
url = "https://cds.climate.copernicus.eu/api"
key = "<JOUW-PERSONAL-ACCESS-TOKEN>"
```

Je token vind je op https://cds.climate.copernicus.eu na het inloggen
(profielpagina → *Personal Access Token*). Vergeet niet de licentie van de
ERA5-dataset te accepteren op de datasetpagina.
"""


def _fail(detail: str | None = None) -> NoReturn:
    """Toon een duidelijke foutmelding in de UI en stop de scriptrun."""
    st.error(_SETUP_HELP + (f"\n\n_Detail: {detail}_" if detail else ""))
    st.stop()


def get_cds_client() -> cdsapi.Client:
    """Geef een geauthenticeerde ``cdsapi.Client`` terug.

    Credentials worden uitgelezen uit ``st.secrets["cds"]["url"]`` en
    ``st.secrets["cds"]["key"]``. Bij ontbrekende of placeholder-waarden wordt
    een ``st.error()`` getoond en wordt de scriptrun gestopt.
    """
    try:
        url = str(st.secrets["cds"]["url"]).strip()
        key = str(st.secrets["cds"]["key"]).strip()
    except (KeyError, FileNotFoundError, AttributeError, TypeError) as exc:
        # FileNotFoundError: er is geen secrets.toml gevonden.
        _fail(f"{type(exc).__name__}: {exc}")

    if not url or not key or "<" in key or "JOUW" in key.upper():
        _fail("`url` en/of `key` is leeg of nog een placeholder.")

    try:
        return cdsapi.Client(url=url, key=key, quiet=True)
    except Exception as exc:  # noqa: BLE001 - UI moet nooit crashen op setup
        _fail(f"Client-initialisatie mislukt: {exc}")
