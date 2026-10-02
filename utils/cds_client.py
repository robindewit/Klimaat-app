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

Je token vind je op https://cds.climate.copernicus.eu/profile na het inloggen.
Accepteer ook de licentie van de ERA5-datasets op de datasetpagina's.
"""


def _fail(detail: str | None = None) -> NoReturn:
    """Toon een duidelijke foutmelding in de UI en stop de scriptrun."""
    st.error(_SETUP_HELP + (f"\n\n_Detail: {detail}_" if detail else ""))
    st.stop()


def get_cds_credentials() -> tuple[str, str]:
    """Lees (url, key) uit ``st.secrets["cds"]``; toont een fout + stopt bij problemen."""
    try:
        url = str(st.secrets["cds"]["url"]).strip()
        key = str(st.secrets["cds"]["key"]).strip()
    except (KeyError, FileNotFoundError, AttributeError, TypeError) as exc:
        _fail(f"{type(exc).__name__}: {exc}")

    placeholder = any(w in key.upper() for w in ("<", "JOUW", "PLAK"))
    if not url or not key or placeholder:
        _fail("`url` en/of `key` is leeg of nog een placeholder.")
    return url, key


def get_cds_client() -> cdsapi.Client:
    """Geef een geauthenticeerde ``cdsapi.Client`` terug."""
    url, key = get_cds_credentials()
    try:
        return cdsapi.Client(url=url, key=key, quiet=True)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Client-initialisatie mislukt: {exc}")
