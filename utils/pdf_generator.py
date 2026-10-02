"""Module 7: PDF-klimaatrapport (fpdf2 + Plotly/kaleido), afgestemd op de gekozen studie-modus."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from fpdf import FPDF

_REPL = {"–": "-", "—": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
         "…": "...", "≈": "~", "→": "->", "≥": ">=", "≤": "<=", "·": "-", "Δ": "d"}
DEFAULT_SIZE = (1000, 450)
MAX_IMG_W_MM = 165.0


def _clean(text) -> str:
    """Maak tekst geschikt voor de ingebouwde lettertypen (latin-1)."""
    text = str(text)
    for old, new in _REPL.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _num(value, decimals: int) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "-" if pd.isna(value) else f"{value:.{decimals}f}".replace(".", ",")
    return _clean(value)


def _coords(lat: float, lon: float) -> str:
    return f"{abs(lat):.4f}° {'N' if lat >= 0 else 'Z'}, {abs(lon):.4f}° {'O' if lon >= 0 else 'W'}"


def _to_png(fig, size: tuple[int, int]) -> bytes | None:
    try:
        return fig.to_image(format="png", engine="kaleido", width=size[0], height=size[1], scale=2)
    except Exception:  # noqa: BLE001 - rapport moet ook zonder grafieken kunnen
        return None


class _ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "Copernicus ERA5 Klimaatrapport", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, 17.5, self.w - self.r_margin, 17.5)
        self.set_text_color(0, 0, 0)
        self.set_y(22)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _key_value(pdf: FPDF, key: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(42, 7, _clean(key + ":"))
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 7, _clean(value), new_x="LMARGIN", new_y="NEXT")


def _ensure_space(pdf: FPDF, needed_mm: float) -> None:
    if pdf.get_y() + needed_mm > pdf.page_break_trigger:
        pdf.add_page()


def _table(pdf: FPDF, title: str, df: pd.DataFrame, decimals: int) -> None:
    n = len(df.columns)
    _ensure_space(pdf, (len(df) + 1) * 5.5 + 12)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
    first = 22.0
    rest = (pdf.epw - first) / max(n - 1, 1)
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=(first, *[rest] * (n - 1)), width=pdf.epw,
                   text_align="CENTER", line_height=5.5) as table:
        head = table.row()
        for c in df.columns:
            head.cell(_clean(c))
        for _, rec in df.iterrows():
            row = table.row()
            for c in df.columns:
                row.cell(_num(rec[c], decimals))
    pdf.ln(4)


def generate_climate_pdf(location_name: str, lat: float, lon: float, years_range: str,
                         mode_label: str, sections: list[dict]) -> bytes:
    """Bouw het rapport voor de gekozen studie-modus.

    Args:
        location_name: Naam/adres van de locatie.
        lat, lon: Coördinaten.
        years_range: Analyseperiode, bv. ``"1991-2020"``.
        mode_label: Naam van de studie, bv. ``"Expert Studie"``.
        sections: ``[{"title": str, "items": [item, ...]}]`` met items
            ``{"kind": "metrics", "metrics": [(label, waarde), ...]}``,
            ``{"kind": "figure", "title": str, "fig": go.Figure, "size": (w, h)}``,
            ``{"kind": "table", "title": str, "df": DataFrame, "decimals": int}`` of
            ``{"kind": "note", "text": str}``. Dit is dezelfde structuur die de app toont.

    Returns:
        PDF als ``bytes``. Lukt de grafiek-export niet (kaleido), dan bevat het
        rapport tabellen en kentallen plus een melding.
    """
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 10, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Klimaatrapport")

    # ---------------- voorpagina
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 14, "Klimaatrapport", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, _clean(mode_label), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)
    _key_value(pdf, "Gezochte locatie", location_name)
    _key_value(pdf, "Coördinaten", _coords(lat, lon))
    _key_value(pdf, "Analyseperiode", years_range)
    _key_value(pdf, "Datum van generatie", datetime.now().strftime("%d-%m-%Y %H:%M"))
    _key_value(pdf, "Bron", "Copernicus C3S - ERA5 (maandgemiddelden, drukvlakken en uurdata)")
    _key_value(pdf, "Onderdelen", ", ".join(s["title"] for s in sections))
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Toelichting", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(0, 5, _clean(
        "Statistieken zijn per kalendermaand berekend over alle jaren in de analyseperiode. "
        "P50 (50% Percentiel) is de middelste waarde. In de boxplots geeft de doos het "
        "percentielbereik P25-P75 (IQR) en reiken de whiskers van P10 tot P90. Afgeleide "
        "indicatoren (Tx/Tn, warmtedagen, WBGT, windroos, IDF) zijn berekend uit uurdata "
        "(UTC-dagen). ERA5 heeft een resolutie van circa 0,25 graden (ca. 28 km): waarden "
        "gelden voor het dichtstbijzijnde gridpunt en zijn geen puntmetingen."),
        new_x="LMARGIN", new_y="NEXT")

    # ---------------- secties
    export_ok = True
    for sec in sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _clean(sec["title"]), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for item in sec["items"]:
            kind = item["kind"]
            if kind == "metrics":
                for label, value in item["metrics"]:
                    _ensure_space(pdf, 7)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(70, 6, _clean(label + ":"))
                    pdf.set_font("Helvetica", "", 10)
                    pdf.cell(0, 6, _clean(value), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
            elif kind == "note":
                pdf.set_font("Helvetica", "I", 8.5)
                pdf.set_text_color(90, 90, 90)
                pdf.multi_cell(0, 4.2, _clean(item["text"]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            elif kind == "table":
                _table(pdf, item["title"], item["df"], item.get("decimals", 2))
            elif kind == "figure":
                w_px, h_px = item.get("size", DEFAULT_SIZE)
                w_mm = MAX_IMG_W_MM * min(w_px / DEFAULT_SIZE[0], 1.0)
                h_mm = w_mm * h_px / w_px
                png = _to_png(item["fig"], (w_px, h_px)) if export_ok else None
                if png is None:
                    export_ok = False
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.multi_cell(0, 5, _clean("[Grafiek kon niet worden geëxporteerd: kaleido niet beschikbaar.]"),
                                   new_x="LMARGIN", new_y="NEXT")
                    continue
                _ensure_space(pdf, h_mm + 4)
                y0 = pdf.get_y()
                pdf.image(io.BytesIO(png), x=pdf.l_margin + (pdf.epw - w_mm) / 2, y=y0, w=w_mm, h=h_mm)
                pdf.set_y(y0 + h_mm + 3)
    return bytes(pdf.output())
