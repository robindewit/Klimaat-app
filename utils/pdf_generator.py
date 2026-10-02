"""Module 7: PDF-klimaatrapport (fpdf2 + Plotly/kaleido)."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

from utils.stats import COL_MEAN, COL_MONTH, COL_P10, COL_P50, COL_P90, number_decimals

EXPORT_W_PX, EXPORT_H_PX = 1000, 450   # PNG-formaat van de grafieken
IMAGE_W_MM = 150.0
IMAGE_H_MM = IMAGE_W_MM * EXPORT_H_PX / EXPORT_W_PX

_REPLACEMENTS = {
    "–": "-", "—": "-", "−": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "≈": "~", "→": "->",
}


def _clean(text: str) -> str:
    """Maak tekst geschikt voor de ingebouwde PDF-lettertypen (latin-1).

    Tekens buiten latin-1 (bv. Cyrillisch of Chinees in plaatsnamen) worden '?'.
    """
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _num(value: float, decimals: int) -> str:
    """Getal met decimale komma; '-' voor ontbrekende waarden."""
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}".replace(".", ",")


def _format_coords(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "Z"
    ew = "O" if lon >= 0 else "W"
    return f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"


def _fig_to_png(fig: go.Figure) -> bytes | None:
    """Exporteer een Plotly-figuur naar PNG; ``None`` als kaleido niet werkt."""
    try:
        return fig.to_image(
            format="png", engine="kaleido",
            width=EXPORT_W_PX, height=EXPORT_H_PX, scale=2,
        )
    except Exception:  # noqa: BLE001 - rapport moet ook zonder grafieken kunnen
        return None


class _ReportPDF(FPDF):
    """FPDF met vaste kop- en voettekst."""

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


def _add_table(pdf: FPDF, df: pd.DataFrame) -> None:
    """Maandtabel met P10, P50 (Percentiel), P90 en Gemiddelde."""
    d = number_decimals(df)
    value_cols = [COL_P10, COL_P50, COL_P90, COL_MEAN]
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(
        col_widths=(20, 30, 40, 30, 34), width=154, text_align="CENTER", line_height=6
    ) as table:
        header = table.row()
        for h in [COL_MONTH, *value_cols]:
            header.cell(_clean(h))
        for _, rec in df.iterrows():
            row = table.row()
            row.cell(_clean(str(rec[COL_MONTH])))
            for c in value_cols:
                row.cell(_num(rec[c], d))


def generate_climate_pdf(
    location_name: str,
    lat: float,
    lon: float,
    years_range: str,
    stats_dict: dict[str, dict],
    figures_dict: dict[str, dict[str, go.Figure]],
) -> bytes:
    """Bouw het klimaatrapport en geef de PDF terug als bytes.

    Args:
        location_name: Naam/adres van de locatie.
        lat, lon: Coördinaten.
        years_range: Analyseperiode als tekst, bv. ``"1991-2020"``.
        stats_dict: ``{parameter_titel: {"df": DataFrame, "unit": str, "note": str}}``
            (DataFrame zoals geleverd door ``calculate_monthly_climatology``).
        figures_dict: ``{parameter_titel: {grafiek_naam: go.Figure}}``.

    Returns:
        De PDF als ``bytes`` (geschikt voor ``st.download_button``). Kan kaleido
        de grafieken niet exporteren, dan bevat het rapport alleen de tabellen
        plus een korte melding.
    """
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 10, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Klimaatrapport")
    pdf.set_author("Copernicus Klimaatapp")

    # ------------------------------------------------ voorpagina
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 14, "Klimaatrapport", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, "Maandelijkse klimatologie op basis van ERA5-reanalyse", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    _key_value(pdf, "Gezochte locatie", location_name)
    _key_value(pdf, "Coördinaten", _format_coords(lat, lon))
    _key_value(pdf, "Analyseperiode", years_range)
    _key_value(pdf, "Datum van generatie", datetime.now().strftime("%d-%m-%Y %H:%M"))
    _key_value(pdf, "Bron", "Copernicus Climate Change Service (C3S), ERA5 monthly averaged data on single levels")
    _key_value(pdf, "Parameters", ", ".join(stats_dict.keys()))

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Toelichting", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(
        0, 5,
        _clean(
            "De statistieken zijn per kalendermaand berekend over alle jaren in de "
            "analyseperiode. P50 (50% Percentiel) is de middelste waarde: de helft van "
            "de jaren ligt eronder. Het percentielbereik P10-P90 geeft het normale bereik "
            "weer waarbinnen 80% van de jaren valt; waarden daarbuiten zijn klimatologische "
            "uitschieters. Het gemiddelde is het rekenkundig gemiddelde.\n\n"
            "ERA5 heeft een ruimtelijke resolutie van circa 0,25° (ongeveer 28 km). De "
            "getoonde waarden gelden voor het dichtstbijzijnde gridpunt en kunnen lokaal "
            "afwijken, bijvoorbeeld in bergachtig of kustgebied."
        ),
        new_x="LMARGIN", new_y="NEXT",
    )

    # ------------------------------------------------ pagina per parameter
    export_ok = True
    for title, info in stats_dict.items():
        df: pd.DataFrame = info["df"]
        unit: str = info.get("unit", "")

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 9, _clean(f"{title} ({unit})"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 6, _clean(f"Statistiek per kalendermaand, periode {years_range}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

        _add_table(pdf, df)
        pdf.ln(4)

        for fig_name, fig in figures_dict.get(title, {}).items():
            png = _fig_to_png(fig) if export_ok else None
            if png is None:
                export_ok = False
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(
                    0, 5,
                    _clean(f"[Grafiek '{fig_name}' kon niet worden geëxporteerd: kaleido niet beschikbaar.]"),
                    new_x="LMARGIN", new_y="NEXT",
                )
                continue
            y0 = pdf.get_y()
            x0 = pdf.l_margin + (pdf.epw - IMAGE_W_MM) / 2
            pdf.image(io.BytesIO(png), x=x0, y=y0, w=IMAGE_W_MM, h=IMAGE_H_MM)
            pdf.set_y(y0 + IMAGE_H_MM + 2)

        if info.get("note"):
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, 4, _clean(info["note"]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())
