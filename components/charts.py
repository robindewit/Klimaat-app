"""Module 4: interactieve Plotly-visualisaties (boxplots, bandgrafieken, windroos, Skew-T, IDF, ...)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.stats import (
    COL_MEAN, COL_MONTH, COL_MONTH_NR, COL_P10, COL_P25, COL_P50, COL_P75, COL_P90,
    MONTH_NAMES, SECTOR_NAMES, number_decimals,
)

PALETTE = ["#1f77b4", "#e4572e", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
COLOR_P50, COLOR_MEAN = "#0b3d91", "#e4572e"
LABEL_P50 = "P50 (50% Percentiel)"
_REQUIRED = [COL_MONTH_NR, COL_MONTH, COL_P10, COL_P25, COL_P50, COL_P75, COL_P90, COL_MEAN]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame mist kolommen: {missing}")
    return df.sort_values(COL_MONTH_NR).reset_index(drop=True)


def _layout(fig: go.Figure, title: str, subtitle: str | None, y_label: str,
            categories: list[str] | None = None, x_title: str = "Maand", hover: str = "x unified") -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title, x=0.0, xanchor="left"),
        hovermode=hover,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=90, b=40),
        autosize=True,
    )
    fig.update_yaxes(title=y_label, zeroline=False)
    x = dict(title=x_title, showgrid=False)
    if categories:
        x.update(categoryorder="array", categoryarray=categories)
    fig.update_xaxes(**x)


def _arr(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float)


# --------------------------------------------------------------------------
def create_monthly_box_chart(df: pd.DataFrame, title: str, unit: str) -> go.Figure:
    """Boxplot per maand: doos = P25–P75 (IQR), whiskers = P10 en P90, markers voor P50 en gemiddelde."""
    df = _validate(df)
    x = df[COL_MONTH].tolist()
    fmt = f".{number_decimals(df)}f"
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x, lowerfence=_arr(df[COL_P10]), q1=_arr(df[COL_P25]), median=_arr(df[COL_P50]),
        q3=_arr(df[COL_P75]), upperfence=_arr(df[COL_P90]),
        name="Doos P25–P75 · whiskers P10–P90",
        fillcolor=_rgba(PALETTE[0], 0.25), line=dict(color=PALETTE[0], width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_arr(df[COL_P50]), mode="markers", name=LABEL_P50,
        marker=dict(symbol="square", size=8, color=COLOR_P50),
        customdata=df[[COL_P10, COL_P25, COL_P75, COL_P90, COL_MEAN]].to_numpy(dtype=float),
        hovertemplate=(
            f"P90: %{{customdata[3]:{fmt}}} {unit}<br>P75: %{{customdata[2]:{fmt}}} {unit}<br>"
            f"<b>{LABEL_P50}: %{{y:{fmt}}} {unit}</b><br>P25: %{{customdata[1]:{fmt}}} {unit}<br>"
            f"P10: %{{customdata[0]:{fmt}}} {unit}<br>Gemiddelde: %{{customdata[4]:{fmt}}} {unit}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_arr(df[COL_MEAN]), mode="markers", name="Gemiddelde",
        marker=dict(symbol="diamond", size=10, color=COLOR_MEAN, line=dict(width=1, color="white")),
        hoverinfo="skip",
    ))
    _layout(fig, f"{title}: maandelijkse klimatologie",
            "Doos: 50% percentielbereik (P25–P75) · Whiskers: P10–P90 · Vierkant: P50 · Ruitje: gemiddelde",
            f"{title} ({unit})", x)
    return fig


def create_climate_fan_chart(df: pd.DataFrame, title: str, unit: str) -> go.Figure:
    """Bandgrafiek: lichte band P10–P90, donkere band P25–P75, lijnen voor P50 en gemiddelde."""
    df = _validate(df)
    x = df[COL_MONTH].tolist()
    fmt = f".{number_decimals(df)}f"

    def tpl(label: str) -> str:
        return f"{label}: %{{y:{fmt}}} {unit}<extra></extra>"

    fig = go.Figure()
    edge = dict(width=0)
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P10]), mode="lines", line=edge, showlegend=False, hovertemplate=tpl("P10")))
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P90]), mode="lines", line=edge, name="P10–P90",
                             fill="tonexty", fillcolor=_rgba(PALETTE[0], 0.15), hovertemplate=tpl("P90")))
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P25]), mode="lines", line=edge, showlegend=False, hovertemplate=tpl("P25")))
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P75]), mode="lines", line=edge, name="P25–P75 (IQR)",
                             fill="tonexty", fillcolor=_rgba(PALETTE[0], 0.35), hovertemplate=tpl("P75")))
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P50]), mode="lines+markers", name=LABEL_P50,
                             line=dict(width=3, color=COLOR_P50), marker=dict(size=6), hovertemplate=tpl(LABEL_P50)))
    fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_MEAN]), mode="lines", name="Gemiddelde",
                             line=dict(width=2, color=COLOR_MEAN, dash="dash"), hovertemplate=tpl("Gemiddelde")))
    _layout(fig, f"{title}: klimaatband per maand", "Banden: P10–P90 en P25–P75 · Lijn: P50 · Gestippeld: gemiddelde",
            f"{title} ({unit})", x)
    return fig


def create_multi_series_chart(series: dict[str, pd.DataFrame], title: str, unit: str,
                              subtitle: str | None = "Lijn: P50 · Band: P25–P75") -> go.Figure:
    """Meerdere reeksen (bv. Tx/Tn, bodemlagen) met P50-lijn en P25–P75-band."""
    fig = go.Figure()
    fmt = f".{max(number_decimals(_validate(d)) for d in series.values())}f"
    for i, (label, raw) in enumerate(series.items()):
        df, col = _validate(raw), PALETTE[i % len(PALETTE)]
        x = df[COL_MONTH].tolist()
        fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P25]), mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", legendgroup=label))
        fig.add_trace(go.Scatter(x=x, y=_arr(df[COL_P75]), mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(col, 0.18), showlegend=False, hoverinfo="skip", legendgroup=label))
        fig.add_trace(go.Scatter(
            x=x, y=_arr(df[COL_P50]), mode="lines+markers", name=label, legendgroup=label,
            line=dict(color=col, width=3), marker=dict(size=6),
            customdata=df[[COL_P25, COL_P75]].to_numpy(dtype=float),
            hovertemplate=f"{label}: %{{y:{fmt}}} {unit} (P25–P75: %{{customdata[0]:{fmt}}} – %{{customdata[1]:{fmt}}})<extra></extra>",
        ))
    _layout(fig, title, subtitle, f"{unit}", MONTH_NAMES)
    return fig


def create_yearly_bar_chart(series: dict[str, pd.Series], title: str, unit: str) -> go.Figure:
    """Gegroepeerde staafgrafiek per jaar (bv. warmte- en vorstdagen)."""
    fig = go.Figure()
    for i, (label, s) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=[str(int(y)) for y in s.index], y=_arr(s), name=label,
                             marker_color=PALETTE[i % len(PALETTE)],
                             hovertemplate=f"{label}: %{{y:.0f}} {unit}<extra></extra>"))
    fig.update_layout(barmode="group")
    _layout(fig, title, None, unit, None, x_title="Jaar")
    fig.update_xaxes(type="category")
    return fig


def create_grouped_month_bars(dfs: dict[str, pd.DataFrame], title: str, unit: str,
                              stacked: bool = False) -> go.Figure:
    """Staven per maand (gemiddelde). Gegroepeerd met P10–P90-whiskers, of gestapeld zonder whiskers."""
    fig = go.Figure()
    fmt = f".{max(number_decimals(_validate(d)) for d in dfs.values())}f"
    for i, (label, raw) in enumerate(dfs.items()):
        df = _validate(raw)
        mean = _arr(df[COL_MEAN])
        err = None
        if not stacked:
            err = dict(type="data", symmetric=False,
                       array=np.clip(_arr(df[COL_P90]) - mean, 0, None),
                       arrayminus=np.clip(mean - _arr(df[COL_P10]), 0, None), thickness=1.5)
        fig.add_trace(go.Bar(x=df[COL_MONTH].tolist(), y=mean, name=label, error_y=err,
                             marker_color=PALETTE[i % len(PALETTE)],
                             hovertemplate=f"{label} (gemiddelde): %{{y:{fmt}}} {unit}<extra></extra>"))
    fig.update_layout(barmode="stack" if stacked else "group")
    _layout(fig, title, "Staven: gemiddelde" + ("" if stacked else " · Whiskers: P10–P90"), unit, MONTH_NAMES)
    return fig


def create_wind_rose(rose: pd.DataFrame, class_labels: list[str], title: str) -> go.Figure:
    """Interactieve windroos (Barpolar): frequentie (% van de uren) per richting en snelheidsklasse."""
    theta = [i * 22.5 for i in range(16)]
    colors = ["#9ecae1", "#4292c6", "#08519c", "#e4572e"]
    fig = go.Figure()
    for c in range(4):
        sub = rose[rose["class"] == c].sort_values("sector")
        fig.add_trace(go.Barpolar(
            r=_arr(sub["pct"]), theta=theta, name=class_labels[c], marker_color=colors[c],
            hovertemplate="%{theta}°: %{r:.2f}% van de uren<extra>" + class_labels[c] + "</extra>",
        ))
    fig.update_layout(
        template="plotly_white", title=dict(text=title, x=0.0, xanchor="left"),
        polar=dict(
            barmode="stack",
            angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                             tickvals=theta, ticktext=SECTOR_NAMES, tickfont=dict(size=10)),
            radialaxis=dict(ticksuffix="%", angle=45),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=70, b=60), autosize=True,
    )
    return fig


def create_threshold_calendar(rows: list[dict], title: str) -> go.Figure:
    """Verkeerslichtkalender. rows: ``[{'label', 'levels': [12 ints -1..2], 'text': [12 str]}]``."""
    colorscale = [[0.0, "#cfcfcf"], [0.25, "#cfcfcf"], [0.25, "#4caf50"], [0.5, "#4caf50"],
                  [0.5, "#ffb300"], [0.75, "#ffb300"], [0.75, "#e53935"], [1.0, "#e53935"]]
    fig = go.Figure(go.Heatmap(
        z=[r["levels"] for r in rows], x=MONTH_NAMES, y=[r["label"] for r in rows],
        text=[r["text"] for r in rows], texttemplate="%{text}", colorscale=colorscale,
        zmin=-1.5, zmax=2.5, showscale=False, xgap=3, ygap=3, hoverinfo="skip",
    ))
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.0, xanchor="left"),
                      margin=dict(l=40, r=20, t=70, b=40), autosize=True,
                      height=max(250, 70 * len(rows) + 120))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return fig


SKEW_FACTOR = 25.0  # °C schuinstand per ln(1000/p)


def create_skewt(profile: pd.DataFrame, month_label: str) -> go.Figure:
    """Skew-T-achtig diagram (log-p, geschuinde isothermen) van het gemiddelde maandprofiel.

    De x-as toont de temperatuur op de 1000 hPa-lijn; de tooltips tonen de echte waarden.
    """
    p = _arr(profile["p"])
    skew = lambda temp, press: temp + SKEW_FACTOR * np.log(1000.0 / press)  # noqa: E731
    fig = go.Figure()
    top_shift = SKEW_FACTOR * np.log(1000.0 / 200.0)
    for iso in range(-80, 50, 10):
        fig.add_trace(go.Scatter(x=[iso, iso + top_shift], y=[1000, 200], mode="lines",
                                 line=dict(color="rgba(150,150,150,0.4)", width=1),
                                 hoverinfo="skip", showlegend=False))
    for col, name, color in (("t", "Temperatuur", "#e4572e"), ("td", "Dauwpunt", "#1f77b4")):
        if col in profile:
            v = _arr(profile[col])
            fig.add_trace(go.Scatter(
                x=skew(v, p), y=p, mode="lines+markers", name=name, line=dict(color=color, width=3),
                customdata=v, hovertemplate="%{y:.0f} hPa: %{customdata:.1f} °C<extra>" + name + "</extra>"))
    levels = [int(v) for v in p]
    fig.update_layout(
        template="plotly_white", hovermode="closest",
        title=dict(text=f"Skew-T profiel (gemiddelde {month_label})<br><sup>Gemiddeld maandprofiel uit ERA5-drukvlakken, 1000–200 hPa</sup>",
                   x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=50, r=20, t=90, b=50), autosize=True,
    )
    fig.update_yaxes(type="log", range=[np.log10(1020), np.log10(190)], tickvals=levels,
                     ticktext=[str(v) for v in levels], title="Druk (hPa)")
    fig.update_xaxes(range=[-30, 45], title="Temperatuur bij 1000 hPa-lijn (°C, geschuind)", showgrid=False)
    return fig


def create_idf_chart(idf: pd.DataFrame) -> go.Figure:
    """IDF-curven: neerslagintensiteit (mm/uur) tegen duur (uur) per herhalingstijd."""
    fig = go.Figure()
    for i, t in enumerate(sorted(idf["T"].unique())):
        sub = idf[idf["T"] == t].sort_values("duur_h")
        fig.add_trace(go.Scatter(
            x=_arr(sub["duur_h"]), y=_arr(sub["intensiteit_mmh"]), mode="lines+markers",
            name=f"T = {int(t)} jaar", line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
            customdata=_arr(sub["hoogte_mm"]),
            hovertemplate="Duur %{x} u: %{y:.2f} mm/u (%{customdata:.1f} mm totaal)<extra>T = " + str(int(t)) + " jaar</extra>"))
    fig.update_layout(
        template="plotly_white", hovermode="closest",
        title=dict(text="IDF-curven (Gumbel-fit op jaarmaxima)<br><sup>Op basis van uurlijkse ERA5-neerslag; extreme buien worden onderschat</sup>",
                   x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=90, b=40), autosize=True,
    )
    fig.update_xaxes(type="log", title="Duur (uur)", tickvals=[1, 3, 6, 12, 24], ticktext=["1", "3", "6", "12", "24"])
    fig.update_yaxes(type="log", title="Intensiteit (mm/uur)")
    return fig
