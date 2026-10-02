"""Module 4: interactieve Plotly-visualisaties van de maandelijkse klimatologie."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils.stats import (
    COL_MEAN, COL_MONTH, COL_MONTH_NR, COL_P10, COL_P50, COL_P90, number_decimals,
)

COLOR_BAND = "rgba(31, 119, 180, 0.20)"
COLOR_RANGE = "#1f77b4"
COLOR_P50 = "#0b3d91"
COLOR_MEAN = "#e4572e"

LABEL_P50 = "50% Percentiel (P50)"
LABEL_RANGE = "Percentielbereik P10–P90"

_REQUIRED = [COL_MONTH_NR, COL_MONTH, COL_P10, COL_P50, COL_P90, COL_MEAN]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame mist kolommen: {missing}")
    return df.sort_values(COL_MONTH_NR).reset_index(drop=True)


def _base_layout(fig: go.Figure, title: str, subtitle: str, y_label: str, cats: list[str]) -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.0, xanchor="left"),
        xaxis=dict(title="Maand", categoryorder="array", categoryarray=cats, showgrid=False),
        yaxis=dict(title=y_label, zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=100, b=40),
        autosize=True,
    )


def create_monthly_quantile_chart(df: pd.DataFrame, variable_title: str, unit: str) -> go.Figure:
    """Whiskerplot: 50% Percentiel (P50) als centrale waarde, whiskers P10–P90.

    Args:
        df: Output van ``calculate_monthly_climatology``.
        variable_title: Naam van de variabele, bv. ``"Temperatuur"``.
        unit: Eenheid, bv. ``"°C"``.
    """
    df = _validate(df)
    x = df[COL_MONTH]
    f = f".{number_decimals(df)}f"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_P50], mode="markers", name=LABEL_P50,
            marker=dict(symbol="square", size=10, color=COLOR_P50),
            error_y=dict(
                type="data", symmetric=False,
                array=(df[COL_P90] - df[COL_P50]).to_numpy(),
                arrayminus=(df[COL_P50] - df[COL_P10]).to_numpy(),
                color=COLOR_RANGE, thickness=3, width=8,
            ),
            customdata=df[[COL_P10, COL_P50, COL_P90, COL_MEAN]].to_numpy(),
            hovertemplate=(
                f"P90: %{{customdata[2]:{f}}} {unit}<br>"
                f"{LABEL_P50}: %{{customdata[1]:{f}}} {unit}<br>"
                f"Gemiddelde: %{{customdata[3]:{f}}} {unit}<br>"
                f"P10: %{{customdata[0]:{f}}} {unit}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_MEAN], mode="markers", name="Gemiddelde",
            marker=dict(symbol="diamond", size=11, color=COLOR_MEAN, line=dict(width=1, color="white")),
            hoverinfo="skip",
        )
    )
    fig.add_trace(  # legenda-item voor de whisker
        go.Scatter(
            x=[None], y=[None], mode="lines", name=LABEL_RANGE,
            line=dict(color=COLOR_RANGE, width=3), hoverinfo="skip",
        )
    )
    _base_layout(
        fig, f"{variable_title}: maandelijkse klimatologie",
        "Whiskers: percentielbereik P10–P90 · Vierkant: 50% Percentiel (P50) · Ruitje: gemiddelde",
        f"{variable_title} ({unit})", x.tolist(),
    )
    return fig


def create_climate_fan_chart(df: pd.DataFrame, variable_title: str, unit: str) -> go.Figure:
    """Bandgrafiek: vlak P10–P90, lijn voor 50% Percentiel (P50) en gemiddelde."""
    df = _validate(df)
    x = df[COL_MONTH]
    f = f".{number_decimals(df)}f"

    def tpl(label: str) -> str:
        return f"{label}: %{{y:{f}}} {unit}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_P10], mode="lines", name="P10",
            line=dict(width=1, color=COLOR_RANGE, dash="dot"),
            showlegend=False, hovertemplate=tpl("P10"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_P90], mode="lines", name=LABEL_RANGE,
            line=dict(width=1, color=COLOR_RANGE, dash="dot"),
            fill="tonexty", fillcolor=COLOR_BAND, hovertemplate=tpl("P90"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_P50], mode="lines+markers", name=LABEL_P50,
            line=dict(width=3, color=COLOR_P50), marker=dict(size=6),
            hovertemplate=tpl(LABEL_P50),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df[COL_MEAN], mode="lines", name="Gemiddelde",
            line=dict(width=2, color=COLOR_MEAN, dash="dash"),
            hovertemplate=tpl("Gemiddelde"),
        )
    )
    _base_layout(
        fig, f"{variable_title}: normaal klimaatbereik per maand",
        "Vlak: percentielbereik P10–P90 · Lijn: 50% Percentiel (P50) · Gestippeld: gemiddelde",
        f"{variable_title} ({unit})", x.tolist(),
    )
    return fig
