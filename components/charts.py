"""Module 4: interactieve Plotly-visualisaties van de maandelijkse klimatologie."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Kleurenpalet (consistent tussen beide grafieken)
COLOR_BAND = "rgba(31, 119, 180, 0.20)"
COLOR_RANGE = "#1f77b4"   # P10-P90 whiskers / bandlijnen
COLOR_MEDIAN = "#0b3d91"  # mediaan
COLOR_MEAN = "#e4572e"    # gemiddelde

_REQUIRED_COLUMNS = ["month", "month_name", "mean", "p50", "p10", "p90"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Controleer kolommen en sorteer op maand."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame mist kolommen: {missing}")
    return df.sort_values("month").reset_index(drop=True)


def _base_layout(fig: go.Figure, title: str, y_label: str, categories: list[str]) -> None:
    """Gedeelde layout-instellingen voor beide grafieken."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left"),
        xaxis=dict(
            title="Maand",
            categoryorder="array",
            categoryarray=categories,
            showgrid=False,
        ),
        yaxis=dict(title=y_label, zeroline=False),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0
        ),
        margin=dict(l=40, r=20, t=80, b=40),
        autosize=True,  # responsive; gebruik use_container_width in Streamlit
    )


def create_monthly_quantile_chart(
    df: pd.DataFrame, variable_title: str, unit: str
) -> go.Figure:
    """Maandoverzicht met whiskers (P10–P90), mediaan en gemiddelde.

    Args:
        df: Output van ``calculate_monthly_climatology``.
        variable_title: Titel/naam van de variabele, bv. ``"Temperatuur"``.
        unit: Eenheid, bv. ``"°C"``.
    """
    df = _validate(df)
    x = df["month_name"]
    y_label = f"{variable_title} ({unit})"

    # Mediaan-marker met asymmetrische errorbars = whisker van P10 tot P90.
    customdata = df[["p10", "p50", "p90", "mean"]].to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["p50"],
            mode="markers",
            name="Mediaan (P50)",
            marker=dict(symbol="square", size=9, color=COLOR_MEDIAN),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(df["p90"] - df["p50"]).to_numpy(),
                arrayminus=(df["p50"] - df["p10"]).to_numpy(),
                color=COLOR_RANGE,
                thickness=3,
                width=8,
            ),
            customdata=customdata,
            hovertemplate=(
                "P90: %{customdata[2]:.2f} " + unit + "<br>"
                "Gemiddelde: %{customdata[3]:.2f} " + unit + "<br>"
                "Mediaan (P50): %{customdata[1]:.2f} " + unit + "<br>"
                "P10: %{customdata[0]:.2f} " + unit + "<extra></extra>"
            ),
        )
    )
    # Gemiddelde als ruitje (tooltip zit al in de mediaan-trace).
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["mean"],
            mode="markers",
            name="Gemiddelde",
            marker=dict(
                symbol="diamond", size=11, color=COLOR_MEAN,
                line=dict(width=1, color="white"),
            ),
            hoverinfo="skip",
        )
    )
    # Legenda-item voor de whisker.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name="P10 – P90 bereik",
            line=dict(color=COLOR_RANGE, width=3),
            hoverinfo="skip",
        )
    )

    _base_layout(
        fig, f"{variable_title}: maandelijkse klimatologie", y_label, x.tolist()
    )
    return fig


def create_climate_fan_chart(
    df: pd.DataFrame, variable_title: str, unit: str
) -> go.Figure:
    """Percentielband (fan chart): P10–P90 vlak, mediaanlijn en gemiddelde.

    Args:
        df: Output van ``calculate_monthly_climatology``.
        variable_title: Titel/naam van de variabele, bv. ``"Temperatuur"``.
        unit: Eenheid, bv. ``"°C"``.
    """
    df = _validate(df)
    x = df["month_name"]
    y_label = f"{variable_title} ({unit})"

    def tpl(label: str) -> str:
        return f"{label}: %{{y:.2f}} {unit}<extra></extra>"

    fig = go.Figure()
    # Ondergrens (onzichtbare lijn, dient als basis voor de vulling).
    fig.add_trace(
        go.Scatter(
            x=x, y=df["p10"], mode="lines", name="P10",
            line=dict(width=1, color=COLOR_RANGE, dash="dot"),
            showlegend=False, hovertemplate=tpl("P10"),
        )
    )
    # Bovengrens, gevuld tot de vorige trace (P10).
    fig.add_trace(
        go.Scatter(
            x=x, y=df["p90"], mode="lines", name="Normaal bereik (P10–P90)",
            line=dict(width=1, color=COLOR_RANGE, dash="dot"),
            fill="tonexty", fillcolor=COLOR_BAND,
            hovertemplate=tpl("P90"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["p50"], mode="lines+markers", name="Mediaan (P50)",
            line=dict(width=3, color=COLOR_MEDIAN), marker=dict(size=6),
            hovertemplate=tpl("Mediaan (P50)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["mean"], mode="lines", name="Gemiddelde",
            line=dict(width=2, color=COLOR_MEAN, dash="dash"),
            hovertemplate=tpl("Gemiddelde"),
        )
    )

    _base_layout(
        fig, f"{variable_title}: normaal klimaatbereik per maand", y_label, x.tolist()
    )
    return fig
