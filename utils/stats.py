"""Module 3: statistiek (P10/P25/P50/P75/P90/gemiddelde), afgeleide indicatoren en eenheden."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]
SECTOR_NAMES = ["N", "NNO", "NO", "ONO", "O", "OZO", "ZO", "ZZO",
                "Z", "ZZW", "ZW", "WZW", "W", "WNW", "NW", "NNW"]

COL_MONTH_NR, COL_MONTH = "Maandnr", "Maand"
COL_P10, COL_P25, COL_P50 = "P10", "P25", "P50 (Percentiel)"
COL_P75, COL_P90, COL_MEAN = "P75", "P90", "Gemiddelde"
VALUE_COLUMNS = [COL_P10, COL_P25, COL_P50, COL_P75, COL_P90, COL_MEAN]

# --------------------------------------------------------------------------
# Eenheden
# --------------------------------------------------------------------------
BFT_LOWER_MS = [0.0, 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
KT_PER_MS = 1.943844
FT_PER_M = 3.280840
UNIT_CHOICES = {"wind": ["m/s", "kt", "Bft"], "solar": ["J/m²/dag", "W/m²"], "cloud": ["ft", "FL"]}


def convert_values(x, kind: str, unit: str) -> np.ndarray:
    """Reken basiswaarden om. Basis: wind m/s, zonnestraling J/m²/dag, wolkenhoogte m."""
    a = np.asarray(x, dtype=float)
    if kind == "wind":
        if unit == "kt":
            return a * KT_PER_MS
        if unit == "Bft":  # continue Beaufort-schaal: v = 0,836 · B^1,5
            return (np.clip(a, 0, None) / 0.836) ** (2.0 / 3.0)
        return a
    if kind == "solar":
        return a / 86400.0 if unit == "W/m²" else a
    if kind == "cloud":
        ft = a * FT_PER_M
        return ft / 100.0 if unit == "FL" else ft
    raise ValueError(f"Onbekend type: {kind}")


def convert_stats_df(df: pd.DataFrame, kind: str, unit: str) -> pd.DataFrame:
    """Converteer alle statistiekkolommen (monotone omrekening: percentielen blijven geldig)."""
    out = df.copy()
    for c in VALUE_COLUMNS:
        out[c] = convert_values(out[c].to_numpy(float), kind, unit)
    return out


def rose_class_labels(unit: str) -> list[str]:
    """Legenda-labels voor de windroos in de gekozen eenheid."""
    if unit == "Bft":
        return ["≤ 3 Bft", "4–5 Bft", "6–7 Bft", "≥ 8 Bft"]
    e = [f"{v:.0f}" for v in convert_values([5.5, 10.8, 17.2], "wind", unit)]
    return [f"< {e[0]} {unit}", f"{e[0]}–{e[1]} {unit}", f"{e[1]}–{e[2]} {unit}", f"≥ {e[2]} {unit}"]


# --------------------------------------------------------------------------
# Maandstatistiek
# --------------------------------------------------------------------------
def percentiles_by_month(long: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Statistiek per kalendermaand over alle jaren.

    Args:
        long: DataFrame met kolommen ``month`` (1-12) en ``value_col``
            (één waarde per jaar-maand; NaN's worden genegeerd).

    Returns:
        12 rijen: ``[Maandnr, Maand, P10, P25, P50 (Percentiel), P75, P90, Gemiddelde]``.
    """
    rows = []
    for m in range(1, 13):
        v = long.loc[long["month"] == m, value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        stats = [np.nan] * 6 if v.size == 0 else [*np.percentile(v, [10, 25, 50, 75, 90]), v.mean()]
        rows.append([m, MONTH_NAMES[m - 1], *stats])
    return pd.DataFrame(rows, columns=[COL_MONTH_NR, COL_MONTH, *VALUE_COLUMNS])


def long_from_dataarray(da) -> pd.DataFrame:
    """DataArray met alleen een ``time``-dimensie -> DataFrame(year, month, value)."""
    s = da.squeeze(drop=True).to_series()
    idx = pd.to_datetime(s.index)
    return pd.DataFrame({"year": idx.year, "month": idx.month, "value": s.to_numpy(dtype=float)})


def _cumulative_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per jaar cumuleren over de maanden (jan->dec) en daarna percentielen per maand."""
    piv = long.pivot_table(index="year", columns="month", values="value").reindex(columns=range(1, 13))
    cum = piv.cumsum(axis=1).reset_index().melt(id_vars="year", var_name="month", value_name="value")
    return percentiles_by_month(cum)


def build_monthly_stats(ds) -> dict[str, pd.DataFrame]:
    """Statistiek voor elke variabele in de maand-dataset, plus ``pe`` en ``cum_pe`` (P − E)."""
    stats: dict[str, pd.DataFrame] = {}
    longs: dict[str, pd.DataFrame] = {}
    for name in ds.data_vars:
        if "time" not in ds[name].dims:
            continue
        longs[name] = long_from_dataarray(ds[name])
        stats[name] = percentiles_by_month(longs[name])
    if "tp" in longs and "e" in longs:
        m = longs["tp"].merge(longs["e"], on=["year", "month"], suffixes=("_p", "_e"))
        m["value"] = m["value_p"] - m["value_e"]
        stats["pe"] = percentiles_by_month(m)
        stats["cum_pe"] = _cumulative_stats(m)
    return stats


def number_decimals(df: pd.DataFrame, cols: list[str] | None = None) -> int:
    """Aantal decimalen op basis van de orde van grootte."""
    cols = cols or [c for c in VALUE_COLUMNS if c in df.columns]
    vals = df[cols].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return 2
    peak = float(np.nanmax(np.abs(vals)))
    return 0 if peak >= 1e5 else 1 if peak >= 100 else 2


# --------------------------------------------------------------------------
# Meteorologische hulpfuncties
# --------------------------------------------------------------------------
def wind_dir(u, v):
    """Meteorologische windrichting (graden, waar de wind VANDAAN komt)."""
    return (180.0 + np.degrees(np.arctan2(u, v))) % 360.0


def wbgt_approx(t, td):
    """Indicatieve WBGT in de schaduw (BoM-benadering): 0,567·T + 0,393·e + 3,94 (e in hPa)."""
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    return 0.567 * t + 0.393 * e + 3.94


ROSE_EDGES_MS = [5.5, 10.8, 17.2]  # klassegrenzen = Bft 4, 6 en 8


def wind_rose_table(speed: pd.Series, direction: pd.Series) -> pd.DataFrame:
    """Frequentie (% van alle uren) per windsector (16) en snelheidsklasse (4)."""
    ok = speed.notna() & direction.notna()
    sp, dr = speed[ok].to_numpy(float), direction[ok].to_numpy(float)
    sector = np.floor(((dr + 11.25) % 360.0) / 22.5).astype(int) % 16
    cls = np.digitize(sp, ROSE_EDGES_MS)
    counts = np.zeros((16, 4))
    np.add.at(counts, (sector, cls), 1)
    pct = counts / max(len(sp), 1) * 100.0
    return pd.DataFrame([(s, c, pct[s, c]) for s in range(16) for c in range(4)],
                        columns=["sector", "class", "pct"])


def idf_table(tp: pd.Series, durations=(1, 3, 6, 12, 24), periods=(2, 5, 10, 20, 50)) -> pd.DataFrame:
    """IDF-gegevens: jaarmaxima per duur + Gumbel-fit (momentenmethode).

    Returns lege DataFrame bij < 5 jaren data.
    """
    tp = tp.asfreq("h")
    rows = []
    for d in durations:
        roll = tp.rolling(d, min_periods=d).sum()
        ann = roll.groupby(roll.index.year).max().dropna()
        if len(ann) < 5:
            continue
        beta = ann.std(ddof=1) * np.sqrt(6.0) / np.pi
        mu = ann.mean() - 0.5772156649 * beta
        for T in periods:
            depth = mu - beta * np.log(-np.log(1.0 - 1.0 / T))
            rows.append((d, T, float(depth), float(depth / d)))
    return pd.DataFrame(rows, columns=["duur_h", "T", "hoogte_mm", "intensiteit_mmh"])


def traffic_levels(values, orange: float, red: float, higher_is_worse: bool = True) -> np.ndarray:
    """Verkeerslichtniveau: -1 = geen data, 0 = groen, 1 = oranje, 2 = rood."""
    v = np.asarray(values, dtype=float)
    if higher_is_worse:
        lvl = np.where(v >= red, 2, np.where(v >= orange, 1, 0))
    else:
        lvl = np.where(v <= red, 2, np.where(v <= orange, 1, 0))
    return np.where(np.isfinite(v), lvl, -1)


def profile_tables(ds) -> dict[int, pd.DataFrame]:
    """Gemiddeld verticaal profiel per kalendermaand (over alle jaren)."""
    clim = ds.groupby("time.month").mean("time")
    out: dict[int, pd.DataFrame] = {}
    for m in clim["month"].values:
        sub = clim.sel(month=m).squeeze(drop=True)
        df = pd.DataFrame({"p": np.asarray(sub["pressure_level"].values, dtype=float)})
        for col in ("t", "td", "q", "u", "v", "z"):
            if col in sub:
                df[col] = np.asarray(sub[col].values, dtype=float)
        if "u" in df and "v" in df:
            df["wind_speed"] = np.hypot(df["u"], df["v"])
            df["wind_dir"] = wind_dir(df["u"], df["v"])
        out[int(m)] = df.sort_values("p", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Afgeleide indicatoren uit uurdata
# --------------------------------------------------------------------------
def _ym(series: pd.Series, how: str) -> pd.DataFrame:
    """Aggregeer een tijdreeks per (jaar, maand) -> DataFrame(year, month, value)."""
    s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["year", "month", "value"])
    res = getattr(s.groupby([s.index.year, s.index.month]), how)()
    res.index = res.index.set_names(["year", "month"])
    return res.rename("value").reset_index()


def _daily(series: pd.Series, how: str, min_count: int = 20) -> pd.Series:
    """Dagwaarde (max/min/mean/…) uit uurdata; dagen met < min_count uren worden NaN."""
    r = series.resample("D").agg([how, "count"])
    return r[how].where(r["count"] >= min_count)


def compute_hourly_products(hourly: pd.DataFrame | None, hot_thr: float = 30.0,
                            bft_thr: int = 6, wbgt_thr: float = 28.0) -> dict:
    """Bereken alle afgeleide indicatoren uit de uurdata.

    Alle ``*_stats`` zijn maandstatistieken over jaren (zie ``percentiles_by_month``);
    dagen zijn UTC-dagen.
    """
    out: dict = {}
    if hourly is None or len(hourly) == 0:
        return out
    h = hourly.sort_index()

    if "t2m" in h:
        tx, tn = _daily(h["t2m"], "max"), _daily(h["t2m"], "min")
        out["tx"] = percentiles_by_month(_ym(tx, "mean"))
        out["tn"] = percentiles_by_month(_ym(tn, "mean"))
        hot = (tx > hot_thr).astype(float).where(tx.notna())
        frost = (tn < 0.0).astype(float).where(tn.notna())
        out["hot_month"] = percentiles_by_month(_ym(hot, "sum"))
        out["frost_month"] = percentiles_by_month(_ym(frost, "sum"))
        out["hot_year"] = hot.dropna().groupby(hot.dropna().index.year).sum()
        out["frost_year"] = frost.dropna().groupby(frost.dropna().index.year).sum()
        if "d2m" in h:
            wd = _daily(wbgt_approx(h["t2m"], h["d2m"]), "max")
            out["wbgt"] = percentiles_by_month(_ym(wd, "mean"))
            days = (wd >= wbgt_thr).astype(float).where(wd.notna())
            out["wbgt_days"] = percentiles_by_month(_ym(days, "sum"))

    if "u10" in h and "v10" in h:
        spd = np.hypot(h["u10"], h["v10"])
        out["wind_speed_h"] = percentiles_by_month(_ym(spd, "mean"))
        thr_ms = BFT_LOWER_MS[int(bft_thr)]
        out["wind_exceed"] = percentiles_by_month(_ym((spd >= thr_ms).astype(float).where(spd.notna()) * 100.0, "mean"))
        out["rose"] = wind_rose_table(spd, wind_dir(h["u10"], h["v10"]))
    if "fg10" in h:
        out["gust_max"] = percentiles_by_month(_ym(h["fg10"], "max"))
    if "tp" in h:
        out["idf"] = idf_table(h["tp"])
    return out
