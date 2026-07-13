"""
Bring-Your-Own-Observations (BYOO): user-supplied recent weather -> forecast.

The judges' question: "what if a user attaches the previous days' weather —
can the twin forecast from THAT?" The scientifically correct entry point for
external data is ASSIMILATION, not replacing model inputs: each user value is
treated as an observation y with error sigma_o, fused into the gridded history
at the cells it covers using the same Optimal-Interpolation mathematics as the
twin loop (Gaussian background-error covariance around the station). The
corrected history then initialises ClimateUNet.

This also unlocks "forecast from today": for days beyond the IMD archive the
background falls back to climatology (zero anomaly), and the user's fresh
observations are the only information that pulls the state away from normal —
exactly how an operational analysis treats a data-sparse day.

No user value is ever used raw: everything is anomalised against the national
climatology and weighted by observation error, so a typo cannot silently
poison the state (it shows up as a large, localised innovation the UI reports).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

# Valid physical ranges (same guards as the IMD ingest).
_RANGES = {"rain": (0.0, 2000.0), "tmax": (-40.0, 60.0), "tmin": (-40.0, 60.0)}

TEMPLATE_CSV = (
    "date,rain,tmax,tmin\n"
    "2024-12-15,0.0,28.5,18.2\n"
    "2024-12-16,2.4,27.9,17.8\n"
    "2024-12-17,0.0,28.8,18.0\n"
)


def parse_csv(text_or_buffer) -> pd.DataFrame:
    """Read a user CSV (date + any of rain/tmax/tmin) and validate it.

    Returns a DataFrame with a parsed `date` column and numeric variable
    columns; out-of-range values are set to NaN (reported, never used).
    """
    if isinstance(text_or_buffer, (str, bytes)):
        text_or_buffer = io.StringIO(
            text_or_buffer.decode() if isinstance(text_or_buffer, bytes) else text_or_buffer)
    df = pd.read_csv(text_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column (YYYY-MM-DD).")
    keep = ["date"] + [v for v in C.VARIABLES if v in df.columns]
    if len(keep) == 1:
        raise ValueError("CSV needs at least one of: rain, tmax, tmin.")
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for v in C.VARIABLES:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors="coerce")
            lo, hi = _RANGES[v]
            df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
    if "tmax" in df.columns and "tmin" in df.columns:
        bad = df["tmin"] > df["tmax"]        # physically impossible ordering
        df.loc[bad, ["tmax", "tmin"]] = np.nan
    return df


def nearest_cell(lat, lon, grid):
    iy = int(np.argmin(np.abs(grid["lat"] - lat)))
    ix = int(np.argmin(np.abs(grid["lon"] - lon)))
    return iy, ix


def build_history(cube, dates, start_ts):
    """(INPUT_DAYS,H,W,3) scaled-anomaly history ending the day before start_ts.

    Days inside the IMD record come from the observed cube; days beyond it
    fall back to climatology (zero anomaly) — the honest background for a
    day with no national analysis yet.
    """
    H, W = cube.shape[1], cube.shape[2]
    hist = np.zeros((C.INPUT_DAYS, H, W, 3), dtype="float32")
    n_real = 0
    for k in range(C.INPUT_DAYS):
        d = pd.Timestamp(start_ts) - pd.Timedelta(days=C.INPUT_DAYS - k)
        if d <= dates[-1] and d >= dates[0]:
            ti = int(np.argmin(np.abs(dates.values - np.datetime64(d))))
            hist[k] = cube[ti]
            n_real += 1
    return hist, n_real


def inject_observations(hist, start_ts, user_df, carr, std, iy, ix,
                        length_scale=2.0, sigma_o=0.5):
    """OI-assimilate user observations into the history window (in place).

    Each user value is anomalised against the climatology at its own date and
    cell, the innovation (user - background) is scaled by the OI gain
    sigma_b^2/(sigma_b^2+sigma_o^2) (sigma_b = 1 in scaled-anomaly space) and
    spread to neighbouring cells with a Gaussian kernel of `length_scale`
    grid cells — one station informs its surroundings, exactly as in the
    twin's assimilation panel.

    Returns (hist, report) where report lists per-day innovations for the UI.
    """
    H, W = hist.shape[1], hist.shape[2]
    yy, xx = np.mgrid[0:H, 0:W]
    kern = np.exp(-((yy - iy) ** 2 + (xx - ix) ** 2)
                  / (2.0 * max(length_scale, 1e-3) ** 2)).astype("float32")
    gain = 1.0 / (1.0 + sigma_o ** 2)
    start_ts = pd.Timestamp(start_ts)
    report = []
    for _, row in user_df.iterrows():
        d = pd.Timestamp(row["date"])
        k = C.INPUT_DAYS - int((start_ts - d).days)
        if not (0 <= k < C.INPUT_DAYS):
            continue                          # outside the 7-day input window
        doy = min(int(d.dayofyear), 365)
        used = {}
        for vi, v in enumerate(C.VARIABLES):
            val = row.get(v, np.nan)
            if val is None or not np.isfinite(val):
                continue
            a_user = (float(val) - carr[v][doy - 1, iy, ix]) / (float(std[vi]) + 1e-6)
            innov = a_user - hist[k, iy, ix, vi]
            hist[k, :, :, vi] += gain * innov * kern
            used[v] = float(innov * std[vi])  # innovation in real units
        if used:
            report.append({"date": str(d.date()), **used})
    return hist, report


def influence_area_pct(delta_field, landmask, thresh):
    """% of land cells where the user's data changed the forecast > thresh."""
    d = np.abs(delta_field[landmask])
    return float(np.mean(d > thresh) * 100.0)
