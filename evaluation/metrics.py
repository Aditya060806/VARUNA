"""
Weather-standard evaluation in real units.

Metrics (per variable, per lead day):
  * RMSE / MAE        in mm/day and deg C
  * ACC               anomaly correlation coefficient (vs climatology)
  * Skill vs persistence and vs persistence-of-anomaly (1 - RMSE/RMSE_ref)

All metrics are computed over land cells only with latitude-area weighting,
exactly as in operational NWP verification.
"""
from __future__ import annotations

import numpy as np


def _wstats(landmask, lat):
    latw = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, landmask.shape[1]))
    w = np.where(landmask, latw, 0.0).astype("float64")
    return w


def wrmse(pred, true, w):
    m = np.isfinite(pred) & np.isfinite(true)
    ww = w * m
    return float(np.sqrt(np.sum(ww * (pred - true) ** 2) / np.sum(ww)))


def wmae(pred, true, w):
    m = np.isfinite(pred) & np.isfinite(true)
    ww = w * m
    return float(np.sum(ww * np.abs(pred - true)) / np.sum(ww))


def wacc(pred, true, clim, w):
    """Anomaly correlation coefficient (area-weighted)."""
    pa, ta = pred - clim, true - clim
    m = np.isfinite(pa) & np.isfinite(ta)
    ww = w * m
    pm = np.sum(ww * pa) / np.sum(ww)
    tm = np.sum(ww * ta) / np.sum(ww)
    cov = np.sum(ww * (pa - pm) * (ta - tm))
    vp = np.sum(ww * (pa - pm) ** 2)
    vt = np.sum(ww * (ta - tm) ** 2)
    return float(cov / np.sqrt(vp * vt + 1e-12))


def skill(rmse, rmse_ref):
    return float(1.0 - rmse / rmse_ref) if rmse_ref > 0 else float("nan")


def event_counts(pred_event, obs_event, w):
    """Area-weighted contingency counts (hits, misses, false alarms, correct
    negatives) for a boolean event field pair — accumulate these across
    windows, then convert with `event_scores` (never average per-window
    ratios)."""
    p = np.asarray(pred_event, bool)
    o = np.asarray(obs_event, bool)
    return np.array([
        float(np.sum(w * (p & o))),          # hits
        float(np.sum(w * (~p & o))),         # misses
        float(np.sum(w * (p & ~o))),         # false alarms
        float(np.sum(w * (~p & ~o))),        # correct negatives
    ])


def event_scores(counts):
    """Contingency counts -> operational categorical verification scores.

    POD (probability of detection), FAR (false-alarm ratio), CSI (critical
    success index) and ETS (equitable threat score — skill relative to random
    hits, the standard headline score for rare events).
    """
    hits, miss, fa, cn = [float(x) for x in counts]
    total = hits + miss + fa + cn
    pod = hits / (hits + miss) if hits + miss > 0 else float("nan")
    far = fa / (hits + fa) if hits + fa > 0 else float("nan")
    csi = hits / (hits + miss + fa) if hits + miss + fa > 0 else float("nan")
    h_rand = (hits + miss) * (hits + fa) / total if total > 0 else 0.0
    denom = hits + miss + fa - h_rand
    ets = (hits - h_rand) / denom if denom > 0 else float("nan")
    base = (hits + miss) / total if total > 0 else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "ets": ets, "base_rate": base}
