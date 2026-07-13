"""
Physical-consistency diagnostics for the twin's forecast state.

The AI core is a statistical emulator, so the twin makes its physics explicit
and *checkable* instead of implicit and assumed. Every forecast can be audited
against first-order physical laws:

  * Thermodynamic ordering   — tmax >= tmin everywhere (diurnal cycle).
  * Mass non-negativity      — rainfall >= 0 (no negative water).
  * Water-budget anchor      — national accumulated rain vs the climatological
                               budget for the same calendar days (an emulator
                               that conserves nothing drifts away from this).
  * Diurnal temperature range— forecast DTR vs the climatological DTR
                               (energy-balance sanity: radiation-driven DTR
                               should not collapse or explode).

These are reported in the dashboard so a reviewer can see the laws being
respected, and by how much, on every forecast — not asserted on a slide.
"""
from __future__ import annotations

import numpy as np


def consistency_report(frames, landmask, clim_tmax_seq=None, clim_tmin_seq=None,
                       clim_rain_seq=None):
    """Audit a forecast {var:(HORIZON,H,W)} against first-order physics.

    Returns a dict of scalar diagnostics (all computed over land cells).
    """
    lm = landmask.astype(bool)
    tmax = np.asarray(frames["tmax"])[:, lm]
    tmin = np.asarray(frames["tmin"])[:, lm]
    rain = np.asarray(frames["rain"])[:, lm]

    dtr = tmax - tmin
    out = {
        "dtr_violation_pct": float(np.mean(dtr < 0) * 100.0),
        "dtr_mean_c": float(np.mean(dtr)),
        "neg_rain_pct": float(np.mean(rain < 0) * 100.0),
        "rain_total_mm": float(np.mean(np.sum(rain, axis=0))),  # per-cell accumulated
    }
    if clim_rain_seq is not None:
        crain = np.asarray(clim_rain_seq)[:, lm]
        c_tot = float(np.mean(np.sum(crain, axis=0)))
        out["rain_clim_total_mm"] = c_tot
        out["rain_budget_departure_pct"] = (
            float(100.0 * (out["rain_total_mm"] - c_tot) / c_tot) if c_tot > 1e-3
            else float("nan"))
    if clim_tmax_seq is not None and clim_tmin_seq is not None:
        cdtr = (np.asarray(clim_tmax_seq)[:, lm] - np.asarray(clim_tmin_seq)[:, lm])
        out["dtr_clim_mean_c"] = float(np.mean(cdtr))
        out["dtr_departure_c"] = out["dtr_mean_c"] - out["dtr_clim_mean_c"]
    return out


def cc_moisture_factor(d_temp_c, rate=0.07):
    """Clausius–Clapeyron scaling: ~7% more atmospheric moisture per degC.

    The saturation vapour pressure follows d(ln e_s)/dT ~ L/(R_v T^2) ~ 7%/K
    near surface temperatures — the thermodynamic law that couples the
    temperature lever to the rainfall regime in the what-if engine.
    """
    return max(1.0 + rate * float(d_temp_c), 0.0)
