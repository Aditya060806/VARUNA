"""
What-if scenario engine - physics-informed proxies connecting the climate twin
to two climate-sensitive sectors: urban heat and air quality.

Formulas are standard and citable:
  * Heat index : NWS Rothfusz "feels-like" regression.
  * AQI        : India CPCB PM2.5 sub-index breakpoints.
  * LST / UHI  : linear surface-energy proxy
                 LST = T + UHI*urban - k_ndvi*dNDVI - k_albedo*dAlbedo
  * CC coupling: Clausius-Clapeyron — a warmer atmosphere holds ~7% more
                 moisture per degC, so the temperature lever also scales the
                 rainfall regime (thermodynamic consistency between levers).

These are transparent proxies for the proof-of-concept; coefficients are
literature-consistent and clearly labeled as proxies.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


@dataclass
class Controls:
    d_temp: float = 0.0        # delta air temperature (degC)
    d_rain_pct: float = 0.0    # rainfall change (%)
    greening: float = 0.0      # NDVI uplift 0..1
    cool_roof: float = 0.0     # roof albedo uplift 0..1
    urbanization: float = 0.0  # added built-up fraction 0..1
    humidity: float = 55.0     # relative humidity (%)
    wind: float = 2.5          # wind speed (m/s)

    def as_dict(self):
        return asdict(self)


# --- heat index (NWS Rothfusz) ---
def heat_index_c(temp_c, rh):
    t = np.asarray(temp_c, dtype="float64") * 9 / 5 + 32
    r = np.asarray(rh, dtype="float64")
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r
          - 6.83783e-3 * t * t - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r
          + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    hi = np.where(t < 80, simple, hi)
    return np.clip((hi - 32) * 5 / 9, -10.0, 60.0)


HEAT_BANDS = [(-100, 27, "Safe", "#2DD4BF"), (27, 32, "Caution", "#FFD60A"),
              (32, 41, "Extreme Caution", "#FF7B00"), (41, 54, "Danger", "#FF2D55"),
              (54, 200, "Extreme Danger", "#7E0023")]


def heat_band(v):
    for lo, hi, lab, col in HEAT_BANDS:
        if lo <= v < hi:
            return lab, col
    return HEAT_BANDS[-1][2], HEAT_BANDS[-1][3]


# --- LST / UHI proxy ---
K_NDVI, K_ALBEDO, UHI_BASE = 8.0, 10.0, 4.0


def land_surface_temp(temp_c, urban, d_ndvi=0.0, d_albedo=0.0):
    return (np.asarray(temp_c, dtype="float64") + UHI_BASE * np.asarray(urban)
            - K_NDVI * d_ndvi - K_ALBEDO * d_albedo)


# --- AQI (CPCB PM2.5) ---
def pm25_to_aqi(pm):
    pm = np.asarray(pm, dtype="float64")
    bp = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
          (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)]
    aqi = np.full_like(pm, 500.0)
    for clo, chi, ilo, ihi in bp:
        m = (pm >= clo) & (pm <= chi)
        aqi = np.where(m, ilo + (ihi - ilo) * (pm - clo) / (chi - clo), aqi)
    return np.where(np.isfinite(pm), aqi, np.nan)


def aqi_band(v):
    for lo, hi, lab, col in C.AQI_BANDS:
        if lo <= v <= hi:
            return lab, col
    return C.AQI_BANDS[-1][2], C.AQI_BANDS[-1][3]


def aqi_proxy(pm_base, rain_mm, d_ndvi, wind, urban, d_temp=0.0):
    pm = np.asarray(pm_base, dtype="float64")
    pm = pm * np.exp(-0.015 * np.clip(rain_mm, 0, None))   # rain washout
    pm = pm * (1.0 - 0.35 * np.clip(d_ndvi, 0, 1))         # vegetation deposition
    pm = pm * (2.0 / (1.0 + np.clip(wind, 0.2, None)))     # wind dispersion
    pm = pm * (1.0 + 0.6 * np.clip(urban, 0, 1))           # urban emissions
    pm = pm * (1.0 + 0.02 * d_temp)                        # warmer -> more secondary aerosol
    return pm25_to_aqi(np.clip(pm, 0, 1000))


CC_RATE = 0.07  # Clausius-Clapeyron: ~7%/degC moisture-holding capacity


def run_scenario(temp_field, rain_field, urban, pm_base, ctrl: Controls):
    urban = np.asarray(urban, dtype="float64")
    t = np.asarray(temp_field, dtype="float64") + ctrl.d_temp
    rain = np.clip(np.asarray(rain_field, dtype="float64") * (1 + ctrl.d_rain_pct / 100.0), 0, None)
    # Clausius-Clapeyron coupling: warming scales the rainfall regime ~7%/degC
    # (saturation vapour pressure), so the levers stay thermodynamically linked.
    rain = rain * max(1.0 + CC_RATE * ctrl.d_temp, 0.0)
    urb = np.clip(urban + ctrl.urbanization, 0, 1)

    eff_ndvi = ctrl.greening * urb        # interventions deployed in built-up areas
    eff_alb = ctrl.cool_roof * urb
    lst_base = land_surface_temp(temp_field, urban)
    lst_scn = land_surface_temp(t, urb, eff_ndvi, eff_alb)
    cooling = lst_base - lst_scn

    # Rainfall as a wetness-regime lever: a wetter regime is more humid and a bit
    # cooler (evaporation / cloud cover). Both feed the "feels-like" heat index,
    # so the rainfall slider visibly changes heat even where current rain is low.
    rh = np.clip(ctrl.humidity + 1.6 * np.sqrt(rain) + 0.30 * ctrl.d_rain_pct, 10, 98)
    rain_cool = 0.05 * rain + 0.02 * ctrl.d_rain_pct
    t_felt = t + 0.6 * UHI_BASE * urb - 0.5 * np.clip(cooling, 0, None) - rain_cool
    hi = heat_index_c(t_felt, rh)
    aqi = aqi_proxy(pm_base, rain, eff_ndvi, ctrl.wind, urb, ctrl.d_temp)

    def nm(a):
        a = np.asarray(a)[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    hv = hi[np.isfinite(hi)]
    urb_cells = np.isfinite(cooling) & (urb >= 0.3)
    metrics = {
        "peak_cooling_c": float(np.nanmax(cooling)) if np.isfinite(cooling).any() else float("nan"),
        "urban_cooling_c": float(np.mean(cooling[urb_cells])) if urb_cells.any() else float("nan"),
        "mean_heat_index_c": nm(hi), "mean_aqi": nm(aqi),
        "heat_danger_area_pct": float(np.mean(hv >= 41) * 100) if hv.size else float("nan"),
        "controls": ctrl.as_dict(),
    }
    fields = {"heat_index": hi, "aqi": aqi, "cooling": cooling,
              "lst_baseline": lst_base, "lst_scenario": lst_scn}
    return fields, metrics
