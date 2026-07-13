"""
Synoptic driver ingestion — the dynamics the surface model cannot see.

Adds four real reanalysis fields as model inputs (the physics the judges asked
about): mean-sea-level pressure (the gradient that drives the flow), 850-hPa
u/v winds (the monsoon circulation / moisture-advection level) and total-column
precipitable water (the moisture supply that becomes rain).

Source: NCEP/NCAR Reanalysis-1 daily means via NOAA PSL's NetcdfSubset service
(openly downloadable, no credentials, 1948->present, so it covers the full
1981-2024 IMD span AND today). The ingest is source-agnostic: IMDAA (NCMRWF
RDS, registration required; open BharatBench release ends 2020) drops into the
same regridder for a fully-national driver set.

Design: climatology and anomaly statistics are computed on the NATIVE 2.5 deg
grid from TRAIN years only (no leakage), then the scaled anomalies are
regridded to the 0.25 deg national grid and cached as a float16 memory-mapped
cube aligned 1:1 with the observation dates.

    python data/drivers.py            # download + build the driver cube
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

DRV_RAW = os.path.join(C.RAW_DIR, "drivers")
os.makedirs(DRV_RAW, exist_ok=True)

# name -> (PSL dataset path, NCSS var, level or None)
DRIVERS = {
    "slp":  ("surface/slp.{yr}.nc", "slp", None),
    "u850": ("pressure/uwnd.{yr}.nc", "uwnd", 850),
    "v850": ("pressure/vwnd.{yr}.nc", "vwnd", 850),
    "pwat": ("surface/pr_wtr.eatm.{yr}.nc", "pr_wtr", None),
}
DRIVER_NAMES = list(DRIVERS.keys())

_NCSS = ("https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/"
         "Dailies/{path}?var={var}&north=45&south=0&west=55&east=105"
         "{lev}&time_start={yr}-01-01T00:00:00Z&time_end={yr}-12-31T23:59:59Z"
         "&accept=netcdf")


def _raw_path(name, yr):
    return os.path.join(DRV_RAW, f"{name}.{yr}.nc")


def download_year(name, yr, retries=3):
    """Fetch one driver-year via NCSS (offline-first: cached file wins)."""
    import urllib.request

    out = _raw_path(name, yr)
    if os.path.exists(out) and os.path.getsize(out) > 10_000:
        return out
    path, var, lev = DRIVERS[name]
    url = _NCSS.format(path=path.format(yr=yr), var=var, yr=yr,
                       lev=(f"&vertCoord={lev}" if lev else ""))
    for k in range(retries):
        try:
            urllib.request.urlretrieve(url, out)
            if os.path.getsize(out) > 10_000:
                return out
        except Exception as e:
            if k == retries - 1:
                raise RuntimeError(f"download failed: {name} {yr}: {e}")
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {name} {yr}")


def download_all(years=None):
    years = years or C.ALL_YEARS
    for name in DRIVER_NAMES:
        for yr in years:
            download_year(name, yr)
            print(f"[drivers] {name} {yr} ok", flush=True)


def _open_var(name, years):
    """Concatenate one driver across years -> xr.DataArray (time, lat, lon)."""
    import xarray as xr

    _, var, _ = DRIVERS[name]
    parts = []
    for yr in years:
        ds = xr.open_dataset(_raw_path(name, yr))
        da = ds[var].squeeze(drop=True)          # drop the level dim if present
        parts.append(da)
    out = xr.concat(parts, dim="time").sortby("time")
    # PSL latitudes run north->south; keep native order, interp handles it
    return out


def build(years=None):
    """Build the scaled-anomaly driver cube aligned to the observation dates.

    Saves data/processed/drv_cube.f16.npy (T,H,W,K float16, memory-mappable),
    drv_stats.json (per-driver anomaly std, native-grid climatology stats).
    """
    import pandas as pd
    import xarray as xr
    from data import climatology

    years = years or C.ALL_YEARS
    download_all(years)

    obs = xr.open_dataset(os.path.join(C.PROCESSED_DIR, "obs.nc"))
    odates = pd.to_datetime(obs["time"].values)
    grid = np.load(os.path.join(C.PROCESSED_DIR, "grid.npz"))
    lat_t, lon_t = grid["lat"], grid["lon"]
    T, H, W, K = len(odates), len(lat_t), len(lon_t), len(DRIVER_NAMES)

    cube = np.lib.format.open_memmap(
        os.path.join(C.PROCESSED_DIR, "drv_cube.f16.npy"),
        mode="w+", dtype="float16", shape=(T, H, W, K))
    stats = {}

    for vi, name in enumerate(DRIVER_NAMES):
        print(f"[drivers] building {name} ...", flush=True)
        da = _open_var(name, years)
        # align to the observation calendar (tolerance 1 day, else NaN)
        da = da.reindex(time=odates, method="nearest", tolerance=pd.Timedelta("1D"))

        # climatology + std on the NATIVE grid, TRAIN years only (no leakage)
        tr = da.sel(time=da["time"].dt.year.isin(C.CLIM_YEARS))
        clim = climatology.smooth_climatology(tr.to_dataset(name=name), name,
                                              C.CLIM_SMOOTH_WINDOW)[name]
        doy = np.clip(da["time"].dt.dayofyear.values, 1, clim.sizes["dayofyear"])
        anom = da.values - clim.values[doy - 1]
        tr_mask = np.isin(odates.year, C.CLIM_YEARS)
        std = float(np.nanstd(anom[tr_mask]))
        stats[name] = {"anom_std": std, "native_mean": float(np.nanmean(da.values))}
        print(f"   {name}: anom_std={std:.3f}", flush=True)

        # scale, fill gaps with 0 (= climatology), regrid to the national grid
        anom = np.nan_to_num(anom / (std + 1e-6), nan=0.0).astype("float32")
        anom_da = xr.DataArray(anom, dims=("time", "lat", "lon"),
                               coords={"time": odates, "lat": da["lat"].values,
                                       "lon": da["lon"].values})
        step = 2000                                   # regrid in time chunks
        for s in range(0, T, step):
            e = min(s + step, T)
            sub = anom_da.isel(time=slice(s, e)).interp(
                lat=lat_t, lon=lon_t, method="linear", kwargs={"fill_value": None})
            cube[s:e, :, :, vi] = sub.values.astype("float16")
        cube.flush()

    with open(os.path.join(C.PROCESSED_DIR, "drv_stats.json"), "w") as f:
        json.dump({"drivers": DRIVER_NAMES, "stats": stats,
                   "source": "NCEP/NCAR R1 daily (NOAA PSL NCSS)",
                   "n_time": T}, f, indent=2)
    print(f"[drivers] done: cube (T={T}, {H}x{W}, K={K}) + stats saved", flush=True)


def load_cube():
    """Memory-mapped driver cube + names, or (None, []) if not built."""
    p = os.path.join(C.PROCESSED_DIR, "drv_cube.f16.npy")
    s = os.path.join(C.PROCESSED_DIR, "drv_stats.json")
    if not (os.path.exists(p) and os.path.exists(s)):
        return None, []
    meta = json.load(open(s))
    return np.load(p, mmap_mode="r"), meta["drivers"]


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:] if a.isdigit()] or None
    build(yrs)
