"""
Turn the processed cache into model-ready sequences (PyTorch).

Builds a scaled-anomaly cube:
    anom_scaled[t, v] = (obs[v, t] - clim[v, doy(t)]) / anom_std[v]

then yields supervised windows (channels-first for PyTorch):
    X = anom_scaled[t-INPUT_DAYS : t]  -> (INPUT_DAYS*3, H, W)
    Y = anom_scaled[t : t+HORIZON]     -> (HORIZON*3,   H, W)

Windows are assigned to train/val/test by the calendar year of their first
forecast day, so there is no temporal leakage across splits.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


def _p(name):
    return os.path.join(C.PROCESSED_DIR, name)


def load_cache():
    obs = xr.open_dataset(_p("obs.nc"))
    clim = {
        "rain": xr.open_dataset(_p("clim_rain.nc"))["rain"],
        "tmax": xr.open_dataset(_p("clim_temp.nc"))["tmax"],
        "tmin": xr.open_dataset(_p("clim_temp.nc"))["tmin"],
    }
    stats = json.load(open(_p("stats.json")))
    landmask = np.load(_p("landmask.npy"))
    grid = np.load(_p("grid.npz"))
    return obs, clim, stats, landmask, grid


def clim_arrays(clim):
    """Climatology indexed [doy-1] -> (366, H, W) per variable."""
    out = {}
    for v in C.VARIABLES:
        arr = clim[v].values.astype("float32")        # (doy, H, W), doy from 1
        if arr.shape[0] < 366:
            pad = np.repeat(arr[-1:], 366 - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        out[v] = arr
    return out


def build_anomaly_cube(obs, clim, stats):
    """Return (anom_scaled (T,H,W,3) float32, dates, clim_arr, std_vec).

    The scaled-anomaly cube is cached to disk so app / eval startup is fast
    instead of rebuilding it (~90s) every time. The cache filename embeds a
    short signature of the inputs (grid shape, date range and anomaly stats),
    so any change to obs/climatology/stats automatically invalidates the cache
    and triggers a rebuild instead of silently loading a stale cube.
    """
    import hashlib

    dates = pd.to_datetime(obs["time"].values)
    doy = dates.dayofyear.values
    carr = clim_arrays(clim)
    std = np.array([stats[v]["anom_std"] for v in C.VARIABLES], dtype="float32")

    T = len(dates)
    H, W = obs["rain"].shape[1], obs["rain"].shape[2]

    sig = hashlib.md5()
    sig.update(np.ascontiguousarray(std).tobytes())
    sig.update(repr((int(T), int(H), int(W))).encode())
    sig.update(str(dates[0]).encode())
    sig.update(str(dates[-1]).encode())
    key = sig.hexdigest()[:12]
    cache = _p(f"anom_cube_{key}.npy")

    if os.path.exists(cache):
        try:
            cube = np.load(cache, mmap_mode=None)
            if cube.shape == (T, H, W, 3):
                return cube, dates, carr, std
        except Exception:
            pass

    cube = np.empty((T, H, W, 3), dtype="float32")
    cidx = np.clip(doy - 1, 0, 365)
    for vi, v in enumerate(C.VARIABLES):
        ov = obs[v].values.astype("float32")
        cube[..., vi] = (ov - carr[v][cidx]) / (std[vi] + 1e-6)
    try:
        # purge stale cubes (old name + superseded signatures) before saving
        for old in glob.glob(_p("anom_cube*.npy")):
            if os.path.abspath(old) != os.path.abspath(cache):
                try:
                    os.remove(old)
                except OSError:
                    pass
        np.save(cache, cube)
    except Exception:
        pass
    return cube, dates, carr, std


def split_indices(dates):
    T = len(dates)
    years = dates.year.values
    splits = {"train": [], "val": [], "test": []}
    for t in range(C.INPUT_DAYS, T - C.HORIZON + 1):
        fy = years[t]
        if fy in C.TRAIN_YEARS:
            splits["train"].append(t)
        elif fy in C.VAL_YEARS:
            splits["val"].append(t)
        elif fy in C.TEST_YEARS:
            splits["test"].append(t)
    return {k: np.array(v) for k, v in splits.items()}


def _poa_channels(last_day):
    """Persistence-of-anomaly prior in scaled-anomaly space.

    last_day: (H,W,3) scaled anomaly of the most recent observed day.
    Returns (HORIZON*3, H, W): rho_var^(k+1) * last_day for each lead k.
    """
    H, W = last_day.shape[0], last_day.shape[1]
    rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
    poa = np.empty((C.HORIZON, 3, H, W), dtype="float32")
    for k in range(C.HORIZON):
        decay = rho ** (k + 1)                       # (3,)
        poa[k] = np.transpose(last_day, (2, 0, 1)) * decay[:, None, None]
    return poa.reshape(C.HORIZON * 3, H, W)


def _window(cube, t, dcube=None):
    """(X, Y) channels-first windows at forecast-start index t.

    X = [7-day history anomalies (21ch)]
        (+ [7-day driver anomalies (7*K ch)] when a driver cube is given)
        + [POA prior for 10 leads (30ch)]
    Y = [10-day target anomalies (30ch)]
    The POA prior lets the model learn a *residual correction* on top of a strong
    baseline, so it can only match-or-beat persistence-of-anomaly.
    """
    H, W = cube.shape[1], cube.shape[2]
    x = cube[t - C.INPUT_DAYS:t]                 # (INPUT_DAYS,H,W,3)
    y = cube[t:t + C.HORIZON]                    # (HORIZON,H,W,3)
    hist = np.transpose(x, (0, 3, 1, 2)).reshape(C.INPUT_DAYS * 3, H, W)
    parts = [hist]
    if dcube is not None:
        d = np.asarray(dcube[t - C.INPUT_DAYS:t], dtype="float32")
        parts.append(np.transpose(d, (0, 3, 1, 2)).reshape(-1, H, W))
    parts.append(_poa_channels(cube[t - 1]))     # (HORIZON*3,H,W)
    X = np.concatenate(parts, axis=0)            # (51 or 51+7K, H, W)
    Y = np.transpose(y, (0, 3, 1, 2)).reshape(C.HORIZON * 3, H, W)
    return X.astype("float32"), Y.astype("float32")


def lead_doys(dates):
    """(T, HORIZON) int32 day-of-year (1..366) for each forecast lead day.

    Used by the physics-informed losses to look up the climatology of each
    target day when reconstructing real-unit fields inside the loss.
    """
    base = dates.dayofyear.values.astype("int32")
    T = len(base)
    out = np.empty((T, C.HORIZON), dtype="int64")
    for k in range(C.HORIZON):
        out[:T - k, k] = base[k:]
        if k:
            out[T - k:, k] = base[-1]            # tail padding (never sampled)
    return np.clip(out, 1, 366)


class WindowDataset(Dataset):
    """Supervised windows for a split, materialised lazily from the cube(s)."""

    def __init__(self, cube, idx, dcube=None, doys=None):
        self.cube = cube
        self.dcube = dcube
        self.doys = doys                          # (T,HORIZON) or None
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        X, Y = _window(self.cube, t, self.dcube)
        if self.doys is not None:
            return torch.from_numpy(X), torch.from_numpy(Y), \
                torch.from_numpy(self.doys[t])
        return torch.from_numpy(X), torch.from_numpy(Y)
