"""
Inference for ClimateUNet: anomalies -> real-world fields.

    pred_anom_scaled = model(input_window)
    field[v, lead] = clim[v, doy(forecast_day)] + pred_anom_scaled[v,lead]*std[v]
    rain is clipped to >= 0.

v2 checkpoints (climate_unet_v2.pt) carry synoptic driver channels (MSLP,
u850, v850, precipitable water); the Forecaster reads the checkpoint config
and consumes the driver cube transparently — all call signatures stay the
same as v1.

Physical-consistency layer (applied at reconstruction):
  * rain >= 0                      (mass non-negativity)
  * tmax >= tmin + 0.1 degC        (thermodynamic ordering of the diurnal cycle)
The pre-fix violation rate is reported in the output dict so the constraint is
a transparent diagnostic, not a silent patch.

Also provides:
  * predict_window   — forecast from an explicit history window (BYOO /
                       user-assimilated states, days beyond the archive)
  * predict_ensemble — perturbed ensemble: initial-condition noise within
                       analysis error (propagated through the POA prior, as in
                       IMD NEPS / NCEP GEFS design) + stochastic MC-dropout.

Returns forecasts in real units (mm/day, deg C) on the national grid.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.architecture import build_model  # noqa: E402
from models import dataset as D  # noqa: E402

DTR_MIN = 0.1        # enforced minimum diurnal range (degC)


def default_checkpoint():
    """Prefer the driver-aware v2 checkpoint when it exists (or $VARUNA_CKPT)."""
    env = os.environ.get("VARUNA_CKPT")
    if env and os.path.exists(env):
        return env
    v2 = os.path.join(C.CKPT_DIR, "climate_unet_v2.pt")
    return v2 if os.path.exists(v2) else os.path.join(C.CKPT_DIR, "climate_unet.pt")


class Forecaster:
    def __init__(self, ckpt=None, device=None, dcube=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = ckpt or default_checkpoint()
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = state.get("config", {})
        self.n_drivers = int(cfg.get("n_drivers", 0))
        self.drivers = cfg.get("drivers", [])
        self.model = build_model(n_drivers=self.n_drivers).to(self.device).eval()
        self.model.load_state_dict(state["state_dict"])
        self.ckpt_name = os.path.basename(ckpt)
        self.dcube = dcube                       # (T,H,W,K) mmap, set by loader
        if self.n_drivers and dcube is None:
            from data import drivers as DRV
            self.dcube, _ = DRV.load_cube()
        if self.n_drivers and self.dcube is None:
            raise RuntimeError(f"{self.ckpt_name} needs the driver cube — "
                               "run `python data/drivers.py` first.")

    @torch.no_grad()
    def _infer(self, X):
        """X (in_ch,H,W) -> scaled-anomaly output (HORIZON,3,H,W)."""
        xb = torch.from_numpy(X[None]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            out = self.model(xb)[0].float().cpu().numpy()
        return out.reshape(C.HORIZON, 3, out.shape[1], out.shape[2])

    def _reconstruct(self, out, start_ts, carr, std):
        """Scaled anomalies -> real-unit frames + physics diagnostics."""
        H, W = out.shape[2], out.shape[3]
        f_dates = [pd.Timestamp(start_ts) + pd.Timedelta(days=k) for k in range(C.HORIZON)]
        frames = {v: np.empty((C.HORIZON, H, W), dtype="float32") for v in C.VARIABLES}
        for lead in range(C.HORIZON):
            doy = min(int(f_dates[lead].dayofyear), 365)
            for vi, v in enumerate(C.VARIABLES):
                field = carr[v][doy - 1] + out[lead, vi] * std[vi]
                if v == "rain":
                    field = np.clip(field, 0.0, None)
                frames[v][lead] = field
        # thermodynamic ordering: report violations, then enforce tmax >= tmin
        viol = frames["tmin"] > frames["tmax"]
        physics = {"dtr_violations_pct": float(np.mean(viol) * 100.0)}
        frames["tmin"] = np.minimum(frames["tmin"], frames["tmax"] - DTR_MIN)
        return {"frames": frames, "dates": f_dates, "physics": physics}

    @torch.no_grad()
    def predict(self, cube, t, carr, std, dates):
        """Forecast HORIZON days from forecast-start index t in the cube.

        Returns dict: frames {var: (HORIZON,H,W) real units}, dates, physics.
        """
        X, _ = D._window(cube, t, self.dcube if self.n_drivers else None)
        out = self._infer(X)
        return self._reconstruct(out, dates[t], carr, std)

    @torch.no_grad()
    def predict_window(self, hist, start_ts, carr, std, drv_hist=None):
        """Forecast from an explicit scaled-anomaly history (INPUT_DAYS,H,W,3).

        Used by the BYOO path: the history may be user-assimilated and may
        extend beyond the IMD archive (climatology background). For driver
        checkpoints, `drv_hist` (INPUT_DAYS,H,W,K) is used when given, else a
        zero-anomaly (climatological) driver background.
        """
        H, W = hist.shape[1], hist.shape[2]
        parts = [np.transpose(hist, (0, 3, 1, 2)).reshape(C.INPUT_DAYS * 3, H, W)]
        if self.n_drivers:
            if drv_hist is None:
                drv_hist = np.zeros((C.INPUT_DAYS, H, W, self.n_drivers), "float32")
            parts.append(np.asarray(drv_hist, dtype="float32")
                         .transpose(0, 3, 1, 2).reshape(-1, H, W))
        parts.append(D._poa_channels(hist[-1]))
        X = np.concatenate(parts, axis=0).astype("float32")
        out = self._infer(X)
        return self._reconstruct(out, start_ts, carr, std)

    @torch.no_grad()
    def predict_ensemble(self, cube, t, carr, std, dates, n=8, ic_sigma=0.12):
        """Perturbed-ensemble forecast at forecast-start index t.

        Two uncertainty sources per member, mirroring operational ensemble
        design (IMD NEPS / NCEP GEFS):
          * initial-condition perturbations — Gaussian noise (sigma `ic_sigma`,
            the training-time input-noise scale, i.e. within analysis error)
            on the history/driver channels, never the POA prior directly;
            the POA prior is rebuilt from the perturbed last day so the
            perturbation propagates through the prior;
          * model uncertainty — the bottleneck dropout is kept stochastic
            (MC-dropout), sampling a different sub-network each pass.

        Returns mean frames + per-cell spread (std, real units) per variable.
        """
        X, _ = D._window(cube, t, self.dcube if self.n_drivers else None)
        hist_ch = self.model.hist_ch
        noise_ch = hist_ch + self.model.drv_ch
        rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
        self.model.drop.train()                       # stochastic bottleneck only
        outs = []
        for i in range(n):
            Xp = X.copy()
            if i > 0 and ic_sigma > 0:                # member 0 = control run
                Xp[:noise_ch] += np.random.default_rng(i).normal(
                    0.0, ic_sigma, Xp[:noise_ch].shape).astype("float32")
                # the POA prior is part of the initial state: rebuild it from
                # the perturbed last observed day so the perturbation
                # propagates through the prior, not only the history
                last = Xp[hist_ch - 3:hist_ch]        # (3,H,W) perturbed last day
                for k in range(C.HORIZON):
                    Xp[noise_ch + k * 3: noise_ch + (k + 1) * 3] = \
                        last * (rho ** (k + 1))[:, None, None]
            outs.append(self._infer(Xp))
        self.model.drop.eval()
        outs = np.stack(outs)                         # (n,HZ,3,H,W)
        res = self._reconstruct(outs.mean(axis=0), dates[t], carr, std)
        spread = outs.std(axis=0)                     # scaled-anomaly spread
        res["spread"] = {v: (spread[:, vi] * float(std[vi])).astype("float32")
                         for vi, v in enumerate(C.VARIABLES)}
        res["members"] = n
        return res


def load_everything(ckpt=None):
    """Convenience: cache + forecaster, ready to predict."""
    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    fc = Forecaster(ckpt)
    return dict(obs=obs, clim=clim, stats=stats, landmask=landmask, grid=grid,
                cube=cube, dates=dates, carr=carr, std=std, forecaster=fc)
