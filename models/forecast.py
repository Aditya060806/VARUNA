"""
Inference for ClimateUNet: anomalies -> real-world fields.

    pred_anom_scaled = model(input_window)
    field[v, lead] = clim[v, doy(forecast_day)] + pred_anom_scaled[v,lead]*std[v]
    rain is clipped to >= 0.

Physical-consistency layer (applied at reconstruction):
  * rain >= 0                      (mass non-negativity)
  * tmax >= tmin + 0.1 degC        (thermodynamic ordering of the diurnal cycle)
The pre-fix violation rate is reported in the output dict so the constraint is
a transparent diagnostic, not a silent patch.

Also provides:
  * predict_window   — forecast from an explicit history window (BYOO /
                       user-assimilated states, days beyond the archive)
  * predict_ensemble — MC-dropout ensemble: the bottleneck dropout is kept
                       stochastic at inference and N members are drawn, giving
                       a per-cell forecast spread (uncertainty source #1; the
                       CNN-vs-XGBoost disagreement is source #2).

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


class Forecaster:
    def __init__(self, ckpt=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model().to(self.device).eval()
        ckpt = ckpt or os.path.join(C.CKPT_DIR, "climate_unet.pt")
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])

    @torch.no_grad()
    def _infer(self, X):
        """X (51,H,W) -> scaled-anomaly output (HORIZON,3,H,W)."""
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
        X, _ = D._window(cube, t)                       # (INPUT_DAYS*3,H,W)
        out = self._infer(X)
        return self._reconstruct(out, dates[t], carr, std)

    @torch.no_grad()
    def predict_window(self, hist, start_ts, carr, std):
        """Forecast from an explicit scaled-anomaly history (INPUT_DAYS,H,W,3).

        Used by the BYOO path: the history may be user-assimilated and may
        extend beyond the IMD archive (climatology background).
        """
        H, W = hist.shape[1], hist.shape[2]
        hist_ch = np.transpose(hist, (0, 3, 1, 2)).reshape(C.INPUT_DAYS * 3, H, W)
        poa = D._poa_channels(hist[-1])
        X = np.concatenate([hist_ch, poa], axis=0).astype("float32")
        out = self._infer(X)
        return self._reconstruct(out, start_ts, carr, std)

    @torch.no_grad()
    def predict_ensemble(self, cube, t, carr, std, dates, n=8, ic_sigma=0.12):
        """Perturbed-ensemble forecast at forecast-start index t.

        Two uncertainty sources per member, mirroring operational ensemble
        design (IMD NEPS / NCEP GEFS):
          * initial-condition perturbations — Gaussian noise (sigma `ic_sigma`,
            the training-time input-noise scale, i.e. within analysis error)
            on the history channels only, never the POA prior;
          * model uncertainty — the bottleneck dropout is kept stochastic
            (MC-dropout), sampling a different sub-network each pass.

        Returns mean frames + per-cell spread (std, real units) per variable.
        """
        X, _ = D._window(cube, t)
        hist_ch = C.INPUT_DAYS * 3
        rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
        self.model.drop.train()                       # stochastic bottleneck only
        outs = []
        for i in range(n):
            Xp = X.copy()
            if i > 0 and ic_sigma > 0:                # member 0 = control run
                Xp[:hist_ch] += np.random.default_rng(i).normal(
                    0.0, ic_sigma, Xp[:hist_ch].shape).astype("float32")
                # the POA prior is part of the initial state: rebuild it from
                # the perturbed last observed day so the perturbation
                # propagates through the prior, not only the history
                last = Xp[hist_ch - 3:hist_ch]        # (3,H,W) perturbed last day
                for k in range(C.HORIZON):
                    Xp[hist_ch + k * 3: hist_ch + (k + 1) * 3] = \
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
