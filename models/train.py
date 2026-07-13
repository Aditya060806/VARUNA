"""
Train ClimateUNet on real IMD anomalies (PyTorch / CUDA).

v2 additions (when the driver cube from data/drivers.py is present):
  * synoptic driver channels — MSLP, u850, v850, precipitable water anomalies
    join the 7-day history (51 -> 51+7K input channels), giving the network
    the pressure-gradient / circulation / moisture information it was blind to;
  * physics-informed loss terms, evaluated in REAL units by reconstructing
    field = anomaly*std + climatology(doy) inside the loss:
      - mass non-negativity   : penalise implied negative rainfall
      - thermodynamic ordering: penalise tmin > tmax (diurnal cycle)
    Both are soft constraints (PINN-style) on top of the data term.

Loss: latitude-area-weighted, land-masked Huber loss in scaled-anomaly space
(+ the physics penalties). Because every variable is divided by its own
anomaly std, the three variables contribute on a comparable scale.

Saves best checkpoint to models/checkpoints/climate_unet.pt (v1, no drivers)
or climate_unet_v2.pt (with drivers), plus a training curve.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.architecture import build_model  # noqa: E402
from models import dataset as D  # noqa: E402

EPOCHS = 60
BATCH = 32
LR = 2.5e-3
WEIGHT_DECAY = 2e-3
PATIENCE = 12
INPUT_NOISE = 0.12   # gaussian augmentation on history+driver inputs (train only)
NUM_WORKERS = 0      # single-process loading: reliable on a 16 GB machine
TRAIN_STRIDE = 3     # consecutive 7-day windows overlap ~86%; stride keeps full
                     # year coverage while cutting epoch cost ~3x

# physics-informed penalty weights (real-unit relu terms; kept modest so the
# data term dominates and the constraints act as regularisers)
W_NEG_RAIN = 0.05    # mass non-negativity of rainfall
W_DTR = 0.05         # thermodynamic ordering tmax >= tmin (+margin)
DTR_MARGIN = 0.1     # degC


def make_weight(landmask, lat):
    """(H,W) loss weight = land x cos(latitude), normalised to mean 1 over land."""
    latw = np.cos(np.deg2rad(lat))[:, None]          # (H,1)
    w = np.where(landmask, latw, 0.0).astype("float32")
    w *= landmask.size / max(w.sum(), 1.0)
    return torch.from_numpy(w)


def weighted_huber(pred, target, w, nv=3, horizon=C.HORIZON, beta=1.0):
    """pred/target: (N, horizon*nv, H, W). w: (H,W)."""
    N = pred.shape[0]
    pred = pred.view(N, horizon, nv, *pred.shape[2:])
    target = target.view(N, horizon, nv, *target.shape[2:])
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)  # (N,hz,nv,H,W)
    wm = w.view(1, 1, 1, *w.shape)
    return (err * wm).mean()


def physics_penalty(pred, doys, carr_t, std_t, w):
    """PINN-style soft constraints in real units.

    pred   : (N, HORIZON*3, H, W) scaled anomalies
    doys   : (N, HORIZON) day-of-year per lead
    carr_t : (366, 3, H, W) climatology on device
    std_t  : (3,) anomaly stds on device
    """
    N, _, H, W = pred.shape
    p = pred.view(N, C.HORIZON, 3, H, W)
    clim = carr_t[doys - 1]                          # (N,HORIZON,3,H,W)
    real = p * std_t.view(1, 1, 3, 1, 1) + clim
    wm = w.view(1, 1, *w.shape)
    neg_rain = F.relu(-real[:, :, 0]) * wm           # implied negative water
    dtr = F.relu(real[:, :, 2] - real[:, :, 1] + DTR_MARGIN) * wm  # tmin > tmax
    return W_NEG_RAIN * neg_rain.mean() + W_DTR * dtr.mean()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {dev}", flush=True)

    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    splits = D.split_indices(dates)

    from data import drivers as DRV
    if os.environ.get("VARUNA_NO_DRIVERS"):        # ablation: same data, no drivers
        dcube, dnames = None, []
    else:
        dcube, dnames = DRV.load_cube()
    n_drv = len(dnames)
    tag = os.environ.get("VARUNA_TAG") or ("v2" if n_drv else "v1")
    print(f"[train] drivers: {dnames or 'none'} -> {tag}", flush=True)
    if dcube is not None and len(dcube) != len(dates):
        raise RuntimeError(f"driver cube T={len(dcube)} != obs T={len(dates)}; "
                           "re-run data/drivers.py after data/prepare.py")

    print(f"[train] windows  train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    w = make_weight(landmask, grid["lat"]).to(dev)
    doys = D.lead_doys(dates)
    # half precision on GPU: the physics penalties are relu means of real-unit
    # fields (0..50 range) — fp16 is ample and halves the gather memory
    _pdt = torch.float16 if dev.type == "cuda" else torch.float32
    carr_t = torch.from_numpy(np.stack([carr[v][:366] for v in C.VARIABLES],
                                       axis=1)).to(dev, _pdt)  # (366,3,H,W)
    std_t = torch.from_numpy(std).to(dev, _pdt)

    train_idx = splits["train"][::TRAIN_STRIDE]
    print(f"[train] using {len(train_idx)} train windows (stride {TRAIN_STRIDE})", flush=True)
    tr = DataLoader(D.WindowDataset(cube, train_idx, dcube, doys), batch_size=BATCH,
                    shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    va = DataLoader(D.WindowDataset(cube, splits["val"], dcube, doys), batch_size=BATCH,
                    shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(n_drivers=n_drv).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_params:,}  in_ch: {model.in_ch}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_val, best_epoch, bad = float("inf"), -1, 0
    hist = {"train": [], "val": [], "phys": []}
    ckpt = os.path.join(C.CKPT_DIR, "climate_unet.pt" if tag == "v1"
                        else f"climate_unet_{tag}.pt")
    noise_ch = model.hist_ch + model.drv_ch          # never noise the POA prior

    for ep in range(EPOCHS):
        model.train()
        t0, tl, pl, nb = time.time(), 0.0, 0.0, 0
        for X, Y, dy in tr:
            X, Y, dy = X.to(dev), Y.to(dev), dy.to(dev)
            if INPUT_NOISE > 0:
                noise = torch.zeros_like(X)
                noise[:, :noise_ch] = INPUT_NOISE * torch.randn_like(X[:, :noise_ch])
                X = X + noise
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                pred = model(X)
                data_loss = weighted_huber(pred, Y, w)
                phys_loss = physics_penalty(pred, dy, carr_t, std_t, w)
                loss = data_loss + phys_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += data_loss.item(); pl += phys_loss.item(); nb += 1
        sched.step()
        tr_loss, ph_loss = tl / max(nb, 1), pl / max(nb, 1)

        model.eval()
        vl, vnb = 0.0, 0
        with torch.no_grad():
            for X, Y, dy in va:
                X, Y = X.to(dev), Y.to(dev)
                with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                    vl += weighted_huber(model(X), Y, w).item(); vnb += 1
        val_loss = vl / max(vnb, 1)
        hist["train"].append(tr_loss); hist["val"].append(val_loss)
        hist["phys"].append(ph_loss)
        print(f"[train] epoch {ep+1:02d}/{EPOCHS}  train {tr_loss:.4f}  "
              f"phys {ph_loss:.5f}  val {val_loss:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s", flush=True)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad = val_loss, ep, 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {"input_days": C.INPUT_DAYS, "horizon": C.HORIZON,
                                   "n_drivers": n_drv, "drivers": dnames},
                        "val_loss": val_loss, "epoch": ep}, ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[train] early stop at epoch {ep+1}", flush=True)
                break

    print(f"[train] best val {best_val:.4f} @ epoch {best_epoch+1}. saved {ckpt}", flush=True)
    with open(os.path.join(C.OUTPUTS_DIR, f"train_history_{tag}.json"), "w") as f:
        json.dump(hist, f)
    _plot(hist, tag)


def _plot(hist, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("weighted Huber loss"); plt.legend()
    plt.title(f"ClimateUNet training {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUTPUTS_DIR, f"training_curve_{tag}.png"), dpi=120)


if __name__ == "__main__":
    main()
