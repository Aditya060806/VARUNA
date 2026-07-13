"""
Full benchmark: ClimateUNet vs persistence / persistence-of-anomaly / climatology
on the held-out TEST years, in real units, with per-lead-day skill.

Produces:
  outputs/eval_metrics.json   machine-readable results
  outputs/skill_curves.png    RMSE & ACC vs lead day
  console table               headline numbers for the deck
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models import baseline, dataset as D  # noqa: E402
from models.forecast import Forecaster, default_checkpoint  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from analytics import extremes  # noqa: E402

HEAVY_RAIN_MM = 64.5      # IMD "heavy" daily rainfall threshold
EVENTS = ["heatwave", "heavy_rain"]


def run(max_windows=None, stride=1, ckpt=None):
    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    splits = D.split_indices(dates)
    test_idx = splits["test"][::stride]
    if max_windows:
        test_idx = test_idx[:max_windows]

    fc = Forecaster(ckpt)
    stem = fc.ckpt_name.replace(".pt", "")
    tag = "" if stem == "climate_unet" else "_" + stem.replace("climate_unet_", "")
    print(f"[eval] evaluating {len(test_idx)} test windows  "
          f"ckpt={fc.ckpt_name} drivers={fc.drivers or 'none'}", flush=True)
    w = M._wstats(landmask, grid["lat"])
    obs_arr = {v: obs[v].values for v in C.VARIABLES}

    methods = ["ai", "poa", "persistence", "climatology"]
    # accumulators: method -> var -> lead -> list
    acc = {m: {v: {"se": np.zeros(C.HORIZON), "ae": np.zeros(C.HORIZON),
                   "acc": np.zeros(C.HORIZON), "n": np.zeros(C.HORIZON)}
               for v in C.VARIABLES} for m in methods}
    # categorical extreme-event contingency counts: method -> event -> (4, HORIZON)
    ev_counts = {m: {e: np.zeros((4, C.HORIZON)) for e in EVENTS}
                 for m in ["ai", "poa", "persistence"]}

    for t in test_idx:
        t = int(t)
        ai = fc.predict(cube, t, carr, std, dates)["frames"]
        poa = baseline.persistence_of_anomaly(obs, carr, dates, t)
        per = baseline.persistence(obs, dates, t)
        cli = baseline.climatology_forecast(carr, dates, t)
        preds = {"ai": ai, "poa": poa, "persistence": per, "climatology": cli}

        for lead in range(C.HORIZON):
            ti = t + lead
            if ti >= len(dates):
                break
            doy = min(int(dates[ti].dayofyear), 365)
            for v in C.VARIABLES:
                truth = obs_arr[v][ti]
                cl = carr[v][doy - 1]
                for m in methods:
                    p = preds[m][v][lead]
                    acc[m][v]["se"][lead] += M.wrmse(p, truth, w) ** 2
                    acc[m][v]["ae"][lead] += M.wmae(p, truth, w)
                    acc[m][v]["acc"][lead] += M.wacc(p, truth, cl, w)
                    acc[m][v]["n"][lead] += 1
            # extreme-event detection (categorical, IMD criteria)
            cl_tmax = carr["tmax"][doy - 1]
            obs_hw = extremes.heatwave_severity(obs_arr["tmax"][ti], cl_tmax) >= 1
            obs_hr = obs_arr["rain"][ti] >= HEAVY_RAIN_MM
            for m in ev_counts:
                pred_hw = extremes.heatwave_severity(preds[m]["tmax"][lead], cl_tmax) >= 1
                pred_hr = preds[m]["rain"][lead] >= HEAVY_RAIN_MM
                ev_counts[m]["heatwave"][:, lead] += M.event_counts(pred_hw, obs_hw, w)
                ev_counts[m]["heavy_rain"][:, lead] += M.event_counts(pred_hr, obs_hr, w)

    # reduce
    results = {}
    for m in methods:
        results[m] = {}
        for v in C.VARIABLES:
            n = np.maximum(acc[m][v]["n"], 1)
            results[m][v] = {
                "rmse": np.sqrt(acc[m][v]["se"] / n).tolist(),
                "mae": (acc[m][v]["ae"] / n).tolist(),
                "acc": (acc[m][v]["acc"] / n).tolist(),
            }

    # skill vs references
    for v in C.VARIABLES:
        rmse_ai = np.array(results["ai"][v]["rmse"])
        results["ai"][v]["skill_vs_persistence"] = [
            M.skill(rmse_ai[k], results["persistence"][v]["rmse"][k]) for k in range(C.HORIZON)]
        results["ai"][v]["skill_vs_poa"] = [
            M.skill(rmse_ai[k], results["poa"][v]["rmse"][k]) for k in range(C.HORIZON)]

    # categorical extreme-event scores (from accumulated contingency counts)
    for m in ev_counts:
        results[m]["events"] = {}
        for e in EVENTS:
            per_lead = [M.event_scores(ev_counts[m][e][:, k]) for k in range(C.HORIZON)]
            results[m]["events"][e] = {
                s: [pl[s] for pl in per_lead]
                for s in ("pod", "far", "csi", "ets", "base_rate")}

    results["meta"] = {"checkpoint": fc.ckpt_name, "drivers": fc.drivers,
                       "n_windows": int(len(test_idx))}
    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    out_json = os.path.join(C.OUTPUTS_DIR, f"eval_metrics{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    _print_events(results)
    _plot(results)
    print(f"[eval] wrote {out_json}", flush=True)
    return results


def _print_events(results):
    print("\n============ EXTREME-EVENT DETECTION (day 1 / day 3) ============")
    for e in EVENTS:
        print(f"\n--- {e} ---")
        print(f"{'method':>12} | {'POD d1':>7} {'FAR d1':>7} {'CSI d1':>7} "
              f"{'ETS d1':>7} | {'POD d3':>7} {'ETS d3':>7}")
        for m in ("ai", "poa", "persistence"):
            ev = results[m]["events"][e]
            print(f"{m:>12} | {ev['pod'][0]:7.3f} {ev['far'][0]:7.3f} "
                  f"{ev['csi'][0]:7.3f} {ev['ets'][0]:7.3f} | "
                  f"{ev['pod'][2]:7.3f} {ev['ets'][2]:7.3f}")


def _print_table(results):
    units = {"rain": "mm", "tmax": "C", "tmin": "C"}
    print("\n================ TEST-SET SKILL (real units) ================")
    for v in C.VARIABLES:
        print(f"\n--- {v} ({units[v]}) ---")
        print(f"{'lead':>4} | {'AI RMSE':>8} {'POA RMSE':>8} {'Pers RMSE':>9} "
              f"{'AI ACC':>7} | {'skill/pers':>10} {'skill/poa':>10}")
        for k in range(C.HORIZON):
            ai = results["ai"][v]
            print(f"{k+1:>4} | {ai['rmse'][k]:8.2f} {results['poa'][v]['rmse'][k]:8.2f} "
                  f"{results['persistence'][v]['rmse'][k]:9.2f} {ai['acc'][k]:7.3f} | "
                  f"{ai['skill_vs_persistence'][k]*100:9.1f}% {ai['skill_vs_poa'][k]*100:9.1f}%")


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads = np.arange(1, C.HORIZON + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, v in enumerate(C.VARIABLES):
        ax = axes[0, j]
        for m, lab in [("ai", "ClimateUNet"), ("poa", "Persist-anom"),
                       ("persistence", "Persistence"), ("climatology", "Climatology")]:
            ax.plot(leads, results[m][v]["rmse"], marker="o", label=lab)
        ax.set_title(f"{v} RMSE"); ax.set_xlabel("lead day"); ax.legend(fontsize=7)
        ax2 = axes[1, j]
        ax2.plot(leads, results["ai"][v]["acc"], marker="o", color="#FF7B00")
        ax2.axhline(0.6, ls="--", color="gray", lw=1)
        ax2.set_title(f"{v} ACC (AI)"); ax2.set_xlabel("lead day"); ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUTPUTS_DIR, "skill_curves.png"), dpi=120)


if __name__ == "__main__":
    _ckpt = next((a for a in sys.argv[1:] if a.endswith(".pt")), None)
    run(ckpt=_ckpt)
