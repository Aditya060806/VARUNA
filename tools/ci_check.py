"""
Lightweight CI check — no heavy data, no GPU, no network.

Runs three fast gates so regressions are caught in CI:
  1. byte-compile every .py in the repo (syntax / indentation errors)
  2. import the pure-Python modules (config, analytics, scenario, twin,
     evaluation, viz, data.proxies, data.insat) — the ones that do not need
     IMD data, torch or xgboost at import time
  3. exercise the core numerical functions on tiny synthetic arrays and assert
     the outputs are finite and shaped correctly

Heavy paths (model training / inference, real IMD download, the full Streamlit
AppTest in tools/test_ui.py) need the processed cache + checkpoints and are run
locally, not in CI.

    python tools/ci_check.py        # exits non-zero on any failure
"""
import compileall
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILURES = []


def gate(label, fn):
    try:
        fn()
        print(f"  [ok]   {label}")
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        FAILURES.append(label)


def compile_all():
    # skip virtualenvs / caches; fail loudly on syntax errors
    ok = compileall.compile_dir(
        ROOT, quiet=1, force=True,
        rx=__import__("re").compile(r"(\.venv|venv|__pycache__|\.git)"),
    )
    if not ok:
        raise RuntimeError("byte-compilation reported syntax errors")


def import_pure_modules():
    import config  # noqa: F401
    from analytics import extremes, physics  # noqa: F401
    from scenario import engine  # noqa: F401
    from twin import assimilate, user_obs  # noqa: F401
    from evaluation import metrics  # noqa: F401
    from data import proxies  # noqa: F401
    from data import insat  # noqa: F401
    from viz import theme, maps  # noqa: F401


def check_scenario():
    from scenario import engine
    hi = engine.heat_index_c(np.array([40.0, 25.0]), np.array([60.0, 40.0]))
    assert np.all(np.isfinite(hi)), "heat index produced non-finite values"
    aqi = engine.pm25_to_aqi(np.array([10.0, 55.0, 200.0]))
    assert np.all((aqi >= 0) & (aqi <= 500)), "AQI out of CPCB range"
    fields, m = engine.run_scenario(
        np.full((4, 4), 38.0), np.full((4, 4), 5.0),
        np.full((4, 4), 0.5), np.full((4, 4), 80.0),
        engine.Controls(d_temp=2.0, greening=0.3))
    assert np.isfinite(m["mean_heat_index_c"]), "scenario heat index NaN"


def check_extremes():
    from analytics import extremes
    tmax = np.stack([np.full((5, 5), 46.0) for _ in range(3)])
    rain = np.stack([np.full((5, 5), 70.0), np.zeros((5, 5)), np.zeros((5, 5))])
    clim = np.stack([np.full((5, 5), 38.0) for _ in range(3)])
    frames = {"tmax": tmax, "rain": rain, "tmin": tmax - 10}
    mask = np.ones((5, 5), dtype=bool)
    summ = extremes.summarize_forecast_hazards(frames, clim, mask)
    assert len(summ["leads"]) == 3, "summary horizon mismatch"
    assert summ["heatwave_area_pct"][0] == 100.0, "expected full heatwave coverage"
    assert summ["heavy_rain_area_pct"][0] == 100.0, "expected full heavy-rain coverage"
    assert summ["dry_spell_mean_days"][-1] >= 1.0, "dry-spell run not accumulating"


def check_assimilate():
    from twin import assimilate
    bg = np.zeros((6, 6))
    ob = np.ones((6, 6))
    ana = assimilate.optimal_interpolation(bg, ob, length_scale=1.0)
    assert ana.shape == bg.shape and np.all(np.isfinite(ana)), "OI output invalid"
    before = assimilate.innovation_stats(bg, ob)["rmse"]
    after = assimilate.innovation_stats(ana, ob)["rmse"]
    assert after <= before + 1e-9, "OI increased RMSE to obs"


def check_user_obs():
    import pandas as pd
    import config as Cc
    from twin import user_obs
    df = user_obs.parse_csv(user_obs.TEMPLATE_CSV)
    assert len(df) == 3 and "rain" in df.columns, "template CSV parse failed"
    # one observation into a zero background -> OI gain + spatial decay
    H = W = 9
    hist = np.zeros((Cc.INPUT_DAYS, H, W, 3), dtype="float32")
    carr = {v: np.zeros((366, H, W), dtype="float32") for v in Cc.VARIABLES}
    std = np.ones(3, dtype="float32")
    user = pd.DataFrame([{"date": "2024-06-10", "rain": 10.0}])
    out, rep = user_obs.inject_observations(hist, "2024-06-11", user, carr, std,
                                            4, 4, length_scale=1.5, sigma_o=0.5)
    assert len(rep) == 1, "innovation report missing"
    cell = out[Cc.INPUT_DAYS - 1, 4, 4, 0]
    assert 0 < cell < 10.0, "OI gain must move toward obs without overshooting"
    assert out[Cc.INPUT_DAYS - 1, 4, 5, 0] < cell, "influence must decay with distance"
    # physically impossible values must be rejected, not ingested
    bad = user_obs.parse_csv("date,rain,tmax,tmin\n2024-06-10,-5,20,25\n")
    assert bad["rain"].isna().all() and bad["tmax"].isna().all(), \
        "invalid values must be masked (negative rain / tmin>tmax)"


def check_physics():
    from analytics import physics
    frames = {"tmax": np.full((3, 5, 5), 30.0), "tmin": np.full((3, 5, 5), 22.0),
              "rain": np.full((3, 5, 5), 4.0)}
    frames["tmin"] = frames["tmin"].copy(); frames["tmin"][0, 0, 0] = 31.0
    rep = physics.consistency_report(frames, np.ones((5, 5), bool))
    assert rep["dtr_violation_pct"] > 0, "DTR violation not detected"
    assert rep["neg_rain_pct"] == 0.0, "false negative-rain flag"
    assert abs(physics.cc_moisture_factor(1.0) - 1.07) < 1e-9, "CC scaling wrong"
    assert physics.cc_moisture_factor(-20.0) == 0.0, "CC factor must floor at 0"


def check_insat_parser():
    from data import insat
    ts = insat.product_datetime("3RIMG_29JUN2026_0815_L2B_LST_V01R00.h5")
    assert ts is not None and ts.month == 6 and ts.day == 29, "INSAT date parse failed"
    assert insat.product_doy("3RIMG_29JUN2026_0815_L2B_LST_V01R00.h5") == 180
    assert insat.product_datetime("no_date_here.h5") is None


def main():
    print("VARUNA CI check")
    gate("byte-compile repo", compile_all)
    gate("import pure modules", import_pure_modules)
    gate("scenario engine", check_scenario)
    gate("extremes analytics", check_extremes)
    gate("assimilation (OI)", check_assimilate)
    gate("user observations (BYOO)", check_user_obs)
    gate("physics diagnostics", check_physics)
    gate("INSAT filename parser", check_insat_parser)

    if FAILURES:
        print(f"\nRESULT: FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("\nRESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
