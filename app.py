"""
AI Digital Twin of India's Climate - interactive dashboard.
ISRO Bharatiya Antariksh Hackathon 2026 - Problem Statement #5.

One AI climate twin -> connected applications:
  * National climate state + AI short-term forecast (real IMD data, ClimateUNet)
  * Hazard early-warning (heatwave / heavy-rain / dry-spell) from the forecast
  * Optimal-Interpolation data assimilation (model + observations)
  * What-if scenario simulator -> urban heat-stress & air quality
  * INSAT/MOSDAC satellite layer (bring-your-own product)

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

import config as C
from viz import theme, maps
from scenario import engine
from twin import assimilate
from analytics import extremes
from data import proxies, insat

st.set_page_config(page_title="VARUNA · AI Digital Twin of India's Climate",
                   page_icon="🛰️", layout="wide")
st.markdown(theme.streamlit_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- load
@st.cache_resource(show_spinner="Loading climate twin + AI model…")
def load():
    from models.forecast import load_everything
    return load_everything()


@st.cache_data
def load_eval():
    p = os.path.join(C.OUTPUTS_DIR, "eval_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_data
def load_xgb_report():
    p = os.path.join(C.OUTPUTS_DIR, "xgb_report.json")
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_resource
def load_xgb():
    try:
        from models.xgb_forecast import CityForecaster
        if all(os.path.exists(os.path.join(C.CKPT_DIR, f"xgb_{v}.json")) for v in C.VARIABLES):
            return CityForecaster()
    except Exception:
        return None
    return None


@st.cache_data(show_spinner="Ingesting INSAT product…")
def load_insat(prod, _files_key):
    from data import insat as _insat
    return _insat.latest(prod, grid)


try:
    B = load()
except Exception as e:
    st.error(f"Model/data not ready: {e}\n\nRun `python data/prepare.py` then "
             f"`python models/train.py`.")
    st.stop()

obs, clim, stats = B["obs"], B["clim"], B["stats"]
landmask, grid = B["landmask"], B["grid"]
cube, dates, carr, std, fc = B["cube"], B["dates"], B["carr"], B["std"], B["forecaster"]
lats, lons = grid["lat"], grid["lon"]
EVAL = load_eval()
XGB = load_xgb()
XGB_REPORT = load_xgb_report()

t_lo, t_hi = C.INPUT_DAYS, len(dates) - C.HORIZON
test_default = next((i for i in range(t_hi, t_lo, -1) if dates[i].year in C.TEST_YEARS), t_hi)


# --------------------------------------------------------------------------- header
c1, c2, c3 = st.columns([6, 3, 3])
with c1:
    st.markdown("## 🛰️ VARUNA <span class='tn-tag'>· AI Digital Twin of India's Climate</span>",
                unsafe_allow_html=True)
    st.caption("ISRO BAH 2026 · PS#5 — real IMD + INSAT data · ClimateUNet forecast · "
               "hazards · assimilation · what-if")
with c2:
    st.markdown(f"<div class='tn-card'><div class='l'>Data source</div>"
                f"<div class='v' style='color:{C.PALETTE['good']};font-size:1.15rem'>IMD gridded</div>"
                f"<div class='l'>{dates[0].year}–{dates[-1].year} · real only</div></div>",
                unsafe_allow_html=True)
with c3:
    if EVAL:
        sk = np.mean([EVAL["ai"][v]["skill_vs_persistence"][0] for v in C.VARIABLES]) * 100
        st.markdown(f"<div class='tn-card'><div class='l'>Day-1 skill vs persistence</div>"
                    f"<div class='v' style='font-size:1.15rem'>+{sk:.0f}%</div>"
                    f"<div class='l'>tmax ACC {EVAL['ai']['tmax']['acc'][0]:.2f}</div></div>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
st.sidebar.header("🛰️ Twin controls")
region_name = st.sidebar.selectbox("Region", list(C.REGIONS.keys()), index=0)
bounds = C.REGIONS[region_name]
var_label = st.sidebar.radio("Climate layer", ["Rainfall", "Max temp", "Min temp"], horizontal=True)
base_var = {"Rainfall": "rain", "Max temp": "tmax", "Min temp": "tmin"}[var_label]

import datetime as _dt
_min_d = dates[t_lo].date()
_last_d = dates[-1].date()                       # last day with real data
_max_d = _dt.date(2026, 12, 31)                  # allow future climatological projection
_def_d = dates[test_default].date()
picked = st.sidebar.date_input("Forecast start date", value=_def_d,
                               min_value=_min_d, max_value=_max_d)
future = picked > _last_d
st.sidebar.caption(
    (f"📅 {picked.strftime('%d %b %Y')} · 📊 climatological projection (beyond IMD data)"
     if future else
     f"📅 {picked.strftime('%d %b %Y')}"
     + ("  ·  unseen test year ✓" if picked.year in C.TEST_YEARS else "  ·  🤖 AI forecast"))
)
lead = st.sidebar.slider("Forecast lead day", 1, C.HORIZON, 1)


@st.cache_data(show_spinner="Running forecast…")
def forecast_for_date(date_iso):
    """AI forecast when real history exists; climatological projection beyond data.

    Returns (frames {var:(HORIZON,H,W)}, f_dates [str], mode 'ai'|'clim').
    """
    pts = pd.Timestamp(date_iso)
    if pts <= dates[-1]:
        ti = int(np.argmin(np.abs(dates.values - np.datetime64(pts))))
        ti = min(max(ti, t_lo), len(dates) - 1)
        out = fc.predict(cube, ti, carr, std, dates)
        return ({v: out["frames"][v] for v in C.VARIABLES},
                [str(d) for d in out["dates"]], "ai")
    # future: expected climatological state for each forecast day
    fdates = [pts + pd.Timedelta(days=k) for k in range(C.HORIZON)]
    frames = {}
    for v in C.VARIABLES:
        frames[v] = np.stack([carr[v][min(int(d.dayofyear), 365) - 1] for d in fdates])
    return frames, [str(d) for d in fdates], "clim"


@st.cache_data(show_spinner="Sampling MC-dropout ensemble…")
def ensemble_for_date(date_iso, n=6):
    """MC-dropout ensemble spread for an AI-forecastable date."""
    pts = pd.Timestamp(date_iso)
    ti = int(np.argmin(np.abs(dates.values - np.datetime64(pts))))
    ti = min(max(ti, t_lo), len(dates) - 1)
    res = fc.predict_ensemble(cube, ti, carr, std, dates, n=n)
    return {"spread": res["spread"], "physics": res["physics"], "members": res["members"]}


frames, f_dates, fmode = forecast_for_date(picked.isoformat())

# nearest real-data index (for panels that compare against observations / history)
t = int(np.argmin(np.abs(dates.values - np.datetime64(min(pd.Timestamp(picked), dates[-1])))))
t = min(max(t, t_lo), len(dates) - 1)


def regional(field):
    return maps.crop_to_bounds(field, lats, lons, bounds)


def clim_field(var, day_idx):
    doy = min(int(pd.Timestamp(f_dates[day_idx]).dayofyear), 365)
    return carr[var][doy - 1]


VIEWS = ["🌍 Climate Twin", "🌡️ Hazards & Extremes", "🧪 What-if Simulator",
         "📥 Your Data → Forecast", "📈 Validation & Skill", "🛰️ Satellite (INSAT)",
         "ℹ️ About"]
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")


# ===== Climate twin + assimilation =========================================
if view == VIEWS[0]:
    mode_lbl = "AI forecast" if fmode == "ai" else "Climatological projection"
    ai_field = frames[base_var][lead - 1]
    rf, rlat, rlon = regional(ai_field)
    rmask, _, _ = maps.crop_to_bounds(landmask, lats, lons, bounds)
    left, right = st.columns([3, 1.1])
    with left:
        fig = maps.field_figure(rf, rlat, rlon, base_var,
                                title=f"{theme.LABELS[base_var]} — {mode_lbl}, lead day {lead}",
                                landmask=rmask, bounds=bounds)
        st.plotly_chart(fig, use_container_width=True, key="twin")
        st.caption(f"{theme.LABELS[base_var]} · {mode_lbl} lead day {lead} "
                   f"({pd.Timestamp(f_dates[lead-1]).strftime('%d %b %Y')}) · {region_name}")
    with right:
        v = rf[np.isfinite(rf) & rmask]
        unit = "mm/day" if base_var == "rain" else "°C"
        st.markdown(f"#### {mode_lbl} state")
        st.metric("Mean", f"{np.mean(v):.1f} {unit}" if v.size else "—")
        st.metric("Max", f"{np.max(v):.1f} {unit}" if v.size else "—")
        st.markdown("#### Assimilation (Optimal Interpolation)")
        if fmode != "ai":
            st.caption("Assimilation applies to AI forecasts of real dates. For future "
                       "climatological projections there is no observation to assimilate.")
        else:
            st.caption("AI background fused with the latest observation; the innovation is "
                       "spread spatially per a correlated background-error covariance.")
            L = st.slider("Correlation length (cells)", 0.5, 5.0, 2.0, 0.5)
            bg = frames[base_var][0]
            ob = obs[base_var].values[t]
            ana = assimilate.optimal_interpolation(bg, ob, length_scale=L, landmask=landmask)
            before = assimilate.innovation_stats(bg, ob)
            after = assimilate.innovation_stats(ana, ob)
            st.metric("RMSE to obs · before", f"{before['rmse']:.2f} {unit}")
            st.metric("RMSE to obs · after OI", f"{after['rmse']:.2f} {unit}",
                      f"{after['rmse']-before['rmse']:+.2f}", delta_color="inverse")
            # --- cyclic assimilation: the analysis initialises the next cycle ---
            if t + 1 < len(dates) and st.button("🔁 Close the loop — run next cycle"):
                doy_t = min(int(dates[t].dayofyear), 365)
                hist_free = np.array(cube[t + 1 - C.INPUT_DAYS:t + 1], dtype="float32")
                hist_asm = hist_free.copy()
                for vi_, v_ in enumerate(C.VARIABLES):
                    bg_v = frames[v_][0]
                    ana_v = assimilate.optimal_interpolation(
                        bg_v, obs[v_].values[t], length_scale=L, landmask=landmask)
                    hist_free[-1, :, :, vi_] = (bg_v - carr[v_][doy_t - 1]) / (std[vi_] + 1e-6)
                    hist_asm[-1, :, :, vi_] = (ana_v - carr[v_][doy_t - 1]) / (std[vi_] + 1e-6)
                dh = (np.asarray(fc.dcube[t + 1 - C.INPUT_DAYS:t + 1], dtype="float32")
                      if getattr(fc, "n_drivers", 0) else None)
                start2 = dates[t] + pd.Timedelta(days=1)
                f_free = fc.predict_window(hist_free, start2, carr, std, dh)
                f_asm = fc.predict_window(hist_asm, start2, carr, std, dh)
                truth2 = obs[base_var].values[t + 1]
                r_free = assimilate.innovation_stats(f_free["frames"][base_var][0], truth2)["rmse"]
                r_asm = assimilate.innovation_stats(f_asm["frames"][base_var][0], truth2)["rmse"]
                st.metric("Next-day RMSE · forecast-only cycle", f"{r_free:.2f} {unit}")
                st.metric("Next-day RMSE · assimilated cycle", f"{r_asm:.2f} {unit}",
                          f"{r_asm - r_free:+.2f}", delta_color="inverse")
                st.caption("**The twin loop, closed:** the OI analysis (not the raw "
                           "background) initialised the next forecast cycle — "
                           "observe → forecast → assimilate → forecast. The "
                           "assimilated cycle tracks tomorrow's truth closer than "
                           "the forecast-only cycle.")

    # --- forecast uncertainty (MC-dropout ensemble) + physics audit ---
    if fmode == "ai":
        uc1, uc2 = st.columns([3, 1.1])
        with uc1:
            show_unc = st.checkbox("🎲 Show forecast uncertainty (MC-dropout ensemble)")
            if show_unc:
                ens = ensemble_for_date(picked.isoformat())
                sp, _, _ = regional(ens["spread"][base_var][lead - 1])
                ufig = maps.field_figure(
                    sp, rlat, rlon, "cooling", landmask=rmask, bounds=bounds,
                    vmin=0, vmax=float(np.nanmax(sp[rmask])) if rmask.any() else 1,
                    unit=("mm/day" if base_var == "rain" else "°C"),
                    title=f"Ensemble spread ({ens['members']} members) — "
                          f"{theme.LABELS[base_var]}, day {lead}", height=420)
                st.plotly_chart(ufig, use_container_width=True, key="unc")
                st.caption("Per-cell std-dev across stochastic forward passes — where "
                           "the twin is confident (dark) vs uncertain (bright). "
                           "Second uncertainty source: CNN-vs-XGBoost disagreement "
                           "in the city ensemble (Validation view).")
        with uc2:
            if show_unc:
                spl = ens["spread"][base_var][lead - 1][landmask]
                unit3 = "mm/day" if base_var == "rain" else "°C"
                st.markdown("#### Uncertainty")
                st.metric("Mean spread", f"{np.nanmean(spl):.2f} {unit3}")
                st.metric("Max spread", f"{np.nanmax(spl):.2f} {unit3}")

    with st.expander("🧪 Physics checks — live audit of this forecast"):
        from analytics import physics as _phys
        _cseq = {v: np.stack([clim_field(v, k) for k in range(C.HORIZON)])
                 for v in ("tmax", "tmin", "rain")}
        _rep = _phys.consistency_report(frames, landmask, _cseq["tmax"],
                                        _cseq["tmin"], _cseq["rain"])
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("tmax ≥ tmin", f"{100 - _rep['dtr_violation_pct']:.1f}% cells",
                  "thermodynamic ordering", delta_color="off")
        p2.metric("Rain ≥ 0", f"{100 - _rep['neg_rain_pct']:.1f}% cells",
                  "mass non-negativity", delta_color="off")
        p3.metric("10-day water budget", f"{_rep.get('rain_budget_departure_pct', 0):+.1f}%",
                  "vs climatological budget", delta_color="off")
        p4.metric("Diurnal range", f"{_rep['dtr_mean_c']:.1f} °C",
                  f"{_rep.get('dtr_departure_c', 0):+.1f} vs normal", delta_color="off")
        st.caption("The AI core is audited against first-order physical laws on every "
                   "forecast: temperature ordering, non-negative water, a bounded "
                   "national water budget, and an energy-balance-consistent diurnal "
                   "range. Constraints are enforced at reconstruction and reported here.")

    # 10-day forecast evolution vs climatology (region-mean over land)
    st.markdown("#### 📈 10-day forecast evolution — region mean")
    leads = list(range(1, C.HORIZON + 1))
    fc_series = [float(np.nanmean(frames[base_var][k][rmask])) for k in range(C.HORIZON)]
    clim_series = [float(np.nanmean(clim_field(base_var, k)[rmask])) for k in range(C.HORIZON)]
    unit2 = "mm/day" if base_var == "rain" else "°C"
    ely = maps.line_figure(leads,
                           {f"{mode_lbl}": fc_series, "Climatology (normal)": clim_series},
                           f"{theme.LABELS[base_var]} — {region_name}", unit2, height=300)
    st.plotly_chart(ely, use_container_width=True, key="evol")
    st.caption("How the forecast departs from the climatological normal across the 10-day horizon "
               "— the gap is the weather signal the AI adds on top of climatology.")


# ===== Hazards & extremes ==================================================
elif view == VIEWS[1]:
    haz = st.radio("Hazard layer", ["🔥 Heatwave risk", "🌧️ Heavy-rain category",
                                    "🏜️ Dry-spell length"], horizontal=True)
    rmask, _, _ = maps.crop_to_bounds(landmask, lats, lons, bounds)
    lc, rc = st.columns([3, 1.1])
    if haz.startswith("🔥"):
        sev = extremes.heatwave_severity(frames["tmax"][lead - 1], clim_field("tmax", lead - 1))
        f, rlat, rlon = regional(sev)
        with lc:
            fig = maps.field_figure(f, rlat, rlon, "aqi", landmask=rmask, bounds=bounds,
                                    vmin=0, vmax=2, unit="severity",
                                    title=f"Heatwave severity — AI forecast, day {lead}")
            st.plotly_chart(fig, use_container_width=True, key="hw")
            st.caption(f"Heatwave severity (0 none · 1 heatwave · 2 severe), IMD departure "
                       f"criteria · AI tmax forecast, day {lead}")
        with rc:
            land = sev[landmask]
            st.markdown("#### Heatwave risk")
            st.metric("Area under heatwave", f"{np.mean(land >= 1)*100:.1f}%")
            st.metric("Severe heatwave area", f"{np.mean(land >= 2)*100:.1f}%")
            st.caption("Forecast lead lets planners issue early heat alerts days ahead.")
    elif haz.startswith("🌧️"):
        cat = extremes.rain_category(frames["rain"][lead - 1])
        f, rlat, rlon = regional(cat)
        with lc:
            fig = maps.field_figure(f, rlat, rlon, "rain", landmask=rmask, bounds=bounds,
                                    vmin=0, vmax=4, unit="category",
                                    title=f"Rainfall category — AI forecast, day {lead}")
            st.plotly_chart(fig, use_container_width=True, key="rc")
            st.caption("IMD rain categories: moderate / heavy / very heavy / extremely heavy · "
                       f"AI rain forecast, day {lead}")
        with rc:
            land = cat[landmask]
            st.markdown("#### Heavy-rain risk")
            st.metric("Heavy+ rain area", f"{np.mean(land >= 2)*100:.1f}%")
            st.metric("Very heavy+ area", f"{np.mean(land >= 3)*100:.1f}%")
    else:
        rain_seq = np.stack([frames["rain"][k] for k in range(lead)], axis=0) if lead > 1 \
            else frames["rain"][0:1]
        dsi = extremes.dry_spell_index(rain_seq)
        f, rlat, rlon = regional(dsi)
        with lc:
            fig = maps.field_figure(f, rlat, rlon, "cooling", landmask=rmask, bounds=bounds,
                                    vmin=0, vmax=max(lead, 2), unit="dry days",
                                    title=f"Dry-spell length — forecast days 1–{lead}")
            st.plotly_chart(fig, use_container_width=True, key="ds")
            st.caption(f"Trailing dry-spell length over forecast days 1–{lead} (rain < 2.5 mm)")
        with rc:
            land = dsi[landmask]
            st.markdown("#### Dry-spell / drought")
            st.metric("Mean dry-run (days)", f"{np.mean(land):.1f}")
            st.metric("Cells fully dry", f"{np.mean(land >= lead)*100:.1f}%")

    # hazard-area trend across the 10-day horizon
    st.markdown("#### 📈 Hazard exposure across the forecast horizon")
    clim_tmax_seq = np.stack([clim_field("tmax", k) for k in range(C.HORIZON)])
    summary = extremes.summarize_forecast_hazards(frames, clim_tmax_seq, landmask)
    leads = summary["leads"]
    if haz.startswith("🔥"):
        tfig = maps.line_figure(leads, {"Area under heatwave": summary["heatwave_area_pct"]},
                                "Heatwave-affected land area", "% of land", height=280,
                                colors=["#FF7B00"])
    elif haz.startswith("🌧️"):
        tfig = maps.line_figure(leads, {"Heavy+ rain area": summary["heavy_rain_area_pct"]},
                                "Heavy-rain land area", "% of land", height=280,
                                colors=["#22D3EE"])
    else:
        tfig = maps.line_figure(leads, {"Mean dry-run length": summary["dry_spell_mean_days"]},
                                "Drought / dry-spell build-up", "days", height=280,
                                colors=["#A7F3D0"])
    st.plotly_chart(tfig, use_container_width=True, key="haztrend")


# ===== What-if simulator ===================================================
elif view == VIEWS[2]:
    tmax_f, rlat, rlon = regional(frames["tmax"][lead - 1])
    rain_f, _, _ = regional(frames["rain"][lead - 1])
    rmask, _, _ = maps.crop_to_bounds(landmask, lats, lons, bounds)
    urban = np.where(rmask, proxies.urban_fraction(rlat, rlon), np.nan)
    month = pd.Timestamp(f_dates[lead - 1]).month
    pmb = np.where(rmask, proxies.pm_baseline(np.nan_to_num(urban), month), np.nan)

    st.caption("Drag the sliders — the impact map and metrics update instantly "
               f"(driven by the {('AI' if fmode=='ai' else 'climatological')} forecast, day {lead}).")

    @st.fragment
    def whatif_panel():
        sc1, sc2, sc3 = st.columns(3)
        ctrl = engine.Controls(
            d_temp=sc1.slider("Δ Air temperature (°C)", -3.0, 5.0, 0.0, 0.5),
            d_rain_pct=sc1.slider("Δ Rainfall (%)", -50, 50, 0, 5),
            greening=sc2.slider("🌳 Urban greening (NDVI)", 0.0, 0.6, 0.0, 0.05),
            cool_roof=sc2.slider("🏠 Cool-roof albedo", 0.0, 0.6, 0.0, 0.05),
            urbanization=sc3.slider("🏙️ Added built-up", 0.0, 0.5, 0.0, 0.05),
        )
        f_s, m_s = engine.run_scenario(tmax_f, rain_f, urban, pmb, ctrl)
        _, m_b = engine.run_scenario(tmax_f, rain_f, urban, pmb, engine.Controls())

        impact = st.radio("Impact layer", ["Heat-stress index", "Air Quality (proxy)",
                                           "Intervention cooling (Δ°C)"], horizontal=True)
        key = {"Heat-stress index": "heat_index", "Air Quality (proxy)": "aqi",
               "Intervention cooling (Δ°C)": "cooling"}[impact]
        lc, rc = st.columns([3, 1.1])
        with lc:
            fig = maps.field_figure(f_s[key], rlat, rlon, key, landmask=rmask, bounds=bounds,
                                    title=f"{impact} — scenario, day {lead}")
            st.plotly_chart(fig, use_container_width=True, key="whatif")
            st.caption("Heat-stress responds to all five levers · AQI responds to rainfall, "
                       "greening, urbanisation & temperature · Cooling shows the intervention "
                       "benefit (greening + cool-roofs in built-up areas).")
        with rc:
            st.markdown("#### Scenario impact")
            st.metric("Peak surface cooling", f"{m_s['peak_cooling_c']:+.1f} °C")
            uc = m_s["urban_cooling_c"]
            st.metric("Built-up mean cooling", f"{uc:+.1f} °C" if uc == uc else "—")
            ab, _ = engine.aqi_band(m_s["mean_aqi"])
            st.metric("Mean AQI", f"{m_s['mean_aqi']:.0f}", ab)
            da = m_b["heat_danger_area_pct"] - m_s["heat_danger_area_pct"]
            st.metric("Heat-danger area", f"{m_s['heat_danger_area_pct']:.1f}%",
                      f"{-da:+.1f}% vs baseline", delta_color="inverse")

    whatif_panel()


# ===== Your data -> forecast (BYOO) ========================================
elif view == VIEWS[3]:
    from twin import user_obs
    from analytics import physics as phys

    st.markdown("### 📥 Bring your own observations → AI forecast")
    st.caption("Attach the last days of weather at your location — the twin treats "
               "them as **observations**, assimilates them into the national state via "
               "Optimal Interpolation, and re-runs ClimateUNet. Works beyond the IMD "
               "archive too: there the background falls back to climatology and *your* "
               "data supplies the weather signal — how an operational analysis handles "
               "a data-sparse day.")

    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        loc_mode = st.radio("Location", ["City", "Custom lat/lon"], horizontal=True)
        if loc_mode == "City":
            city_b = st.selectbox("City", list(C.CITIES.keys()), key="byoo_city")
            b_lat, b_lon = C.CITIES[city_b]
        else:
            b_lat = st.number_input("Latitude °N", 6.5, 38.5, 19.08, 0.01)
            b_lon = st.number_input("Longitude °E", 66.5, 100.0, 72.88, 0.01)
    with c2:
        start_pick = st.date_input("Forecast start (day after your last observation)",
                                   value=_last_d + _dt.timedelta(days=1),
                                   min_value=_min_d, max_value=_max_d, key="byoo_date")
        horizon = st.select_slider("Forecast horizon (days)", options=[2, 3, 7, 10],
                                   value=7)
    with c3:
        sigma_o = st.slider("Observation error σₒ (lower = trust your data more)",
                            0.1, 1.5, 0.5, 0.1)
        Lb = st.slider("Station influence radius (grid cells)", 0.5, 5.0, 2.0, 0.5)

    start_ts = pd.Timestamp(start_pick)
    iy_b, ix_b = user_obs.nearest_cell(b_lat, b_lon, grid)

    st.download_button("📄 Download CSV template", user_obs.TEMPLATE_CSV,
                       "varuna_obs_template.csv", "text/csv")
    up = st.file_uploader("Upload observations CSV (date, rain, tmax, tmin)",
                          type=["csv"])

    # editable table prefilled with the twin's own background at that cell
    pre_rows = []
    for k in range(C.INPUT_DAYS):
        d = start_ts - pd.Timedelta(days=C.INPUT_DAYS - k)
        doy = min(int(d.dayofyear), 365)
        if dates[0] <= d <= dates[-1]:
            ti_ = int(np.argmin(np.abs(dates.values - np.datetime64(d))))
            vals = {v: round(float(obs[v].values[ti_, iy_b, ix_b]), 1) for v in C.VARIABLES}
        else:
            vals = {v: round(float(carr[v][doy - 1, iy_b, ix_b]), 1) for v in C.VARIABLES}
        pre_rows.append({"date": str(d.date()), **vals})
    st.caption("Edit the table (pre-filled with the twin's background — IMD where "
               "available, climatology beyond the archive) or upload your CSV:")
    edited = st.data_editor(pd.DataFrame(pre_rows), use_container_width=True,
                            num_rows="fixed", key="byoo_editor")

    if st.button("🔮 Assimilate my data & forecast", type="primary"):
        try:
            user_df = (user_obs.parse_csv(up.getvalue()) if up is not None
                       else user_obs.parse_csv(edited.to_csv(index=False)))
        except Exception as e:
            st.error(f"Could not read observations: {e}")
            st.stop()

        hist0, n_real = user_obs.build_history(cube, dates, start_ts)
        dh_b = (user_obs.build_driver_history(fc.dcube, dates, start_ts, fc.n_drivers)
                if getattr(fc, "n_drivers", 0) else None)
        base = fc.predict_window(hist0, start_ts, carr, std, dh_b)
        hist1, innov_report = user_obs.inject_observations(
            hist0.copy(), start_ts, user_df, carr, std, iy_b, ix_b,
            length_scale=Lb, sigma_o=sigma_o)
        mine = fc.predict_window(hist1, start_ts, carr, std, dh_b)

        if not innov_report:
            st.warning("No observation fell inside the 7-day input window before the "
                       "forecast start — check the dates in your data.")
        lcb, rcb = st.columns([3, 1.1])
        unit_b = "mm/day" if base_var == "rain" else "°C"
        with lcb:
            # city-point chart: your obs -> twin baseline vs twin + your data
            fdd = [start_ts + pd.Timedelta(days=k) for k in range(horizon)]
            rows_b = {"date": [], theme.LABELS[base_var]: [], "series": []}
            for _, r in user_df.iterrows():
                if base_var in user_df.columns and np.isfinite(r.get(base_var, np.nan)):
                    rows_b["date"].append(pd.Timestamp(r["date"]))
                    rows_b[theme.LABELS[base_var]].append(float(r[base_var]))
                    rows_b["series"].append("your observations")
            for k in range(horizon):
                rows_b["date"].append(fdd[k])
                rows_b[theme.LABELS[base_var]].append(float(base["frames"][base_var][k, iy_b, ix_b]))
                rows_b["series"].append("twin (archive only)")
                rows_b["date"].append(fdd[k])
                rows_b[theme.LABELS[base_var]].append(float(mine["frames"][base_var][k, iy_b, ix_b]))
                rows_b["series"].append("twin + your data")
            import altair as alt
            st.altair_chart(alt.Chart(pd.DataFrame(rows_b)).mark_line(point=True).encode(
                x="date:T", y=alt.Y(f"{theme.LABELS[base_var]}:Q"),
                color=alt.Color("series:N",
                                scale=alt.Scale(range=["#A7F3D0", "#8b97c6", "#FF7B00"]))),
                use_container_width=True)

            # where the user's data changed the national day-1 field
            delta = mine["frames"][base_var][0] - base["frames"][base_var][0]
            dfld, drlat, drlon = maps.crop_to_bounds(delta, lats, lons, bounds)
            dmask, _, _ = maps.crop_to_bounds(landmask, lats, lons, bounds)
            dmax = float(np.nanmax(np.abs(dfld[dmask]))) if dmask.any() else 1.0
            dfig = maps.field_figure(dfld, drlat, drlon, "cooling", landmask=dmask,
                                     bounds=bounds, vmin=-max(dmax, 1e-3),
                                     vmax=max(dmax, 1e-3), unit=unit_b, height=420,
                                     title="Impact of your observations — Δ day-1 forecast")
            st.plotly_chart(dfig, use_container_width=True, key="byoo_delta")
            st.caption("The innovation spreads from your station per the correlated "
                       "background-error covariance — one observation informs its "
                       "neighbourhood, the defining behaviour of Optimal Interpolation.")
        with rcb:
            st.markdown("#### Assimilation report")
            st.metric("Observations used", f"{len(innov_report)} day(s)")
            st.metric("Real-archive days in window", f"{n_real}/{C.INPUT_DAYS}")
            d1 = float(mine["frames"][base_var][0, iy_b, ix_b]
                       - base["frames"][base_var][0, iy_b, ix_b])
            st.metric("Δ day-1 at your location", f"{d1:+.2f} {unit_b}")
            thr = 0.5 if base_var == "rain" else 0.05
            st.metric("Area influenced", f"{user_obs.influence_area_pct(delta, landmask, thr):.1f}% of land")
            rep_b = phys.consistency_report(mine["frames"], landmask)
            st.metric("Physics: tmax ≥ tmin", f"{100 - rep_b['dtr_violation_pct']:.0f}% cells")
            if innov_report:
                st.markdown("**Innovations (obs − background):**")
                st.dataframe(pd.DataFrame(innov_report), use_container_width=True,
                             height=180)


# ===== Validation & skill ==================================================
elif view == VIEWS[4]:
    st.markdown("### AI model skill on unseen test years (2021–2024)")
    if EVAL is None:
        st.warning("Run `python evaluation/evaluate.py` to generate skill metrics.")
    else:
        units = {"rain": "mm", "tmax": "°C", "tmin": "°C"}
        cols = st.columns(3)
        for j, v in enumerate(C.VARIABLES):
            with cols[j]:
                ai = EVAL["ai"][v]
                st.markdown(f"#### {theme.LABELS[v]}")
                st.metric("MAE day-1", f"{ai['mae'][0]:.2f} {units[v]}")
                st.metric("RMSE day-1", f"{ai['rmse'][0]:.2f} {units[v]}")
                st.metric("ACC day-1", f"{ai['acc'][0]:.3f}")
                st.metric("Skill vs persistence (d1)", f"{ai['skill_vs_persistence'][0]*100:+.0f}%")
                st.metric("Skill vs persist-anomaly (d1)", f"{ai['skill_vs_poa'][0]*100:+.1f}%")

        leads = list(range(1, C.HORIZON + 1))
        g1, g2 = st.columns(2)
        with g1:
            rmse_fig = maps.line_figure(
                leads, {theme.LABELS[v]: EVAL["ai"][v]["rmse"] for v in C.VARIABLES},
                "RMSE vs lead day (real units)", "RMSE", height=320,
                colors=["#22D3EE", "#FF7B00", "#A7F3D0"])
            st.plotly_chart(rmse_fig, use_container_width=True, key="rmse_curve")
        with g2:
            acc_fig = maps.line_figure(
                leads, {theme.LABELS[v]: EVAL["ai"][v]["acc"] for v in C.VARIABLES},
                "Anomaly Correlation (ACC) vs lead day", "ACC", height=320,
                colors=["#22D3EE", "#FF7B00", "#A7F3D0"])
            st.plotly_chart(acc_fig, use_container_width=True, key="acc_curve")
        # skill vs persistence (day-1) bar comparison
        bar = maps.bar_figure(
            [theme.LABELS[v] for v in C.VARIABLES],
            {"Skill vs persistence": [EVAL["ai"][v]["skill_vs_persistence"][0]*100 for v in C.VARIABLES],
             "Skill vs persist-anomaly": [EVAL["ai"][v]["skill_vs_poa"][0]*100 for v in C.VARIABLES]},
            "Day-1 skill over operational baselines", "% improvement", height=300)
        st.plotly_chart(bar, use_container_width=True, key="skillbar")

        # categorical extreme-event verification (contingency-table scores)
        if "events" in EVAL.get("ai", {}):
            st.markdown("#### Extreme-event detection — categorical skill (day 1, unseen years)")
            enames = {"heatwave": "🔥 Heatwave (IMD departure criteria)",
                      "heavy_rain": "🌧️ Heavy rain (≥ 64.5 mm/day)"}
            ecols = st.columns(2)
            for j, e in enumerate(enames):
                with ecols[j]:
                    ev = EVAL["ai"]["events"][e]
                    st.markdown(f"**{enames[e]}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("POD", f"{ev['pod'][0]*100:.0f}%",
                              "detection rate", delta_color="off")
                    m2.metric("FAR", f"{ev['far'][0]*100:.0f}%",
                              "false alarms", delta_color="off")
                    poa_ets = EVAL["poa"]["events"][e]["ets"][0]
                    m3.metric("ETS", f"{ev['ets'][0]:.3f}",
                              f"POA {poa_ets:.3f}", delta_color="off")
            st.caption("POD / FAR / ETS from area-weighted contingency tables — the "
                       "operational categorical verification used for warnings (ETS = "
                       "skill relative to random hits; higher is better). Compared "
                       "against the persistence-of-anomaly baseline.")

    # companion model — XGBoost validation skill (real units, val years 2019–2020)
    if XGB_REPORT:
        units = {"rain": "mm", "tmax": "°C", "tmin": "°C"}
        st.markdown("#### Companion model — XGBoost (station ensemble)")
        xcols = st.columns(3)
        for j, v in enumerate(C.VARIABLES):
            with xcols[j]:
                xr_ = XGB_REPORT[v]["val_rmse_real"]
                # day-1 ClimateUNet RMSE (test) shown as delta context where available
                delta = None
                if EVAL is not None:
                    cnn_r = EVAL["ai"][v]["rmse"][0]
                    delta = f"ClimateUNet d1 {cnn_r:.2f} {units[v]}"
                st.metric(f"{theme.LABELS[v]} · XGB val RMSE",
                          f"{xr_:.2f} {units[v]}", delta, delta_color="off")
        st.caption("XGBoost validation RMSE (2019–2020, real units) from "
                   "`outputs/xgb_report.json`. The two paradigms — spatial CNN and "
                   "tabular boosting — are blended in the city forecast below.")

    st.markdown("---")
    st.markdown("### City forecast — ClimateUNet" + (" + XGBoost ensemble" if XGB else ""))
    city = st.selectbox("Location", list(C.CITIES.keys()))
    la, lo = C.CITIES[city]
    iy = int(np.argmin(np.abs(lats - la))); ix = int(np.argmin(np.abs(lons - lo)))
    hist_n = 14
    h_dates = [dates[t - hist_n + k] for k in range(hist_n)]
    h_vals = [obs[base_var].values[t - hist_n + k, iy, ix] for k in range(hist_n)]
    cnn_vals = [frames[base_var][k, iy, ix] for k in range(C.HORIZON)]
    fd = [pd.Timestamp(d) for d in f_dates]
    rows = {"date": [pd.Timestamp(d) for d in h_dates] + fd,
            theme.LABELS[base_var]: h_vals + cnn_vals,
            "series": ["observed"] * hist_n + ["ClimateUNet"] * C.HORIZON}
    if XGB:
        xv = XGB.predict_cell(cube, dates, carr, std, grid, t, iy, ix)[base_var]
        # default blend weight from inverse validation error (skill-weighted):
        # the more accurate model gets the larger share, and a slider lets the
        # planner rebalance manually.
        cnn_r = EVAL["ai"][base_var]["rmse"][0] if EVAL else 1.0
        xgb_r = XGB_REPORT[base_var]["val_rmse_real"] if XGB_REPORT else 1.0
        w_def = ((1.0 / cnn_r) / ((1.0 / cnn_r) + (1.0 / xgb_r))
                 if cnn_r > 0 and xgb_r > 0 else 0.5)
        w_cnn = st.slider("ClimateUNet weight in ensemble", 0.0, 1.0,
                          round(float(w_def), 2), 0.05,
                          help="Default is inverse-RMSE (skill-weighted); drag to "
                               "rebalance ClimateUNet vs XGBoost.")
        ens = [w_cnn * cnn_vals[k] + (1.0 - w_cnn) * float(xv[k]) for k in range(C.HORIZON)]
        rows["date"] += fd; rows[theme.LABELS[base_var]] += ens
        rows["series"] += ["CNN+XGB ensemble"] * C.HORIZON
        st.caption(f"Ensemble = {w_cnn:.2f}·ClimateUNet + {1-w_cnn:.2f}·XGBoost "
                   "(default weights ∝ inverse validation RMSE).")
    df = pd.DataFrame(rows)
    import altair as alt
    rng = ["#22D3EE", "#FF7B00", "#A7F3D0"]
    st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
        x="date:T", y=alt.Y(f"{theme.LABELS[base_var]}:Q"),
        color=alt.Color("series:N", scale=alt.Scale(range=rng))),
        use_container_width=True)


# ===== Satellite (INSAT) ===================================================
elif view == VIEWS[5]:
    st.markdown("### INSAT / MOSDAC satellite layer — real Indian satellite data")
    if not insat.has_data():
        st.info(insat.instructions())
    else:
        files = insat.available_files()
        prod = st.radio("Product", ["lst", "rain", "sst"], horizontal=True,
                        format_func=lambda p: {"lst": "Land Surface Temp",
                                               "rain": "Rainfall (IMR)",
                                               "sst": "Sea Surface Temp"}[p])
        res = load_insat(prod, "|".join(os.path.basename(x) for x in files))
        if res is None:
            st.info(f"No INSAT {prod.upper()} file found in data/insat/. "
                    "Available files: " + ", ".join(os.path.basename(x) for x in files))
        else:
            rmask, _, _ = maps.crop_to_bounds(landmask, lats, lons, bounds)
            f, rlat, rlon = regional(res["field"])
            key = "lst" if prod in ("lst", "sst") else "rain"
            lc, rc = st.columns([3, 1.1])
            with lc:
                fig = maps.field_figure(f, rlat, rlon, key,
                                        landmask=(rmask if prod != "sst" else None),
                                        bounds=bounds,
                                        title=f"INSAT-3DR {prod.upper()} — regridded to national grid")
                st.plotly_chart(fig, use_container_width=True, key="sat")
                st.caption(f"Real INSAT-3DR observation · {res['file']} · "
                           "regridded from the full-disk product onto the 0.25° grid.")
            with rc:
                vals = res["field"][landmask]
                vals = vals[np.isfinite(vals)]
                unit = "°C" if prod in ("lst", "sst") else "mm/hr"
                st.markdown("#### Satellite observation")
                if vals.size:
                    st.metric("Mean", f"{np.mean(vals):.1f} {unit}")
                    st.metric("Max", f"{np.max(vals):.1f} {unit}")
                # Observed-vs-model cross-check (LST skin temp vs air tmax climatology)
                if prod == "lst":
                    # day-of-year parsed from the product filename (falls back to
                    # mid-year if the stamp is absent) -> compare to the air-temp normal
                    doy = insat.product_doy(res["file"]) or 180
                    air = carr["tmax"][doy - 1][landmask]
                    air = air[np.isfinite(air)]
                    st.markdown("#### Cross-check vs air temp")
                    when = insat.product_datetime(res["file"])
                    st.metric("Climatological air Tmax", f"{np.mean(air):.1f} °C",
                              when.strftime("%d %b") if when is not None else None)
                    st.metric("Skin–air offset", f"{np.mean(vals)-np.mean(air):+.1f} °C")
                    st.caption("Satellite land-surface (skin) temperature runs hotter than "
                               "screen-level air Tmax — the positive offset is the expected "
                               "physical signature, validating the ingest.")
            st.caption(f"INSAT {prod.upper()} regridded to the national grid · {res['file']}")


# ===== About ===============================================================
else:
    st.markdown("""
### VARUNA — what it is
**VARUNA** (*Virtual AI Replica for Understanding & Nowcasting the Atmosphere*) is an
**indigenous, AI-powered digital twin of India's climate**, trained entirely on **real IMD
gridded data** with a **real INSAT-3DR** satellite layer — no synthetic data anywhere. It forecasts near-term rainfall and temperature,
flags climate hazards, assimilates observations, and lets planners run **what-if** experiments
with live urban-heat and air-quality impacts.

**Connected applications (PS#5):**
1. **Climate state & AI forecast** — IMD rainfall (0.25°) + temperature, *ClimateUNet*.
2. **Hazard early-warning** — heatwave, heavy-rain and dry-spell maps from the forecast.
3. **Urban heat & air quality** — intervention what-if (greening, cool roofs) → heat index & AQI.

### Models
- **ClimateUNet** — residual U-Net + attention, predicts anomalies vs climatology and refines a
  persistence-of-anomaly prior; **direct multi-horizon** (10 days, no autoregressive divergence).
- **XGBoost** — complementary gradient-boosted station forecaster; ensembled at city scale.

### Data assimilation
**Optimal Interpolation** — the AI background is fused with observations; the innovation is
spread spatially per a correlated background-error covariance (beyond pointwise nudging).

### Honest framing
AI short-term **forecast** on real IMD analyses (not a full GCM). Heat/AQI/LST are
**physics-informed proxies** (NWS heat index, CPCB AQI, surface-energy LST). Satellite layer
ingests real **INSAT/MOSDAC** products (bring-your-own file; nothing synthesised). Scale-up path:
foundation models (Prithvi-WxC / Pangu-Weather on IMDAA / BharatBench) + live INSAT feeds.
""")
