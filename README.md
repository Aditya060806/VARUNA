<div align="center">

# 🛰️ VARUNA
### Virtual AI Replica for Understanding & Nowcasting the Atmosphere
**An AI‑Powered Digital Twin of India's Climate — built entirely on India's national data.**

*ISRO Bharatiya Antariksh Hackathon (BAH) 2026 · Problem Statement #5*

[![Data](https://img.shields.io/badge/Data-IMD%20%2B%20INSAT--3DR-2DD4BF)](#data-sources)
[![Model](https://img.shields.io/badge/Model-ClimateUNet%20%2B%20XGBoost-FF7B00)](#the-ai-model--climateunet)
[![Framework](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C)](#technology-stack)
[![Coverage](https://img.shields.io/badge/Coverage-National%200.25%C2%B0-1E5BFF)](#data-sources)
[![Data Policy](https://img.shields.io/badge/Synthetic%20data-NONE-2DC937)](#honest-framing)

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-FFD21E)](https://huggingface.co/spaces/Aditya1002/VARUNA)
[![Status](https://img.shields.io/badge/Status-Live-2DC937)](https://huggingface.co/spaces/Aditya1002/VARUNA)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#technology-stack)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)](#the-dashboard)
[![License](https://img.shields.io/badge/License-MIT-blue)](#citations--license)
[![Reproducible](https://img.shields.io/badge/Pipeline-prepare%E2%86%92train%E2%86%92eval%E2%86%92serve-7E57C2)](#quickstart--reproduce)
[![CI](https://github.com/Aditya060806/VARUNA/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditya060806/VARUNA/actions/workflows/ci.yml)

*Real data only · trained on‑device · reproducible end‑to‑end · nothing fake, nothing static.*

</div>

![VARUNA dashboard — AI Digital Twin of India's Climate](assets/ui-1.png)

<div align="center"><sub>VARUNA dashboard — national AI forecast over India, rendered live from real IMD + INSAT data.</sub></div>

---

> **VARUNA** is a dynamic, high‑fidelity virtual replica of India's climate system. It continuously
> ingests real **IMD** ground observations and **INSAT‑3DR** satellite data, predicts near‑term
> **rainfall and temperature** with a GPU‑trained deep model, fuses observations through **data
> assimilation**, and lets a planner run **what‑if** scenarios whose impacts on **urban heat** and
> **air quality** are seen live on an interactive map. *Named after Varuṇa — the Vedic deity of the
> sky, waters and cosmic order.*

### Why VARUNA matters
India's monsoon, heatwaves and droughts directly shape the lives of over 1.4 billion people, yet
actionable climate intelligence remains fragmented across agencies and formats. VARUNA unifies the
nation's own datasets into **one living, queryable twin** — *observe → forecast → assimilate →
simulate* — so a planner can see today's climate state, a 10‑day outlook, the emerging hazards,
and the downstream effect of an intervention **before spending a single rupee**. It is built to be
**Atmanirbhar** (self‑reliant): every byte of data and every model weight is Indian and on‑device.

> **Repository:** https://github.com/Aditya060806/VARUNA · **Live demo:** https://huggingface.co/spaces/Aditya1002/VARUNA (Hugging Face Space, Docker).

### The 60‑second pitch
> Point VARUNA at any day from **1981 to 2026** and it reconstructs India's climate state from real
> IMD observations, fires a **10‑day rain + temperature forecast** from a GPU‑trained deep network,
> blends in **INSAT‑3DR satellite** truth through **optimal‑interpolation assimilation**, flags the
> **heatwaves, heavy‑rain and dry‑spells** hiding in that forecast, and then lets you pull a lever —
> *"+2 °C, greener city, cool roofs"* — and watch **urban heat‑stress and air‑quality** maps redraw in
> real time. One model, four jobs, **zero synthetic data**, all on a laptop GPU.

> **Who is it for?** City planners sizing cooling shelters, disaster cells watching for heat/rain
> hazards, water managers tracking dry spells, and researchers needing a reproducible, India‑native
> climate sandbox — anyone who needs to *see the consequence of a decision before making it.*

---

## Table of Contents
1. [Highlights](#highlights)
2. [Problem Statement Alignment](#problem-statement-alignment)
3. [System Architecture](#system-architecture)
4. [Data Sources](#data-sources)
5. [Data Pipeline](#data-pipeline)
6. [The AI Model — ClimateUNet](#the-ai-model--climateunet)
7. [What Makes This Technically Hard](#what-makes-this-technically-hard)
8. [Companion Model — XGBoost](#companion-model--xgboost)
9. [Training Configuration](#training-configuration)
10. [Results & Validation](#results--validation)
11. [Digital‑Twin Assimilation](#digital-twin-assimilation)
12. [Connected Applications](#connected-applications)
13. [Satellite Integration (INSAT/MOSDAC)](#satellite-integration-insatmosdac)
14. [The Dashboard](#the-dashboard)
15. [A Walk Through the Twin](#a-walk-through-the-twin)
16. [Technology Stack](#technology-stack)
17. [Repository Structure](#repository-structure)
18. [Quickstart & Reproduce](#quickstart--reproduce)
19. [Deployment](#deployment)
20. [Honest Framing](#honest-framing)
21. [Scale‑up Roadmap](#scale-up-roadmap)
22. [Evaluation‑Parameter Map](#evaluation-parameter-map)
23. [FAQ](#faq)
24. [Glossary](#glossary)
25. [Citations & License](#citations--license)
26. [Acknowledgements](#acknowledgements)
27. [Team Vandalizers](#team-vandalizers)

---

## Highlights

| | Capability |
|---|---|
| 🇮🇳 | **100% national data** — IMD gridded rainfall + temperature, INSAT‑3DR satellite. No synthetic data anywhere. |
| 🧠 | **ClimateUNet** — a residual, attention‑based spatiotemporal CNN that forecasts **10 days** of rain/tmax/tmin in one pass. |
| 📈 | **Validated skill** — temperature MAE **0.6–0.8 °C**, anomaly correlation **0.76**, beats persistence by **6–28 %** on unseen 2021–2024 data. |
| 🌪️ | **Hazard early‑warning** — heatwave, heavy‑rain and dry‑spell maps straight from the forecast. |
| 🔭 | **Real satellite fusion** — INSAT‑3DR LST ingested, regridded, cross‑checked against the model. |
| 🧪 | **What‑if simulator** — temperature/rainfall/greening/cool‑roof levers → live urban‑heat & air‑quality impact. |
| 🛰️ | **Optimal‑Interpolation assimilation** — model state fused with observations (beyond naive nudging). |
| 🗺️ | **Interactive dashboard** — six views, dark "orbital" theme, instant client‑side maps. |
| ♻️ | **Fully reproducible** — `prepare → train → evaluate → serve`, all in one repo on a single GPU. |

### At a Glance

| | |
|---|---|
| **Domain** | All‑India, 0.25° grid (129 × 135), 4,964 land cells |
| **Data span** | 1981–2024 · **16,071 daily fields** · 3 IMD variables + INSAT‑3DR LST |
| **Model** | ClimateUNet — **7.42 M** params, residual U‑Net + attention, 10‑day direct horizon |
| **Trained on** | NVIDIA RTX 4050 (6 GB), PyTorch + CUDA, mixed precision |
| **Day‑1 accuracy** | Tmax **MAE 0.82 °C**, Tmin **0.59 °C**, Rain **3.16 mm** · ACC **0.76** |
| **Skill** | Beats persistence by **6–28 %**; beats persistence‑of‑anomaly on temperature |
| **Synthetic data** | **None** — 100 % real national datasets |
| **Deploy** | **Live** on Hugging Face Spaces (Docker) · Streamlit dashboard |

---

## Problem Statement Alignment

PS#5 asks for *"a high‑fidelity, dynamic virtual replica of India's climate… integrating multi‑source
national data… leveraging AI/ML and data assimilation… with applications for climate‑sensitive
sectors and a what‑if interface."* VARUNA implements every clause:

| PS requirement | VARUNA delivery |
|---|---|
| Multi‑source national data | **IMD** (rain 0.25°, tmax/tmin 1°) + **INSAT‑3DR** (LST) via MOSDAC |
| AI/ML short‑term prediction | **ClimateUNet** (deep CNN) + **XGBoost** ensemble |
| Data assimilation | **Optimal Interpolation** (correlated background‑error covariance) |
| Continuously evolving climate state | Forecast → assimilation → state loop on the national grid |
| Applications for climate‑sensitive sectors | **Urban heat** (NWS heat index, LST/UHI) + **air quality** (CPCB AQI) |
| What‑if simulation | Live scenario sliders → instant impact maps |
| Interactive geospatial dashboard | Streamlit + Plotly, national + pilot, six views |
| Scalable national framework | National 0.25° grid + documented foundation‑model path |
| Extreme phenomena (monsoon, heat, drought) | Heatwave / heavy‑rain / dry‑spell hazard layers |

---

## System Architecture

```mermaid
flowchart LR
    subgraph SRC["① National Data Sources"]
        A1["IMD Rainfall 0.25°<br/>1981–2024"]
        A2["IMD Max/Min Temp 1.0°<br/>1981–2024"]
        A3["INSAT-3DR LST<br/>MOSDAC L2B"]
    end
    subgraph PRE["② Pre-processing"]
        B1["imdlib ingest"]
        B2["Regrid temp → 0.25°"]
        B3["Day-of-year climatology<br/>(train years only)"]
        B4["Anomaly cube + land mask"]
    end
    subgraph MODEL["③ AI Models"]
        C1["ClimateUNet<br/>residual U-Net + attention"]
        C2["XGBoost<br/>station ensemble"]
    end
    subgraph TWIN["④ Digital Twin"]
        D1["10-day forecast"]
        D2["Optimal-Interpolation<br/>assimilation"]
        D3["Hazard analytics"]
    end
    subgraph APP["⑤ Applications & UI"]
        E1["Climate Twin map"]
        E2["What-if: Heat & AQI"]
        E3["Validation & Skill"]
        E4["Satellite layer"]
    end
    A1 & A2 & A3 --> B1 --> B2 --> B3 --> B4
    B4 --> C1 & C2
    C1 --> D1 --> D2 --> D3
    A3 -. observation .-> D2
    D1 & D2 & D3 --> E1 & E2 & E3 & E4
```

**Data‑flow tensor shapes**

```
IMD daily grids ─► obs cube (16071, 129, 135, 3)  [rain, tmax, tmin]
        │ subtract smoothed climatology, ÷ anomaly-std
        ▼
anomaly cube (16071, 129, 135, 3)
        │ window: 7 history days + persistence-of-anomaly prior
        ▼
ClimateUNet input  (N, 51, 129, 135)   →   output (N, 30, 129, 135)
        │ reshape + add climatology + ReLU(rain)
        ▼
forecast frames {rain,tmax,tmin} × 10 lead days  (real units)
```

---

## Data Sources

All data is **real** and **national**. Ground data via [`imdlib`](https://imdlib.readthedocs.io/),
satellite via [MOSDAC](https://www.mosdac.gov.in/).

| Parameter | Product | Native grid | Period used | Role |
|---|---|---|---|---|
| Rainfall | IMD Gridded 0.25° | 135 × 129 (lon×lat) | 1981–2024 | model + validation |
| Max temperature | IMD 1.0° | 31 × 31 → regridded 0.25° | 1981–2024 | model + validation |
| Min temperature | IMD 1.0° | 31 × 31 → regridded 0.25° | 1981–2024 | model + validation |
| Land Surface Temp | INSAT‑3DR `3RIMG_L2B_LST` | full‑disk → regridded 0.25° | 2026 snapshot | satellite layer + cross‑check |
| (ready) Rainfall | INSAT‑3DR `3RIMG_L2B_IMC` | full‑disk | — | drop‑in supported |
| (ready) Sea Surface Temp | INSAT‑3DR `3RIMG_L2B_SST` | full‑disk | — | drop‑in supported |

**Coverage:** national bounding box 66.5–100.0 °E, 6.5–38.5 °N · **4,964 land cells** on the 129×135 grid.

**Chronological split (no leakage):**

| Split | Years | Daily windows |
|---|---|---|
| Train | 1981–2018 (38 yr) | 13,872 |
| Validation | 2019–2020 | 731 |
| Test (unseen) | 2021–2024 | 1,452 |

---

## Data Pipeline

Implemented in `data/` — robust, cached, reproducible:

1. **Ingest** (`download_imd.py`) — pulls IMD rain/tmax/tmin for 1981–2024 via `imdlib`, masks fill sentinels (`-999, 99.9, -99.9`) to NaN.
2. **Regrid** (`prepare.py`) — bilinear‑interpolates 1° temperature onto the 0.25° rainfall grid → one common national grid; derives a **land mask** from rainfall validity.
3. **Climatology** (`climatology.py`) — smoothed **day‑of‑year** mean (21‑day circular window) built **from training years only** → no leakage.
4. **Anomalies** — `anomaly = (obs − climatology) / anomaly_std`; per‑variable std computed on train years.
5. **Cache** — compact NetCDF + a memory‑mapped anomaly cube for fast startup.

**Anomaly normalisation statistics (train years):**

| Variable | Anomaly σ | Units |
|---|---|---|
| Rainfall | 9.93 | mm/day |
| Max temp | 2.03 | °C |
| Min temp | 1.60 | °C |

---

## The AI Model — ClimateUNet

A modern, fully‑convolutional spatiotemporal forecaster engineered for **stability** and **skill**.

### Design principles
- **Anomaly forecasting** — predicts deviations from climatology, not raw fields → rollouts relax toward climate instead of diverging.
- **Residual‑over‑baseline** — the network is *given* a **persistence‑of‑anomaly (POA)** prior and learns only the **correction** (additive skip, zero‑initialised head). It therefore **matches‑or‑beats** a strong operational baseline by construction.
- **Direct multi‑horizon** — emits all **10 lead days** in a single forward pass → **no autoregressive feedback loop** (the divergence failure mode of naive ConvLSTMs).
- **Physics constraints** — rainfall reconstructed non‑negative; latitude‑area‑weighted, land‑masked loss.

### Architecture

```mermaid
flowchart TD
    I["Input (51, 129, 135)<br/>7 history days × 3 vars (21)<br/>+ POA prior 10 days × 3 vars (30)"]
    S["Stem Conv 3×3 → 48"]
    E1["ResBlock 48 + SE"]
    E2["ResBlock 96 + SE ↓"]
    E3["ResBlock 192 + SE ↓"]
    B["Bottleneck 384 ↓<br/>Dropout 0.2"]
    D3["Up + Skip → ResBlock 192"]
    D2["Up + Skip → ResBlock 96"]
    D1["Up + Skip → ResBlock 48"]
    H["Head Conv 1×1 → 30"]
    O["Output = residual + POA prior<br/>(30, 129, 135)"]
    I-->S-->E1-->E2-->E3-->B-->D3-->D2-->D1-->H-->O
```

| Property | Value |
|---|---|
| Type | Residual U‑Net + Squeeze‑Excite channel attention |
| Parameters | **7,424,898** |
| Input | `(51, 129, 135)` — history (21) + POA prior (30) |
| Output | `(30, 129, 135)` — 10 lead days × 3 variables (anomalies) |
| Normalisation | GroupNorm · activation GELU |
| Framework | PyTorch 2.5 + CUDA 12.1 (mixed precision) |
| Hardware | NVIDIA RTX 4050 (6 GB) |

---

## What Makes This Technically Hard

Climate forecasting is not "just another image task." These are the traps that sink naive models —
and the concrete design choice in VARUNA that defuses each one.

| Challenge | Why it breaks naive models | VARUNA's answer |
|---|---|---|
| **Autoregressive divergence** | Feeding a model its own output compounds error; rollouts blow up after ~6 days. | **Direct multi‑horizon** — all 10 lead days emitted in one forward pass, no feedback loop. |
| **Non‑stationary fields** | Raw rain/temperature have huge seasonal swings the network wastes capacity learning. | Forecast **anomalies** vs a smoothed day‑of‑year climatology; predictions relax toward climate, never explode. |
| **Beating a strong baseline** | Persistence‑of‑anomaly is already hard to beat; many ML models quietly lose to it. | **Residual learning** — the POA prior is fed in and added back via a zero‑init head, so the net only learns the *correction*. It matches‑or‑beats POA by construction. |
| **Class imbalance in rain** | Most cells are dry most days; MSE collapses to "predict zero." | **Huber loss** + non‑negativity reconstruction + anomaly normalisation keep rare wet events in the gradient. |
| **Area distortion of a lat/lon grid** | Equal‑pixel loss over‑weights the far north. | **Latitude‑area‑weighted, land‑masked** loss — every km² counts equally, oceans ignored. |
| **Small‑GPU memory** | 6 GB can't hold large climate transformers. | A compact **7.4 M‑param** residual U‑Net + **mixed‑precision (AMP)** — trains comfortably on an RTX 4050. |
| **Data leakage** | Climatology/normalisation computed over all years inflates scores. | Climatology and σ use **training years only (1981–2018)**; test years 2021–2024 are fully unseen. |

> The net effect: a model that is **stable by design**, **honest by construction**, and **beats the
> operational persistence baseline at every lead day** — on four years it never saw.

---

## Companion Model — XGBoost

A second, complementary paradigm for **station‑level precision**, ensembled with ClimateUNet at city scale.

- **Features (per cell, per lead):** 7‑day lagged anomalies (×3 vars), 7‑day mean & trend, the POA prior, day‑of‑year harmonics, lat/lon, lead.
- **Training:** 3 GPU‑accelerated regressors (one per variable), ~3 M rows, early stopping.
- **Use:** the dashboard's city forecast shows **observed → ClimateUNet → CNN+XGB ensemble**.

| Variable | XGBoost val RMSE (real units) |
|---|---|
| Rainfall | 10.65 mm |
| Max temp | 1.98 °C |
| Min temp | 1.46 °C |

---

## Training Configuration

| Hyper‑parameter | Value |
|---|---|
| Optimiser | AdamW (lr 2.5e‑3, weight‑decay 2e‑3) |
| Schedule | Cosine annealing, 60 epochs |
| Batch size | 32 |
| Loss | Latitude‑area‑weighted, land‑masked **Huber** (on scaled anomalies) |
| Regularisation | Dropout 0.2 · input‑noise augmentation 0.12 · early stopping (patience 12) |
| Precision | Automatic Mixed Precision (AMP) |
| Best checkpoint | epoch 9 (early‑stopped at 21) · `models/checkpoints/climate_unet.pt` |
| Train windows | 13,872 (stride‑3 over 1981–2018) |

---

## Results & Validation

Evaluated on the **held‑out test years 2021–2024** (1,452 daily forecasts the model never saw),
land‑masked and **latitude‑area‑weighted**, in **real‑world units**. Metrics follow operational NWP
practice: **RMSE, MAE, Anomaly Correlation Coefficient (ACC)**, and **skill vs reference baselines**.

### Headline — day‑1 accuracy

| Variable | MAE | RMSE | ACC | Skill vs persistence | Skill vs persistence‑of‑anomaly |
|---|---|---|---|---|---|
| **Max temp** | **0.82 °C** | 1.20 °C | **0.759** | +6 % | +1.6 % |
| **Min temp** | **0.59 °C** | 0.84 °C | **0.762** | +14 % | **+10.1 %** |
| **Rainfall** | **3.16 mm** | 9.95 mm | 0.399 | +16 % | +0.6 % |

> *ACC > 0.6 is the accepted threshold for "useful" forecast skill — VARUNA reaches **0.76** for temperature.*

### Skill vs lead day (RMSE, real units)

| Lead (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Rain (mm)** | 9.95 | 10.48 | 10.57 | 10.59 | 10.63 | 10.64 | 10.64 | 10.65 | 10.63 | 10.61 |
| **Tmax (°C)** | 1.20 | 1.61 | 1.81 | 1.93 | 1.99 | 2.03 | 2.05 | 2.07 | 2.09 | 2.10 |
| **Tmin (°C)** | 0.84 | 1.12 | 1.30 | 1.41 | 1.47 | 1.52 | 1.55 | 1.57 | 1.58 | 1.59 |

### Anomaly Correlation (ACC) vs lead day

| Lead (days) | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| **Rain** | 0.40 | 0.25 | 0.21 | 0.18 | 0.17 | 0.16 |
| **Tmax** | 0.76 | 0.58 | 0.45 | 0.32 | 0.27 | 0.22 |
| **Tmin** | 0.76 | 0.60 | 0.48 | 0.37 | 0.31 | 0.27 |

### Skill over plain persistence (%) — grows with lead time

| Lead (days) | 1 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| **Rain** | +16 | +24 | +26 | +27 | +28 |
| **Tmax** | +6 | +15 | +20 | +23 | +27 |
| **Tmin** | +14 | +21 | +23 | +25 | +28 |

**Benchmarked against three references** (`models/baseline.py`): persistence, persistence‑of‑anomaly,
and climatology. Full machine‑readable results in `outputs/eval_metrics.json`; skill curves in
`outputs/skill_curves.png`.

### How to read these numbers
- **MAE / RMSE** — average error in real units (°C, mm/day); lower is better. RMSE punishes big misses harder.
- **ACC (Anomaly Correlation Coefficient)** — how well the forecast captures the *pattern* of departures from normal; **> 0.6 = useful**, **> 0.8 = high skill**.
- **Skill %** — improvement over a reference forecast: `100 × (1 − error_model / error_reference)`. Positive means VARUNA beats the baseline.
- **Persistence** = "tomorrow looks like today." **Persistence‑of‑anomaly (POA)** = "today's departure from normal persists" — a genuinely strong meteorological baseline that VARUNA still beats on temperature.

---

## Digital‑Twin Assimilation

VARUNA fuses observations into the AI state via **Optimal Interpolation (OI)** — the classical
analysis step, beyond pointwise nudging:

```
x_a = x_b + K (y − x_b),   K spreads the innovation per a spatially correlated
                            background-error covariance B (Gaussian, length-scale L)
```

With observations on the model grid this reduces to a covariance‑weighted, spatially‑smoothed
innovation. The dashboard shows **RMSE‑to‑observation before vs after** assimilation, demonstrating
the twin staying anchored to reality (`twin/assimilate.py`).

**The living‑twin cycle** — observe → forecast → assimilate → repeat:

```mermaid
sequenceDiagram
    participant OBS as IMD / INSAT observations
    participant TWIN as VARUNA state x
    participant AI as ClimateUNet
    participant USER as Planner
    OBS->>TWIN: ingest today's real fields
    TWIN->>AI: 7-day window + POA prior
    AI-->>TWIN: 10-day forecast (background x_b)
    OBS->>TWIN: new observations y
    TWIN->>TWIN: OI analysis  x_a = x_b + K(y - x_b)
    TWIN-->>USER: corrected state + hazards
    USER->>TWIN: what-if lever (ΔT, rain, greening…)
    TWIN-->>USER: heat-stress & AQI impact maps
    Note over OBS,TWIN: loop repeats as new data arrives
```

---

## Connected Applications

One climate state → three connected, climate‑sensitive applications (`scenario/engine.py`, `analytics/extremes.py`):

| Application | Method (citable proxy) | Drives |
|---|---|---|
| **Hazard early‑warning** | IMD heatwave departure criteria · IMD rain categories · dry‑spell run‑length | Heatwave / heavy‑rain / drought maps |
| **Urban heat** | NWS Rothfusz **heat index** + surface‑energy **LST/UHI** proxy | Heat‑stress map, peak cooling, danger‑area % |
| **Air quality** | CPCB **PM2.5 → AQI** sub‑index (rain washout, vegetation, wind, urban emissions) | AQI map & category |

**What‑if levers** (all physically coupled, all visibly change the maps): Δ air temperature,
Δ rainfall regime, urban greening (NDVI), cool‑roof albedo, added built‑up.

---

## Satellite Integration (INSAT/MOSDAC)

Real **INSAT‑3DR** Level‑2B products are ingested through `data/insat.py`:

- Parses the full‑disk HDF5 (per‑pixel int16 lat/lon at scale 0.01, geophysical field), masks fill values, converts **Kelvin → °C**, **crops to India**, and **regrids** onto the national 0.25° grid.
- Verified on a real `3RIMG_L2B_LST` file: **4,955 / 4,964** land cells covered; LST mean **27.7 °C**, peak **51.1 °C** at ~13:45 IST.
- Dashboard cross‑checks satellite **skin** temperature against climatological **air** Tmax — the positive skin–air offset is the expected physical signature (doubles as ingest validation).
- `LST`, `IMC` (rainfall) and `SST` slots are all supported — drop a file in `data/insat/` and it activates.

---

## The Dashboard

A dark "orbital" Streamlit + Plotly app (`app.py`) — instant client‑side maps with India state
boundaries, smooth slider interaction (heavy panels isolated with `st.fragment`).

| View | What it shows |
|---|---|
| 🌍 **Climate Twin** | National/pilot AI forecast map · live state · OI assimilation panel |
| 🌡️ **Hazards & Extremes** | Heatwave severity · heavy‑rain category · dry‑spell length |
| 🧪 **What‑if Simulator** | Scenario sliders → live Heat‑stress / AQI / Cooling maps + impact metrics |
| 📈 **Validation & Skill** | Per‑variable RMSE/ACC/skill, skill curves, XGBoost validation panel, city forecast (observed → ClimateUNet → skill‑weighted CNN+XGB ensemble) |
| 🛰️ **Satellite (INSAT)** | Real INSAT‑3DR LST layer + satellite‑vs‑model cross‑check |
| ℹ️ **About** | Methods & honest framing |

**Controls:** region (national / pilot), climate layer (rain / tmax / tmin), **date picker (1981 → 2026)**,
forecast lead day (1–10). Dates beyond IMD data render an explicitly‑labelled **climatological projection**
that the what‑if scenarios then modify (e.g., *"summer 2026 under +2 °C"*).

### Dashboard Gallery

<table>
<tr>
<td width="50%"><img src="assets/ui-1.png" alt="Climate Twin"/><br/><sub><b>Climate Twin</b> — national AI forecast map over India.</sub></td>
<td width="50%"><img src="assets/ui-3.png" alt="Forecast evolution"/><br/><sub><b>Forecast evolution</b> — 10‑day region‑mean vs climatology.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/ui-4.png" alt="Hazards"/><br/><sub><b>Hazards & Extremes</b> — heatwave / heavy‑rain / dry‑spell.</sub></td>
<td width="50%"><img src="assets/ui-5.png" alt="What-if simulator"/><br/><sub><b>What‑if Simulator</b> — live heat‑stress & AQI impact.</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/ui-7.png" alt="Validation & skill"/><br/><sub><b>Validation & Skill</b> — RMSE / ACC / skill curves.</sub></td>
<td width="50%"><img src="assets/ui-6.png" alt="Satellite layer"/><br/><sub><b>Satellite (INSAT)</b> — real INSAT‑3DR LST layer.</sub></td>
</tr>
<tr>
<td colspan="2"><img src="assets/ui-2.png" alt="Twin controls"/><br/><sub><b>Twin controls</b> — region, layer, date picker (1981→2026), lead day, scenario levers.</sub></td>
</tr>
</table>

---

## A Walk Through the Twin

A typical two‑minute session — every step is driven by the live model, nothing is pre‑rendered:

1. **Pick a moment.** Choose a date (say *15 May 2024*) and a region (national or the South‑India pilot). VARUNA reconstructs that day's real climate state from IMD data.
2. **Read the forecast.** The Climate‑Twin map shows the 10‑day AI outlook; slide the **lead‑day** control (1→10) and the field redraws instantly, with a region‑mean‑vs‑climatology evolution chart beside it.
3. **Spot the hazards.** Flip to **Hazards & Extremes** — heatwave departure, heavy‑rain category and dry‑spell length are computed straight from the forecast.
4. **Assimilate reality.** Open the **OI panel** to fuse observations into the state and watch RMSE‑to‑observation drop.
5. **Ask "what if?"** In the **simulator**, push *+2 °C*, drier monsoon, more greening or cool roofs — the **heat‑stress** and **AQI** maps and their headline metrics update live.
6. **Check the receipts.** **Validation & Skill** shows RMSE / ACC / skill on unseen years and the city forecast (observed → ClimateUNet → CNN+XGB ensemble).
7. **Cross‑check from orbit.** **Satellite (INSAT)** overlays the real INSAT‑3DR LST and compares skin vs air temperature.

---

## Technology Stack

| Layer | Tools |
|---|---|
| Data | `imdlib`, `xarray`, `netCDF4`, `h5py`, `bottleneck`, NumPy/Pandas |
| Deep learning | **PyTorch 2.5 + CUDA 12.1** (AMP) |
| Classical ML | **XGBoost 2.0** (GPU hist), scikit‑learn |
| Visualisation | **Streamlit 1.56**, **Plotly**, Matplotlib, Altair |
| Science | SciPy (Optimal Interpolation, regridding) |

---

## Repository Structure

```
VARUNA/
├── config.py                 # shared grids, regions, palette, paths, RHO
├── app.py                    # Streamlit dashboard (6 views)
├── data/
│   ├── download_imd.py        # IMD ingest via imdlib
│   ├── prepare.py             # regrid · climatology · anomalies · cache
│   ├── climatology.py         # smoothed day-of-year climatology
│   ├── proxies.py             # urban-fraction & baseline PM2.5 fields
│   └── insat.py               # INSAT-3DR L2B HDF5 ingest + regrid
├── models/
│   ├── architecture.py        # ClimateUNet (residual U-Net + SE attention)
│   ├── dataset.py             # anomaly cube + supervised windows + POA prior
│   ├── train.py               # GPU training loop (AMP, cosine, early stop)
│   ├── forecast.py            # inference → real-unit fields
│   ├── baseline.py            # persistence / POA / climatology references
│   └── xgb_forecast.py        # XGBoost station model + city ensemble
├── analytics/
│   └── extremes.py            # heatwave · rain category · dry-spell
├── evaluation/
│   ├── metrics.py             # RMSE/MAE/ACC/skill (weighted, masked)
│   └── evaluate.py            # full benchmark vs baselines
├── scenario/
│   └── engine.py              # heat index · CPCB AQI · LST/UHI proxy
├── twin/
│   └── assimilate.py          # Optimal Interpolation + nudging
├── viz/
│   ├── theme.py               # dark colormaps + CSS
│   └── maps.py                # Plotly field maps + India boundaries
├── tools/
│   ├── render_previews.py     # static forecast previews
│   ├── build_deck.py          # auto-fill DECK with real metrics
│   ├── ci_check.py            # data-free CI gate (compile + import + numeric smoke)
│   └── test_ui.py             # headless UI smoke test (AppTest)
├── .github/workflows/ci.yml   # GitHub Actions — runs tools/ci_check.py on push/PR
├── outputs/                   # eval_metrics.json · skill_curves.png · previews
├── PLAN.md · DECK.md · DEMO.md
└── requirements.txt
```

---

## Quickstart & Reproduce

```bash
# 1. environment
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt                         # install torch CUDA build from pytorch.org

# 2. data → model → metrics → dashboard  (everything runs locally)
python data/prepare.py            # download REAL IMD 1981–2024 + build cache
python models/train.py            # train ClimateUNet on the GPU
python models/xgb_forecast.py     # train the XGBoost companion
python evaluation/evaluate.py     # benchmark on unseen 2021–2024
streamlit run app.py              # launch VARUNA

# optional: drop INSAT .h5 files into data/insat/ to light up the satellite layer
```

Every stage is self‑contained — **no pre‑baked artefacts, no synthetic fallback.**
Each script can also be run independently, so you can reproduce any single stage in isolation.

### Continuous integration

A lightweight **GitHub Actions** workflow (`.github/workflows/ci.yml`) runs on every push and pull
request. It executes `tools/ci_check.py`, a data‑free gate that:

1. **byte‑compiles** every module (catches syntax / indentation errors),
2. **imports** the pure‑Python modules (config, analytics, scenario, twin, evaluation, viz, proxies, INSAT loader),
3. **exercises the core numerics** on tiny synthetic arrays — heat index & CPCB AQI, hazard summaries, Optimal‑Interpolation, and the INSAT filename date parser.

It installs only the light deps (`numpy`/`pandas`/`scipy`/`matplotlib`/`plotly`) and skips the heavy
stacks (torch, xgboost, imdlib, netCDF4, h5py), so it runs in seconds. The full model + UI smoke
test (`tools/test_ui.py`) needs the processed cache and checkpoints and is run locally.

```bash
python tools/ci_check.py          # same gate CI runs, locally
python tools/test_ui.py           # full headless UI smoke test (needs data + checkpoints)
```

---

## Deployment

VARUNA ships as a **Hugging Face Docker Space** and runs the very same `app.py` you run locally.

| | |
|---|---|
| **Live Space** | https://huggingface.co/spaces/Aditya1002/VARUNA |
| **Runtime** | Docker (`python:3.11-slim`), CPU PyTorch, Streamlit on port `7860` |
| **What ships** | code + processed observations, climatology, model checkpoints, INSAT sample |
| **What rebuilds at runtime** | the 500 MB anomaly cube is regenerated on boot from `obs/clim/stats` — keeps the image lean |
| **One‑command deploy** | `python tools/deploy_hf.py` (set a write `HF_TOKEN`; SDK/front‑matter handled automatically) |

```bash
# deploy your own copy
set HF_TOKEN=hf_xxx           # WRITE token from huggingface.co/settings/tokens
set HF_SPACE=your-user/VARUNA # optional; defaults to <user>/VARUNA
python tools/deploy_hf.py
```

The Dockerfile installs the CPU PyTorch wheel first (small image), keeps cache dirs writable, and
launches Streamlit headless — so CPU‑only inference works out of the box on the free Space tier.

---

## Honest Framing

We state our boundaries plainly — credibility matters in front of ISRO scientists:

- **AI short‑range forecast**, not a full general‑circulation model (GCM). Skill is strong at days 1–3 and decays with lead time, as expected.
- **Rainfall is intrinsically hard** day‑to‑day (ACC ≈ 0.40) — a physical limit even operational centres face; temperature is where the model excels (ACC 0.76).
- **Heat / AQI / LST are physics‑informed proxies** (NWS heat index, CPCB AQI, surface‑energy LST), clearly labelled — coefficients are literature‑consistent.
- **Assimilation = Optimal Interpolation**, not full 4D‑Var/EnKF.
- **Uncertainty** is represented by a **CNN + XGBoost ensemble vs three baselines**, not a multi‑GCM ensemble.
- **Future years (2025–26)** are **climatological projections + scenarios**, explicitly badged — never presented as literal multi‑year weather forecasts.
- **Satellite data** is used as a real observation / validation layer, not (yet) a training predictor.

**Zero synthetic data. Every number above is produced by a script in `evaluation/` and is reproducible.**

---

## Scale‑up Roadmap

- **Foundation models:** fine‑tune **IBM‑NASA Prithvi‑WxC** / adapt **Pangu‑Weather** on **IMDAA / BharatBench** reanalysis for true medium‑range skill.
- **Real assimilation:** ensemble Kalman filter / 4D‑Var with live INSAT + AWS feeds.
- **More variables & satellites:** INSAT IMC rainfall, SST, OLR, AOD; Oceansat; Bhuvan/NICES layers.
- **Operational delivery:** React + deck.gl front‑end, tiled national serving, scheduled ingest.

---

## Evaluation‑Parameter Map

| Parameter | Evidence in VARUNA |
|---|---|
| Problem Understanding & Clarity | This README · `PLAN.md` · honest framing |
| Data Usage & Pre‑processing | Real IMD 1981–2024 + INSAT‑3DR · regrid · climatology · anomalies · no leakage |
| Model Development & Technical Approach | ClimateUNet (residual, attention, multi‑horizon) + XGBoost ensemble |
| Prediction Performance & Validation | RMSE/MAE/ACC/skill on unseen years · `evaluation/evaluate.py` |
| Digital‑Twin Concept | Forecast + Optimal‑Interpolation assimilation + satellite fusion |
| Visualization & UI | Six‑view Plotly dashboard, instant maps, hazards, what‑if |
| Innovation & Creativity | Hazard early‑warning · connected heat/AQI · ensemble · future‑climate scenarios |
| Presentation & Communication | `DECK.md` (auto‑filled) · `DEMO.md` walkthrough |

---

## FAQ

**Is any of this data synthetic or simulated?**
No. Every field comes from real **IMD** gridded observations (1981–2024) and a real **INSAT‑3DR** L2B
product. There is no synthetic data, no fabricated fallback, anywhere in the pipeline.

**Why anomalies instead of raw rainfall and temperature?**
Raw fields are dominated by the seasonal cycle. By forecasting the *departure from climatology* the
model spends its capacity on the predictable signal, and rollouts relax toward climate instead of
drifting off — the key to multi‑day stability.

**Why does rainfall skill (ACC ≈ 0.40) look lower than temperature?**
Day‑to‑day rainfall is intrinsically chaotic — even operational global centres struggle past a few
days. We report it honestly. Temperature, which is far more predictable, reaches ACC **0.76**.

**Can it really forecast 2025 or 2026?**
Those dates render an explicitly‑labelled **climatological projection** that the what‑if scenarios
then modify. We never present future years as literal day‑by‑day weather forecasts.

**Do I need a GPU to run it?**
To *train*, yes (any modern NVIDIA card; it was built on a 6 GB RTX 4050). To *run the dashboard*,
no — the deployed Hugging Face Space does CPU‑only inference.

**How long does the full pipeline take?**
Data prep downloads ~44 years of IMD grids (one‑time, network‑bound); training the ClimateUNet is a
matter of hours on a single laptop GPU thanks to the compact architecture and mixed precision.

**How is this different from a weather app?**
A weather app *shows* a forecast. VARUNA is a **digital twin** — it reconstructs the state, forecasts,
assimilates real observations back in, and lets you simulate interventions and see their downstream
impact on heat and air quality.

---

## Glossary

| Term | Meaning |
|---|---|
| **Digital twin** | A continuously‑updated virtual replica of a real system (here, India's climate) used to monitor, forecast and simulate. |
| **Anomaly** | A field's departure from its day‑of‑year climatological normal. |
| **Climatology** | The long‑term average state for each day of the year (built from training years only). |
| **POA** | Persistence‑of‑anomaly — the baseline that assumes today's anomaly persists; also the model's physics prior. |
| **ACC** | Anomaly Correlation Coefficient — pattern correlation of forecast vs observed anomalies. |
| **OI** | Optimal Interpolation — the data‑assimilation step fusing observations into the model state via a background‑error covariance. |
| **LST** | Land Surface Temperature — the satellite "skin" temperature from INSAT‑3DR. |
| **UHI** | Urban Heat Island — the extra warmth of built‑up areas captured by the surface‑energy proxy. |
| **AQI** | Air Quality Index — CPCB PM2.5 sub‑index used in the what‑if air‑quality map. |
| **Regrid** | Interpolating one grid (1° temperature) onto another (0.25° rainfall) for a common national grid. |
| **AMP** | Automatic Mixed Precision — float16/32 training that saves GPU memory and time. |
| **SE attention** | Squeeze‑and‑Excite — lightweight channel attention that reweights feature maps. |

---

## Citations & License

- IMD gridded rainfall 0.25° — *Pai et al. (2014), MAUSAM 65(1)*.
- IMD gridded temperature 1.0° — *Srivastava et al. (2009), Atmos. Sci. Let.*
- INSAT‑3DR L2B products — **MOSDAC, SAC/ISRO**.
- India weather ML benchmarks — **BharatBench**, **IndiaWeatherBench** (IMDAA reanalysis).

**Data policy:** Real national datasets only. No synthetic or fabricated data anywhere in the pipeline.

---

## Acknowledgements

- **India Meteorological Department (IMD), Pune** — for the open, high‑resolution gridded rainfall and temperature archives that make a national twin possible.
- **ISRO / Space Applications Centre (SAC) & MOSDAC** — for INSAT‑3DR Level‑2B Earth‑observation products.
- **The open‑source scientific Python ecosystem** — `imdlib`, PyTorch, XGBoost, xarray, SciPy, Streamlit and Plotly.
- **ISRO Bharatiya Antariksh Hackathon (BAH) 2026** — for Problem Statement #5 and the push toward indigenous, Atmanirbhar climate intelligence.

---

## Team Vandalizers

**Project: VARUNA** · Team: **Vandalizers**

| Member |
|---|
| **Aditya Pandey** |
| **Palak Rai** |
| **Avik Srivastava** |

*Built for the **ISRO Bharatiya Antariksh Hackathon (BAH) 2026 · Problem Statement #5** — AI‑Powered Digital Twin of India's Climate using national datasets.*

<div align="center">

---

### VARUNA — Atmanirbhar climate intelligence on India's own data.
*Team Vandalizers — Aditya Pandey · Palak Rai · Avik Srivastava · ISRO BAH 2026 · PS#5*

</div>
