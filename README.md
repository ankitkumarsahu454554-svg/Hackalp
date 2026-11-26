# CYBERSENTINEL AI — Temporal Threat Radar

**Project**: AI-Based Network Attack Forecasting from Network Traffic Data

> **Important Notice**: Every simulation in this project is a **historical replay** of recorded network flows (never a live capture). Forecasts are **calibrated onset-risk estimates**, never confirmed attacks.

## 1. Overview & Research Question
CYBERSENTINEL AI investigates one falsifiable cybersecurity ML research question:

> *"Can machine learning forecast a new network-attack onset a few minutes BEFORE it starts, better than simple baselines?"*

Three independent analytical outputs remain separated throughout code, metrics, and UI:
1. **Detection** (Module A): Classifies observed flows as benign or a specific attack category (Random Forest & XGBoost).
2. **Anomaly Scoring** (Module B): Measures deviation from a verified benign baseline period (out-of-time Isolation Forest). Anomaly scores are **not** probabilities of attack.
3. **Forecasting** (Core Research Component): Estimates calibrated risk of a **new** attack onset within the next $H = 5$ minutes using strictly causal past-only features.

## 2. Installation
```bash
pip install -r requirements.txt
```

## 3. Data Sources & Provenance (from `DATA_MANIFEST.json`)
- **Dataset Version**: CICIDS2017_improved (IEEE CNS 2022)
- **Source Name**: DistriNet CNS2022 Improved CICIDS2017 (Liu et al., 2022)
- **Source URL**: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
- **Retrieved UTC**: 2026-11-25T13:46:52.144228+00:00
- **Verification Status**: `verified`
- **Synthetic Mode**: `False`
- **License / Citation Note**: Academic research release by DistriNet / KU Leuven (Liu, Engelen, Lynar, Essam, Joosen, IEEE CNS 2022), derived from UNB CIC-IDS-2017. Cite Liu et al. (2022) & Sharafaldin et al. (2018).

| File | Bytes | SHA-256 | Status |
| --- | ---: | --- | --- |
| `CICIDS2017_improved.zip` | 343,549,013 | `97fdb91d339e2d8c...` | verified |
| `friday.csv` | 285,188,226 | `ebd499e6f23bd59f...` | verified |
| `monday.csv` | 207,875,155 | `51fe5dc962626efb...` | verified |
| `thursday.csv` | 189,519,159 | `78a4d11eaf473d09...` | verified |
| `tuesday.csv` | 178,397,720 | `e2a0a5b631dfc6b4...` | verified |
| `wednesday.csv` | 291,290,505 | `bf46c5f3c792e881...` | verified |

## 4. Dataset Acquisition & Version Requirements
- Run `python -m scripts.fetch_data` to discover, download, extract, and SHA-256 verify the DistriNet CNS 2022 `CICIDS2017_improved.zip` archive automatically into `data/raw/`.
- Timestamped, labelled flow data is required (`Timestamp`, `Flow Duration`, `Src/Dst IP`, `Dst Port`, `Label`). Pre-stripped versions such as `MachineLearningCVE` lack timestamps/IPs and cannot be used for temporal forecasting.
- In `CICIDS2017_improved` (DistriNet IEEE CNS 2022), timestamps are UTC with microsecond precision and `Attempted Category` flags intended attack flows without malicious behaviour; per author instructions, all `11,979` `*- Attempted` rows are relabelled to `BENIGN` rather than kept as separate classes.
- Audit totals: `2,099,976` raw flows, `0` duplicates removed, `2,099,976` clean flows across `5` UTC capture days.

## 5. Architecture & Methodology
- **End-Time Causal Availability**: A flow record is emitted only when the flow terminates (`end_ts = start_ts + Flow Duration`). Minute $t$ features use **only** flows with `end_ts < t + 60s`.
- **Attack-Active & Onset Definition**: Minute $m$ is `attack_active` if `>= 5` attack-labelled flows **start** in minute $m$. An `onset` occurs at the first `attack_active` minute preceded by `>= 5` consecutive non-attack-active minutes within the same segment (breaks `> 60` min start a new segment; zero rolling across breaks).
- **Target & Exclusions**: $y_{5}(t) = 1$ if any onset occurs in $(t, t+5]$ within the same segment. Active-attack minutes, initial `30`-min segment warm-up, and censored tails are excluded. Identifiers (`Flow ID`, IPs, `Src Port`, `Timestamp`, `Attempted Category`) are strictly excluded.
- **Chronological Splitting, 1-to-1 Episode Matching & Leakage Verification**:
  - Detection uses a **globally chronological split** across the full 5-day UTC timeline (`30`-min temporal embargo gaps; training strictly precedes validation, which strictly precedes test).
  - Forecasting uses `35`-min gaps (`max(FEATURE_WINDOWS) + max(HORIZONS)`). Platt calibration and false-alert-budget thresholding (`<= 1.0/hr`, rejecting degenerate `>35%` alert-rate solutions) are frozen on validation only.
  - Alert episodes (`gap <= 2` min) use **strict one-to-one chronological matching** to onsets (multiple alert episodes cannot claim credit for one onset), and false alerts/hour are divided by **eligible observation hours**.

### Observed Attack Onsets Per Day (`K_MIN = 5`)
| Day | UTC Date | Observed Minutes | Attack-Active Minutes | Onset Events | Attack Flow Starts |
| --- | --- | ---: | ---: | ---: | ---: |
| monday | 2017-07-03 | 487 | 0 | 0 | 0 |
| tuesday | 2017-07-04 | 488 | 125 | 2 | 6,933 |
| wednesday | 2017-07-05 | 509 | 65 | 3 | 171,645 |
| thursday | 2017-07-06 | 486 | 39 | 4 | 71,907 |
| friday | 2017-07-07 | 484 | 84 | 8 | 254,946 |

Total onset events: **17** (Split onsets — Train: `6`, Validation: `5`, Test: `6`; eligible positives: `25` / `25` / `30`).

## 6. Actual Results

### Stage 2 Detection (Module A — Globally Chronological Split) & Out-of-Time Anomaly (Module B)
- **Split**: `globally_chronological_60_15_25_with_30min_embargo` (`200,000` train / `150,000` val / `150,000` test rows; Train end `2017-07-06 13:41:56.761173+00:00`, Val `2017-07-06 14:11:57.088913+00:00 -> 2017-07-07 13:01:49.164669+00:00`, Test `2017-07-07 13:31:49.481709+00:00 -> 2017-07-07 20:02:40.977798+00:00`).
  - **Unseen-in-training test attack classes**: `['Botnet', 'DDoS', 'Portscan']` — under a globally chronological split, later-day attack families never appear in earlier-day training (`macro_f1_all` reflects zero exact-family recall on unseen classes; binary ROC-AUC/PR-AUC reflects only benign-vs-attack feature separability, not unseen-family classification).
  - **Random Forest (83 features)**: Binary (`attack_f1 = 0.1838`, `roc_auc = 0.9855`, `pr_auc = 0.9878`) | Multiclass (`macro_f1_all = 0.1058`).
  - **XGBoost (83 features)**: Binary (`attack_f1 = 0.5119`, `roc_auc = 0.9361`, `pr_auc = 0.9400`) | Multiclass (`macro_f1_all = 0.1006`).
  - **XGBoost Ablation without `Dst Port` (82 features)**: Binary (`attack_f1 = 0.4054`, `roc_auc = 0.9637`, `pr_auc = 0.9634`).
  - **Hold-One-Family-Out (`DoS Hulk`)**: XGBoost flagged `0.0914` of `50,000` unseen flows as attack; Isolation Forest flagged `0.7856`.
- **Out-of-Time Isolation Forest** (fitted on verified BENIGN flows `flows_monday.parquet` (`150,000` fit rows; `371,624` fitting/calibration rows excluded from out-of-time evaluation), `95th` percentile threshold `0.5181`):
  - Across `1,728,352` out-of-time Tue–Fri flows (`prevalence = 0.2924`): **ROC-AUC = `0.8691`**, **PR-AUC = `0.6061`**.

### Stage 3/4 Attack-Onset Forecasting (`H = 5` min)
- Test prevalence baseline: `0.0955` (`30/314` eligible test minutes).
- **5-Seed Ensemble vs Single-Seed Metrics** (`SEEDS = [0, 1, 2, 3, 4]`):
  - **5-Seed Calibrated Ensemble Test PR-AUC**: `0.3224` (point estimate reported for `xgboost_forecaster` below)
  - **Single-Seed Calibrated Test PR-AUC (mean ± SD)**: `0.3162 ± 0.0067`

| Model | Scope | Calibration | Point Test PR-AUC | Bootstrap PR-AUC Mean [95% CI] | ROC-AUC | Brier | Precision | 1-to-1 Event Recall [95% CI] | Mean Lead (min) | False Alerts/hr (Elig.) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `B0_base_rate` | Deployable | Uncalibrated | 0.0955 | 0.0962 [0.030, 0.173] | 0.5000 | 0.0922 | 0.0000 | 0.0000 [0.000, 0.000] | N/A | 0.000 |
| `B1_oracle_persistence` | ORACLE (Not Deployable) | Platt | 0.2603 | 0.2844 [0.047, 0.579] | 0.6785 | 0.0812 | 0.3667 | 0.5000 [0.000, 1.000] | 4.33 | 0.382 |
| `B2_anomaly_persistence` | Deployable | Platt | 0.1189 | 0.1347 [0.034, 0.268] | 0.5934 | 0.0912 | 0.0000 | 0.0000 [0.000, 0.000] | N/A | 0.382 |
| `B3_logistic_regression` | Deployable | Uncalibrated | 0.2043 | 0.2161 [0.042, 0.446] | 0.6840 | 0.1979 | 0.3000 | 0.5000 [0.000, 1.000] | 5.00 | 0.191 |
| `xgboost_forecaster` | Deployable | Platt | 0.3224 | 0.3014 [0.056, 0.597] | 0.7546 | 0.0832 | 0.2708 | 0.6667 [0.000, 1.000] | 3.25 | 1.720 |

#### Paired Block Bootstrap Differences (`30`-min blocks, `500` reps)
- `xgboost_forecaster_minus_B0_base_rate`: PR-AUC diff = `0.2052` (95% CI `[0.0056, 0.4833]`, excludes 0: `True`), Brier diff = `-0.0089` (95% CI `[-0.0229, 0.0022]`, excludes 0: `False`)
- `xgboost_forecaster_minus_B2_anomaly_persistence`: PR-AUC diff = `0.1667` (95% CI `[-0.0777, 0.4515]`, excludes 0: `False`), Brier diff = `-0.0082` (95% CI `[-0.0181, 0.0001]`, excludes 0: `False`)
- `xgboost_forecaster_minus_B3_logistic_regression`: PR-AUC diff = `0.0853` (95% CI `[-0.0201, 0.2433]`, excludes 0: `False`), Brier diff = `-0.1140` (95% CI `[-0.2534, -0.0218]`, excludes 0: `True`)

#### `K_MIN` Sensitivity Analysis (`K_MIN` ∈ {1, 3, 5})
| `K_MIN` | Total Onsets | Train/Val/Test Onsets | Test Prev | B2 PR-AUC | B3 PR-AUC | XGB 5-Seed PR-AUC | Single-Seed Mean ± SD | 1-to-1 Event Recall | False Alerts/hr | Verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `1` | 20 | 9/5/6 | 0.0974 | 0.0882 | 0.1444 | 0.1056 | 0.1073 ± 0.0094 | 0.5000 | 3.312 | `NOT DISTINGUISHABLE` |
| `3` | 17 | 6/5/6 | 0.0958 | 0.1181 | 0.1882 | 0.2956 | 0.2784 ± 0.0388 | 0.5000 | 1.342 | `NOT DISTINGUISHABLE` |
| `5` | 17 | 6/5/6 | 0.0955 | 0.1189 | 0.2043 | 0.3224 | 0.3162 ± 0.0067 | 0.6667 | 1.720 | `NOT DISTINGUISHABLE` |

#### Secondary Expanding Walk-Forward Folds
- Segment `2` (test positives=15, prevalence=0.0367): XGBoost PR-AUC = `0.0376` vs B3 LogReg = `0.0609`
- Segment `3` (test positives=20, prevalence=0.0485): XGBoost PR-AUC = `0.0966` vs B3 LogReg = `0.0605`
- Segment `4` (test positives=40, prevalence=0.1099): XGBoost PR-AUC = `0.1478` vs B3 LogReg = `0.2681`

### Stage 5 Explainability (`shap_tree_explainer`)
- **Top Forecasting Features**: `unique_src_ips_roll15_mean` (0.802), `unique_dst_ips_roll15_burst` (0.461), `short_flow_fraction_roll30_slope` (0.278), `mean_packet_length_roll5_max` (0.274), `unique_dst_ips_roll5_mean` (0.224)
- **Top Detection Features**: `Dst Port` (0.630), `Fwd Packet Length Max` (0.570), `Bwd Packet Length Max` (0.474), `Active Min` (0.384), `Bwd Init Win Bytes` (0.317)

## 7. Automated Verdict
> **VERDICT: `NOT DISTINGUISHABLE`**
> XGBoost forecaster PR-AUC 95% paired bootstrap intervals overlap zero against deployable baselines (vs B2: [-0.07765824828958812, 0.4514594258844709], vs B3: [-0.020092461615253635, 0.24333346368866587]).

*(Separate non-deployable reference — `B1_oracle_persistence`: PR-AUC = `0.2603`, Brier = `0.0812`, 1-to-1 Event Recall = `0.5000`.)*

## 8. Limitations
1. **Controlled Historical Benchmark**: CICIDS2017 is a scripted 5-day lab capture with scheduled morning/afternoon attacks rather than organic adversary dwell times.
2. **Few Onset Events**: Only `17` dataset-defined attack onsets (`6` in the test partition) yield wide bootstrap intervals and limit statistical power.
3. **Verdict Is `NOT DISTINGUISHABLE`**: Under 30-min block bootstrap, XGBoost forecasting gains over deployable baselines `B2` and `B3` have 95% CIs overlapping zero.
4. **No Production Generalization**: Captures reflect one testbed topology; IP/port baselines and concept drift limit transferability to real production networks.
5. **Label Sensitivity**: Even after CNS 2022 fixes and `Attempted -> BENIGN` relabelling, onset boundaries depend on `K_MIN` and `QUIET_GAP_MIN`.
6. **Flow Metadata & Encrypted Payloads**: Only aggregate flow statistics are observed (no packet payloads, DNS, or endpoint logs).

## 9. Web Application, Brand Identity & Design System
- **Entry Point & Design System**: `app.py`, `assets/styles/theme.css`, `components/ui.py`, `utils/data_loaders.py`.
- **Original Brand Assets (`assets/brand/`)**: `cybersentinel_logo_primary.svg`, `cybersentinel_logo_light.svg`, `cybersentinel_logo_sidebar.svg`, `cybersentinel_icon.svg`, `favicon.svg`, `favicon.png`.
- **Five Application Pages (`dashboard/`)**:
  1. **Security Command Centre** (`command_centre.py`) — Unified SOC KPI strip, readiness overview, family/anomaly distributions, temporal timelines, recent alerts, and decoupled module status cards.
  2. **Temporal Threat Radar** (`threat_radar.py`) — Polar temporal threat scope, historical replay controls (`Start`, `Pause`, `Step +1m`, `Step +10m`, `Reset`, `Speed`), observed-vs-forecast timeline, and warning feeds.
  3. **Explainable AI** (`explainable_ai.py`) — Strictly separated Forecasting (Module M) & Detection (Module A) global and per-alert SHAP attributions with plain-English reason cards.
  4. **Alert Management** (`alert_management.py`) — SOC triage queue with severity KPI summary, search/filter/sort, SHAP inspector, and persistent analyst workflow (`New`, `Acknowledged`, `Escalated`, `Resolved`, `Dismissed`).
  5. **Model Evaluation** (`model_evaluation.py`) — Scientific verdict (`NOT DISTINGUISHABLE`), mandatory limitations, baseline comparison, paired bootstrap CIs, `K_MIN in {1, 3, 5}` sensitivity, global chronological classification, and out-of-time Isolation Forest audit.
- **Screenshots**: Captured across all five pages and responsive mobile viewport in `docs/screenshots/`.

## 10. Reproduction Commands
```bash
python -m scripts.fetch_data
python -m src.pipeline --stage all
python -m src.make_readme
python -m pytest -v
streamlit run app.py
```

## 11. Privacy Notes
Analysis uses aggregate network-flow metadata (packet counts, byte volumes, inter-arrival timing, TCP flags, destination ports, and testbed IPs solely for entropy and unseen-source ratios); no packet payloads or PII are inspected or stored.
