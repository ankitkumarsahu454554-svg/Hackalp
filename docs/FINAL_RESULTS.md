# CYBERSENTINEL AI (Temporal Threat Radar) — Verified Final Results

All metrics below were read directly from `results/` JSON/CSV files and `data/raw/DATA_MANIFEST.json` produced after the Phase 2 methodological corrections. Zero synthetic metrics or estimates are reported.

---

## 1. Dataset Provenance & Stage 1 Audit (`DATA_MANIFEST.json`, `data_audit.md`, `onset_counts.csv`)
- **Dataset**: `CICIDS2017_improved` (DistriNet IEEE CNS 2022, Liu et al., 2022)
- **Source URL**: `https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip`
- **Archive**: `343,549,013` bytes | SHA-256 `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883` (`verified`)
- **Total Clean Flows**: `2,099,976` across 5 UTC CSVs (`0` duplicates removed)
- **Label Hygiene**: `11,979` `*- Attempted` flows relabelled to `BENIGN` per DistriNet guidance

| Day | UTC Date | Observed Min | Attack-Active Min (`K_MIN=5`) | Onset Events | Attack Flow Starts |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Monday | 2017-07-03 | 487 | 0 | 0 | 0 |
| Tuesday | 2017-07-04 | 488 | 125 | 2 | 6,933 |
| Wednesday | 2017-07-05 | 509 | 65 | 3 | 171,645 |
| Thursday | 2017-07-06 | 486 | 39 | 4 | 71,907 |
| Friday | 2017-07-07 | 484 | 84 | 8 | 254,946 |
| **Total** | **5 days** | **2,454** | **313** | **17** | **505,431** |

---

## 2. Module A — Flow Classification (`classification_metrics.json`)
- **Protocol**: `globally_chronological_60_15_25_with_30min_embargo` (`200,000` train / `150,000` val / `150,000` test sampled rows; `56.14%` test attack prevalence)
- **Unseen-in-Training Test Attack Classes**: `['Botnet', 'DDoS', 'Portscan']` (validation unseen: `['Infiltration', 'Infiltration - Portscan']`)

| Model & Feature Scope | Test Binary F1 | Test Binary ROC-AUC | Test Binary PR-AUC | Test Multiclass Macro-F1 (All) | Test Multiclass Macro-F1 (Seen) | Test Weighted-F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest (`with_dst_port`, 83 features) | 0.1838 | 0.9855 | 0.9878 | 0.1058 | 0.2116 | 0.2785 |
| XGBoost (`with_dst_port`, 83 features) | 0.5119 | 0.9361 | 0.9400 | 0.1006 | 0.1761 | 0.3089 |
| Random Forest (`without_dst_port_ablation`, 82 features) | 0.2581 | 0.9912 | 0.9923 | 0.1079 | 0.2157 | 0.2839 |
| XGBoost (`without_dst_port_ablation`, 82 features) | 0.4054 | 0.9637 | 0.9634 | 0.0967 | 0.1692 | 0.2969 |

- **Hold-One-Family-Out (`DoS Hulk`, `50,000` unseen flows)**:
  - XGBoost binary detection rate (`1 - P(BENIGN) >= 0.5`): **`0.09144`** (`4,572 / 50,000`; `45,428` predicted `BENIGN`, `4,427` `DoS GoldenEye`, `142` `DoS Slowloris`, `3` `DoS Slowhttptest`)
- **Legacy Reference (`results/exploratory_intraday_classification.json`, `exploratory_only: true`)**: Preserved separately; NOT directly comparable due to intra-day calendar overlap.

---

## 3. Module B — Strict Out-of-Time Anomaly Scoring (`anomaly_metrics.json`, `anomaly_per_family.csv`)
- **Fitting Baseline (`flows_monday.parquet`)**: Verified `0` attack / `0` non-benign raw/clean rows (`371,624` total Monday rows: `150,000` fit, `74,325` benign-tail calibration, `147,299` excess discarded — **all `371,624` strictly excluded from out-of-time evaluation**).
- **Threshold (`95th` percentile on benign calibration tail)**: `0.518115` (`99th` percentile ref: `0.597722`).
- **Out-of-Time Deployment Performance (Tue–Fri, `1,728,352` flows, `29.24%` attack prevalence)**:
  - Overall Out-of-Time **ROC-AUC = `0.8691`**, **PR-AUC = `0.6061`**
  - Per Day ROC-AUC / PR-AUC: Tue (`0.8102` / `0.0564`), Wed (`0.9527` / `0.8004`), Thu (`0.8046` / `0.3581`), Fri (`0.8417` / `0.7079`)
  - Out-of-Time `BENIGN` false positive rate (`1,222,921` flows): `0.0550` (`67,252` flagged)
  - High detection rates on unseen volumetric/web attacks: `DoS Hulk` (`0.7856`), `DoS Slowloris` (`0.8012`), `DoS Slowhttptest` (`0.8011`), `Web Attack - Brute Force` (`0.9863`), `Web Attack - XSS` (`1.0000`), `Heartbleed` (`1.0000`), `Infiltration` (`0.9167`)
  - Low detection rates on stealth/scan traffic blended in flow aggregates: `Portscan` (`0.0005`), `Infiltration - Portscan` (`0.0007`), `FTP-Patator` (`0.0000`), `SSH-Patator` (`0.0007`)

---

## 4. Core Research Component — Temporal Forecasting (`H = 5` min)
- **Split Gaps**: `35` min (`>= max(30) + 5`); Eligible minutes: `1,267` Train (`6` onsets, `25` pos) / `349` Val (`5` onsets, `25` pos) / `314` Test (`6` onsets, `30` pos, prevalence `0.0955`).
- **Frozen Validation Calibration & Alert-Budget Thresholds**:
  - `B0_base_rate`: Uncalibrated (`thr=0.500000`)
  - `B1_oracle_persistence` (Oracle Reference): Platt (`slope=0.5190`, `thr=0.160000`)
  - `B2_anomaly_persistence`: Platt (`slope=0.1540`, `thr=0.124931`)
  - `B3_logistic_regression`: **Uncalibrated** (Platt skipped: non-positive slope `-0.0050`, `thr=1.000000`)
  - `xgboost_forecaster`: Platt (`slope=0.1945`, `thr=0.106607`; Risk tiers: `Medium=0.1066`, `High=0.1125`, `Critical=0.1174`)

### 4.1 Test Evaluation Table (`forecast_metrics.csv`, `bootstrap_ci.json`)
| Model | Scope | Calibrated | Point Test PR-AUC | Bootstrap Mean [95% CI] | ROC-AUC | Brier | Precision | 1-to-1 Event Recall [95% CI] | Mean Lead (min) | False Alerts/hr (Elig.) |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| `B0_base_rate` | Deployable | No | 0.0955 | 0.0962 [0.0304, 0.1725] | 0.5000 | 0.0922 | 0.0000 | 0.0000 [0.000, 0.000] | N/A | 0.000 |
| `B1_oracle_persistence` | **Oracle** | Yes | 0.2603 | 0.2844 [0.0466, 0.5794] | 0.6785 | 0.0812 | 0.3667 | 0.5000 [0.000, 1.000] | 4.33 | 0.382 |
| `B2_anomaly_persistence` | Deployable | Yes | 0.1189 | 0.1347 [0.0344, 0.2684] | 0.5934 | 0.0912 | 0.0000 | 0.0000 [0.000, 0.000] | N/A | 0.382 |
| `B3_logistic_regression` | Deployable | **No** | 0.2043 | 0.2161 [0.0416, 0.4458] | 0.6840 | 0.1979 | 0.3000 | 0.5000 [0.000, 1.000] | 5.00 | 0.191 |
| `xgboost_forecaster` | Deployable | Yes (5-seed) | **0.3224** | 0.3014 [0.0562, 0.5966] | **0.7546** | **0.0832** | 0.2708 | **0.6667** [0.000, 1.000] | 3.25 | 1.720 |

### 4.2 Single-Seed (`SEEDS = [0..4]`) vs 5-Seed Ensemble (`forecast_training_summary.json`)
- **Single-Seed Calibrated Test PR-AUC (Mean ± SD)**: **`0.3162 ± 0.0067`** (Seeds `0..4`: `0.3167`, `0.3138`, `0.3156`, `0.3268`, `0.3083`)
- **5-Seed Ensemble Calibrated Test PR-AUC**: **`0.3224`**
- **30-Min Block Bootstrap PR-AUC Mean (`500` reps, `498` valid)**: **`0.3014`**

### 4.3 Paired Block Bootstrap Differences (`bootstrap_ci.json`)
- `xgb - B0_base_rate`: PR-AUC diff = `+0.2052` (95% CI `[+0.0056, +0.4833]`, **excludes 0**); Brier diff = `-0.0089` (`[-0.0229, +0.0022]`)
- `xgb - B2_anomaly_persistence`: PR-AUC diff = `+0.1667` (95% CI `[-0.0777, +0.4515]`, **includes 0**); Brier diff = `-0.0082` (`[-0.0181, +0.0001]`)
- `xgb - B3_logistic_regression`: PR-AUC diff = `+0.0853` (95% CI `[-0.0201, +0.2433]`, **includes 0**); Brier diff = `-0.1140` (`[-0.2534, -0.0218]`, **excludes 0**)

### 4.4 `K_MIN` Sensitivity (`k_min_sensitivity.csv`)
| `K_MIN` | Total Onsets | Train/Val/Test Onsets | Test Elig. Min | Test Prev | B2 PR-AUC | B3 PR-AUC | XGB 5-Seed PR-AUC | Single-Seed Mean ± SD | 1-to-1 Event Recall | False Alerts/hr | Verdict |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :--- |
| `1` | 20 | 9 / 5 / 6 | 308 | 0.0974 | 0.0882 | 0.1444 | 0.1056 | 0.1073 ± 0.0094 | 0.5000 | 3.312 | `NOT DISTINGUISHABLE` |
| `3` | 17 | 6 / 5 / 6 | 313 | 0.0958 | 0.1181 | 0.1882 | 0.2956 | 0.2784 ± 0.0388 | 0.5000 | 1.342 | `NOT DISTINGUISHABLE` |
| `5` | 17 | 6 / 5 / 6 | 314 | 0.0955 | 0.1189 | 0.2043 | 0.3224 | 0.3162 ± 0.0067 | 0.6667 | 1.720 | `NOT DISTINGUISHABLE` |

---

## 5. Explainability (`shap_forecaster_global.json`, `shap_detector_global.json`)
- **Top Forecasting SHAP Features** (`663` validation+test eligible minutes):
  1. `unique_src_ips_roll15_mean` (`0.8016`)
  2. `unique_dst_ips_roll15_burst` (`0.4615`)
  3. `short_flow_fraction_roll30_slope` (`0.2775`)
  4. `mean_packet_length_roll5_max` (`0.2742`)
  5. `unique_dst_ips_roll5_mean` (`0.2244`)
- **Top Detection SHAP Features** (`2,000` sampled test flows):
  1. `Dst Port` (`0.6296`)
  2. `Fwd Packet Length Max` (`0.5701`)
  3. `Bwd Packet Length Max` (`0.4743`)
  4. `Active Min` (`0.3842`)
  5. `Bwd Init Win Bytes` (`0.3165`)
