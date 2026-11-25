# CYBERSENTINEL AI (Temporal Threat Radar) — Final Scientific Audit & Correction Report

## 1. Executive Summary
A complete end-to-end audit (`CHECK -> CORRECT -> RETEST -> RELEASE` preparation) was performed across the entire CYBERSENTINEL AI project (`config/`, `src/`, `scripts/`, `dashboard/`, `tests/`, `results/`, `README.md`, `.gitignore`, and Git remote state).

Five genuine methodological and reporting issues identified in the initial inspection were corrected in source code, verified with new regression unit tests, and executed end-to-end across all pipelines against the verified DistriNet IEEE CNS 2022 `CICIDS2017_improved.zip` dataset (`2,099,976` clean flows; `0` synthetic shortcuts).

The core scientific conclusion remains unchanged and strictly evidenced:
- **Verdict**: **`NOT DISTINGUISHABLE`**
- With only **`17`** dataset-defined attack onset events (`6` train / `5` validation / `6` test; `K_MIN = 5`, `QUIET_GAP_MIN = 5`), the 5-seed calibrated XGBoost forecaster achieves **`0.3224`** 5-seed ensemble test PR-AUC (single-seed mean±SD: **`0.3162 ± 0.0067`**; 30-min block bootstrap mean: **`0.3014`**, 95% CI `[0.0562, 0.5966]`), exceeding point estimates of deployable baselines `B0` (`0.0955`), `B2` (`0.1189`), and `B3` (`0.2043`), **but paired 95% bootstrap intervals overlap zero** against both `B2` (`[-0.0777, 0.4515]`) and `B3` (`[-0.0201, 0.2433]`).

---

## 2. Methodological Defects Found & Corrected

### 2.1 Detection Split Protocol (`src/classification.py`)
- **Finding (FAIL -> FIXED)**: The prior classifier pipeline partitioned each weekday CSV independently (`60/15/25`), causing Thursday and Friday morning flows to enter `train` while Monday–Wednesday afternoon flows entered `val` and `test` (future-day leakage across calendar time).
- **Correction**:
  - Archived the legacy result cleanly at `results/exploratory_intraday_classification.json` stamped with `"exploratory_only": true` and a prominent methodological warning.
  - Implemented `compute_global_chronological_split_bounds` enforcing strict global UTC ordering across all 5 days with `30`-minute temporal embargoes:
    - `Train`: `2017-07-03 11:55:58+00:00` -> `2017-07-06 13:41:56+00:00` (`1,259,985` raw flows; `200,000` sampled train rows)
    - `Embargo 1`: `16,967` rows excluded
    - `Validation`: `2017-07-06 14:11:57+00:00` -> `2017-07-07 13:01:49+00:00` (`314,996` raw flows; `150,000` sampled val rows)
    - `Embargo 2`: `34,511` rows excluded
    - `Test`: `2017-07-07 13:31:49+00:00` -> `2017-07-07 20:02:40+00:00` (`473,517` raw flows; `150,000` sampled test rows)
  - Separated `binary_detection` (`1 - P(BENIGN)`) from `multiclass_classification` (`macro_f1_all_classes`, `macro_f1_seen_classes_only`, `weighted_f1`, `unseen_in_training_classes`), and recorded `unseen_family_limitation_note` (`Botnet`, `DDoS`, `Portscan` in global test never occurred in earlier-day training, yielding `0.0` exact-family recall and lowering XGBoost `macro_f1_all_classes` to `0.1006`, while binary benign-vs-attack ROC-AUC is `0.9361`).

### 2.2 Anomaly Detection Verification & Strict Out-of-Time Scope (`src/anomaly_detection.py`)
- **Finding (FAIL -> FIXED)**: The prior script fitted Isolation Forest on the first 80% of `flows_monday.parquet` without an explicit raw-label benign assertion, and mixed Monday rows into `results/anomaly_per_family.csv` under `BENIGN`.
- **Correction**:
  - Added explicit per-file benign assertions on `flows_monday.parquet`: `attack_rows = 0`, `non_benign_clean_rows = 0`, and `non_benign_raw_rows = 0`.
  - Logged exact row accounting: `371,624` Monday flows (`150,000` fit, `147,299` excess fit discarded, `74,325` benign-tail threshold calibration) -> `discarded_from_out_of_time_eval_rows = 371,624`.
  - Restricted `results/anomaly_per_family.csv` (`per_family_out_of_time_only`) strictly to Tue–Fri (`1,728,352` flows: `1,222,921` out-of-time `BENIGN` + `505,431` attacks across 15 families; overall out-of-time `ROC-AUC = 0.8691`, `PR-AUC = 0.6061`).

### 2.3 Alert Episode One-to-One Matching & Eligible Duration (`src/evaluation.py`)
- **Finding (FAIL -> FIXED)**:
  1. `compute_alert_episode_metrics` previously flagged multiple separate alert episodes preceding one onset as separate hits (`ep_hit = True`), distorting hit counts and bootstrap recall.
  2. `evaluate_model_predictions` previously computed `false_alerts_per_hour` by dividing by `timeline_df.height` (`361` total test minutes including excluded attack-active minutes) instead of `elig_df.height` (`314` eligible decision minutes).
- **Correction**:
  - Enforced strict greedy chronological 1-to-1 matching (`matched_onsets` set ensuring at most one alert episode claims each onset within `(ep_start, ep_start + H]` within the same segment; `hit_episodes == warned_onsets` invariant).
  - Updated `compute_block_bootstrap` to resample over `warned_onset_indices` consistently.
  - Divided `false_alerts_per_hour` by observed eligible duration (`314 / 60 = 5.2333` eligible hours), correcting XGBoost's false-alert rate from `1.496` to `1.7197` (`9` false episodes / `5.2333` hr).

### 2.4 Calibration Transparency & 5-Seed vs Ensemble Consistency (`src/forecasting.py`, `src/evaluation.py`)
- **Finding (FAIL -> FIXED)**:
  1. `B3_logistic_regression` skipped Platt scaling due to non-positive validation slope (`-0.0050`), yet `forecast_metrics.csv` lacked an `is_calibrated` indicator.
  2. `forecast_training_summary.json` reported raw uncalibrated single-seed PR-AUC (`0.3162 ± 0.0067`) without matching per-seed calibration/thresholding, creating apparent ambiguity against the `0.3224` 5-seed ensemble and `0.3014` bootstrap mean.
- **Correction**:
  - Added `is_calibrated`, `estimator_type`, `bootstrap_pr_auc_mean`, `single_seed_pr_auc_mean`, and `single_seed_pr_auc_sd` columns to `results/forecast_metrics.csv` and labelled plots explicitly (`[Uncalibrated]`, `(Platt-calibrated)`, `(5-seed ensemble)`).
  - Updated `src/forecasting.py` to Platt-calibrate and threshold each individual seed (`0..4`) identically on validation before computing single-seed summary statistics (`0.3162 ± 0.0067`), alongside `5-seed_ensemble` PR-AUC (`0.3224`) and 30-min block bootstrap mean (`0.3014`).

### 2.5 Onset Threshold Sensitivity (`K_MIN in {1, 3, 5}`) (`src/forecasting.py`, `src/pipeline.py`)
- **Finding (NOT VERIFIED -> FIXED)**: Only `K_MIN = 5` had been executed.
- **Correction**: Added `run_k_min_sensitivity(cfg, k_values=(1, 3, 5))` re-running the end-to-end feature-target, train/val/test split, Platt calibration, and paired block bootstrap pipeline cleanly for `K_MIN = 1, 3, 5` (`results/k_min_sensitivity.csv` & `.json`). At every setting (`20`, `17`, and `17` onsets), paired 95% CIs vs `B2` and `B3` overlap zero -> `NOT DISTINGUISHABLE`.

---

## 3. Project Identity & Privacy Audit
- Scanned all workspace source, dashboard, test, script, configuration, and documentation files for team identifiers, project IDs, student/faculty names, university names, competition names, secrets, API tokens, and `.env*` files.
- Removed non-product project identifiers from `app.py`, `src/make_readme.py`, and regenerated `README.md`.
- Retained `CYBERSENTINEL AI`, `Temporal Threat Radar`, and legitimate academic citations (`Liu et al., 2022`, `Sharafaldin et al., 2018`, DistriNet / KU Leuven, UNB CIC-IDS-2017).
- Post-cleanup verification scan: **`0` branding hits**, **`0` secret hits**, **`0` `.env` files**.

---

## 4. Application Quality Audit (5-Page Streamlit UI)
- Verified all 5 pages (`Command Centre`, `Temporal Threat Radar`, `Explainable AI`, `Alert Management`, `Model Evaluation`) render real saved outputs with zero fabricated metrics and strict separation of Detection, Anomaly, and Forecast outputs.
- Replaced deprecated `use_container_width=True` with `width="stretch"` across all 5 modules.
- Enhanced `dashboard/model_evaluation.py` to display the verbatim verdict banner, all core dataset/methodology limitations visibly on screen, 1-to-1 matching & eligible-duration definitions, 5-seed ensemble vs single-seed breakdown, `K_MIN in {1, 3, 5}` table, global chronological binary vs multiclass classification tables (plus archived intra-day reference expander), and strict out-of-time Isolation Forest counts.
- Verified with programmatic Streamlit `AppTest` across all 5 pages + Replay `Step +10 min` interaction, plus live `streamlit run app.py` HTTP `200 ok` on `/_stcore/health`.

---

## 5. Complete List of Modified and Added Files
- **Modified Source & Dashboard**:
  - `app.py`
  - `src/classification.py`
  - `src/anomaly_detection.py`
  - `src/evaluation.py`
  - `src/forecasting.py`
  - `src/pipeline.py`
  - `src/make_readme.py`
  - `dashboard/command_centre.py`
  - `dashboard/threat_radar.py`
  - `dashboard/explainable_ai.py`
  - `dashboard/alert_management.py`
  - `dashboard/model_evaluation.py`
- **Modified / Added Tests**:
  - `tests/test_stage2.py`
  - `tests/test_forecasting_and_eval.py`
  - `tests/test_splits.py` *(new)*
  - `tests/test_replay.py` *(new)*
- **Added / Updated Documentation & Repo Hygiene**:
  - `.gitignore` *(new)*
  - `README.md` *(regenerated from `src/make_readme.py`)*
  - `docs/FINAL_AUDIT_CHECKLIST.md`
  - `docs/FINAL_AUDIT_REPORT.md` / `final_audit_report.md`
  - `docs/FINAL_TEST_REPORT.md`
  - `docs/FINAL_RESULTS.md`
  - `docs/IMPLEMENTATION_PLAN.md`
- **Regenerated Model / Result Artifacts**:
  - `results/exploratory_intraday_classification.json` *(archived prior intra-day classification metrics)*
  - `results/classification_metrics.json`, `results/classification_per_class.csv`, `results/classifier_cm_*.png`
  - `results/anomaly_metrics.json`, `results/anomaly_per_family.csv`
  - `results/forecast_metrics.csv`, `results/forecast_metrics.json`, `results/forecast_training_summary.json`, `results/bootstrap_ci.json`, `results/verdict.json`
  - `results/k_min_sensitivity.csv`, `results/k_min_sensitivity.json`
  - `results/calibration_h5.png`, `results/forecast_comparison_h5.png`, `results/lead_time_hist_h5.png`, `results/pr_curves_h5.png`
  - `results/shap_detector_global.json`, `results/shap_forecaster_global.json`

---

## 6. Remaining Honest Scientific & Engineering Limitations
1. **Controlled Scripted Lab Benchmark**: `CICIDS2017_improved` spans 5 scheduled lab capture days with deterministic morning/afternoon attack windows rather than organic multi-week enterprise intrusion dwell times.
2. **Low Statistical Power (`17` Onsets / `6` Test Onsets)**: Small onset count yields wide 30-min block bootstrap intervals (XGBoost PR-AUC 95% CI `[0.0562, 0.5966]`; event recall CI `[0.000, 1.000]`).
3. **Verdict Is `NOT DISTINGUISHABLE`**: Across `K_MIN in {1, 3, 5}`, paired 95% bootstrap PR-AUC differences vs `B2` and `B3` include zero.
4. **Chronological Class Novelty vs Unseen Families**: In the globally chronological classification split, test-only attack families (`Botnet`, `DDoS`, `Portscan`) never appear during training, resulting in `0.0` exact-family recall (`macro_f1_all_classes = 0.1006` for XGBoost). High binary ROC-AUC (`0.9361`–`0.9855`) reflects feature separability rather than true unseen-family recognition; Hold-One-Family-Out (`DoS Hulk`) flagged only `9.14%` under XGBoost (`78.56%` under out-of-time Isolation Forest).
5. **Onset Parameter & Flow Metadata Scope**: Results depend on operational `K_MIN` / `QUIET_GAP_MIN` definitions and see flow metadata only (no packet payloads, DNS logs, or endpoint telemetry).

---

## 7. Release Readiness Gate Status
- Raw `.zip`/`.csv` archives, processed `.parquet` data, `.joblib` weights, caches, and `.env*` files are excluded via `.gitignore`.
- Target remote repository was verified reachable and currently empty (`git ls-remote` exit `0`).
- Local push was **explicitly withheld** pending user authorization.
