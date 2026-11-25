# CYBERSENTINEL AI — Final Test & Verification Report

## 1. Automated Unit & Integration Suite (`python -m pytest -v`)
Executed on Python `3.13.7`, `pytest 9.1.1` (Windows x64) after all Phase 2–4 code and dashboard fixes:
- **Outcome**: **`25 passed, 7 warnings in 40.41s`** (`0` failed, `0` skipped)
- Warnings observed: 1 `RequestsDependencyWarning` (installed `urllib3`/`charset_normalizer` minor mismatch), 6 synthetic single-label `confusion_matrix` `UserWarning`s in `test_stage2.py`.

### Verified Test Cases (All 25 Passed)
1. `tests/test_acquisition.py::test_offline_writes_data_needed_and_synthetic_flag` — **PASSED**
2. `tests/test_acquisition.py::test_local_existing_files_verified_and_skipped_without_overwrite` — **PASSED**
3. `tests/test_acquisition.py::test_size_or_hash_mismatch_detected_without_overwrite` — **PASSED**
4. `tests/test_acquisition.py::test_zip_slip_blocked` — **PASSED**
5. `tests/test_dashboard_and_docs.py::test_config_env_overrides` — **PASSED**
6. `tests/test_dashboard_and_docs.py::test_make_readme_from_results_and_manifest` — **PASSED**
7. `tests/test_dashboard_and_docs.py::test_dashboard_pages_handle_missing_results_gracefully` — **PASSED**
8. `tests/test_forecasting_and_eval.py::test_hand_computed_alert_episode_metrics_one_to_one` — **PASSED** *(verifies two separate alert episodes before one onset yield `hit_episodes=1`, `warned_onsets=1`, `false_episodes=1`, `lead_times=[5.0]`)*
9. `tests/test_forecasting_and_eval.py::test_eligible_observation_duration_used_for_false_alert_rate` — **PASSED** *(verifies division by `observed_minutes` rather than wall-clock length)*
10. `tests/test_forecasting_and_eval.py::test_chronological_split_and_pipeline_smoke` — **PASSED**
11. `tests/test_leakage.py::test_no_forbidden_columns_in_deployable_features` — **PASSED**
12. `tests/test_leakage.py::test_leakage_equivalence_under_end_time_truncation` — **PASSED**
13. `tests/test_leakage.py::test_config_override_horizons_changes_targets` — **PASSED**
14. `tests/test_preprocessing.py::test_label_dash_normalization` — **PASSED**
15. `tests/test_preprocessing.py::test_timestamp_12h_ampm_disambiguation` — **PASSED**
16. `tests/test_preprocessing.py::test_feature_column_exclusion` — **PASSED**
17. `tests/test_preprocessing.py::test_onset_grid_k_min_quiet_gap_and_segments` — **PASSED**
18. `tests/test_preprocessing.py::test_audit_end_to_end_on_tiny_csv` — **PASSED**
19. `tests/test_replay.py::test_replay_causality_and_controls_step_pause_reset` — **PASSED** *(verifies replay causal masking and `Step +10 min` / `Pause` / `Reset` semantics)*
20. `tests/test_splits.py::test_global_classification_split_strictly_chronological` — **PASSED** *(verifies `train < embargo < val < embargo < test`)*
21. `tests/test_splits.py::test_forecast_split_no_overlap_and_required_gaps` — **PASSED** *(verifies `>= 35`-min gaps on real forecast dataset)*
22. `tests/test_stage2.py::test_global_chronological_split_bounds_have_embargo` — **PASSED**
23. `tests/test_stage2.py::test_stage2_global_classification_and_strict_oot_anomaly` — **PASSED** *(verifies Monday fit exclusion from OOT evaluation)*
24. `tests/test_stages_5_to_7.py::test_explainability_and_alert_manager_separation` — **PASSED**
25. `tests/test_stages_5_to_7.py::test_replay_never_reads_future_and_hides_future_labels` — **PASSED**

---

## 2. Dashboard End-to-End & Runtime Verification (`AppTest` + Live Health Check)
- **Streamlit `AppTest` across all 5 views + replay stepping** (`timeout=60s` per view):
  - `1. Security Command Centre`: **PASSED** (`5` KPI metrics rendered, `Step +10 min` causal replay button clicked with zero exceptions)
  - `2. Temporal Threat Radar`: **PASSED** (`4` metrics, `0` exceptions)
  - `3. Explainable AI`: **PASSED** (`0` exceptions)
  - `4. Alert Management`: **PASSED** (`1` alert dataframe rendered, `0` exceptions)
  - `5. Model Evaluation`: **PASSED** (`6` evaluation tables & sensitivity breakdowns rendered, `0` exceptions)
- **Live Server Startup Check**:
  - Started `python -m streamlit run app.py --server.headless true --server.port 8509`
  - Queried `http://127.0.0.1:8509/_stcore/health` -> **HTTP `200` (`ok`)**, then terminated cleanly.

---

## 3. Data, Experiments & Hygiene Verification
- **Manifest & Provenance**: `CICIDS2017_improved.zip` and 5 extracted CSVs SHA-256 verified against `data/raw/DATA_MANIFEST.json`.
- **Pipeline Re-runs Completed After Fixes**:
  - `classify` -> Global chronological split (`2,099,976` clean rows across 5 days, `200k`/`150k`/`150k` samples, `30`-min embargoes)
  - `anomaly` -> Benign-verified `flows_monday.parquet` fit (`371,624` excluded), strict Tue–Fri out-of-time eval (`1,728,352` flows)
  - `forecast` + `evaluate` + `k_min_sensitivity` -> Primary `H=5` min + `K_MIN in {1, 3, 5}` + 30-min block bootstrap (`500` reps) + Detection/Forecast SHAP + `README.md` regeneration
- **Security, Secrets & Branding Hygiene**:
  - `.env*` file count: **`0`**
  - Secret / credential pattern hits: **`0`**
  - Non-product branding/institutional/team hits: **`0`**
