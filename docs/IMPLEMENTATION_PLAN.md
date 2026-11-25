# CYBERSENTINEL AI (Temporal Threat Radar) — Architecture & Corrected Implementation Plan

## 1. Research Objective & Design Principles
Tests one falsifiable research question:
> *"Can machine learning forecast a new network-attack onset a few minutes BEFORE it starts, better than simple baselines?"*

Three outputs remain strictly separated across code, evaluation, and UI:
1. **Detection (Module A)**: Classifies individual terminated network flows (`BENIGN` vs attack family + binary attack score `1 - P(BENIGN)`).
2. **Anomaly Scoring (Module B)**: Measures deviation from a verified benign reference period (out-of-time Isolation Forest anomaly score; NOT a probability of attack).
3. **Temporal Forecasting (Core)**: Estimates calibrated risk of a **new** attack onset within `(t, t + 5]` min from strictly causal flow-end-time aggregated features.

---

## 2. Verified Hardware, Data & Provenance
- **Host Constraints**: Windows 11 x64 CPU workstation (`~7.93 GB` RAM); streaming via Polars/DuckDB and batched numpy inference keep memory bounded cleanly.
- **Author-Verified Dataset**: DistriNet IEEE CNS 2022 `CICIDS2017_improved.zip` (`343,549,013` B, SHA-256 `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883`), yielding `2,099,976` clean flows (`0` duplicates) across Monday–Friday (UTC microseconds, `11,979` `Attempted -> BENIGN` rows normalized per authors' instructions).

---

## 3. Corrected Pipeline Architecture (Stages 0–8)

1. **Stage 0 — Configuration & CLI (`config/settings.py`, `src/pipeline.py`)**: Typed configuration constants (`HORIZONS=[5]`, `FEATURE_WINDOWS=[5, 15, 30]`, `QUIET_GAP_MIN=5`, `K_MIN=5`, `SEGMENT_BREAK_MIN=60`, `SEEDS=[0..4]`) and stage CLI dispatcher.
2. **Stage 1 — Resumable Acquisition & Temporal Audit (`src/data_acquisition.py`, `src/data_loader.py`, `src/preprocessing.py`)**: HEAD/disk pre-flight checks, SHA-256 verification, traversal-safe extraction, normalization, per-day Parquet export, `DATA_MANIFEST.json`, `results/data_audit.md`, `results/onset_counts.csv`.
3. **Stage 2 — Globally Chronological Detection & Strict Out-of-Time Anomaly (`src/classification.py`, `src/anomaly_detection.py`)**:
   - *Module A (`classification.py`)*: **Globally chronological 60/15/25 split** across the full Mon–Fri timeline with `30`-minute temporal embargo gaps (`compute_global_chronological_split_bounds`); separates `binary_detection` from `multiclass_classification`, preserves legacy intra-day artifact at `results/exploratory_intraday_classification.json`, runs no-`Dst Port` ablation (`82` features) and hold-one-family-out (`DoS Hulk`).
   - *Module B (`anomaly_detection.py`)*: Verifies `attack_rows == 0` and `non_benign_raw_rows == 0` on `flows_monday.parquet`, fits Isolation Forest only on earliest `150,000` benign flows, calibrates `95th` percentile on the Monday benign tail (`74,325` flows), and excludes all `371,624` Monday rows from out-of-time Tue–Fri evaluation (`1,728,352` flows).
4. **Stage 3 — End-Time Causal Feature Engineering & Forecasting (`src/feature_engineering.py`, `src/forecasting.py`)**:
   - Enforces `end_ts < t + 60s` availability for features, `start_ts` ground-truth attack-active/onset labels, and zero rolling across `SEGMENT_BREAK_MIN > 60` boundaries.
   - Splits with `max(FEATURE_WINDOWS) + max(HORIZONS) = 35` min gaps (`Train` 6 onsets / `Val` 5 onsets / `Test` 6 onsets).
   - Evaluates `B0` (base rate), `B1` (non-deployable oracle past-label persistence), `B2` (anomaly persistence), `B3` (Logistic Regression), and `M` (XGBoost `SEEDS=[0..4]`) with **validation-only** Platt scaling (skipped cleanly if slope <= 0 and explicitly flagged `is_calibrated=False`) and validation false-alert-budget thresholding (`<= 1.0/hr`).
   - Computes both calibrated per-seed summary (`mean ± SD`) and 5-seed ensemble predictions plus **`run_k_min_sensitivity(k_values=(1, 3, 5))`**.
5. **Stage 4 — Rigorous Evaluation & Statistical Testing (`src/evaluation.py`)**:
   - Alert-episode merging (`gap <= 2` min) with **strict 1-to-1 greedy chronological matching** (`hit_episodes == warned_onsets`).
   - False-alert episodes per **eligible observation hour** (`elig_df.height / 60`).
   - `30`-min block bootstrap (`500` reps) over metrics and paired differences vs deployable baselines; automated `NOT DISTINGUISHABLE` verdict when paired PR-AUC 95% CIs overlap zero.
6. **Stage 5–7 — Explainability, Alert Separation & Historical Replay (`src/explainability.py`, `src/alert_manager.py`, `src/replay_engine.py`)**:
   - Independent TreeSHAP explanations for Detector and Forecaster; strict separation of `DETECTION` and `FORECAST WARNING` alerts; minute-by-minute causal historical replay with delayed label reveal.
7. **Stage 8 — Dashboard, Docs & Release Hygiene (`app.py`, `dashboard/*.py`, `src/make_readme.py`, `.gitignore`, `docs/*`)**:
   - Streamlit 5-page dashboard reading exclusively real saved outputs with explicit verdict and benchmark/methodological caveats; automated `README.md` generator; clean Git ignore rules excluding raw/processed data, `.joblib` binaries, caches, and `.env*` files.
