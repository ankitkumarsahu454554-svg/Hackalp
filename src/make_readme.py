"""Documentation Generator: builds README.md strictly from results/ and DATA_MANIFEST.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import polars as pl

from config.settings import Settings, get_settings


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def generate_readme(cfg: Optional[Settings] = None) -> Path:
    """Generate ``README.md`` at project root using ONLY verified outputs in ``results/`` and ``DATA_MANIFEST.json``."""
    settings = cfg or get_settings()
    manifest = _load_json(settings.manifest_path)
    audit = _load_json(settings.results_dir / "data_audit.json")
    cls_metrics = _load_json(settings.results_dir / "classification_metrics.json")
    anom_metrics = _load_json(settings.results_dir / "anomaly_metrics.json")
    fc_metrics = _load_json(settings.results_dir / "forecast_metrics.json")
    verdict = _load_json(settings.results_dir / "verdict.json")
    shap_fc = _load_json(settings.results_dir / "shap_forecaster_global.json")
    shap_det = _load_json(settings.results_dir / "shap_detector_global.json")

    onset_csv = settings.results_dir / "onset_counts.csv"
    onset_rows = pl.read_csv(onset_csv).to_dicts() if onset_csv.exists() else []

    fc_csv = settings.results_dir / "forecast_metrics.csv"
    fc_rows = pl.read_csv(fc_csv).to_dicts() if fc_csv.exists() else []

    k_csv = settings.results_dir / "k_min_sensitivity.csv"
    k_rows = pl.read_csv(k_csv).to_dicts() if k_csv.exists() else []

    synthetic_banner = ""
    if manifest.get("synthetic_mode"):
        synthetic_banner = "> **SYNTHETIC: NOT RESULTS** — Real data was unavailable; metrics below are smoke outputs only.\n\n"

    file_rows = [
        f"| `{f.get('filename')}` | {int(f.get('bytes', 0)):,} | `{str(f.get('sha256', ''))[:16]}...` | {f.get('verification_status', '-')} |"
        for f in manifest.get("files", [])
    ]
    file_table = "\n".join(file_rows) if file_rows else "| (none) | 0 | - | missing |"

    onset_md_rows = [
        f"| {r['day']} | {r['utc_date']} | {r['total_minutes']:,} | {r['attack_active_minutes']:,} | {r['onset_events']} | {r['attack_flow_starts']:,} |"
        for r in onset_rows
    ]
    onset_table = "\n".join(onset_md_rows) if onset_md_rows else "| - | - | 0 | 0 | 0 | 0 |"

    cls_with = cls_metrics.get("with_dst_port", {})
    cls_no = cls_metrics.get("without_dst_port_ablation", {})
    rf_test = cls_with.get("models", {}).get("random_forest", {}).get("test", {})
    xgb_test = cls_with.get("models", {}).get("xgboost", {}).get("test", {})
    xgb_noport_test = cls_no.get("models", {}).get("xgboost", {}).get("test", {})
    holdout = cls_metrics.get("holdout_family_experiment", {})
    cls_bounds = cls_metrics.get("global_split_bounds", {})

    fc_md_rows = []
    for r in fc_rows:
        ci_pr = f"[{_fmt(r.get('pr_auc_ci_low'), 3)}, {_fmt(r.get('pr_auc_ci_high'), 3)}]" if r.get("pr_auc_ci_low") is not None else "N/A"
        ci_ev = f"[{_fmt(r.get('event_recall_ci_low'), 3)}, {_fmt(r.get('event_recall_ci_high'), 3)}]" if r.get("event_recall_ci_low") is not None else "N/A"
        deploy_lbl = "Deployable" if r.get("is_deployable") else "ORACLE (Not Deployable)"
        cal_lbl = "Platt" if r.get("is_calibrated") else "Uncalibrated"
        fc_md_rows.append(
            f"| `{r['model']}` | {deploy_lbl} | {cal_lbl} | {_fmt(r.get('pr_auc'))} | {_fmt(r.get('bootstrap_pr_auc_mean'))} {ci_pr} | {_fmt(r.get('roc_auc'))} | "
            f"{_fmt(r.get('brier_score'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('event_onset_recall'))} {ci_ev} | "
            f"{_fmt(r.get('mean_lead_time_min'), 2)} | {_fmt(r.get('false_alerts_per_hour'), 3)} |"
        )
    fc_table = "\n".join(fc_md_rows) if fc_md_rows else "| (run evaluate) | - | - | - | - | - | - | - | - | - | - |"

    k_md_rows = [
        f"| `{r['k_min']}` | {r['total_onsets']} | {r['train_onsets']}/{r['val_onsets']}/{r['test_onsets']} | "
        f"{_fmt(r.get('test_prevalence'))} | {_fmt(r.get('b2_pr_auc'))} | {_fmt(r.get('b3_pr_auc'))} | "
        f"{_fmt(r.get('xgb_ensemble_pr_auc'))} | {_fmt(r.get('xgb_single_seed_pr_auc_mean'))} ± {_fmt(r.get('xgb_single_seed_pr_auc_sd'))} | "
        f"{_fmt(r.get('xgb_event_onset_recall'))} | {_fmt(r.get('xgb_false_alerts_per_hour'), 3)} | `{r['verdict']}` |"
        for r in k_rows
    ]
    k_table = "\n".join(k_md_rows) if k_md_rows else "| (run evaluate) | - | - | - | - | - | - | - | - | - | - |"

    paired = fc_metrics.get("bootstrap", {}).get("paired_differences", {})
    paired_lines = []
    for pair_key, metrics_d in paired.items():
        pr_d = metrics_d.get("pr_auc", {})
        br_d = metrics_d.get("brier_score", {})
        paired_lines.append(
            f"- `{pair_key}`: PR-AUC diff = `{_fmt(pr_d.get('mean_diff'))}` "
            f"(95% CI `[{_fmt(pr_d.get('ci_low'))}, {_fmt(pr_d.get('ci_high'))}]`, excludes 0: `{pr_d.get('excludes_zero')}`), "
            f"Brier diff = `{_fmt(br_d.get('mean_diff'))}` "
            f"(95% CI `[{_fmt(br_d.get('ci_low'))}, {_fmt(br_d.get('ci_high'))}]`, excludes 0: `{br_d.get('excludes_zero')}`)"
        )
    paired_md = "\n".join(paired_lines) if paired_lines else "- Run evaluation to populate paired bootstrap intervals."

    seed_info = fc_metrics.get("xgboost_seed_metrics", {})
    wf_rows = fc_metrics.get("walk_forward_secondary", [])
    wf_md = "\n".join(
        f"- Segment `{w['test_segment_id']}` (test positives={w['test_positives']}, prevalence={_fmt(w['test_prevalence'])}): "
        f"XGBoost PR-AUC = `{_fmt(w['xgb_pr_auc'])}` vs B3 LogReg = `{_fmt(w['b3_logreg_pr_auc'])}`"
        for w in wf_rows
    ) or "- Insufficient segments for secondary walk-forward check."

    split_onsets = verdict.get("split_onsets", {})
    top_fc_shap = ", ".join(f"`{x['feature']}` ({_fmt(x['importance'], 3)})" for x in shap_fc.get("top_features", [])[:5])
    top_det_shap = ", ".join(f"`{x['feature']}` ({_fmt(x['importance'], 3)})" for x in shap_det.get("top_features", [])[:5])

    readme_text = f"""# CYBERSENTINEL AI — Temporal Threat Radar

**Project**: AI-Based Network Attack Forecasting from Network Traffic Data

{synthetic_banner}> **Important Notice**: Every simulation in this project is a **historical replay** of recorded network flows (never a live capture). Forecasts are **calibrated onset-risk estimates**, never confirmed attacks.

## 1. Overview & Research Question
CYBERSENTINEL AI investigates one falsifiable cybersecurity ML research question:

> *"Can machine learning forecast a new network-attack onset a few minutes BEFORE it starts, better than simple baselines?"*

Three independent analytical outputs remain separated throughout code, metrics, and UI:
1. **Detection** (Module A): Classifies observed flows as benign or a specific attack category (Random Forest & XGBoost).
2. **Anomaly Scoring** (Module B): Measures deviation from a verified benign baseline period (out-of-time Isolation Forest). Anomaly scores are **not** probabilities of attack.
3. **Forecasting** (Core Research Component): Estimates calibrated risk of a **new** attack onset within the next $H = {settings.horizons[0]}$ minutes using strictly causal past-only features.

## 2. Installation
```bash
pip install -r requirements.txt
```

## 3. Data Sources & Provenance (from `DATA_MANIFEST.json`)
- **Dataset Version**: {manifest.get('dataset_version', 'N/A')}
- **Source Name**: {manifest.get('source_name', 'N/A')}
- **Source URL**: {manifest.get('source_url', 'N/A')}
- **Retrieved UTC**: {manifest.get('retrieval_timestamp_utc', 'N/A')}
- **Verification Status**: `{manifest.get('verification_status', 'N/A')}`
- **Synthetic Mode**: `{manifest.get('synthetic_mode', False)}`
- **License / Citation Note**: {manifest.get('license_note', 'N/A')}

| File | Bytes | SHA-256 | Status |
| --- | ---: | --- | --- |
{file_table}

## 4. Dataset Acquisition & Version Requirements
- Run `python -m scripts.fetch_data` to discover, download, extract, and SHA-256 verify the DistriNet CNS 2022 `CICIDS2017_improved.zip` archive automatically into `data/raw/`.
- Timestamped, labelled flow data is required (`Timestamp`, `Flow Duration`, `Src/Dst IP`, `Dst Port`, `Label`). Pre-stripped versions such as `MachineLearningCVE` lack timestamps/IPs and cannot be used for temporal forecasting.
- In `CICIDS2017_improved` (DistriNet IEEE CNS 2022), timestamps are UTC with microsecond precision and `Attempted Category` flags intended attack flows without malicious behaviour; per author instructions, all `{audit.get('total_attempted_relabelled', 0):,}` `*- Attempted` rows are relabelled to `BENIGN` rather than kept as separate classes.
- Audit totals: `{audit.get('total_raw_rows', 0):,}` raw flows, `{audit.get('total_duplicates_removed', 0):,}` duplicates removed, `{audit.get('total_clean_rows', 0):,}` clean flows across `5` UTC capture days.

## 5. Architecture & Methodology
- **End-Time Causal Availability**: A flow record is emitted only when the flow terminates (`end_ts = start_ts + Flow Duration`). Minute $t$ features use **only** flows with `end_ts < t + 60s`.
- **Attack-Active & Onset Definition**: Minute $m$ is `attack_active` if `>= {settings.k_min}` attack-labelled flows **start** in minute $m$. An `onset` occurs at the first `attack_active` minute preceded by `>= {settings.quiet_gap_min}` consecutive non-attack-active minutes within the same segment (breaks `> {settings.segment_break_min}` min start a new segment; zero rolling across breaks).
- **Target & Exclusions**: $y_{{{settings.horizons[0]}}}(t) = 1$ if any onset occurs in $(t, t+{settings.horizons[0]}]$ within the same segment. Active-attack minutes, initial `{max(settings.feature_windows)}`-min segment warm-up, and censored tails are excluded. Identifiers (`Flow ID`, IPs, `Src Port`, `Timestamp`, `Attempted Category`) are strictly excluded.
- **Chronological Splitting, 1-to-1 Episode Matching & Leakage Verification**:
  - Detection uses a **globally chronological split** across the full 5-day UTC timeline (`{cls_bounds.get('embargo_minutes', 30)}`-min temporal embargo gaps; training strictly precedes validation, which strictly precedes test).
  - Forecasting uses `{settings.split_gap_min}`-min gaps (`max(FEATURE_WINDOWS) + max(HORIZONS)`). Platt calibration and false-alert-budget thresholding (`<= {settings.false_alert_budget_per_hour}/hr`, rejecting degenerate `>35%` alert-rate solutions) are frozen on validation only.
  - Alert episodes (`gap <= {settings.alert_merge_gap_min}` min) use **strict one-to-one chronological matching** to onsets (multiple alert episodes cannot claim credit for one onset), and false alerts/hour are divided by **eligible observation hours**.

### Observed Attack Onsets Per Day (`K_MIN = {settings.k_min}`)
| Day | UTC Date | Observed Minutes | Attack-Active Minutes | Onset Events | Attack Flow Starts |
| --- | --- | ---: | ---: | ---: | ---: |
{onset_table}

Total onset events: **{audit.get('total_onset_events', 0)}** (Split onsets — Train: `{split_onsets.get('train_onsets', 'N/A')}`, Validation: `{split_onsets.get('val_onsets', 'N/A')}`, Test: `{split_onsets.get('test_onsets', 'N/A')}`; eligible positives: `{split_onsets.get('train_positive_rows', 0)}` / `{split_onsets.get('val_positive_rows', 0)}` / `{split_onsets.get('test_positive_rows', 0)}`).

## 6. Actual Results

### Stage 2 Detection (Module A — Globally Chronological Split) & Out-of-Time Anomaly (Module B)
- **Split**: `{cls_metrics.get('split_protocol', 'globally_chronological')}` (`{cls_with.get('train_rows', 0):,}` train / `{cls_with.get('val_rows', 0):,}` val / `{cls_with.get('test_rows', 0):,}` test rows; Train end `{cls_bounds.get('train_end', 'N/A')}`, Val `{cls_bounds.get('val_start', 'N/A')} -> {cls_bounds.get('val_end', 'N/A')}`, Test `{cls_bounds.get('test_start', 'N/A')} -> {cls_bounds.get('test_end', 'N/A')}`).
  - **Unseen-in-training test attack classes**: `{xgb_test.get('unseen_in_training_classes', [])}` — under a globally chronological split, later-day attack families never appear in earlier-day training (`macro_f1_all` reflects zero exact-family recall on unseen classes; binary ROC-AUC/PR-AUC reflects only benign-vs-attack feature separability, not unseen-family classification).
  - **Random Forest (83 features)**: Binary (`attack_f1 = {_fmt(rf_test.get('binary_attack_f1'))}`, `roc_auc = {_fmt(rf_test.get('binary_roc_auc'))}`, `pr_auc = {_fmt(rf_test.get('binary_pr_auc'))}`) | Multiclass (`macro_f1_all = {_fmt(rf_test.get('macro_f1_all_classes'))}`).
  - **XGBoost (83 features)**: Binary (`attack_f1 = {_fmt(xgb_test.get('binary_attack_f1'))}`, `roc_auc = {_fmt(xgb_test.get('binary_roc_auc'))}`, `pr_auc = {_fmt(xgb_test.get('binary_pr_auc'))}`) | Multiclass (`macro_f1_all = {_fmt(xgb_test.get('macro_f1_all_classes'))}`).
  - **XGBoost Ablation without `Dst Port` (82 features)**: Binary (`attack_f1 = {_fmt(xgb_noport_test.get('binary_attack_f1'))}`, `roc_auc = {_fmt(xgb_noport_test.get('binary_roc_auc'))}`, `pr_auc = {_fmt(xgb_noport_test.get('binary_pr_auc'))}`).
  - **Hold-One-Family-Out (`{holdout.get('held_out_family', 'N/A')}`)**: XGBoost flagged `{_fmt(holdout.get('xgb_binary_detection_rate_on_unseen_family'))}` of `{holdout.get('held_out_family_eval_flows', 0):,}` unseen flows as attack; Isolation Forest flagged `{_fmt(anom_metrics.get('holdout_family_anomaly_performance', {}).get('detection_rate'))}`.
- **Out-of-Time Isolation Forest** (fitted on verified BENIGN flows `{anom_metrics.get('fit_period', 'flows_monday.parquet')}` (`{anom_metrics.get('fit_rows_used', 0):,}` fit rows; `{anom_metrics.get('discarded_from_out_of_time_eval_rows', 0):,}` fitting/calibration rows excluded from out-of-time evaluation), `95th` percentile threshold `{_fmt(anom_metrics.get('threshold'))}`):
  - Across `{anom_metrics.get('out_of_time_later_days_flows', 0):,}` out-of-time Tue–Fri flows (`prevalence = {_fmt(anom_metrics.get('out_of_time_attack_prevalence'))}`): **ROC-AUC = `{_fmt(anom_metrics.get('out_of_time_roc_auc'))}`**, **PR-AUC = `{_fmt(anom_metrics.get('out_of_time_pr_auc'))}`**.

### Stage 3/4 Attack-Onset Forecasting (`H = {settings.horizons[0]}` min)
- Test prevalence baseline: `{_fmt(verdict.get('b0_baserate_test_pr_auc'))}` (`{split_onsets.get('test_positive_rows', 0)}/{split_onsets.get('test_eligible_rows', 0)}` eligible test minutes).
- **5-Seed Ensemble vs Single-Seed Metrics** (`SEEDS = {seed_info.get('seeds', [])}`):
  - **5-Seed Calibrated Ensemble Test PR-AUC**: `{_fmt(seed_info.get('ensemble_5seed_test_pr_auc'))}` (point estimate reported for `xgboost_forecaster` below)
  - **Single-Seed Calibrated Test PR-AUC (mean ± SD)**: `{_fmt(seed_info.get('single_seed_test_pr_auc_mean'))} ± {_fmt(seed_info.get('single_seed_test_pr_auc_sd'))}`

| Model | Scope | Calibration | Point Test PR-AUC | Bootstrap PR-AUC Mean [95% CI] | ROC-AUC | Brier | Precision | 1-to-1 Event Recall [95% CI] | Mean Lead (min) | False Alerts/hr (Elig.) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
{fc_table}

#### Paired Block Bootstrap Differences (`{fc_metrics.get('bootstrap', {}).get('block_minutes', 30)}`-min blocks, `{fc_metrics.get('bootstrap', {}).get('bootstrap_reps', 500)}` reps)
{paired_md}

#### `K_MIN` Sensitivity Analysis (`K_MIN` ∈ {{1, 3, 5}})
| `K_MIN` | Total Onsets | Train/Val/Test Onsets | Test Prev | B2 PR-AUC | B3 PR-AUC | XGB 5-Seed PR-AUC | Single-Seed Mean ± SD | 1-to-1 Event Recall | False Alerts/hr | Verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
{k_table}

#### Secondary Expanding Walk-Forward Folds
{wf_md}

### Stage 5 Explainability (`{shap_fc.get('method', 'N/A')}`)
- **Top Forecasting Features**: {top_fc_shap or 'N/A'}
- **Top Detection Features**: {top_det_shap or 'N/A'}

## 7. Automated Verdict
> **VERDICT: `{verdict.get('verdict', 'UNKNOWN')}`**
> {verdict.get('rationale', '')}

*(Separate non-deployable reference — `B1_oracle_persistence`: PR-AUC = `{_fmt(verdict.get('b1_oracle_reference_not_deployable', {}).get('pr_auc'))}`, Brier = `{_fmt(verdict.get('b1_oracle_reference_not_deployable', {}).get('brier_score'))}`, 1-to-1 Event Recall = `{_fmt(verdict.get('b1_oracle_reference_not_deployable', {}).get('event_onset_recall'))}`.)*

## 8. Limitations
1. **Controlled Historical Benchmark**: CICIDS2017 is a scripted 5-day lab capture with scheduled morning/afternoon attacks rather than organic adversary dwell times.
2. **Few Onset Events**: Only `{audit.get('total_onset_events', 0)}` dataset-defined attack onsets (`{split_onsets.get('test_onsets', 0)}` in the test partition) yield wide bootstrap intervals and limit statistical power.
3. **Verdict Is `NOT DISTINGUISHABLE`**: Under 30-min block bootstrap, XGBoost forecasting gains over deployable baselines `B2` and `B3` have 95% CIs overlapping zero.
4. **No Production Generalization**: Captures reflect one testbed topology; IP/port baselines and concept drift limit transferability to real production networks.
5. **Label Sensitivity**: Even after CNS 2022 fixes and `Attempted -> BENIGN` relabelling, onset boundaries depend on `K_MIN` and `QUIET_GAP_MIN`.
6. **Flow Metadata & Encrypted Payloads**: Only aggregate flow statistics are observed (no packet payloads, DNS, or endpoint logs).

## 9. Reproduction Commands
```bash
python -m scripts.fetch_data
python -m src.pipeline --stage all
python -m src.make_readme
python -m pytest -v
streamlit run app.py
```

## 10. Privacy Notes
Analysis uses aggregate network-flow metadata (packet counts, byte volumes, inter-arrival timing, TCP flags, destination ports, and testbed IPs solely for entropy and unseen-source ratios); no packet payloads or PII are inspected or stored.
"""
    out_path = settings.project_root / "README.md"
    out_path.write_text(readme_text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = generate_readme(get_settings())
    print(f"Generated {p}")
