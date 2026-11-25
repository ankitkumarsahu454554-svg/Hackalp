# Stage 1B Data Audit Report

## Dataset Provenance & Version
- **Dataset Version**: CICIDS2017_improved (IEEE CNS 2022)
- **Source Name**: DistriNet CNS2022 Improved CICIDS2017 (Liu et al., 2022)
- **Source URL**: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
- **Synthetic Mode**: False
- **Total Raw Flows**: 2,099,976 | **Duplicates Removed**: 0 | **Clean Flows**: 2,099,976
- **Attempted Flows Relabelled to BENIGN**: 11,979 (per CNS 2022 author guidance: `Attempted Category != -1` or `*- Attempted` -> `BENIGN`, never kept as separate attack classes)

## Feature Column Governance & Leakage Prevention
- **Excluded Columns (14)**: `['Attempted Category', 'Dst IP', 'Flow ID', 'Label', 'Src IP', 'Src Port', 'Timestamp', 'day', 'end_ts', 'id', 'is_attack', 'label', 'raw_label', 'start_ts']`
- **Classifier Features (with Dst Port)**: 83 columns
- **Classifier Features (without Dst Port, ablation)**: 82 columns (`Dst Port` excluded)

## Attack-Onset Counts Per Day (Stage 3 Definition)
- Parameters: `BIN_SECONDS=60`, `K_MIN=5`, `QUIET_GAP_MIN=5`, `SEGMENT_BREAK_MIN=60`
- **Total Onset Events Across Dataset**: **17**

| Day | UTC Date | Observed Minutes | Segments | Attack-Active Minutes | Onset Events | Attack Flow Starts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| monday | 2017-07-03 | 487 | 1 | 0 | 0 | 0 |
| tuesday | 2017-07-04 | 488 | 1 | 125 | 2 | 6,933 |
| wednesday | 2017-07-05 | 509 | 1 | 65 | 3 | 171,645 |
| thursday | 2017-07-06 | 486 | 1 | 39 | 4 | 71,907 |
| friday | 2017-07-07 | 484 | 1 | 84 | 8 | 254,946 |

## Per-Day File Audit & Quality Diagnostics

### Monday (`monday.csv`)
- **Encoding**: `utf-8-sig` | **Raw rows**: 371,624 | **Duplicates removed**: 0 | **Clean rows**: 371,624
- **UTC Range**: `2017-07-03 11:55:58.598308+00:00` -> `2017-07-03 20:01:34.352593+00:00`
- **Timestamp Format & Timezone**: ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff) — UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)
- **AM/PM Inferred Rows**: 0 (monotonic disagreements: 0)
- **Flow Duration Unit**: microseconds verified=True (median `(pkts/dur_s) / Flow Packets/s` ratio = 1.000000)
- **Flow Duration Anomalies**: negative=0, zero=3, max=119,999,987 us
- **Dash-Normalized Label Rows**: 0 | **Attempted -> BENIGN**: 0
- **NaN / Null Columns**: None
- **`+inf` Columns**: {'Flow Bytes/s': 3, 'Flow Packets/s': 3}
- **`-inf` Columns**: None
- **Unexpected Negative-Value Columns**: {'Fwd Header Length': 46, 'Bwd Header Length': 85}

**Clean Label Distribution:**

| Label | Count |
| --- | ---: |
| `BENIGN` | 371,624 |

**Observed Attack Windows vs Published Schedule:**

- Benign-only baseline day (no attack labels present).

### Tuesday (`tuesday.csv`)
- **Encoding**: `utf-8-sig` | **Raw rows**: 322,078 | **Duplicates removed**: 0 | **Clean rows**: 322,078
- **UTC Range**: `2017-07-04 11:53:44.398274+00:00` -> `2017-07-04 20:00:30.041663+00:00`
- **Timestamp Format & Timezone**: ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff) — UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)
- **AM/PM Inferred Rows**: 0 (monotonic disagreements: 0)
- **Flow Duration Unit**: microseconds verified=True (median `(pkts/dur_s) / Flow Packets/s` ratio = 1.000000)
- **Flow Duration Anomalies**: negative=0, zero=0, max=119,999,977 us
- **Dash-Normalized Label Rows**: 0 | **Attempted -> BENIGN**: 39
- **NaN / Null Columns**: None
- **`+inf` Columns**: None
- **`-inf` Columns**: None
- **Unexpected Negative-Value Columns**: {'Fwd Header Length': 60, 'Bwd Header Length': 82}

**Clean Label Distribution:**

| Label | Count |
| --- | ---: |
| `BENIGN` | 315,145 |
| `FTP-Patator` | 3,972 |
| `SSH-Patator` | 2,961 |

**Observed Attack Windows vs Published Schedule:**

| Clean Attack Label | Flows | Observed First Start (UTC) | Observed Last Start (UTC) |
| --- | ---: | --- | --- |
| `FTP-Patator` | 3,972 | `2017-07-04 12:19:20.152476+00:00` | `2017-07-04 13:20:13.627945+00:00` |
| `SSH-Patator` | 2,961 | `2017-07-04 17:09:01.049616+00:00` | `2017-07-04 19:04:15.353690+00:00` |

Published reference windows:
  - `FTP-Patator`: `2017-07-04 12:17:52` -> `2017-07-04 13:20:16` UTC (09:17-10:20 ADT)
  - `SSH-Patator`: `2017-07-04 17:09:01` -> `2017-07-04 19:04:19` UTC (14:09-16:04 ADT)

### Wednesday (`wednesday.csv`)
- **Encoding**: `utf-8-sig` | **Raw rows**: 496,641 | **Duplicates removed**: 0 | **Clean rows**: 496,641
- **UTC Range**: `2017-07-05 11:42:42.084372+00:00` -> `2017-07-05 20:10:14.557249+00:00`
- **Timestamp Format & Timezone**: ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff) — UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)
- **AM/PM Inferred Rows**: 0 (monotonic disagreements: 0)
- **Flow Duration Unit**: microseconds verified=True (median `(pkts/dur_s) / Flow Packets/s` ratio = 1.000000)
- **Flow Duration Anomalies**: negative=0, zero=1, max=119,999,998 us
- **Dash-Normalized Label Rows**: 0 | **Attempted -> BENIGN**: 5,876
- **NaN / Null Columns**: None
- **`+inf` Columns**: {'Flow Bytes/s': 1, 'Flow Packets/s': 1}
- **`-inf` Columns**: None
- **Unexpected Negative-Value Columns**: {'Fwd Header Length': 64, 'Bwd Header Length': 88}

**Clean Label Distribution:**

| Label | Count |
| --- | ---: |
| `BENIGN` | 324,996 |
| `DoS Hulk` | 158,468 |
| `DoS GoldenEye` | 7,567 |
| `DoS Slowloris` | 3,859 |
| `DoS Slowhttptest` | 1,740 |
| `Heartbleed` | 11 |

**Observed Attack Windows vs Published Schedule:**

| Clean Attack Label | Flows | Observed First Start (UTC) | Observed Last Start (UTC) |
| --- | ---: | --- | --- |
| `DoS Slowloris` | 3,859 | `2017-07-05 12:48:54.539220+00:00` | `2017-07-05 13:10:11.354291+00:00` |
| `DoS Slowhttptest` | 1,740 | `2017-07-05 13:15:37.936810+00:00` | `2017-07-05 13:35:18.764958+00:00` |
| `DoS Hulk` | 158,468 | `2017-07-05 13:43:23.194704+00:00` | `2017-07-05 14:01:34.512067+00:00` |
| `DoS GoldenEye` | 7,567 | `2017-07-05 14:10:03.231753+00:00` | `2017-07-05 14:19:49.515445+00:00` |
| `Heartbleed` | 11 | `2017-07-05 18:12:15.650811+00:00` | `2017-07-05 18:32:18.242051+00:00` |

Published reference windows:
  - `DoS Slowloris`: `2017-07-05 12:47:00` -> `2017-07-05 13:10:00` UTC (09:47-10:10 ADT)
  - `DoS Slowhttptest`: `2017-07-05 13:14:00` -> `2017-07-05 13:35:00` UTC (10:14-10:35 ADT)
  - `DoS Hulk`: `2017-07-05 13:43:00` -> `2017-07-05 14:07:00` UTC (10:43-11:07 ADT)
  - `DoS GoldenEye`: `2017-07-05 14:10:00` -> `2017-07-05 14:24:00` UTC (11:10-11:24 ADT)
  - `Heartbleed`: `2017-07-05 18:12:00` -> `2017-07-05 18:32:00` UTC (15:12-15:32 ADT)

### Thursday (`thursday.csv`)
- **Encoding**: `utf-8-sig` | **Raw rows**: 362,076 | **Duplicates removed**: 0 | **Clean rows**: 362,076
- **UTC Range**: `2017-07-06 11:59:00.490140+00:00` -> `2017-07-06 20:04:36.513760+00:00`
- **Timestamp Format & Timezone**: ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff) — UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)
- **AM/PM Inferred Rows**: 0 (monotonic disagreements: 0)
- **Flow Duration Unit**: microseconds verified=True (median `(pkts/dur_s) / Flow Packets/s` ratio = 1.000000)
- **Flow Duration Anomalies**: negative=0, zero=1, max=119,999,993 us
- **Dash-Normalized Label Rows**: 0 | **Attempted -> BENIGN**: 1,997
- **NaN / Null Columns**: None
- **`+inf` Columns**: {'Flow Bytes/s': 1, 'Flow Packets/s': 1}
- **`-inf` Columns**: None
- **Unexpected Negative-Value Columns**: {'Fwd Header Length': 99, 'Bwd Header Length': 128}

**Clean Label Distribution:**

| Label | Count |
| --- | ---: |
| `BENIGN` | 290,169 |
| `Infiltration - Portscan` | 71,767 |
| `Web Attack - Brute Force` | 73 |
| `Infiltration` | 36 |
| `Web Attack - XSS` | 18 |
| `Web Attack - SQL Injection` | 13 |

**Observed Attack Windows vs Published Schedule:**

| Clean Attack Label | Flows | Observed First Start (UTC) | Observed Last Start (UTC) |
| --- | ---: | --- | --- |
| `Web Attack - Brute Force` | 73 | `2017-07-06 12:19:27.660566+00:00` | `2017-07-06 12:59:41.197757+00:00` |
| `Web Attack - XSS` | 18 | `2017-07-06 13:16:16.603214+00:00` | `2017-07-06 13:34:28.136241+00:00` |
| `Web Attack - SQL Injection` | 13 | `2017-07-06 13:35:45.732950+00:00` | `2017-07-06 13:42:50.310628+00:00` |
| `Infiltration - Portscan` | 71,767 | `2017-07-06 17:00:31.706755+00:00` | `2017-07-06 18:46:04.331747+00:00` |
| `Infiltration` | 36 | `2017-07-06 17:19:02.547210+00:00` | `2017-07-06 18:44:24.375292+00:00` |

Published reference windows:
  - `Web Attack - Brute Force`: `2017-07-06 12:15:00` -> `2017-07-06 13:00:00` UTC (09:15-10:00 ADT)
  - `Web Attack - XSS`: `2017-07-06 13:15:00` -> `2017-07-06 13:35:00` UTC (10:15-10:35 ADT)
  - `Web Attack - SQL Injection`: `2017-07-06 13:40:00` -> `2017-07-06 13:42:00` UTC (10:40-10:42 ADT)
  - `Infiltration / Portscan`: `2017-07-06 17:19:00` -> `2017-07-06 18:45:00` UTC (14:19-15:45 ADT)

### Friday (`friday.csv`)
- **Encoding**: `utf-8-sig` | **Raw rows**: 547,557 | **Duplicates removed**: 0 | **Clean rows**: 547,557
- **UTC Range**: `2017-07-07 11:59:50.315195+00:00` -> `2017-07-07 20:02:40.977798+00:00`
- **Timestamp Format & Timezone**: ISO-24h UTC with microseconds (YYYY-MM-DD HH:MM:SS.ffffff) — UTC (09:00-17:00 ADT corresponds to ~12:00-20:00 UTC in July)
- **AM/PM Inferred Rows**: 0 (monotonic disagreements: 0)
- **Flow Duration Unit**: microseconds verified=True (median `(pkts/dur_s) / Flow Packets/s` ratio = 1.000000)
- **Flow Duration Anomalies**: negative=0, zero=0, max=119,999,993 us
- **Dash-Normalized Label Rows**: 0 | **Attempted -> BENIGN**: 4,067
- **NaN / Null Columns**: None
- **`+inf` Columns**: None
- **`-inf` Columns**: None
- **Unexpected Negative-Value Columns**: {'Fwd Header Length': 40, 'Bwd Header Length': 89}

**Clean Label Distribution:**

| Label | Count |
| --- | ---: |
| `BENIGN` | 292,611 |
| `Portscan` | 159,066 |
| `DDoS` | 95,144 |
| `Botnet` | 736 |

**Observed Attack Windows vs Published Schedule:**

| Clean Attack Label | Flows | Observed First Start (UTC) | Observed Last Start (UTC) |
| --- | ---: | --- | --- |
| `Botnet` | 736 | `2017-07-07 13:04:13.990571+00:00` | `2017-07-07 14:02:02.808963+00:00` |
| `Portscan` | 159,066 | `2017-07-07 16:55:32.117090+00:00` | `2017-07-07 18:23:53.256235+00:00` |
| `DDoS` | 95,144 | `2017-07-07 18:56:31.796937+00:00` | `2017-07-07 19:16:10.645771+00:00` |

Published reference windows:
  - `Botnet`: `2017-07-07 13:02:00` -> `2017-07-07 14:02:00` UTC (10:02-11:02 ADT)
  - `Portscan`: `2017-07-07 16:55:00` -> `2017-07-07 18:24:00` UTC (13:55-15:24 ADT)
  - `DDoS`: `2017-07-07 18:56:00` -> `2017-07-07 19:16:00` UTC (15:56-16:16 ADT)

## Label-Quality & Methodological Limitations Recorded
- CICIDS2017 is a scripted 5-day lab capture; attack windows occur in concentrated morning/afternoon blocks.
- CNS 2022 fixed CICFlowMeter TCP termination (`FIN`/`RST`), UTC timestamps, and `Attempted` labels, but `Flow Bytes/s` and `Flow Packets/s` can still exhibit `NaN`/`inf` if `Flow Duration == 0` and `-1` sentinels exist in `FWD/Bwd Init Win Bytes` and `ICMP Code/Type`.
- Because onset events are sparse, forecasting statistical power depends directly on the observed onset counts reported above.
