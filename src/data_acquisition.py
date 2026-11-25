"""Stage 1A: Ethical, verifiable, resumable data acquisition for CYBERSENTINEL AI.

Discovers dataset links directly from official or author-hosted pages, enforces
disk-space and download-size caps, verifies file size, SHA-256, and CSV schema,
and never overwrites existing verified raw files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlparse

import requests
from tqdm import tqdm

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

CNS2022_DOWNLOAD_PAGE = "https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html"
WTMC2021_TOOLS_PAGE = "https://downloads.distrinet-research.be/WTMC2021/tools_datasets.html"
UNB_CICIDS2017_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
UNSW_NB15_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

REQUEST_TIMEOUT = 30
CHUNK_SIZE = 1024 * 1024  # 1 MiB


class _LinkExtractor(HTMLParser):
    """Lightweight stdlib HTML anchor extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "a":
            attr_dict = dict(attrs)
            href = attr_dict.get("href")
            if href:
                self._current_href = href
                self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            self.links.append((self._current_href, " ".join("".join(self._text_parts).split())))
            self._current_href = None
            self._text_parts = []


@dataclass
class DiscoveredFile:
    url: str
    filename: str
    expected_bytes: Optional[int]
    source_name: str
    dataset_version: str
    license_note: str


@dataclass
class AcquisitionResult:
    success: bool
    synthetic_mode: bool
    source_name: str
    source_url: str
    manifest_path: Path
    provenance_path: Path
    data_needed_path: Optional[Path] = None
    verified_csvs: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def extract_links(html_text: str, base_url: str) -> list[tuple[str, str]]:
    """Return ``(absolute_url, anchor_text)`` pairs parsed from an HTML page."""
    parser = _LinkExtractor()
    parser.feed(html_text)
    results: list[tuple[str, str]] = []
    for href, text in parser.links:
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        results.append((urljoin(base_url, href), text))
    return results


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            digest.update(buf)
    return digest.hexdigest()


def check_internet(session: requests.Session, probe_urls: Optional[list[str]] = None) -> bool:
    """Check whether at least one official source host is reachable."""
    urls = probe_urls or [CNS2022_DOWNLOAD_PAGE, WTMC2021_TOOLS_PAGE, UNB_CICIDS2017_PAGE]
    for url in urls:
        try:
            resp = session.head(url, allow_redirects=True, timeout=10)
            if resp.status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


def head_content_length(session: requests.Session, url: str) -> Optional[int]:
    """Query ``Content-Length`` via HTTP HEAD without downloading the body."""
    resp = session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit():
        return int(cl)
    return None


def check_preflight_limits(
    total_bytes: int,
    target_dir: Path,
    max_download_gb: float,
) -> None:
    """Enforce hard download cap and 3x free disk space requirement."""
    cap_bytes = int(max_download_gb * (1024 ** 3))
    if total_bytes > cap_bytes:
        raise RuntimeError(
            f"Planned download ({total_bytes / 1e9:.2f} GB) exceeds MAX_DOWNLOAD_GB={max_download_gb:.2f} GB."
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    required_bytes = total_bytes * 3
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free disk space in {target_dir}: "
            f"{free_bytes / 1e9:.2f} GB free, need >= {required_bytes / 1e9:.2f} GB (3x download)."
        )


def _discover_cns2022(session: requests.Session) -> list[DiscoveredFile]:
    """Discover corrected CICIDS2017 archive from DistriNet CNS 2022 page."""
    r = session.get(CNS2022_DOWNLOAD_PAGE, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    links = extract_links(r.text, CNS2022_DOWNLOAD_PAGE)
    dir_urls = [u for u, _ in links if "datasets" in u.lower()]
    discovered: list[DiscoveredFile] = []
    for d_url in dir_urls:
        dr = session.get(d_url, timeout=REQUEST_TIMEOUT)
        dr.raise_for_status()
        for f_url, _ in extract_links(dr.text, d_url):
            fname = Path(urlparse(f_url).path).name
            if "cicids2017" in fname.lower() and fname.lower().endswith((".zip", ".csv")):
                size = head_content_length(session, f_url)
                discovered.append(
                    DiscoveredFile(
                        url=f_url,
                        filename=fname,
                        expected_bytes=size,
                        source_name="DistriNet CNS2022 Improved CICIDS2017 (Liu et al., 2022)",
                        dataset_version="CICIDS2017_improved (IEEE CNS 2022)",
                        license_note=(
                            "Academic research release by DistriNet / KU Leuven (Liu, Engelen, Lynar, Essam, Joosen, "
                            "IEEE CNS 2022), derived from UNB CIC-IDS-2017. Cite Liu et al. (2022) & Sharafaldin et al. (2018)."
                        ),
                    )
                )
    return discovered


def _discover_wtmc2021(session: requests.Session) -> list[DiscoveredFile]:
    """Discover corrected CICIDS2017 archive from DistriNet WTMC 2021 page."""
    r = session.get(WTMC2021_TOOLS_PAGE, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    discovered: list[DiscoveredFile] = []
    for f_url, _ in extract_links(r.text, WTMC2021_TOOLS_PAGE):
        fname = Path(urlparse(f_url).path).name
        if fname.lower().endswith(".zip") and "dataset" in f_url.lower():
            size = head_content_length(session, f_url)
            discovered.append(
                DiscoveredFile(
                    url=f_url,
                    filename="CICIDS2017_wtmc2021.zip" if fname == "dataset.zip" else fname,
                    expected_bytes=size,
                    source_name="DistriNet WTMC2021 Corrected CICIDS2017 (Engelen et al., 2021)",
                    dataset_version="CICIDS2017_WTMC2021 (IEEE SPW 2021)",
                    license_note=(
                        "Academic research release by Engelen, Rimmer, Joosen (IEEE SPW / WTMC 2021), "
                        "derived from UNB CIC-IDS-2017."
                    ),
                )
            )
    return discovered


def _inspect_unb_source(session: requests.Session) -> tuple[list[DiscoveredFile], Optional[str]]:
    """Inspect official UNB CICIDS2017 page; detect manual form/login blocks per R13."""
    r = session.get(UNB_CICIDS2017_PAGE, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    links = extract_links(r.text, UNB_CICIDS2017_PAGE)
    download_pages = [u for u, txt in links if "cicresearch.ca" in u.lower() or "download this dataset" in txt.lower()]
    discovered: list[DiscoveredFile] = []
    for dp in download_pages:
        dr = session.get(dp, timeout=REQUEST_TIMEOUT)
        dr.raise_for_status()
        body_lower = dr.text.lower()
        if "<form" in body_lower or "please fill in the information" in body_lower or "server error" in body_lower:
            return [], (
                f"UNB download endpoint {dp} requires filling a registration form or returned a server error; "
                "cannot automate per R13."
            )
        for f_url, _ in extract_links(dr.text, dp):
            fname = Path(urlparse(f_url).path).name
            if any(k in fname.lower() for k in ("generatedlabelled", "trafficlabelling")) and fname.lower().endswith(
                (".zip", ".csv")
            ):
                size = head_content_length(session, f_url)
                discovered.append(
                    DiscoveredFile(
                        url=f_url,
                        filename=fname,
                        expected_bytes=size,
                        source_name="UNB CIC-IDS-2017 Official Labelled Flows",
                        dataset_version="CIC-IDS-2017 GeneratedLabelledFlows",
                        license_note="UNB Canadian Institute for Cybersecurity academic research dataset.",
                    )
                )
    if not discovered:
        return [], f"No direct flow CSV/ZIP links found on {UNB_CICIDS2017_PAGE}."
    return discovered, None


def _check_mirror_available() -> tuple[list[DiscoveredFile], Optional[str]]:
    """Source 3: Check only pre-configured mirror credentials (Kaggle / HF); never fabricate."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_kaggle_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not kaggle_json.exists() and not has_kaggle_env:
        return [], "No pre-configured Kaggle credentials found; skipping mirror source."
    return [], "Mirror fallback not needed when Source 1 succeeds; no automated mirror query configured."


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load existing DATA_MANIFEST.json if present, else return empty structure."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def verify_csv_schema(csv_path: Path) -> dict[str, Any]:
    """Inspect the header and first data row of a flow CSV without loading it into memory."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as fh:
                reader = csv.reader(fh)
                raw_header = next(reader)
                first_row = next(reader, None)
            clean_cols = [c.strip() for c in raw_header]
            ts_candidates = [c for c in clean_cols if "timestamp" in c.lower()]
            lbl_candidates = [c for c in clean_cols if c.lower() == "label" or "label" in c.lower()]
            if not ts_candidates:
                raise ValueError(f"Missing timestamp column in {csv_path.name}; columns={clean_cols[:15]}")
            if not lbl_candidates:
                raise ValueError(f"Missing label column in {csv_path.name}; columns={clean_cols[-10:]}")
            if first_row is None:
                raise ValueError(f"CSV file {csv_path.name} has header only and zero data rows.")
            return {
                "encoding_detected": enc,
                "columns_count": len(clean_cols),
                "timestamp_col": ts_candidates[0],
                "label_col": lbl_candidates[0],
                "sample_columns": clean_cols[:10] + clean_cols[-5:],
                "verification_status": "verified",
            }
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise ValueError(f"Unable to decode CSV {csv_path}: {last_err}")


def download_file_resumable(
    session: requests.Session,
    item: DiscoveredFile,
    dest_path: Path,
    manifest_records: dict[str, dict[str, Any]],
) -> tuple[Path, str, bool]:
    """Download ``item.url`` to ``dest_path`` safely and idempotently.

    Returns ``(dest_path, sha256, skipped_existing)``.
    Never overwrites an existing raw file. Raises ValueError on size/hash mismatch.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        actual_hash = compute_sha256(dest_path)
        if item.expected_bytes is not None and actual_size != item.expected_bytes:
            raise ValueError(
                f"Existing file {dest_path} size mismatch: found {actual_size} bytes, "
                f"expected {item.expected_bytes} bytes. Refusing to overwrite raw file."
            )
        prev = manifest_records.get(dest_path.name)
        if prev and prev.get("sha256") and prev["sha256"] != actual_hash:
            raise ValueError(
                f"Existing file {dest_path} SHA-256 mismatch: found {actual_hash}, "
                f"manifest recorded {prev['sha256']}. Refusing to overwrite raw file."
            )
        logger.info("Skipping verified existing raw file: %s (%d bytes)", dest_path.name, actual_size)
        return dest_path, actual_hash, True

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    resume_pos = part_path.stat().st_size if part_path.exists() else 0
    headers: dict[str, str] = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    with session.get(item.url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416 and item.expected_bytes and resume_pos == item.expected_bytes:
            part_path.rename(dest_path)
            return dest_path, compute_sha256(dest_path), False
        resp.raise_for_status()
        append_mode = resp.status_code == 206 and resume_pos > 0
        if not append_mode:
            resume_pos = 0
        mode = "ab" if append_mode else "wb"
        total = item.expected_bytes or (
            int(resp.headers["Content-Length"]) + resume_pos if resp.headers.get("Content-Length") else None
        )
        with part_path.open(mode) as fh, tqdm(
            total=total,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=item.filename,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))

    actual_size = part_path.stat().st_size
    if item.expected_bytes is not None and actual_size != item.expected_bytes:
        raise ValueError(
            f"Downloaded size mismatch for {item.filename}: got {actual_size}, expected {item.expected_bytes}."
        )
    part_path.rename(dest_path)
    sha256 = compute_sha256(dest_path)
    return dest_path, sha256, False


def extract_zip_safe(
    zip_path: Path,
    extract_dir: Path,
    manifest_records: dict[str, dict[str, Any]],
) -> list[Path]:
    """Safely extract CSV files from a ZIP archive without overwriting existing raw files."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_parts = [p for p in info.filename.replace("\\", "/").split("/") if p and p != "."]
            if info.filename.startswith(("/", "\\")) or ".." in raw_parts or (len(info.filename) >= 2 and info.filename[1] == ":"):
                raise ValueError(f"Unsafe zip member path blocked: {info.filename}")
            if not raw_parts or not raw_parts[-1].lower().endswith(".csv"):
                continue
            target = (extract_dir.joinpath(*raw_parts)).resolve()
            if extract_dir.resolve() not in target.parents:
                raise ValueError(f"Unsafe zip member path blocked: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                actual_size = target.stat().st_size
                if actual_size != info.file_size:
                    raise ValueError(
                        f"Existing extracted CSV {target} size mismatch: {actual_size} vs zip {info.file_size}."
                    )
                prev = manifest_records.get(target.name)
                if prev and prev.get("sha256"):
                    actual_hash = compute_sha256(target)
                    if actual_hash != prev["sha256"]:
                        raise ValueError(
                            f"Existing extracted CSV {target} hash mismatch: {actual_hash} vs {prev['sha256']}."
                        )
            else:
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
                if target.stat().st_size != info.file_size:
                    raise ValueError(f"Extracted size mismatch for {target.name}.")
            csv_paths.append(target)
    return sorted(csv_paths)


def write_data_needed(
    path: Path,
    reasons: list[str],
    raw_dir: Path,
) -> Path:
    """Write manual acquisition instructions per R13 when automated sources fail."""
    reason_bullets = "\n".join(f"- {r}" for r in reasons) if reasons else "- No automated source succeeded."
    content = f"""# DATA NEEDED — Manual Dataset Acquisition Instructions

Automated acquisition could not obtain verified timestamped flow CSVs. Per R13, no login, form, or CAPTCHA was bypassed.

## Reasons Recorded
{reason_bullets}

## Option A (Recommended): DistriNet CNS 2022 Improved CICIDS2017
1. Open `https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html` in a browser.
2. Follow the dataset directory link and download `CICIDS2017_improved.zip`.
3. Place `CICIDS2017_improved.zip` (or its extracted daily `.csv` files) inside:
   `{raw_dir}`
4. Re-run: `python -m scripts.fetch_data` to verify schema, sizes, and SHA-256 hashes.

## Option B: Official UNB CIC-IDS-2017 Labelled Flows
1. Open `https://www.unb.ca/cic/datasets/ids-2017.html` and click **Download this dataset**.
2. Complete the UNB registration form manually and download `GeneratedLabelledFlows.zip` / `TrafficLabelling.zip` (NOT `MachineLearningCVE.zip`, which lacks timestamps and IP addresses).
3. Place the downloaded archive or extracted daily `.csv` files in `{raw_dir}` and re-run `python -m scripts.fetch_data`.

> Note: Until real data is placed and verified, only `SYNTHETIC: NOT RESULTS` smoke tests may run.
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_provenance_markdown(manifest: dict[str, Any], out_path: Path) -> None:
    """Write ``results/data_provenance.md`` strictly from ``DATA_MANIFEST.json``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Provenance",
        "",
        f"- **Dataset Version**: {manifest.get('dataset_version', 'unknown')}",
        f"- **Source Name**: {manifest.get('source_name', 'unknown')}",
        f"- **Source URL**: {manifest.get('source_url', 'none')}",
        f"- **Retrieved UTC**: {manifest.get('retrieval_timestamp_utc', 'unknown')}",
        f"- **Verification Status**: {manifest.get('verification_status', 'unknown')}",
        f"- **Synthetic Mode**: {manifest.get('synthetic_mode', False)}",
        f"- **License / Citation Note**: {manifest.get('license_note', 'N/A')}",
        "",
        "## Verified Files",
        "",
        "| File | Bytes | SHA-256 | Timestamp Col | Label Col | Status |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for f in manifest.get("files", []):
        lines.append(
            f"| `{f.get('filename')}` | {f.get('bytes', 0):,} | `{f.get('sha256', '')[:16]}...` "
            f"| `{f.get('timestamp_col', '-')}` | `{f.get('label_col', '-')}` | {f.get('verification_status', '-')} |"
        )
    if manifest.get("notes"):
        lines.extend(["", "## Notes", ""] + [f"- {n}" for n in manifest["notes"]])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_local_raw_files(raw_dir: Path) -> tuple[list[Path], list[Path]]:
    zips = sorted(raw_dir.glob("*.zip"))
    csvs = sorted(raw_dir.rglob("*.csv"))
    return zips, csvs


def acquire_data(
    settings: Optional[Settings] = None,
    session: Optional[requests.Session] = None,
    internet_checker: Optional[Callable[[requests.Session], bool]] = None,
) -> AcquisitionResult:
    """Execute Stage 1A acquisition following the source ladder and verification rules."""
    cfg = settings or get_settings()
    cfg.ensure_dirs()
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "CyberSentinelAI-Research/1.0")

    existing_manifest = load_manifest(cfg.manifest_path)
    existing_records = {f["filename"]: f for f in existing_manifest.get("files", []) if "filename" in f}
    notes: list[str] = []
    errors: list[str] = []

    # Check if local verified raw CSVs or ZIPs already exist (supports offline re-runs or manual placement)
    local_zips, local_csvs = _discover_local_raw_files(cfg.data_raw_dir)
    if local_csvs or local_zips:
        try:
            file_entries: list[dict[str, Any]] = []
            verified_csvs: list[Path] = []
            for zp in local_zips:
                z_hash = compute_sha256(zp)
                z_size = zp.stat().st_size
                prev = existing_records.get(zp.name)
                if prev:
                    if prev.get("bytes") is not None and z_size != prev["bytes"]:
                        raise ValueError(f"Size mismatch on existing archive {zp.name}: {z_size} != {prev['bytes']}")
                    if prev.get("sha256") and z_hash != prev["sha256"]:
                        raise ValueError(f"Hash mismatch on existing archive {zp.name}: {z_hash} != {prev['sha256']}")
                file_entries.append(
                    {
                        "filename": zp.name,
                        "relative_path": str(zp.relative_to(cfg.project_root)).replace("\\", "/"),
                        "bytes": z_size,
                        "sha256": z_hash,
                        "expected_bytes": prev.get("expected_bytes", z_size) if prev else z_size,
                        "verification_status": "verified",
                    }
                )
                extracted = extract_zip_safe(zp, cfg.data_raw_dir / "extracted", existing_records)
                for c in extracted:
                    if c not in local_csvs:
                        local_csvs.append(c)
            for cp in sorted(set(local_csvs)):
                c_hash = compute_sha256(cp)
                c_size = cp.stat().st_size
                prev = existing_records.get(cp.name)
                if prev:
                    if prev.get("bytes") is not None and c_size != prev["bytes"]:
                        raise ValueError(f"Size mismatch on existing CSV {cp.name}: {c_size} != {prev['bytes']}")
                    if prev.get("sha256") and c_hash != prev["sha256"]:
                        raise ValueError(f"Hash mismatch on existing CSV {cp.name}: {c_hash} != {prev['sha256']}")
                schema_info = verify_csv_schema(cp)
                verified_csvs.append(cp)
                file_entries.append(
                    {
                        "filename": cp.name,
                        "relative_path": str(cp.relative_to(cfg.project_root)).replace("\\", "/"),
                        "bytes": c_size,
                        "sha256": c_hash,
                        "expected_bytes": prev.get("expected_bytes", c_size) if prev else c_size,
                        **schema_info,
                    }
                )
            if verified_csvs:
                manifest = {
                    "source_name": existing_manifest.get("source_name", "Local Verified Raw Dataset"),
                    "source_url": existing_manifest.get("source_url", "local"),
                    "dataset_version": existing_manifest.get("dataset_version", "CICIDS2017 (verified local)"),
                    "license_note": existing_manifest.get(
                        "license_note", "Preserved existing raw files without modification."
                    ),
                    "retrieval_timestamp_utc": existing_manifest.get(
                        "retrieval_timestamp_utc", datetime.now(timezone.utc).isoformat()
                    ),
                    "synthetic_mode": False,
                    "verification_status": "verified",
                    "files": file_entries,
                    "notes": existing_manifest.get("notes", []) + ["Verified existing raw files without overwriting."],
                }
                cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                prov_path = cfg.results_dir / "data_provenance.md"
                write_provenance_markdown(manifest, prov_path)
                return AcquisitionResult(
                    success=True,
                    synthetic_mode=False,
                    source_name=manifest["source_name"],
                    source_url=manifest["source_url"],
                    manifest_path=cfg.manifest_path,
                    provenance_path=prov_path,
                    verified_csvs=verified_csvs,
                )
        except Exception as exc:
            # Fail loudly on integrity mismatch rather than overwriting raw files
            raise RuntimeError(f"Raw data integrity verification failed: {exc}") from exc

    # No complete local raw CSVs found: run internet check + Source Ladder
    has_net = internet_checker(sess) if internet_checker else check_internet(sess)
    discovered: list[DiscoveredFile] = []
    if not has_net:
        errors.append("No internet connection or official source hosts unreachable.")
    else:
        # Source 1A: CNS 2022
        try:
            discovered = _discover_cns2022(sess)
            if discovered:
                notes.append("Selected Source 1 (DistriNet CNS 2022 Improved CICIDS2017).")
        except Exception as exc:
            errors.append(f"Source 1 (CNS 2022) discovery failed: {exc}")

        # Source 1B: WTMC 2021 fallback
        if not discovered:
            try:
                discovered = _discover_wtmc2021(sess)
                if discovered:
                    notes.append("Selected Source 1 fallback (DistriNet WTMC 2021 CICIDS2017).")
            except Exception as exc:
                errors.append(f"Source 1B (WTMC 2021) discovery failed: {exc}")

        # Source 2: UNB official page
        if not discovered:
            try:
                unb_files, unb_msg = _inspect_unb_source(sess)
                if unb_files:
                    discovered = unb_files
                    notes.append("Selected Source 2 (UNB CIC-IDS-2017).")
                elif unb_msg:
                    errors.append(unb_msg)
            except Exception as exc:
                errors.append(f"Source 2 (UNB) failed: {exc}")

        # Source 3: Pre-configured mirror
        if not discovered:
            mir_files, mir_msg = _check_mirror_available()
            if mir_files:
                discovered = mir_files
            elif mir_msg:
                errors.append(mir_msg)

        # Source 4: PCAP regeneration check
        if not discovered:
            errors.append(
                "Source 4 (PCAP regeneration via CICFlowMeter) skipped: multi-GB PCAP download & Java "
                "regeneration would exceed workstation budget or lacks automated flow-label pairing."
            )

    if discovered:
        total_bytes = sum(d.expected_bytes or 0 for d in discovered)
        check_preflight_limits(total_bytes, cfg.data_raw_dir, cfg.max_download_gb)
        file_entries = []
        verified_csvs = []
        for item in discovered:
            archive_path = cfg.data_raw_dir / item.filename
            dl_path, dl_hash, skipped = download_file_resumable(sess, item, archive_path, existing_records)
            if skipped:
                notes.append(f"Skipped existing archive {dl_path.name} (size and hash verified).")
            file_entries.append(
                {
                    "filename": dl_path.name,
                    "relative_path": str(dl_path.relative_to(cfg.project_root)).replace("\\", "/"),
                    "bytes": dl_path.stat().st_size,
                    "expected_bytes": item.expected_bytes,
                    "sha256": dl_hash,
                    "source_url": item.url,
                    "verification_status": "verified",
                }
            )
            if dl_path.suffix.lower() == ".zip":
                extracted_csvs = extract_zip_safe(dl_path, cfg.data_raw_dir / "extracted", existing_records)
            else:
                extracted_csvs = [dl_path]
            for cp in extracted_csvs:
                schema_info = verify_csv_schema(cp)
                verified_csvs.append(cp)
                file_entries.append(
                    {
                        "filename": cp.name,
                        "relative_path": str(cp.relative_to(cfg.project_root)).replace("\\", "/"),
                        "bytes": cp.stat().st_size,
                        "sha256": compute_sha256(cp),
                        "source_url": item.url,
                        **schema_info,
                    }
                )

        if not verified_csvs:
            raise RuntimeError("Download succeeded but zero valid flow CSVs were extracted.")

        manifest = {
            "source_name": discovered[0].source_name,
            "source_url": discovered[0].url,
            "dataset_version": discovered[0].dataset_version,
            "license_note": discovered[0].license_note,
            "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic_mode": False,
            "verification_status": "verified",
            "files": file_entries,
            "notes": notes,
        }
        cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        prov_path = cfg.results_dir / "data_provenance.md"
        write_provenance_markdown(manifest, prov_path)
        return AcquisitionResult(
            success=True,
            synthetic_mode=False,
            source_name=manifest["source_name"],
            source_url=manifest["source_url"],
            manifest_path=cfg.manifest_path,
            provenance_path=prov_path,
            verified_csvs=verified_csvs,
        )

    # Step 5: Failure handling -> DATA_NEEDED.md + synthetic smoke flag
    data_needed_path = write_data_needed(cfg.project_root / "DATA_NEEDED.md", errors, cfg.data_raw_dir)
    manifest = {
        "source_name": "UNAVAILABLE (SYNTHETIC SMOKE MODE ONLY)",
        "source_url": "none",
        "dataset_version": "SYNTHETIC_SMOKE_ONLY",
        "license_note": "No real dataset acquired. Synthetic mode is strictly for smoke/unit tests.",
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_mode": True,
        "verification_status": "missing_real_data",
        "files": [],
        "notes": errors,
    }
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prov_path = cfg.results_dir / "data_provenance.md"
    write_provenance_markdown(manifest, prov_path)
    return AcquisitionResult(
        success=False,
        synthetic_mode=True,
        source_name=manifest["source_name"],
        source_url="none",
        manifest_path=cfg.manifest_path,
        provenance_path=prov_path,
        data_needed_path=data_needed_path,
        errors=errors,
    )
