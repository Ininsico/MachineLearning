"""Public wastewater 16S rRNA acquisition from MGnify.

The study specification calls for the Bioconductor package MGnifyR::

    MgnifyClient() |> doQuery(biome = "wastewater") |> getResult(get.taxa = TRUE, output = "phyloseq")

That exact workflow is implemented in ``r/mgnify_retrieval.R`` and is used
automatically when ``Rscript`` is available. This module is a behaviour-preserving
pure-Python equivalent that talks to the same MGnify REST API, so the pipeline
remains runnable on hosts without R (as here). Both paths emit the same tidy
``genus x sample`` count table, so downstream stages are agnostic to provenance.

MGnify API notes discovered against the live service:
  * query-string filters (``biome_lineage=``) are silently ignored on
    ``/studies`` and ``/samples``; the relationship endpoint
    ``/biomes/{lineage}/studies`` is the effective filter;
  * ``/samples/{acc}/analyses`` does not exist; analyses are reached through
    ``/studies/{acc}/analyses`` and carry a ``sample`` relationship;
  * genus-level counts come from ``/analyses/{acc}/taxonomy/ssu`` (SSU = 16S).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import file_sha256
from .logging_utils import get_logger

TAXONOMY_COLUMNS = ["super kingdom", "phylum", "class", "order", "family", "genus", "species"]

RANK_PREFIXES = ("sk__", "k__", "p__", "c__", "o__", "f__", "g__", "s__")

UNASSIGNED_TOKENS = ("unclassified", "uncultured", "unknown", "unidentified", "incertae", "ambiguous")

PLACEHOLDER_GENERA = {
    "",
    "unclassified",
    "unclassified bacteria",
    "unclassified archaea",
    "unidentified",
    "uncultured",
    "unknown",
    "na",
    "none",
    "other",
}


def parse_silva_lineage(taxonomy: str) -> dict:
    """Parse an MGnify/SILVA lineage string using its fixed rank prefixes.

    The strings look like ``sk__Bacteria;k__;p__Proteobacteria;...;g__Zoogloea``.
    Positional splitting is unreliable because empty ranks are still emitted, so
    the rank prefix is what actually determines the level.
    """
    parsed = {"domain": "", "phylum": "", "genus": "", "species": "", "genus_assigned": False}
    for segment in str(taxonomy).split(";"):
        segment = segment.strip()
        for prefix in RANK_PREFIXES:
            if not segment.startswith(prefix):
                continue
            value = segment[len(prefix):].strip()
            if prefix == "sk__":
                parsed["domain"] = value
            elif prefix == "p__":
                parsed["phylum"] = value
            elif prefix == "g__":
                lowered = value.lower()
                if value and not any(token in lowered for token in UNASSIGNED_TOKENS):
                    parsed["genus"] = value
                    parsed["genus_assigned"] = True
            elif prefix == "s__":
                parsed["species"] = value
            break
    return parsed


@dataclass(frozen=True)
class AnalysisRecord:
    analysis_accession: str
    sample_accession: str
    study_accession: str
    experiment_type: str
    pipeline_version: str
    instrument_platform: str
    instrument_model: str
    biome: str


class MgnifyRestClient:
    """Thin, retrying, paginating client for the MGnify REST API."""

    def __init__(self, base_url: str, timeout: float = 90.0, retries: int = 3,
                 backoff: float = 2.0, sleep_between: float = 0.15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_between = sleep_between
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "heavy-metal-removal-mlp/1.0 (reproducible research pipeline)"})
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.request_count = 0

    def get_json(self, path: str, **params) -> dict:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        self.request_count += 1
        if self.sleep_between:
            time.sleep(self.sleep_between)
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        if "json" not in (response.headers.get("content-type") or ""):
            return {}
        return response.json()

    def paginate(self, path: str, page_size: int = 100, max_pages: int = 500) -> Iterable[dict]:
        page = 1
        while page <= max_pages:
            payload = self.get_json(path, page=page, page_size=page_size)
            data = payload.get("data") or []
            if not data:
                return
            data = data if isinstance(data, list) else [data]
            yield from data
            pagination = (payload.get("meta") or {}).get("pagination") or {}
            if not pagination.get("next"):
                return
            page += 1


def discover_studies(client: MgnifyRestClient, biome_lineages: Sequence[str], max_studies: int) -> List[str]:
    logger = get_logger()
    seen: List[str] = []
    for lineage in biome_lineages:
        payload = client.get_json(f"/biomes/{lineage}/studies", page_size=100)
        studies = [item["id"] for item in (payload.get("data") or [])]
        total = (payload.get("meta") or {}).get("pagination", {}).get("count")
        if studies:
            logger.info("Biome %s -> %s studies on page 1 (API count=%s); harvesting up to %d",
                        lineage, len(studies), total, max_studies)
        else:
            logger.warning("Biome %s returned no studies (lineage may not exist in MGnify)", lineage)

        page = 1
        while len(seen) < max_studies:
            for study in studies:
                if study not in seen:
                    seen.append(study)
            pagination = (payload.get("meta") or {}).get("pagination") or {}
            if not pagination.get("next") or len(seen) >= max_studies:
                break
            page += 1
            payload = client.get_json(f"/biomes/{lineage}/studies", page=page, page_size=100)
            studies = [item["id"] for item in (payload.get("data") or [])]
            if not studies:
                break
        if len(seen) >= max_studies:
            break
    return seen[:max_studies]


def list_amplicon_analyses(client: MgnifyRestClient, study_accession: str,
                           experiment_type: str = "amplicon") -> List[AnalysisRecord]:
    records: List[AnalysisRecord] = []
    for item in client.paginate(f"/studies/{study_accession}/analyses", page_size=100, max_pages=50):
        attrs = item.get("attributes") or {}
        if str(attrs.get("experiment-type", "")).lower() != experiment_type.lower():
            continue
        sample = ((item.get("relationships") or {}).get("sample") or {}).get("data") or {}
        sample_accession = sample.get("id")
        if not sample_accession:
            continue
        records.append(
            AnalysisRecord(
                analysis_accession=item["id"],
                sample_accession=sample_accession,
                study_accession=study_accession,
                experiment_type=str(attrs.get("experiment-type")),
                pipeline_version=str(attrs.get("pipeline-version")),
                instrument_platform=str(attrs.get("instrument-platform")),
                instrument_model=str(attrs.get("instrument-model")),
                biome=str(attrs.get("biome-name") or ""),
            )
        )
    return records


def fetch_taxonomy_rows(client: MgnifyRestClient, analysis_accession: str,
                        cache_dir: Path, page_size: int = 100) -> pd.DataFrame:
    """Fetch the SSU (16S) taxonomy table for one analysis, with on-disk caching."""
    cache_file = cache_dir / f"{analysis_accession}.taxonomy.csv.gz"
    if cache_file.exists():
        return pd.read_csv(cache_file)

    rows: List[dict] = []
    for item in client.paginate(f"/analyses/{analysis_accession}/taxonomy/ssu",
                                page_size=page_size, max_pages=50):
        attrs = item.get("attributes") or {}
        hierarchy = attrs.get("hierarchy") or {}
        row = {"analysis_accession": analysis_accession,
               "count": attrs.get("count"),
               "rank": attrs.get("rank"),
               "lineage": attrs.get("lineage")}
        for column in TAXONOMY_COLUMNS:
            row[column] = hierarchy.get(column) or ""
        rows.append(row)

    frame = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_csv(cache_file, index=False, compression="gzip")
    return frame


def fetch_otu_genus_counts(client: MgnifyRestClient, analysis_accession: str,
                           cache_dir: Path) -> pd.DataFrame:
    """Genus-level counts from an analysis' SSU OTU table.

    Why this exists: the ``/taxonomy/ssu`` endpoint returns a rank-collapsed
    summary that resolves only a few dozen genera per sample and aggregates the
    rest at family/order level. The per-analysis ``*_SSU_OTU.tsv`` download carries
    the full MAPseq assignment per OTU, which resolves roughly four times as many
    genera. More genera means more chances to recover a curated metal-active taxon,
    which is the binding constraint on the supervised task.

    Returns a frame with columns ``genus``, ``phylum``, ``count``.
    """
    cache_file = cache_dir / f"{analysis_accession}.otu_genus.csv.gz"
    if cache_file.exists():
        return pd.read_csv(cache_file)

    downloads = client.get_json(f"/analyses/{analysis_accession}/downloads", page_size=100)
    candidates = [
        item for item in (downloads.get("data") or [])
        if str(item.get("id", "")).endswith("SSU_OTU.tsv")
    ]
    if not candidates:
        return pd.DataFrame(columns=["genus", "phylum", "count"])

    url = (candidates[0].get("links") or {}).get("self")
    if not url:
        return pd.DataFrame(columns=["genus", "phylum", "count"])

    response = client.session.get(url, timeout=client.timeout)
    client.request_count += 1
    if response.status_code != 200:
        return pd.DataFrame(columns=["genus", "phylum", "count"])

    rows: List[dict] = []
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            count = float(parts[1])
        except (TypeError, ValueError):
            continue
        parsed = parse_silva_lineage(parts[2])
        if not parsed["genus_assigned"] or count <= 0:
            continue
        rows.append(
            {
                "genus": parsed["genus"],
                "phylum": parsed["phylum"] or "unclassified_phylum",
                "count": count,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["genus", "phylum", "count"])
    frame = frame.groupby(["genus", "phylum"], as_index=False)["count"].sum()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_file, index=False, compression="gzip")
    return frame


def _genus_counts_for_analysis(client: MgnifyRestClient, analysis_accession: str,
                               cache_dir: Path, source: str) -> pd.DataFrame:
    """Genus-level counts for one analysis from the configured taxonomy source."""
    if source == "taxonomy_ssu":
        frame = fetch_taxonomy_rows(client, analysis_accession, cache_dir)
        return pd.DataFrame(columns=["genus", "phylum", "count"]) if frame.empty else clean_genera(frame)

    otu = fetch_otu_genus_counts(client, analysis_accession, cache_dir)
    if not otu.empty:
        return otu

    # The OTU download may be absent for older pipelines; fall back to the summary.
    frame = fetch_taxonomy_rows(client, analysis_accession, cache_dir)
    return pd.DataFrame(columns=["genus", "phylum", "count"]) if frame.empty else clean_genera(frame)


def clean_genera(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["genus"] = frame["genus"].astype(str).str.strip()
    frame["phylum"] = frame["phylum"].astype(str).str.strip()
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0.0)
    frame["is_placeholder"] = frame["genus"].str.lower().isin(PLACEHOLDER_GENERA)
    frame = frame[(frame["genus"] != "") & (~frame["is_placeholder"])]
    frame = frame[frame["count"] > 0]
    frame["phylum"] = frame["phylum"].replace("", "unclassified_phylum")
    return frame


def acquire_mgnify(cfg, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Harvest a genus x sample count table for wastewater amplicon samples."""
    logger = get_logger()
    mcfg = cfg.acquisition.mgnify
    client = MgnifyRestClient(
        base_url=str(mcfg.base_url),
        timeout=float(mcfg.request_timeout),
        retries=int(mcfg.retries),
        backoff=float(mcfg.backoff_seconds),
        sleep_between=float(mcfg.sleep_between_requests),
    )

    cache_dir = paths["raw"] / "mgnify_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    studies = discover_studies(client, list(mcfg.biome_lineage_hints), int(mcfg.max_studies))
    if not studies:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_studies"}

    records: List[AnalysisRecord] = []
    for study in studies:
        try:
            records.extend(list_amplicon_analyses(client, study, str(mcfg.experiment_type)))
        except requests.RequestException as exc:
            logger.warning("Failed to list analyses for %s: %s", study, exc)
        if len(records) >= int(mcfg.max_analyses):
            break

    if not records:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_amplicon_analyses", "studies": len(studies)}

    # one analysis per sample: prefer the largest recovered taxonomy table
    per_sample: Dict[str, AnalysisRecord] = {}
    for record in records:
        current = per_sample.get(record.sample_accession)
        if current is None or record.pipeline_version > current.pipeline_version:
            per_sample[record.sample_accession] = record
    selected = list(per_sample.values())[: int(mcfg.max_analyses)]
    logger.info("Selected %d unique amplicon samples across %d wastewater studies",
                len(selected), len({r.study_accession for r in selected}))

    tables: List[pd.DataFrame] = []
    provenance: List[dict] = []
    source = str(mcfg.get("taxonomy_source", "otu_table")).lower()
    logger.info("Genus-level taxonomy source: %s", source)

    for index, record in enumerate(selected, start=1):
        try:
            genera = _genus_counts_for_analysis(client, record.analysis_accession, cache_dir, source)
        except requests.RequestException as exc:
            logger.warning("[%d/%d] taxonomy fetch failed for %s: %s",
                           index, len(selected), record.analysis_accession, exc)
            continue
        if genera.empty:
            continue
        aggregated = genera.groupby(["genus", "phylum"], as_index=False)["count"].sum()
        aggregated["sample_accession"] = record.sample_accession
        tables.append(aggregated)
        provenance.append(
            {
                "sample_accession": record.sample_accession,
                "analysis_accession": record.analysis_accession,
                "study_accession": record.study_accession,
                "biome": record.biome,
                "pipeline_version": record.pipeline_version,
                "instrument_platform": record.instrument_platform,
                "instrument_model": record.instrument_model,
                "taxonomy_source": source,
                "n_genera": int(aggregated.shape[0]),
                "total_counts": float(aggregated["count"].sum()),
            }
        )
        if index % 25 == 0 or index == len(selected):
            logger.info("  fetched %d/%d taxonomy tables (%d genera so far across %d samples)",
                        index, len(selected), sum(t.shape[0] for t in tables), len(tables))

    if not tables:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_taxonomy_tables"}

    long = pd.concat(tables, ignore_index=True)
    wide = (
        long.pivot_table(index="genus", columns="sample_accession", values="count",
                         aggfunc="sum", fill_value=0.0)
    )
    phylum_map = (
        long.groupby("genus")["phylum"]
        .agg(lambda values: values.value_counts().index[0])
    )
    wide = wide.sort_index()
    wide.attrs["phylum_map"] = phylum_map.to_dict()

    metadata = pd.DataFrame(provenance).sort_values("sample_accession").reset_index(drop=True)
    info = {
        "status": "ok",
        "source": "MGnify REST API",
        "n_studies": len({r.study_accession for r in selected}),
        "n_samples": int(wide.shape[1]),
        "n_genera": int(wide.shape[0]),
        "n_http_requests": client.request_count,
        "biome_lineages": list(mcfg.biome_lineage_hints),
    }
    return wide, metadata, info


def load_or_acquire_mgnify(cfg, paths: Dict[str, Path], force: bool = False
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    logger = get_logger()
    counts_path = paths["raw"] / "mgnify_genus_counts.csv.gz"
    meta_path = paths["raw"] / "mgnify_sample_metadata.csv"
    info_path = paths["raw"] / "mgnify_acquisition_info.json"

    if not force and counts_path.exists() and meta_path.exists():
        logger.info("Using cached MGnify table: %s", counts_path)
        wide = pd.read_csv(counts_path, index_col=0)
        metadata = pd.read_csv(meta_path)
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        info["status"] = "ok_cached"
        info["cache_sha256_16"] = file_sha256(counts_path)[:16]
        return wide, metadata, info

    wide, metadata, info = acquire_mgnify(cfg, paths)
    if info.get("status") == "ok":
        wide.to_csv(counts_path, compression="gzip")
        metadata.to_csv(meta_path, index=False)
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        logger.info("Saved MGnify genus table: %d genera x %d samples -> %s",
                    wide.shape[0], wide.shape[1], counts_path.name)
    return wide, metadata, info
