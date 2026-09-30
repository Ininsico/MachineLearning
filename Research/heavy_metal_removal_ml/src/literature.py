"""Literature-derived genus annotations used to build supervised labels.

IMPORTANT SCIENTIFIC NOTE
-------------------------
The labels produced here are NOT experimental measurements. They encode a
curated reading of the published literature stating that a given genus has been
reported to remove or tolerate heavy metals. Consequently:

* label 1 = "documented metal removal/tolerance reported in literature"
* label 0 = "no such documentation located" -- this is NOT proof of incapacity.

This distinction is the single largest source of label noise in the pipeline and
is reported explicitly in the README limitations section. The curated table
(data/literature/genus_labels.csv) is the single source of truth and its SHA-256
hash is recorded with every trained model for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import file_sha256
from .logging_utils import get_logger

REQUIRED_LABEL_COLUMNS = {"genus", "phylum", "label", "evidence_level", "mechanism", "needs_verification"}
REQUIRED_EFFICIENCY_COLUMNS = {"genus", "metal", "efficiency_pct"}


def load_genus_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Curated genus label table not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_LABEL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["genus"] = frame["genus"].astype(str).str.strip()
    frame["phylum"] = frame["phylum"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(int)
    frame["needs_verification"] = (
        frame["needs_verification"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False}).fillna(True)
    )
    duplicates = frame["genus"][frame["genus"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate genus entries in curated label table: {duplicates}")

    logger = get_logger()
    logger.info(
        "Curated label table loaded: %d genera (%d positive / %d negative); %d entries flagged needs_verification",
        len(frame),
        int((frame["label"] == 1).sum()),
        int((frame["label"] == 0).sum()),
        int(frame["needs_verification"].sum()),
    )
    logger.info("Curated label table SHA-256: %s", file_sha256(path)[:16])
    return frame


def load_metal_efficiency(path: str | Path) -> pd.DataFrame:
    """Load quantitative per-metal removal efficiencies mined from literature."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metal efficiency table not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_EFFICIENCY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["genus"] = frame["genus"].astype(str).str.strip()
    frame["metal"] = frame["metal"].astype(str).str.strip()
    frame["efficiency_pct"] = frame["efficiency_pct"].astype(float)
    if frame["efficiency_pct"].between(0, 100).all() is False:
        raise ValueError("efficiency_pct values must lie in [0, 100]")
    get_logger().info(
        "Quantitative metal-efficiency table loaded: %d rows spanning %d genera and %d metals",
        len(frame),
        frame["genus"].nunique(),
        frame["metal"].nunique(),
    )
    return frame


def label_lookup(labels: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(labels["genus"], labels["label"]))


def phylum_lookup(labels: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(labels["genus"], labels["phylum"]))


def efficiency_lookup(efficiencies: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Genus x metal -> mean documented efficiency (averaged across strains)."""
    grouped = efficiencies.groupby(["genus", "metal"])["efficiency_pct"].mean()
    return {(genus, metal): float(value) for (genus, metal), value in grouped.items()}
