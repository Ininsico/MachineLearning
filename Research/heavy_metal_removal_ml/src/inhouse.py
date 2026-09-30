"""In-house biofilm samples (SM, OS): community composition and measured metal
concentrations before/after treatment, plus the removal-efficiency arithmetic.

Removal efficiency follows the protocol requested for this study:

    efficiency(%) = (before - after) / before * 100

with two explicit detection-limit conventions:
  * a metal that is present before treatment and below detection afterwards is
    scored as 100% removal (it left solution);
  * a metal that is below detection before treatment and appears afterwards is
    scored as 0% removal.

Negative efficiencies are retained (they indicate net release from the biofilm
or matrix rather than removal) unless ``clamp_negative_removal`` is enabled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

METAL_COLUMNS = ["sample_id", "metal", "before_mg_L", "after_mg_L", "before_bdl", "after_bdl"]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES"})


def load_metal_concentrations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(METAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["before_bdl"] = _as_bool(frame["before_bdl"])
    frame["after_bdl"] = _as_bool(frame["after_bdl"])
    frame["metal"] = frame["metal"].astype(str).str.strip()
    frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
    return frame


def load_communities(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "genus", "rel_abundance_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
    frame["genus"] = frame["genus"].astype(str).str.strip()
    frame["rel_abundance_pct"] = frame["rel_abundance_pct"].astype(float)
    if frame["rel_abundance_pct"].between(0, 100).all() is False:
        raise ValueError("rel_abundance_pct must lie in [0, 100]")
    totals = frame.groupby("sample_id")["rel_abundance_pct"].sum()
    if ((totals - 100.0).abs() > 0.5).any():
        get_logger().warning(
            "Community compositions do not sum to 100%%: %s", totals.round(3).to_dict()
        )
    return frame


def removal_efficiency(
    before: float | None,
    after: float | None,
    before_bdl: bool,
    after_bdl: bool,
    treat_disappearance_as_full_removal: bool = True,
    treat_appearance_as_zero_removal: bool = True,
    clamp_negative: bool = False,
) -> float:
    """Compute removal efficiency for one metal under the stated conventions."""
    if before_bdl and after_bdl:
        return 0.0
    if before_bdl:
        return 0.0 if treat_appearance_as_zero_removal else float("nan")
    if before is None or before <= 0:
        return float("nan")
    if after_bdl:
        return 100.0 if treat_disappearance_as_full_removal else float("nan")
    if after is None:
        return float("nan")

    value = (before - after) / before * 100.0
    if clamp_negative and value < 0:
        return 0.0
    return value


def compute_observed_removal(concentrations: pd.DataFrame, cfg) -> pd.DataFrame:
    vals = cfg.validation
    frame = concentrations.copy()
    frame["removal_pct"] = [
        removal_efficiency(
            row.before_mg_L,
            row.after_mg_L,
            row.before_bdl,
            row.after_bdl,
            treat_disappearance_as_full_removal=bool(vals.treat_bdl_disappearance_as_full_removal),
            treat_appearance_as_zero_removal=bool(vals.treat_bdl_appearance_as_zero_removal),
            clamp_negative=bool(vals.clamp_negative_removal),
        )
        for row in frame.itertuples()
    ]
    frame["outcome"] = frame["removal_pct"].apply(
        lambda v: "removed" if v > 5 else ("released" if v < -5 else "unchanged")
    )
    return frame


def genus_relative_abundance(communities: pd.DataFrame) -> pd.DataFrame:
    """Collapse species-level in-house compositions to genus-level percentages."""
    frame = communities.copy()
    frame["is_unclassified"] = frame["genus"].str.contains("unclassified|unspecified|other", case=False, regex=True)
    assigned = frame[~frame["is_unclassified"]]
    collapsed = (
        assigned.groupby(["sample_id", "genus"], as_index=False)["rel_abundance_pct"].sum()
    )
    collapsed["rel_abundance_fraction"] = collapsed["rel_abundance_pct"] / 100.0
    return collapsed


def describe_inhouse(concentrations: pd.DataFrame, communities: pd.DataFrame) -> None:
    logger = get_logger()
    logger.info("In-house validation samples:")
    for sample_id, group in communities.groupby("sample_id"):
        total = group["rel_abundance_pct"].sum()
        n_taxa = group["genus"].nunique()
        dominant = group.loc[group["rel_abundance_pct"].idxmax()]
        logger.info(
            "  %s: %d taxa, %.2f%% composition accounted, dominant=%s (%.2f%%)",
            sample_id,
            n_taxa,
            total,
            dominant["genus"],
            dominant["rel_abundance_pct"],
        )
    logger.info("Measured metal removal (before -> after):")
    for row in concentrations.itertuples():
        before = "BDL" if row.before_bdl else f"{row.before_mg_L:.3f}"
        after = "BDL" if row.after_bdl else f"{row.after_mg_L:.3f}"
        logger.info("  %-3s %-3s %8s -> %8s mg/L", row.sample_id, row.metal, before, after)
