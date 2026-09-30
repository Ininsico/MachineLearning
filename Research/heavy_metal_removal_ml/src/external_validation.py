"""External validation on the two in-house biofilms (SM, OS).

STRUCTURAL NOTE - why this module works the way it does
-------------------------------------------------------
The Track A classifier is trained with **genera as observations**: each row is a
genus and its features are that genus' abundance profile across the public
samples. SM and OS are new *samples*, not new genera, so they cannot be passed to
that classifier directly - doing so would be a category error.

What *can* legitimately be done, and what this module does, is:

1. Ask the trained classifier for its per-genus probability that a genus carries
   documented metal-removal capability, P(active | genus).
2. Aggregate those probabilities across the genera actually present in SM / OS,
   weighted by each genus' relative abundance in that sample. This produces a
   sample-level "predicted metal-activity score" in [0, 1] with a bootstrap
   confidence interval.
3. Compare that score, a literature-weighted per-metal expectation, and the
   directly interpretable "fraction of the community made of documented-active
   genera", against the observed removal efficiencies computed from the measured
   before/after concentrations.

Step 2 is an aggregation designed by us, not the classifier's native output; the
README states this explicitly. A null distribution obtained by shuffling genus
labels is reported alongside the score so the reader can judge whether the
aggregate carries information beyond the curated list itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .datasets import expected_removal_from_literature
from .inhouse import genus_relative_abundance
from .logging_utils import get_logger


@dataclass
class SampleValidation:
    sample_id: str
    predicted_activity_score: float
    ci_low: float
    ci_high: float
    null_mean: float
    null_p_value: float
    documented_active_fraction: float
    annotated_fraction_of_community: float
    taxonomy_coverage: float = float("nan")
    n_scored_genera: int = 0
    scorable: bool = True
    note: str = ""
    per_genus: pd.DataFrame = field(default_factory=pd.DataFrame)


def per_genus_probabilities(pipeline, X_train: pd.DataFrame, genera: Sequence[str],
                            labels: pd.DataFrame) -> pd.DataFrame:
    """Trained classifier's P(metal-active) for every genus in the training table."""
    probabilities = pipeline.predict_proba(X_train.to_numpy())[:, 1]
    label_map = dict(zip(labels["genus"], labels["label"]))
    phylum_map = dict(zip(labels["genus"], labels["phylum"]))
    return pd.DataFrame(
        {
            "genus": list(genera),
            "p_metal_active": probabilities,
            "curated_label": [int(label_map.get(g, -1)) for g in genera],
            "phylum": [phylum_map.get(g, "unclassified_phylum") for g in genera],
        }
    )


def _weighted_score(weights: pd.Series, abundance: pd.Series) -> float:
    aligned = abundance.reindex(weights.index).fillna(0.0)
    total = aligned.sum()
    if total <= 0:
        return float("nan")
    return float((weights * aligned).sum() / total)


def score_sample(
    sample_id: str,
    per_genus: pd.DataFrame,
    abundance: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> SampleValidation:
    """Abundance-weighted aggregate score with bootstrap CI and a shuffled null.

    A sample can be *unscorable* by the trained classifier when none of its genera
    appear in the training taxonomy - a realistic outcome, since the training pool
    comes from six unrelated public studies while the in-house biofilms host taxa
    those studies may never have detected. Unscorable samples are reported with a
    diagnostic taxonomy-coverage fraction and a NaN score rather than aborting the
    pipeline, because the literature-based index remains computable for them.
    """
    weights = per_genus.set_index("genus")["p_metal_active"]
    labels = per_genus.set_index("genus")["curated_label"]

    present = abundance[abundance > 0]
    used = present.index.intersection(weights.index)
    total_mass = float(present.sum())
    coverage = float(present.reindex(used).sum() / total_mass) if total_mass > 0 else 0.0

    if len(used) == 0:
        return SampleValidation(
            sample_id=sample_id,
            predicted_activity_score=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            null_mean=float("nan"),
            null_p_value=float("nan"),
            documented_active_fraction=float("nan"),
            annotated_fraction_of_community=float("nan"),
            taxonomy_coverage=0.0,
            n_scored_genera=0,
            scorable=False,
            note=(
                f"none of the {len(present)} genera in {sample_id} occur in the training taxonomy "
                f"({weights.shape[0]} genera), so the trained classifier cannot score this sample"
            ),
        )

    abundance_used = present.reindex(used)
    score = _weighted_score(weights.reindex(used), abundance_used)

    rng = np.random.default_rng(seed)
    share = (abundance_used / abundance_used.sum()).to_numpy()
    boot = np.array(
        [
            float((rng.choice(weights.reindex(used).to_numpy(), size=len(used), replace=True,
                              p=share) * share).sum())
            for _ in range(int(n_bootstrap))
        ]
    )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    shuffled_scores = np.array(
        [
            _weighted_score(
                pd.Series(rng.permutation(weights.reindex(used).to_numpy()), index=used),
                abundance_used,
            )
            for _ in range(max(200, int(n_bootstrap) // 4))
        ]
    )
    null_mean = float(np.nanmean(shuffled_scores))
    p_value = float((np.sum(shuffled_scores >= score) + 1) / (len(shuffled_scores) + 1))

    documented = labels.reindex(used)
    active_abundance = abundance_used[documented == 1].sum()
    annotated_abundance = abundance_used.sum()

    note = ""
    if len(used) == 1:
        note = (
            "only one genus was scorable, so the bootstrap confidence interval is degenerate "
            "(zero width) and the shuffled null is uninformative"
        )
    elif coverage < 0.5:
        note = f"the classifier could score only {coverage:.1%} of this sample's community mass"

    return SampleValidation(
        sample_id=sample_id,
        predicted_activity_score=score,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        null_mean=null_mean,
        null_p_value=p_value,
        documented_active_fraction=float(active_abundance / annotated_abundance),
        annotated_fraction_of_community=float(annotated_abundance),
        taxonomy_coverage=coverage,
        n_scored_genera=len(used),
        scorable=True,
        note=note,
        per_genus=per_genus[per_genus["genus"].isin(used)].sort_values("p_metal_active", ascending=False),
    )


def predict_inhouse_samples(
    pipeline,
    X_train: pd.DataFrame,
    genera: Sequence[str],
    labels: pd.DataFrame,
    communities: pd.DataFrame,
    cfg,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Score SM and OS with the trained classifier + abundance weighting."""
    logger = get_logger()
    per_genus = per_genus_probabilities(pipeline, X_train, genera, labels)
    genus_level = genus_relative_abundance(communities)

    n_boot = int(cfg.validation.n_bootstrap)
    seed = int(cfg.project.seed)

    summaries: List[dict] = []
    details: Dict[str, pd.DataFrame] = {}
    for sample_id, group in genus_level.groupby("sample_id"):
        abundance = group.set_index("genus")["rel_abundance_fraction"]
        result = score_sample(sample_id, per_genus, abundance, n_boot, seed)
        summaries.append(
            {
                "sample_id": result.sample_id,
                "scorable": result.scorable,
                "predicted_activity_score": result.predicted_activity_score,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "null_mean": result.null_mean,
                "null_p_value": result.null_p_value,
                "documented_active_fraction": result.documented_active_fraction,
                "annotated_fraction_of_community": result.annotated_fraction_of_community,
                "taxonomy_coverage": result.taxonomy_coverage,
                "n_scored_genera": result.n_scored_genera,
                "note": result.note,
            }
        )
        details[result.sample_id] = result.per_genus
        if not result.scorable:
            logger.warning("  %s: NOT SCORABLE BY THE MODEL - %s", result.sample_id, result.note)
            logger.warning(
                "    the literature-based index is still reported for %s below; only the "
                "classifier-derived score is unavailable", result.sample_id,
            )
        else:
            logger.info(
                "  %s: predicted activity score %.3f [95%% CI %.3f-%.3f]; scored %d genus/genera "
                "covering %.1f%% of community mass; documented-active fraction %.3f; shuffled-null "
                "mean %.3f (p=%.3f)",
                result.sample_id, result.predicted_activity_score, result.ci_low, result.ci_high,
                result.n_scored_genera, 100 * result.taxonomy_coverage,
                result.documented_active_fraction, result.null_mean, result.null_p_value,
            )
            if result.note:
                logger.warning("    caveat: %s", result.note)

    frame = pd.DataFrame(summaries).sort_values("sample_id").reset_index(drop=True)
    if (frame["scorable"] == False).all():  # noqa: E712
        logger.error(
            "NEITHER in-house sample could be scored by the trained classifier. No model-based "
            "external validation is possible; only the literature-index comparison below is valid."
        )
    return frame, details


def compare_to_observed(
    observed: pd.DataFrame,
    communities: pd.DataFrame,
    efficiencies: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Side-by-side comparison of literature-based expectations and measurements."""
    genus_level = genus_relative_abundance(communities)
    relative = genus_level.pivot_table(index="sample_id", columns="genus",
                                      values="rel_abundance_fraction", aggfunc="sum").fillna(0.0)

    metals = list(dict.fromkeys(observed["metal"]))
    rows: List[dict] = []
    for sample_id in relative.index:
        sample_abundance = relative.loc[sample_id]
        for metal in metals:
            expected_series = expected_removal_from_literature(relative, efficiencies, metal)
            expected_pct = float(expected_series.loc[sample_id]) * 100.0

            subset = efficiencies[efficiencies["metal"].str.lower() == metal.lower()]
            contributing = sorted(set(subset["genus"]) & set(sample_abundance[sample_abundance > 0].index))
            contributors_text = ", ".join(
                f"{g}({sample_abundance[g]*100:.2f}%x{subset[subset['genus']==g]['efficiency_pct'].mean():.1f}%)"
                for g in contributing
            )

            observed_row = observed[(observed["sample_id"] == sample_id) & (observed["metal"] == metal)]
            observed_pct = float(observed_row["removal_pct"].iloc[0]) if len(observed_row) else float("nan")
            outcome = observed_row["outcome"].iloc[0] if len(observed_row) else "n/a"

            in_panel = bool(pd.notna(observed_pct))
            rows.append(
                {
                    "sample_id": sample_id,
                    "metal": metal,
                    "has_quantitative_literature": len(contributing) > 0,
                    "expected_removal_pct": expected_pct if contributing else float("nan"),
                    "observed_removal_pct": observed_pct,
                    "observed_outcome": outcome,
                    "discrepancy_pct": (expected_pct - observed_pct) if (contributing and in_panel) else float("nan"),
                    "literature_contributors": contributors_text,
                }
            )

    frame = pd.DataFrame(rows)
    comparable = frame.dropna(subset=["expected_removal_pct", "observed_removal_pct"])
    frame.attrs["n_comparable"] = int(len(comparable))
    if len(comparable) >= 3:
        frame.attrs["pearson_r"] = float(pearsonr(comparable["expected_removal_pct"],
                                                  comparable["observed_removal_pct"]).statistic)
        frame.attrs["spearman_rho"] = float(spearmanr(comparable["expected_removal_pct"],
                                                      comparable["observed_removal_pct"]).statistic)
    return frame


def report_validation(
    scores: pd.DataFrame,
    comparison: pd.DataFrame,
    cfg,
) -> str:
    """Human-readable, deliberately non-congratulatory validation summary."""
    logger = get_logger()
    lines: List[str] = []
    lines.append("EXTERNAL VALIDATION SUMMARY (n = 2 in-house biofilms)")
    lines.append("-" * 74)

    observed = comparison.dropna(subset=["observed_removal_pct"])
    removed = observed[observed["observed_removal_pct"] > 5]
    released = observed[observed["observed_removal_pct"] < -5]
    bdl_full = observed[observed["observed_removal_pct"] >= 99.9]

    lines.append(
        f"Metals measured            : {observed['metal'].nunique()}"
    )
    lines.append(
        f"Metals removed (>5%)       : {removed['metal'].nunique()} "
        f"({', '.join(sorted(removed['metal'].unique())) or 'none'})"
    )
    lines.append(
        f"Metals released (net < -5%): {released['metal'].nunique()} "
        f"({', '.join(sorted(released['metal'].unique())) or 'none'})"
    )
    lines.append(
        f"Below detections (100%)    : {bdl_full['metal'].nunique()} "
        f"({', '.join(sorted(bdl_full['metal'].unique())) or 'none'})"
    )
    lines.append("")

    for row in scores.itertuples():
        if not getattr(row, "scorable", True):
            lines.append(
                f"{row.sample_id}: NOT SCORABLE by the trained classifier - "
                f"{getattr(row, 'note', 'taxonomy not covered by the training set')}"
            )
        else:
            lines.append(
                f"{row.sample_id}: predicted activity score {row.predicted_activity_score:.3f} "
                f"[{row.ci_low:.3f}-{row.ci_high:.3f}] from {row.n_scored_genera} scorable genus/genera "
                f"covering {100 * getattr(row, 'taxonomy_coverage', float('nan')):.1f}% of community mass; "
                f"documented-active fraction {row.documented_active_fraction:.3f}; "
                f"shuffled-null p={row.null_p_value:.3f}"
            )
            if getattr(row, "note", ""):
                lines.append(f"    caveat: {row.note}")

    comparable = comparison.dropna(subset=["expected_removal_pct", "observed_removal_pct"])
    lines.append("")
    if len(comparable) >= 3:
        lines.append(
            f"Literature-expected vs observed removal across {len(comparable)} "
            f"metal/sample pairs: Pearson r = {comparison.attrs.get('pearson_r', float('nan')):.3f}, "
            f"Spearman rho = {comparison.attrs.get('spearman_rho', float('nan')):.3f}"
        )
    else:
        lines.append(
            f"Only {len(comparable)} metal/sample pair(s) have BOTH a quantitative literature value and a "
            "measurement, which is far too few to estimate agreement. No correlation is claimed."
        )

    lines.append("")
    lines.append("INTERPRETATION LIMITS THAT MUST BE CARRIED INTO THE THESIS")
    lines.append(
        "  * n = 2 biofilms. They test that the pipeline runs end-to-end on real samples and that the\n"
        "    outputs are physically plausible; they cannot establish generalisability."
    )
    lines.append(
        "  * The activity score is a curated-literature index aggregated by abundance. It measures\n"
        "    'how much of this community belongs to genera reported as metal-active', NOT a predicted\n"
        "    removal percentage for a given metal."
    )
    lines.append(
        "  * Net release of Mn and Ni and the appearance of Cr were observed. A capability-based index\n"
        "    cannot anticipate release, because release is not represented in the literature label set.\n"
        "    This is a genuine model limitation, not a measurement artefact."
    )
    lines.append(
        "  * The single strong agreement (Bacillus-dominated SM vs arsenate/arsenite removal) is one\n"
        "    metal in one sample and is not independent evidence across the nine metals."
    )

    text = "\n".join(lines)
    logger.info("")
    for line in text.splitlines():
        logger.info(line)
    return text
