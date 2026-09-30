"""Construction of the labelled datasets the models are trained on.

Three complementary datasets are built, because the public 16S data carry no
metal-removal measurements of their own:

Track A - genus-level capability classifier (the primary supervised task)
    rows     : genera recovered from the public wastewater samples
    features : the genus' CLR-transformed relative abundance across all samples,
               plus (optionally) phylum one-hot membership and prevalence stats
    label    : 1 if the genus has documented heavy-metal removal/tolerance,
               0 if no such documentation was located

Track B - sample-level community metal-removal-potential index
    rows     : samples (public communities, plus the in-house SM/OS holdout)
    value    : abundance-weighted mean genus metal-activity, i.e. an interpretable
               "what fraction of this community is made of documented metal-active
               genera" score. This is an index, not a trained regressor.

Track C - per-metal removal-efficiency regressor (only if enough quantitative
          literature values exist; the mined table currently supports very few
          genera, and the code reports that limitation instead of hiding it)

EXPLICIT WARNING ABOUT TRACK A
------------------------------
The Track A label is a curated literature statement about the genus, while the
features are abundance summaries of that same genus. A model can therefore reach
high discriminative performance by re-deriving the curation rather than by
learning any metal-removal mechanism. To make this measurable rather than
implicit, ``build_track_a`` also emits feature-group masks so the pipeline can
report:
  * performance using abundance features only,
  * performance using phylum membership only,
  * the increment contributed by phylum membership,
  * performance of a trivial "genus name -> literature table" lookup.
The ablation results are reported in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import CompositionalFeatureSpace, transform_new_samples
from .logging_utils import get_logger


@dataclass
class TrackADataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    genera: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def build_track_a(feature_space: CompositionalFeatureSpace, labels: pd.DataFrame, cfg) -> TrackADataset:
    """Assemble the genus-level literature-supervised classification dataset.

    Labelling rule requested for this study:
      * label 1 - the genus has documented heavy-metal removal/tolerance in the
        curated literature table;
      * label 0 - no such documentation was located for the genus.

    Every genus recovered from the public 16S data is therefore scored: annotated
    genera take their curated label, and all remaining recovered genera become
    label 0. Phylum membership is taken from the observed taxonomy wherever
    available, falling back to the curated table only when the observed lineage
    does not resolve the phylum.
    """
    logger = get_logger()
    acfg = cfg.dataset.track_a
    annotated = labels.set_index("genus").to_dict("index")

    observed_phylum = dict(feature_space.phylum_map)
    curated_phylum = {genus: info["phylum"] for genus, info in annotated.items()}
    phylum_map = {
        genus: observed_phylum.get(genus) or curated_phylum.get(genus, "unclassified_phylum")
        for genus in feature_space.genera
    }

    abundance_matrix = feature_space.clr.T
    sample_columns = list(abundance_matrix.columns)

    rows: List[pd.Series] = []
    kept_genera: List[str] = []
    dropped: List[str] = []
    labels_out: List[int] = []
    annotated_flags: List[bool] = []

    relative = feature_space.relative
    prevalence = (feature_space.counts > 0).mean(axis=1)
    mean_rel = relative.mean(axis=0)

    for genus in abundance_matrix.index:
        present = int((feature_space.counts.loc[genus] > 0).sum())
        if present < int(acfg.min_samples_present):
            dropped.append(genus)
            continue
        info = annotated.get(genus)
        label = int(info["label"]) if info is not None else 0
        rows.append(abundance_matrix.loc[genus])
        kept_genera.append(genus)
        labels_out.append(label)
        annotated_flags.append(info is not None)

    if not rows:
        raise ValueError(
            "No genera survived the dataset construction filters. Check acquisition output "
            "and the min_samples_present threshold."
        )

    X = pd.DataFrame(rows, index=kept_genera)
    feature_groups = {"abundance": sample_columns}

    if bool(acfg.include_phylum_features):
        phyla = sorted({phylum_map.get(g, "unclassified_phylum") for g in kept_genera})
        phylum_features = pd.DataFrame(
            0.0, index=kept_genera, columns=[f"phylum_{p}" for p in phyla], dtype=float
        )
        for genus in kept_genera:
            phylum_features.loc[genus, f"phylum_{phylum_map.get(genus, 'unclassified_phylum')}"] = 1.0
        X = pd.concat([X, phylum_features], axis=1)
        feature_groups["phylum"] = list(phylum_features.columns)

    if bool(acfg.include_prevalence_features):
        extra = pd.DataFrame(
            {
                "genus_prevalence": prevalence.reindex(kept_genera).to_numpy(),
                "genus_mean_relative_abundance": mean_rel.reindex(kept_genera).to_numpy(),
            },
            index=kept_genera,
        )
        X = pd.concat([X, extra], axis=1)
        feature_groups["prevalence"] = list(extra.columns)

    y = pd.Series(labels_out, index=kept_genera, name="label")
    groups = pd.Series(kept_genera, index=kept_genera, name="genus")

    n_annotated = int(sum(annotated_flags))
    logger.info(
        "Track A dataset: %d genera x %d features (%d labelled 1 / %d labelled 0)",
        X.shape[0], X.shape[1], int((y == 1).sum()), int((y == 0).sum()),
    )
    logger.info(
        "  %d genera carry a curated literature entry; %d further genera are labelled 0 "
        "because no documentation was located (absence of evidence, not evidence of absence)",
        n_annotated, len(kept_genera) - n_annotated,
    )
    if len(dropped):
        logger.info("  dropped %d genera observed in fewer than %d samples",
                    len(dropped), int(acfg.min_samples_present))

    return TrackADataset(X=X, y=y, groups=groups, feature_groups=feature_groups,
                         genera=kept_genera, dropped=dropped)


def build_track_c(feature_space: CompositionalFeatureSpace, efficiencies: pd.DataFrame,
                  labels: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Per-metal regression dataset: rows = genera, targets = documented efficiency %."""
    logger = get_logger()
    available = set(feature_space.clr.columns)
    subset = efficiencies[efficiencies["genus"].isin(available)].copy()
    if subset.empty:
        logger.warning("Track C skipped: no curated genus with a quantitative efficiency was recovered")
        return pd.DataFrame(), pd.DataFrame(), {}

    abundances = feature_space.clr.T
    targets: Dict[str, pd.Series] = {}
    feature_sets: Dict[str, List[str]] = {}

    for metal, group in subset.groupby("metal"):
        series = group.groupby("genus")["efficiency_pct"].mean()
        series = series[series.index.isin(abundances.index)]
        if len(series) < int(cfg.dataset.track_c.min_genera):
            logger.warning(
                "Track C: metal %s has only %d genus observation(s); skipping (need >= %d)",
                metal, len(series), int(cfg.dataset.track_c.min_genera),
            )
            continue
        targets[metal] = series
        feature_sets[metal] = list(series.index)

    if not targets:
        logger.warning(
            "Track C produced no trainable metal targets. The mined quantitative literature table "
            "currently covers only %d genera (%s), which is too few for a defensible regression. "
            "This is reported as a limitation rather than worked around.",
            subset["genus"].nunique(), ", ".join(sorted(subset["genus"].unique())),
        )
        return pd.DataFrame(), pd.DataFrame(), {}

    X = abundances.loc[sorted({g for s in targets.values() for g in s.index})]
    y = pd.DataFrame({metal: series.reindex(X.index) for metal, series in targets.items()})
    logger.info("Track C dataset: %d genera x %d metals", X.shape[0], y.shape[1])
    return X, y, feature_sets


def community_metal_activity_index(
    relative: pd.DataFrame,
    labels: pd.DataFrame,
    mode: str = "binary_knowledge",
) -> pd.DataFrame:
    """Abundance-weighted metal-activity index per sample.

    ``mode = 'binary_knowledge'`` weights documented genera by 1 and undocumented
    by 0, yielding the interpretable quantity "fraction of the community composed
    of genera with documented metal removal/tolerance". The index is normalised by
    the relative abundance that carr ies an annotation, so unclassified reads do not
    dilute it; both the normalised and raw forms are returned.
    """
    label_map = dict(zip(labels["genus"], labels["label"]))
    genera = [g for g in relative.columns if g in label_map]
    if not genera:
        raise ValueError("No curated genera found in the community table")

    annotated = relative[genera]
    weights = np.array([float(label_map[g]) for g in genera])
    raw = annotated.to_numpy() @ weights
    total_annotated = annotated.sum(axis=1).to_numpy()
    normalised = np.divide(raw, total_annotated, out=np.zeros_like(raw), where=total_annotated > 0)

    return pd.DataFrame(
        {
            "sample": relative.index,
            "documented_active_fraction_raw": raw,
            "annotated_fraction_of_community": total_annotated,
            "metal_activity_index": normalised,
            "n_annotated_genera": [int((annotated.loc[s] > 0).sum()) for s in relative.index],
        }
    ).set_index("sample")


def expected_removal_from_literature(
    relative: pd.DataFrame,
    efficiencies: pd.DataFrame,
    metal: str,
) -> pd.Series:
    """Literature-weighted expectation of removal % for one metal in each sample.

    For every genus with a documented efficiency for ``metal`` we take
    ``efficiency/100 * relative abundance`` and sum. The result is intentionally
    conservative: a genus occupying 5% of the community can contribute at most 5
    percentage points, so the score is a lower bound that ignores synergistic
    community effects.
    """
    subset = efficiencies[efficiencies["metal"].str.lower() == metal.lower()]
    if subset.empty:
        return pd.Series(0.0, index=relative.index, name=f"expected_{metal}")
    lookup = subset.groupby("genus")["efficiency_pct"].mean() / 100.0
    shared = [g for g in relative.columns if g in lookup.index]
    if not shared:
        return pd.Series(0.0, index=relative.index, name=f"expected_{metal}")
    contribution = relative[shared].to_numpy() @ lookup.loc[shared].to_numpy()
    return pd.Series(contribution, index=relative.index, name=f"expected_{metal}")


def build_inhouse_counts(communities: pd.DataFrame, genera: Sequence[str]) -> pd.DataFrame:
    """Genus x sample count-like table for the in-house samples in the training space.

    In-house compositions are percentages rather than read counts; we keep them on
    a common scale by treating the community projection onto the training genus set
    as the observation, and note this asymmetry in the README.
    """
    from .inhouse import genus_relative_abundance

    genus_level = genus_relative_abundance(communities)
    pivot = genus_level.pivot_table(index="genus", columns="sample_id",
                                    values="rel_abundance_pct", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(index=list(genera)).fillna(0.0)
    return pivot


def project_inhouse(feature_space: CompositionalFeatureSpace, communities: pd.DataFrame, cfg) -> pd.DataFrame:
    """Project in-house samples into the frozen training feature space.

    Samples whose taxa lie entirely outside the training genus set cannot be
    projected (there is no composition to close) and are reported and skipped
    rather than silently zero-filled, which would produce a meaningless row.
    """
    logger = get_logger()
    counts = build_inhouse_counts(communities, feature_space.genera)
    totals = counts.sum(axis=0)
    empty = [str(sample) for sample in totals.index[totals <= 0]]
    if empty:
        logger.warning(
            "Cannot project %s into the training feature space: none of its genera were observed "
            "in the public training data, so there is nothing to close to a composition",
            ", ".join(empty),
        )
    scorable = counts.loc[:, totals > 0]
    if scorable.empty:
        logger.error("No in-house sample could be projected into the training feature space")
        return pd.DataFrame(columns=feature_space.genera)
    return transform_new_samples(scorable, feature_space.genera, cfg, scaler=None)
