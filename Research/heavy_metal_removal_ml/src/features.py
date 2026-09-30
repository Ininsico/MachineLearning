"""Feature engineering for compositional 16S data.

Ordering of operations (deliberate, and discussed in the README):

1. counts -> within-sample relative abundance (closure)
2. zero replacement (multiplicative replacement or additive pseudocount)
3. centered log-ratio (CLR) transform, applied *within each sample across genera*
4. genus prevalence / mean-abundance filtering
5. StandardScaler (fit on training data only)
6. optional dimensionality reduction when the feature count exceeds a threshold

Steps 5-6 are exposed as sklearn transformers so they can live inside a
``Pipeline`` and be re-fit inside every cross-validation fold. Performing feature
selection outside the CV loop would leak test information into training and
inflate the reported scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .logging_utils import get_logger


def to_relative_abundance(counts: pd.DataFrame) -> pd.DataFrame:
    """Close each sample to a relative-abundance composition summing to 1."""
    totals = counts.sum(axis=0)
    empty = totals[totals <= 0]
    if len(empty):
        raise ValueError(f"Samples with zero total counts: {list(empty.index)}")
    return counts.divide(totals, axis=1)


def multiplicative_replacement(compositions: pd.DataFrame, delta_scale: float = 1.0) -> pd.DataFrame:
    """Replace zeros with a detection-limit value while preserving closure.

    Uses the standard multiplicative replacement: a zero is replaced by
    ``delta = delta_scale / N`` (N = total counts of that sample, i.e. roughly one
    sequencing read) and the non-zero entries are rescaled by ``1 - z * delta`` so
    each composition still sums to 1.
    """
    frame = compositions.astype(float).copy()
    values = frame.to_numpy(copy=True)
    n_samples = values.shape[0]
    for i in range(n_samples):
        row = values[i]
        zeros = row <= 0
        n_zero = int(zeros.sum())
        if n_zero == 0:
            continue
        total = row.sum()
        delta = delta_scale / max(total, 1.0)
        if n_zero * delta >= 1.0:
            delta = 0.5 / n_zero
        row[zeros] = delta
        row[~zeros] = row[~zeros] * (1.0 - n_zero * delta)
        values[i] = row
    replaced = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    return replaced.divide(replaced.sum(axis=1), axis=0)


def additive_pseudocount(compositions: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    frame = compositions.astype(float).copy()
    frame[frame <= 0] = pseudocount
    return frame.divide(frame.sum(axis=1), axis=0)


def zero_replacement(compositions: pd.DataFrame, method: str = "multiplicative_replacement",
                     pseudocount: float = 1e-6) -> pd.DataFrame:
    method = (method or "").strip().lower()
    if method in {"multiplicative_replacement", "multiplicative", "mrr"}:
        return multiplicative_replacement(compositions)
    if method in {"pseudocount", "additive", "additive_pseudocount"}:
        return additive_pseudocount(compositions, pseudocount)
    raise ValueError(f"Unknown zero-handling method: {method}")


def clr_transform(compositions: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio transform applied within each row (sample)."""
    values = np.asarray(compositions, dtype=float)
    if np.any(values <= 0):
        raise ValueError("CLR requires strictly positive values; apply zero replacement first")
    log_values = np.log(values)
    centred = log_values - log_values.mean(axis=1, keepdims=True)
    return pd.DataFrame(centred, index=compositions.index, columns=compositions.columns)


def drop_low_depth_samples(counts: pd.DataFrame, min_library_size: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """Discard samples with too few assigned reads to give a stable composition.

    Real MGnify amplicon analyses vary by orders of magnitude in sequencing depth,
    and a handful of reads cannot support a CLR transform. Samples removed here are
    reported so the loss of n is visible rather than silent.
    """
    totals = counts.sum(axis=0)
    keep = totals >= int(min_library_size)
    dropped = [str(sample) for sample in totals.index[~keep]]
    if dropped:
        get_logger().warning(
            "Dropped %d sample(s) with fewer than %d assigned reads: %s",
            len(dropped), int(min_library_size),
            ", ".join(dropped[:12]) + (" ..." if len(dropped) > 12 else ""),
        )
    return counts.loc[:, keep], dropped


def drop_empty_samples(counts: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove samples left with zero counts after taxon filtering."""
    totals = counts.sum(axis=0)
    keep = totals > 0
    dropped = [str(sample) for sample in totals.index[~keep]]
    if dropped:
        get_logger().warning(
            "Dropped %d sample(s) whose entire composition was removed by taxon filtering: %s",
            len(dropped), ", ".join(dropped[:12]) + (" ..." if len(dropped) > 12 else ""),
        )
    return counts.loc[:, keep], dropped


def filter_taxa(counts: pd.DataFrame, min_mean_relative_abundance: float = 0.001,
                min_prevalence: float = 0.0, abundance_metric: str = "mean_overall"
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop low-abundance / low-prevalence taxa; return (kept, summary table).

    ``abundance_metric`` selects how "mean relative abundance" is computed, which
    matters a great deal for data pooled across heterogeneous studies:

    ``mean_overall``      mean over *all* samples, treating a non-detection as zero.
                          This is the strict reading of the study specification. A
                          genus that dominates one study but is absent elsewhere is
                          penalised heavily.
    ``mean_when_present`` mean over only the samples where the genus was detected,
                          paired with a separate prevalence requirement. This is the
                          conventional ecological reading and is far less punitive
                          for multi-study pools.

    Both are reported so the sensitivity of the results to this choice is visible.
    """
    relative = to_relative_abundance(counts)
    mean_overall = relative.mean(axis=1)
    detected = counts > 0
    prevalence = detected.mean(axis=1)
    mean_when_present = (relative.where(detected).mean(axis=1)).fillna(0.0)

    metric = (abundance_metric or "mean_overall").strip().lower()
    if metric == "mean_when_present":
        abundance_used = mean_when_present
    elif metric == "mean_overall":
        abundance_used = mean_overall
    else:
        raise ValueError(f"Unknown abundance_metric: {abundance_metric!r}")

    keep = (abundance_used >= min_mean_relative_abundance) & (prevalence >= min_prevalence)
    summary = pd.DataFrame(
        {
            "genus": counts.index,
            "mean_relative_abundance": mean_overall.reindex(counts.index).to_numpy(),
            "mean_relative_abundance_when_present": mean_when_present.reindex(counts.index).to_numpy(),
            "prevalence": prevalence.reindex(counts.index).to_numpy(),
            "abundance_metric_used": metric,
            "total_counts": counts.sum(axis=1).reindex(counts.index).to_numpy(),
            "retained": keep.reindex(counts.index).to_numpy(),
        }
    ).sort_values("mean_relative_abundance", ascending=False)
    get_logger().info(
        "Taxon filter (%s): kept %d/%d genera (abundance >= %.5f, prevalence >= %.3f)",
        metric, int(keep.sum()), len(counts), min_mean_relative_abundance, min_prevalence,
    )
    return counts.loc[keep], summary


@dataclass
class CompositionalFeatureSpace:
    """Container for the CLR-transformed genus features of every sample."""

    clr: pd.DataFrame
    relative: pd.DataFrame
    counts: pd.DataFrame
    phylum_map: Dict[str, str]
    simulated: bool = False
    provenance: dict = field(default_factory=dict)

    @property
    def genera(self) -> List[str]:
        return list(self.clr.columns)

    @property
    def samples(self) -> List[str]:
        return list(self.clr.index)


def build_feature_space(
    counts: pd.DataFrame,
    cfg,
    phylum_map: Optional[Dict[str, str]] = None,
    simulated: bool = False,
    provenance: Optional[dict] = None,
) -> CompositionalFeatureSpace:
    """Run the full feature-engineering chain on a genus x sample count table."""
    logger = get_logger()
    pcfg = cfg.preprocessing
    phylum_map = dict(phylum_map or {})
    initial_samples = int(counts.shape[1])

    counts, low_depth = drop_low_depth_samples(
        counts, int(getattr(pcfg, "min_library_size", 50))
    )
    if counts.empty:
        raise ValueError("Every sample was removed by the library-size filter")

    kept, summary = filter_taxa(
        counts,
        min_mean_relative_abundance=float(pcfg.min_mean_relative_abundance),
        min_prevalence=float(pcfg.min_prevalence),
        abundance_metric=str(getattr(pcfg, "abundance_metric", "mean_overall")),
    )
    if kept.empty:
        raise ValueError("No genera survived filtering; loosen preprocessing thresholds")

    kept, emptied = drop_empty_samples(kept)
    if kept.empty:
        raise ValueError("No samples retained any counts after taxon filtering")

    if len(kept.columns) < initial_samples:
        logger.info(
            "Sample retention: %d/%d samples usable after QC (%d low-depth, %d emptied by taxon filter)",
            kept.shape[1], initial_samples, len(low_depth), len(emptied),
        )

    relative = to_relative_abundance(kept).T
    replaced = zero_replacement(
        relative,
        method=str(pcfg.clr_zero_handling),
        pseudocount=float(pcfg.clr_pseudocount),
    )
    clr = clr_transform(replaced)
    logger.info(
        "Feature space: %d samples x %d CLR-transformed genera (zero handling: %s)",
        clr.shape[0], clr.shape[1], pcfg.clr_zero_handling,
    )
    return CompositionalFeatureSpace(
        clr=clr,
        relative=relative,
        counts=kept,
        phylum_map={g: phylum_map.get(g, "unclassified_phylum") for g in kept.index},
        simulated=simulated,
        provenance=provenance or {},
    )


def build_selector(cfg, task: str, n_features: int) -> Optional[object]:
    """Return an sklearn selector when dimensionality exceeds the configured threshold."""
    scfg = cfg.preprocessing.feature_selection
    mode = str(scfg.mode).lower()
    triggered = mode == "always" or (mode == "auto" and n_features > int(scfg.dim_threshold))
    if mode == "never" or not triggered:
        get_logger().info(
            "Feature selection: skipped (%d features, threshold %d, mode=%s)",
            n_features, int(scfg.dim_threshold), mode,
        )
        return None

    method = str(scfg.method).lower()
    if method == "variance_threshold":
        selector = VarianceThreshold(threshold=float(scfg.variance_threshold))
    elif method == "select_k_best":
        k = max(1, min(int(scfg.k), n_features))
        score_func = f_classif if task == "classification" else f_regression
        # Drop constant columns first: an all-zero one-hot phylum column has zero
        # within-group variance and makes the ANOVA F-score undefined.
        selector = Pipeline(
            steps=[
                ("drop_constant", VarianceThreshold(threshold=0.0)),
                ("k_best", SelectKBest(score_func=score_func, k=k)),
            ]
        )
    else:
        raise ValueError(f"Unknown feature-selection method: {method}")
    get_logger().info(
        "Feature selection: %s applied (%d features > threshold %d)", method, n_features, int(scfg.dim_threshold)
    )
    return selector


def make_scaler(cfg, enabled: Optional[bool] = None) -> Optional[StandardScaler]:
    use = bool(cfg.preprocessing.standardize) if enabled is None else enabled
    return StandardScaler(with_mean=True, with_std=True) if use else None


def align_to_genera(frame: pd.DataFrame, genera: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    """Reindex a genus x sample table onto a fixed genus set (missing genera -> 0)."""
    aligned = frame.reindex(index=list(genera))
    return aligned.fillna(fill_value)


def transform_new_samples(
    counts: pd.DataFrame,
    genera: Sequence[str],
    cfg,
    scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    """Project new samples (e.g. in-house SM/OS) into the training feature space.

    The zero-replacement + CLR step is recomputed within each new sample, exactly
    as during training, and the *frozen* training scaler is then applied.
    """
    aligned = align_to_genera(counts, genera)
    relative = to_relative_abundance(aligned).T
    replaced = zero_replacement(
        relative,
        method=str(cfg.preprocessing.clr_zero_handling),
        pseudocount=float(cfg.preprocessing.clr_pseudocount),
    )
    clr = clr_transform(replaced)
    if scaler is not None:
        scaled = scaler.transform(clr)
        clr = pd.DataFrame(scaled, index=clr.index, columns=clr.columns)
    return clr
