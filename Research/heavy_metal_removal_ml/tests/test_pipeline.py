"""Unit tests for the heavy-metal-removal ML pipeline.

These cover the parts where a silent error would corrupt the science: the
compositional transform, the detection-limit conventions in the removal formula,
the labelling rule, leakage prevention, and end-to-end determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.acquire_mgnify import clean_genera
from src.datasets import build_track_a, community_metal_activity_index, expected_removal_from_literature
from src.external_validation import score_sample
from src.features import (
    CompositionalFeatureSpace,
    clr_transform,
    filter_taxa,
    multiplicative_replacement,
    to_relative_abundance,
    zero_replacement,
)
from src.inhouse import (
    compute_observed_removal,
    genus_relative_abundance,
    load_communities,
    load_metal_concentrations,
    removal_efficiency,
)
from src.literature import load_genus_labels, load_metal_efficiency
from src.config import load_config, set_global_seed

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture()
def cfg():
    config = load_config(PROJECT_ROOT / "config.yaml")
    config.acquisition.simulate.n_samples = 40
    config.acquisition.simulate.n_genera = 60
    config.preprocessing.feature_selection.mode = "never"
    set_global_seed(42)
    return config


@pytest.fixture()
def labels():
    return load_genus_labels(PROJECT_ROOT / "data" / "literature" / "genus_labels.csv")


@pytest.fixture()
def efficiencies():
    return load_metal_efficiency(PROJECT_ROOT / "data" / "literature" / "genus_metal_efficiency.csv")


# --------------------------------------------------------------------- CLR
def test_clr_rows_sum_to_zero():
    frame = pd.DataFrame([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], columns=list("abc"))
    transformed = clr_transform(frame)
    assert np.allclose(transformed.sum(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_clr_matches_closed_form():
    frame = pd.DataFrame([[0.5, 0.25, 0.25]], columns=list("abc"))
    transformed = clr_transform(frame).iloc[0]
    logs = np.log([0.5, 0.25, 0.25])
    expected = logs - logs.mean()
    assert np.allclose(transformed.to_numpy(), expected)


def test_clr_rejects_non_positive_values():
    frame = pd.DataFrame([[0.5, 0.0, 0.5]], columns=list("abc"))
    with pytest.raises(ValueError, match="strictly positive"):
        clr_transform(frame)


def test_relative_abundance_closes_to_one():
    counts = pd.DataFrame({"s1": [10, 30, 60], "s2": [1, 1, 8]}, index=list("abc"))
    relative = to_relative_abundance(counts)
    assert np.allclose(relative.sum(axis=0).to_numpy(), 1.0)


def test_relative_abundance_rejects_empty_sample():
    counts = pd.DataFrame({"s1": [0, 0, 0]}, index=list("abc"))
    with pytest.raises(ValueError, match="zero total counts"):
        to_relative_abundance(counts)


def test_multiplicative_replacement_preserves_closure_and_positivity():
    frame = pd.DataFrame([[0.6, 0.4, 0.0], [0.0, 0.5, 0.5]], columns=list("abc"))
    replaced = multiplicative_replacement(frame)
    assert (replaced.to_numpy() > 0).all()
    assert np.allclose(replaced.sum(axis=1).to_numpy(), 1.0)


def test_zero_replacement_methods_dispatch():
    frame = pd.DataFrame([[0.5, 0.5, 0.0]], columns=list("abc"))
    assert (zero_replacement(frame, "pseudocount").to_numpy() > 0).all()
    assert (zero_replacement(frame, "multiplicative_replacement").to_numpy() > 0).all()
    with pytest.raises(ValueError, match="Unknown zero-handling"):
        zero_replacement(frame, "nonsense")


def test_taxon_filter_drops_rare_genera():
    counts = pd.DataFrame(
        {"s1": [100, 1, 0], "s2": [100, 0, 0], "s3": [100, 1, 0]}, index=["common", "rare", "absent"]
    )
    kept, summary = filter_taxa(counts, min_mean_relative_abundance=0.01, min_prevalence=0.0)
    assert list(kept.index) == ["common"]
    assert summary.loc[summary["genus"] == "absent", "retained"].iloc[0] == False  # noqa: E712


# ------------------------------------------------- removal-efficiency rules
def test_removal_efficiency_basic():
    assert removal_efficiency(10.0, 2.5, False, False) == pytest.approx(75.0)


def test_removal_efficiency_below_detection_disappearance_is_full_removal():
    assert removal_efficiency(0.155, None, False, True) == pytest.approx(100.0)


def test_removal_efficiency_appearance_from_below_detection_is_zero():
    assert removal_efficiency(None, 0.030, True, False) == pytest.approx(0.0)


def test_removal_efficiency_negative_when_metal_is_released():
    assert removal_efficiency(0.006, 0.011, False, False) == pytest.approx(-83.3333, rel=1e-3)


def test_removal_efficiency_clamps_when_requested():
    assert removal_efficiency(0.006, 0.011, False, False, clamp_negative=True) == pytest.approx(0.0)


def test_observed_removal_matches_published_in_house_values(cfg):
    concentrations = load_metal_concentrations(
        PROJECT_ROOT / "data" / "literature" / "inhouse_metal_concentrations.csv"
    )
    observed = compute_observed_removal(concentrations, cfg)
    single = observed[observed["sample_id"] == "SM"].set_index("metal")["removal_pct"]
    assert single["Hg"] == pytest.approx(79.6671, rel=1e-4)
    assert single["As"] == pytest.approx(96.7949, rel=1e-4)
    assert single["Pb"] == pytest.approx(60.0)
    assert single["Ca"] == pytest.approx(37.1646, rel=1e-4)
    assert single["Fe"] == pytest.approx(100.0)
    assert single["Zn"] == pytest.approx(100.0)
    assert single["Cr"] == pytest.approx(0.0)
    assert single["Ni"] < 0
    assert single["Mn"] < 0


def test_inhouse_compositions_close_to_100():
    communities = load_communities(PROJECT_ROOT / "data" / "literature" / "inhouse_communities.csv")
    totals = communities.groupby("sample_id")["rel_abundance_pct"].sum()
    assert np.allclose(totals.to_numpy(), 100.0, atol=0.5)


def test_genus_relative_abundance_merges_pseudomonas_species():
    communities = load_communities(PROJECT_ROOT / "data" / "literature" / "inhouse_communities.csv")
    collapsed = genus_relative_abundance(communities)
    os_row = collapsed[(collapsed["sample_id"] == "OS") & (collapsed["genus"] == "Pseudomonas")]
    assert os_row["rel_abundance_pct"].iloc[0] == pytest.approx(32.12, abs=0.01)


# ------------------------------------------------------------- labelling
def test_literature_labels_loaded_with_expected_genera(labels):
    lookup = dict(zip(labels["genus"], labels["label"]))
    for genus in ["Raoultella", "Pseudomonas", "Bacillus", "Klebsiella", "Enterobacter", "Delftia",
                  "Chryseobacterium"]:
        assert lookup[genus] == 1, f"{genus} should be labelled metal-active"
    assert lookup["Microvirgula"] == 0


def test_literature_labels_have_no_duplicate_genera(labels):
    assert labels["genus"].is_unique


def test_efficiency_table_covers_documented_values(efficiencies):
    lookup = dict(zip(zip(efficiencies["genus"], efficiencies["metal"]), efficiencies["efficiency_pct"]))
    assert lookup[("Raoultella", "Pb")] == pytest.approx(89.0)
    assert lookup[("Raoultella", "Ni")] == pytest.approx(55.6)
    assert lookup[("Pseudomonas", "Cu")] == pytest.approx(80.0)


def test_track_a_labels_unannotated_genera_as_zero(cfg, labels):
    rng = np.random.default_rng(0)
    genera = ["Bacillus", "Pseudomonas", "TotallyUnknownGenus", "AnotherUnknown"]
    counts = pd.DataFrame(
        rng.integers(1, 500, size=(len(genera), 30)), index=genera,
        columns=[f"S{i:03d}" for i in range(30)],
    )
    space = CompositionalFeatureSpace(
        clr=clr_transform(multiplicative_replacement(to_relative_abundance(counts).T)),
        relative=to_relative_abundance(counts).T,
        counts=counts,
        phylum_map={g: "Proteobacteria" for g in genera},
    )
    dataset = build_track_a(space, labels, cfg)
    annotation = dict(zip(dataset.genera, dataset.y))
    assert annotation["Bacillus"] == 1
    assert annotation["Pseudomonas"] == 1
    assert annotation["TotallyUnknownGenus"] == 0
    assert annotation["AnotherUnknown"] == 0


def test_track_a_feature_groups_partition_columns(cfg, labels):
    rng = np.random.default_rng(1)
    genera = ["Bacillus", "Raoultella", "Unknown1"]
    counts = pd.DataFrame(
        rng.integers(1, 100, size=(len(genera), 25)), index=genera,
        columns=[f"S{i:03d}" for i in range(25)],
    )
    space = CompositionalFeatureSpace(
        clr=clr_transform(multiplicative_replacement(to_relative_abundance(counts).T)),
        relative=to_relative_abundance(counts).T,
        counts=counts,
        phylum_map={g: "Firmicutes" for g in genera},
    )
    dataset = build_track_a(space, labels, cfg)
    grouped = [c for cols in dataset.feature_groups.values() for c in cols]
    assert set(grouped) == set(dataset.X.columns)
    assert len(grouped) == len(set(grouped))


# ------------------------------------------------------- leakage controls
def test_feature_selection_lives_inside_the_pipeline(cfg, labels):
    """Selection must be an in-fold step, never applied before cross-validation."""
    from src.models import build_pipeline, make_mlp_classifier

    cfg.preprocessing.feature_selection.mode = "always"
    cfg.preprocessing.feature_selection.k = 3
    pipeline = build_pipeline(make_mlp_classifier(cfg, 42), cfg, "classification", 20)
    assert "select" in pipeline.named_steps
    assert pipeline.steps[-1][0] == "model"


def test_selector_is_fitted_only_during_fit(cfg, labels):
    from src.models import build_pipeline, make_mlp_classifier

    cfg.preprocessing.feature_selection.mode = "always"
    cfg.preprocessing.feature_selection.k = 2
    pipeline = build_pipeline(make_mlp_classifier(cfg, 42), cfg, "classification", 10)
    selector = pipeline.named_steps["select"]
    assert not hasattr(selector, "n_features_in_") or not getattr(selector, "n_features_in_", None)


# --------------------------------------------------------- external validation
def test_score_sample_is_abundance_weighted_mean():
    per_genus = pd.DataFrame(
        {"genus": ["A", "B"], "p_metal_active": [1.0, 0.0], "curated_label": [1, 0],
         "phylum": ["P", "P"]}
    )
    abundance = pd.Series({"A": 0.75, "B": 0.25})
    result = score_sample("X", per_genus, abundance, n_bootstrap=200, seed=42)
    assert result.predicted_activity_score == pytest.approx(0.75)
    assert result.documented_active_fraction == pytest.approx(0.75)


def test_score_sample_reports_unscorable_instead_of_raising():
    """A sample whose taxa are absent from the training taxonomy must be reported.

    This is a real failure mode: the in-house SM biofilm is a Bacillus monoculture and
    Bacillus never appears in the public wastewater training corpus, so the classifier
    has no weight to apply. Aborting the whole pipeline would be wrong; the literature
    index is still computable for that sample.
    """
    per_genus = pd.DataFrame(
        {"genus": ["A"], "p_metal_active": [1.0], "curated_label": [1], "phylum": ["P"]}
    )
    result = score_sample("X", per_genus, pd.Series({"Z": 1.0}), n_bootstrap=200, seed=0)
    assert result.scorable is False
    assert np.isnan(result.predicted_activity_score)
    assert result.taxonomy_coverage == 0.0
    assert result.n_scored_genera == 0
    assert "training taxonomy" in result.note


def test_score_sample_flags_degenerate_single_genus_interval():
    per_genus = pd.DataFrame(
        {"genus": ["A"], "p_metal_active": [0.9], "curated_label": [1], "phylum": ["P"]}
    )
    result = score_sample("X", per_genus, pd.Series({"A": 0.4}), n_bootstrap=200, seed=0)
    assert result.scorable is True
    assert result.n_scored_genera == 1
    assert result.taxonomy_coverage == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(result.ci_high)
    assert "degenerate" in result.note


def test_community_metal_activity_index_is_bounded(labels):
    relative = pd.DataFrame(
        {"Bacillus": [0.5, 0.0], "UnknownX": [0.5, 1.0]}, index=["S1", "S2"]
    )
    index = community_metal_activity_index(relative, labels)
    assert index.loc["S1", "metal_activity_index"] == pytest.approx(1.0)
    assert index.loc["S2", "metal_activity_index"] == pytest.approx(0.0)
    assert np.all(index["metal_activity_index"].to_numpy() >= 0)
    assert np.all(index["metal_activity_index"].to_numpy() <= 1)


def test_expected_removal_is_abundance_scaled(efficiencies):
    relative = pd.DataFrame({"Raoultella": [0.5], "Bacillus": [0.5]}, index=["S1"])
    expected = expected_removal_from_literature(relative, efficiencies, "Pb")
    assert expected.loc["S1"] == pytest.approx(0.89 * 0.5)


def test_expected_removal_is_zero_for_unmined_metal(efficiencies):
    relative = pd.DataFrame({"Raoultella": [1.0]}, index=["S1"])
    expected = expected_removal_from_literature(relative, efficiencies, "Hg")
    assert expected.loc["S1"] == pytest.approx(0.0)


# ------------------------------------------------------------- acquisition
def test_clean_genera_drops_placeholders_and_zeros():
    frame = pd.DataFrame(
        {
            "genus": ["Bacillus", "unclassified", "", "uncultured", "Pseudomonas"],
            "phylum": ["Firmicutes", "x", "y", "z", "Proteobacteria"],
            "count": [10, 5, 5, 5, 0],
        }
    )
    cleaned = clean_genera(frame)
    assert set(cleaned["genus"]) == {"Bacillus"}


def test_clean_genera_backfills_missing_phylum():
    frame = pd.DataFrame({"genus": ["Bacillus"], "phylum": [""], "count": [7]})
    cleaned = clean_genera(frame)
    assert cleaned["phylum"].iloc[0] == "unclassified_phylum"


# ------------------------------------------------------------- determinism
def test_simulation_is_deterministic(cfg, labels):
    from src.acquire_fallback import simulate_wastewater_communities

    phylum_map = dict(zip(labels["genus"], labels["phylum"]))
    first, _, info_a = simulate_wastewater_communities(cfg, 42, list(labels["genus"]), phylum_map)
    second, _, info_b = simulate_wastewater_communities(cfg, 42, list(labels["genus"]), phylum_map)
    assert first.equals(second)
    assert info_a["seed"] == info_b["seed"]
    assert info_a.get("simulated") is True


def test_simulation_warns_that_it_is_simulated(cfg, labels):
    from src.acquire_fallback import simulate_wastewater_communities

    phylum_map = dict(zip(labels["genus"], labels["phylum"]))
    _, _, info = simulate_wastewater_communities(cfg, 42, list(labels["genus"]), phylum_map)
    assert "warning" in info and "SYNTHETIC" in info["warning"]


def test_config_fingerprint_is_stable(cfg):
    from src.config import config_fingerprint

    assert config_fingerprint(cfg) == config_fingerprint(cfg)


def test_config_mutation_persists(cfg):
    """Guards against the DotDict copy-on-read bug: overrides must stick."""
    cfg.model.grid.enabled = False
    assert cfg.model.grid.enabled is False


# --------------------------------------------------------- torch backend
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.torch_backend import TorchMLPClassifier  # noqa: E402


def test_torch_estimator_is_recognised_as_a_classifier():
    """Regression test for the mixin-order bug.

    Declaring ``(BaseEstimator, ClassifierMixin)`` leaves ``estimator_type`` unset, so
    scikit-learn does not treat the estimator as a classifier and passes the raw
    ``(n, 2)`` probability matrix to scoring functions, which then raise
    "y should be a 1d array, got an array of shape (n, 2)".
    """
    from sklearn.base import is_classifier

    assert is_classifier(TorchMLPClassifier()) is True
    tags = TorchMLPClassifier().__sklearn_tags__()
    assert tags.estimator_type == "classifier"
    assert tags.classifier_tags is not None


def test_torch_estimator_survives_cross_validation_with_roc_auc():
    from sklearn.model_selection import cross_validate

    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.4, size=60) > 0).astype(int)
    estimator = TorchMLPClassifier(hidden_layer_sizes=(16,), max_iter=60,
                                  early_stopping=False, random_state=0, device="cpu")
    scores = cross_validate(estimator, X, y, cv=3, scoring="roc_auc")
    assert len(scores["test_score"]) == 3
    assert np.all(np.isfinite(scores["test_score"]))


def test_torch_estimator_probabilities_are_valid():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int)
    estimator = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=40,
                                  early_stopping=False, random_state=0, device="cpu")
    estimator.fit(X, y)
    probabilities = estimator.predict_proba(X)
    assert probabilities.shape == (40, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(estimator.predict(X))).issubset({0, 1})
    assert len(estimator.loss_curve_) > 0


def test_torch_estimator_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int)
    first = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=50, early_stopping=False,
                              random_state=99, device="cpu").fit(X, y).predict_proba(X)
    second = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=50, early_stopping=False,
                               random_state=99, device="cpu").fit(X, y).predict_proba(X)
    assert np.allclose(first, second, atol=1e-6)


# ------------------------------------------------------- confusion summary
def test_balanced_accuracy_is_the_mean_of_sensitivity_and_specificity():
    """Regression test for a ternary/`+` precedence bug.

    The original expression was `((a) if cond else 0.0 + (b) if cond else 0.0) / 2`,
    which Python parses as `a if cond else (0.0 + (b if cond else 0.0))`, so the result
    was divided by two without ever adding the specificity term.
    """
    from src.evaluate import summarise_confusion

    cm = np.array([[21, 0], [0, 2]])  # perfect classifier
    summary = summarise_confusion(cm)
    assert summary["sensitivity"] == pytest.approx(1.0)
    assert summary["specificity"] == pytest.approx(1.0)
    assert summary["balanced_accuracy"] == pytest.approx(1.0)

    cm_skewed = np.array([[8, 2], [0, 2]])  # always catches positives, misses 2 negatives
    summary = summarise_confusion(cm_skewed)
    assert summary["sensitivity"] == pytest.approx(1.0)
    assert summary["specificity"] == pytest.approx(0.8)
    assert summary["balanced_accuracy"] == pytest.approx(0.9)


def test_balanced_accuracy_handles_degenerate_folds():
    from src.evaluate import summarise_confusion

    summary = summarise_confusion(np.array([[10, 0], [0, 0]]))
    assert np.isnan(summary["sensitivity"])
    assert summary["specificity"] == pytest.approx(1.0)
    assert np.isnan(summary["balanced_accuracy"])
