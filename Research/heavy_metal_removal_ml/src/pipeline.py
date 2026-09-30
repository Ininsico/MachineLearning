"""End-to-end orchestration of the heavy-metal-removal ML pipeline."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .acquire_fallback import (
    acquire_via_r_dada2,
    acquire_via_r_mgnify,
    rscript_available,
    simulate_wastewater_communities,
)
from .acquire_mgnify import load_or_acquire_mgnify
from .config import config_fingerprint, ensure_dirs, load_config, set_global_seed
from .datasets import (
    build_track_a,
    build_track_c,
    community_metal_activity_index,
    project_inhouse,
)
from .evaluate import (
    classification_metrics,
    confusion_matrix_frame,
    majority_class_reference,
    out_of_fold_scores,
    permutation_importance_table,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_precision_recall_curve,
    plot_ranked_importance,
    plot_roc_curve,
    regression_metrics,
    shap_importance_table,
    summarise_confusion,
    tune_threshold,
)
from .external_validation import compare_to_observed, predict_inhouse_samples, report_validation
from .features import build_feature_space, build_selector, make_scaler
from .inhouse import (
    compute_observed_removal,
    describe_inhouse,
    load_communities,
    load_metal_concentrations,
)
from .literature import load_genus_labels, load_metal_efficiency
from .logging_utils import banner, format_metric_table, get_logger, kv_table, section, setup_logging
from .models import (
    assess_class_balance,
    cross_validate_model,
    learning_curve_history,
    make_baselines,
    make_mlp_classifier,
    resolve_backend,
    stratified_holdout_split,
    tune_mlp,
)
from sklearn.pipeline import Pipeline


@dataclass
class PipelineState:
    cfg: object
    paths: Dict[str, Path]
    acquisition: dict = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    results: dict = field(default_factory=dict)


class HeavyMetalPipeline:
    def __init__(self, cfg, force_acquisition: bool = False, skip_grid: bool = False,
                 strategy: Optional[str] = None, verbose: bool = False) -> None:
        self.cfg = cfg
        self.force_acquisition = force_acquisition
        self.skip_grid = skip_grid
        self.forced_strategy = strategy
        self.paths = ensure_dirs(cfg)
        self.log = setup_logging(self.paths["logs"] / "pipeline.log", verbose=verbose)
        self.seed = int(cfg.project.seed)
        set_global_seed(self.seed)
        self.state = PipelineState(cfg=cfg, paths=self.paths)
        self.fingerprint = config_fingerprint(cfg)
        self.state.device_description = self._describe_device()
        self.state.confusion_summary = {}

    def _describe_device(self) -> str:
        backend = resolve_backend(self.cfg)
        if backend != "torch":
            return "cpu (scikit-learn MLPClassifier)"
        try:
            from .torch_backend import describe_device, resolve_device

            preference = str(getattr(self.cfg.model.torch, "device", "auto"))
            return describe_device(resolve_device(preference))
        except Exception:
            return "torch (device unknown)"

    # ------------------------------------------------------------------ utils
    def save_table(self, frame: pd.DataFrame, name: str, index: bool = True) -> Path:
        path = self.paths["tables"] / name
        frame.to_csv(path, index=index)
        self.state.artifacts[name] = path
        self.log.info("  saved table: %s", path.name)
        return path

    def save_json(self, payload: dict, name: str) -> Path:
        path = self.paths["tables"] / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.state.artifacts[name] = path
        self.log.info("  saved json : %s", path.name)
        return path

    # ------------------------------------------------------------ stage 1
    def stage_acquisition(self, labels: pd.DataFrame, phylum_map: Dict[str, str]) -> None:
        banner("Stage 1 / 6 - Public wastewater 16S data acquisition", self.log)
        acfg = self.cfg.acquisition
        order = [self.forced_strategy] if self.forced_strategy else list(acfg.strategy_order)
        min_samples = int(acfg.mgnify.min_samples)

        self.log.info("Strategy order            : %s", " -> ".join(order))
        self.log.info("Minimum acceptable samples: %d", min_samples)
        self.log.info("R/Rscript available       : %s", rscript_available(self.cfg))

        counts = metadata = pd.DataFrame()
        info: dict = {"status": "not_attempted"}
        attempts: List[dict] = []

        for strategy in order:
            section(f"Strategy: {strategy}", self.log)
            started = time.perf_counter()
            try:
                if strategy == "mgnify_r":
                    counts, metadata, info = acquire_via_r_mgnify(self.cfg, self.paths)
                elif strategy == "mgnify_rest":
                    counts, metadata, info = load_or_acquire_mgnify(
                        self.cfg, self.paths, force=self.force_acquisition
                    )
                elif strategy == "sra_dada2":
                    counts, metadata, info = acquire_via_r_dada2(self.cfg, self.paths)
                elif strategy == "simulate":
                    counts, metadata, info = simulate_wastewater_communities(
                        self.cfg, self.seed, list(labels["genus"]), phylum_map
                    )
                else:
                    info = {"status": "unknown_strategy"}
            except Exception as exc:
                info = {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
                self.log.warning("Strategy %s raised: %s", strategy, info["error"])

            elapsed = time.perf_counter() - started
            n_samples = int(counts.shape[1]) if not counts.empty else 0
            accepted = n_samples >= min_samples
            attempts.append(
                {"strategy": strategy, "status": info.get("status"), "n_samples": n_samples,
                 "n_genera": int(counts.shape[0]) if not counts.empty else 0,
                 "accepted": accepted, "seconds": round(elapsed, 2)}
            )
            self.log.info("Strategy %s -> status=%s, samples=%d, accepted=%s (%.1fs)",
                          strategy, info.get("status"), n_samples, accepted, elapsed)

            if accepted and strategy != "simulate":
                break
            if strategy == "simulate" and accepted:
                self.log.warning(
                    "Falling back to the SIMULATED surrogate. Results below demonstrate pipeline "
                    "mechanics only and must not be interpreted biologically."
                )
                break

        if counts.empty:
            raise RuntimeError(
                "All acquisition strategies failed. Check network access, R availability, or set "
                "acquisition.strategy_order: [simulate] to run offline."
            )

        self.state.acquisition = {
            "info": info,
            "attempts": attempts,
            "n_samples": int(counts.shape[1]),
            "n_genera": int(counts.shape[0]),
            "simulated": bool(info.get("simulated", False)),
        }

        simulated = self.state.acquisition["simulated"]
        if simulated:
            self.log.warning("DATA PROVENANCE: SIMULATED - not real wastewater observations")
        else:
            self.log.info("DATA PROVENANCE: %s", info.get("source", "unknown"))

        phylum_map = counts.attrs.get("phylum_map", phylum_map) or phylum_map
        self.state.feature_space = build_feature_space(
            counts, self.cfg, phylum_map=phylum_map, simulated=simulated, provenance=info
        )
        self.state.metadata = metadata
        self.save_table(counts, "public_genus_counts.csv")
        if not metadata.empty:
            self.save_table(metadata, "public_sample_metadata.csv", index=False)

    # ------------------------------------------------------------ stage 2
    def stage_datasets(self, labels: pd.DataFrame, efficiencies: pd.DataFrame,
                       communities: pd.DataFrame) -> None:
        banner("Stage 2 / 6 - Dataset construction", self.log)
        space = self.state.feature_space

        section("Track A - genus-level literature-supervised classification", self.log)
        track_a = build_track_a(space, labels, self.cfg)
        self.state.track_a = track_a
        labelled = track_a.X.copy()
        labelled.insert(0, "label", track_a.y)
        self.save_table(labelled, "track_a_labelled_dataset.csv")
        self.save_table(
            pd.DataFrame(
                {
                    "genus": track_a.genera,
                    "label": track_a.y.to_numpy(),
                    "n_features": track_a.n_features,
                }
            ),
            "track_a_genera.csv",
            index=False,
        )

        section("Track A - feature group inventory", self.log)
        for group, features in track_a.feature_groups.items():
            self.log.info("  %-12s %4d features", group, len(features))

        section("Track B - sample-level community metal-activity index", self.log)
        space.relative.index.name = "sample"
        index_frame = community_metal_activity_index(space.relative, labels)
        self.state.track_b = index_frame
        self.save_table(index_frame, "track_b_metal_activity_index.csv")
        self.log.info(
            "  public samples scored: %d; index mean %.3f, median %.3f, range %.3f-%.3f",
            len(index_frame), index_frame["metal_activity_index"].mean(),
            index_frame["metal_activity_index"].median(),
            index_frame["metal_activity_index"].min(), index_frame["metal_activity_index"].max(),
        )

        inhouse_index = community_metal_activity_index(self._inhouse_relative(communities), labels)
        self.save_table(inhouse_index, "track_b_inhouse_metal_activity_index.csv")
        self.log.info("  in-house index (SM/OS):")
        for row in inhouse_index.itertuples():
            self.log.info(
                "    %-3s index=%.3f  documented-active fraction=%.3f  annotated share of community=%.3f",
                row.Index, row.metal_activity_index, row.documented_active_fraction_raw,
                row.annotated_fraction_of_community,
            )

        section("Track C - per-metal quantitative regression", self.log)
        X_c, y_c, feature_sets = build_track_c(space, efficiencies, labels, self.cfg)
        self.state.track_c = (X_c, y_c, feature_sets)
        if not X_c.empty:
            self.save_table(y_c, "track_c_targets.csv")
            self.save_table(X_c, "track_c_features.csv")
        else:
            self.log.warning(
                "Track C has no trainable target and will be reported as data-limited."
            )
        self.state.inhouse_communities = communities

    def _inhouse_relative(self, communities: pd.DataFrame,
                          restrict_to: Optional[List[str]] = None) -> pd.DataFrame:
        """Genus-level relative-abundance table for the in-house samples.

        ``restrict_to`` projects onto the training genus set and is only
        appropriate when feeding a model. The literature index deliberately uses
        the unrestricted composition, because dropping a genus merely because the
        public training data never contained it would understate the sample.
        """
        from .inhouse import genus_relative_abundance

        collapsed = genus_relative_abundance(communities)
        pivot = collapsed.pivot_table(index="sample_id", columns="genus",
                                      values="rel_abundance_fraction", aggfunc="sum").fillna(0.0)
        if restrict_to is not None:
            pivot = pivot.reindex(columns=list(restrict_to), fill_value=0.0)
        return pivot

    # ------------------------------------------------------------ stage 3
    def stage_training(self) -> None:
        banner("Stage 3 / 6 - Model training and hyper-parameter tuning", self.log)
        track_a = self.state.track_a
        X, y = track_a.X, track_a.y

        section("Train/test split (80/20, stratified)", self.log)
        X_train, X_test, y_train, y_test = stratified_holdout_split(X, y, self.cfg, "classification")
        self.log.info("  train: %d genera (%d positive)", len(y_train), int(y_train.sum()))
        self.log.info("  test : %d genera (%d positive)", len(y_test), int(y_test.sum()))
        self.state.split = (X_train, X_test, y_train, y_test)

        section("Trainability assessment", self.log)
        minority, folds, message = assess_class_balance(
            y_train, int(self.cfg.model.cv.folds)
        )
        self.log.info("  %s", message)
        self.state.cv_folds = folds
        self.state.minority_class_count = minority

        if folds < 2:
            self.log.error(
                "INSUFFICIENT POSITIVE EXAMPLES: %s. Cross-validation, hyper-parameter search and "
                "held-out metrics are all unavailable for this dataset. The pipeline will fit a "
                "single model so that the external validation stage can still run, and will report "
                "the classification results as not estimable. This is a data limitation, not a "
                "pipeline failure - see the README section on statistical power.", message,
            )
            self.state.trainable = False
            self.state.baseline_results = []
            self.state.mlp_default_cv = None
            self.state.mlp_tuned = None
            steps = [("scaler", make_scaler(self.cfg)),
                     ("select", build_selector(self.cfg, "classification", X.shape[1])),
                     ("model", make_mlp_classifier(self.cfg, self.seed))]
            pipeline = Pipeline([(name, step) for name, step in steps if step is not None])
            pipeline.fit(X.to_numpy(), y.to_numpy())
            self.state.final_model = pipeline
            self.state.best_params = {"note": "not tuned - insufficient positive examples"}
            self.state.model_comparison = pd.DataFrame()
            self.state.test_metrics = {"not_estimable": float("nan")}
            self.state.confusion_matrix = confusion_matrix_frame(
                np.zeros(len(y_test)), np.zeros(len(y_test))
            )
            self.state.ablation = pd.DataFrame()
            self.state.threshold = 0.5
            self.save_json(
                {"trainable": False, "reason": message, "minority_class_count": minority},
                "trainability_report.json",
            )
            return

        self.state.trainable = True

        section(f"Baseline models - {folds}-fold stratified cross-validation", self.log)
        baselines = make_baselines(self.cfg, self.seed)
        baseline_results = []
        for name, estimator in baselines.items():
            if not name.endswith("classifier"):
                continue
            result = cross_validate_model(name, estimator, X_train, y_train, self.cfg,
                                          "classification", n_splits=folds)
            baseline_results.append(result)
        self.state.baseline_results = baseline_results

        section(f"Multilayer perceptron - {folds}-fold cross-validation at configured defaults", self.log)
        mlp_default = make_mlp_classifier(self.cfg, self.seed)
        self.state.mlp_default_cv = cross_validate_model(
            "mlp_default", mlp_default, X_train, y_train, self.cfg, "classification", n_splits=folds
        )

        section("Multilayer perceptron - GridSearchCV", self.log)
        if self.skip_grid or not bool(self.cfg.model.grid.enabled):
            self.log.info("  grid search disabled; using configured defaults")
            steps = [("scaler", make_scaler(self.cfg)),
                     ("select", build_selector(self.cfg, "classification", X.shape[1])),
                     ("model", make_mlp_classifier(self.cfg, self.seed))]
            pipeline = Pipeline([(name, step) for name, step in steps if step is not None])
            pipeline.fit(X_train.to_numpy(), y_train.to_numpy())
            self.state.mlp_tuned = None
            self.state.final_model = pipeline
            self.state.best_params = {"note": "grid search skipped"}
        else:
            result, search = tune_mlp(self.cfg, X_train, y_train, "classification", n_splits=folds)
            self.state.mlp_tuned = result
            self.state.final_model = result.fitted_pipeline
            self.state.best_params = result.best_params
            self.save_table(result.extra["cv_results"], "gridsearch_cv_results.csv", index=False)

        section("Model comparison - cross-validated ROC-AUC", self.log)
        rows = []
        for result in baseline_results + [self.state.mlp_default_cv]:
            row = result.cv_summary.set_index("metric")["mean"].to_dict()
            row["model"] = result.name
            rows.append(row)
        if self.state.mlp_tuned is not None:
            row = self.state.mlp_tuned.cv_summary.set_index("metric")["mean"].to_dict()
            row["model"] = "mlp_tuned"
            rows.append(row)
        comparison = pd.DataFrame(rows).set_index("model")
        ordered = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
                   if c in comparison.columns]
        comparison = comparison[ordered].sort_values("roc_auc", ascending=False)
        self.state.model_comparison = comparison
        self.save_table(comparison, "model_comparison_cv.csv")

        metrics = ["accuracy", "f1", "roc_auc"] if "roc_auc" in comparison.columns else ["accuracy"]
        table = format_metric_table(
            ["model"] + metrics, [[idx] + [comparison.loc[idx, m] for m in metrics] for idx in comparison.index]
        )
        for line in table.splitlines():
            self.log.info("  %s", line)

        section("Final model evaluation on the held-out 20% test set", self.log)
        model = self.state.final_model
        y_score = model.predict_proba(X_test.to_numpy())[:, 1]

        reference = majority_class_reference(y_train)
        self.log.info(
            "  majority-class reference (always predict %d): accuracy %.4f, f1 %.4f; "
            "positive rate in training %.4f",
            reference["majority_class"], reference["accuracy"], reference["f1"], reference["positive_rate"],
        )
        self.log.info("  NOTE: with a %.1f%% positive rate, accuracy alone is uninformative - a trivial "
                      "majority classifier already scores %.3f", 100 * reference["positive_rate"],
                      reference["accuracy"])

        oof_scores = out_of_fold_scores(model, X_train, y_train, self.cfg)
        threshold, oof_best_f1 = tune_threshold(y_train, oof_scores, "f1")
        self.log.info(
            "  decision threshold tuned on out-of-fold training scores: %.4f (out-of-fold F1 %.4f)",
            threshold, oof_best_f1,
        )
        y_pred = (y_score >= threshold).astype(int)

        test_metrics = classification_metrics(y_test, y_pred, y_score)
        test_metrics["threshold"] = float(threshold)
        self.state.test_metrics = test_metrics
        self.state.threshold = float(threshold)
        for key, value in test_metrics.items():
            self.log.info("  %-20s %.4f", key, value)

        cm = confusion_matrix_frame(y_test, y_pred)
        self.state.confusion_matrix = cm
        self.save_table(cm, "confusion_matrix.csv")
        self.log.info("  confusion matrix (threshold %.3f):\n%s", threshold, cm.to_string())
        extra = summarise_confusion(cm.to_numpy())
        self.state.confusion_summary = extra
        self.log.info("  sensitivity %.3f | specificity %.3f | balanced accuracy %.3f | n=%d",
                      extra["sensitivity"], extra["specificity"], extra["balanced_accuracy"],
                      int(extra["n"]))

        section("Feature-group ablation - how much is explained by lineage alone?", self.log)
        self.state.ablation = self._run_ablation(track_a)

    def _run_ablation(self, track_a) -> pd.DataFrame:
        """Quantify how much of the signal comes from phylum membership."""
        groups = track_a.feature_groups
        abundance_only = [c for c in groups.get("abundance", [])]
        phylum_only = [c for c in groups.get("phylum", [])]
        prevalence_only = [c for c in groups.get("prevalence", [])]

        configurations = {
            "full_model": list(track_a.X.columns),
            "abundance_only": abundance_only,
            "phylum_only": phylum_only or None,
            "prevalence_only": prevalence_only or None,
            "abundance_plus_prevalence": abundance_only + prevalence_only,
        }
        rows = []
        for name, columns in configurations.items():
            if not columns:
                continue
            subset = track_a.X[columns]
            estimator = make_mlp_classifier(self.cfg, self.seed)
            result = cross_validate_model(f"ablation_{name}", estimator, subset, track_a.y,
                                          self.cfg, "classification")
            summary = result.cv_summary.set_index("metric")["mean"].to_dict()
            rows.append({"configuration": name, "n_features": len(columns), **summary})
        frame = pd.DataFrame(rows).set_index("configuration")
        self.save_table(frame, "feature_group_ablation.csv")
        for line in format_metric_table(
            ["configuration", "n_features", "roc_auc", "accuracy"],
            [[idx, int(frame.loc[idx, "n_features"]), frame.loc[idx, "roc_auc"], frame.loc[idx, "accuracy"]]
             for idx in frame.index],
        ).splitlines():
            self.log.info("  %s", line)

        full = frame.loc["full_model", "roc_auc"]
        if "phylum_only" in frame.index:
            phylum = frame.loc["phylum_only", "roc_auc"]
            self.log.warning(
                "  Phylum membership alone reaches ROC-AUC %.3f versus %.3f for the full model. "
                "A large share of Track A performance is therefore lineage memorisation rather than "
                "community-composition signal. This is reported as a limitation.", phylum, full
            )
        return frame

    # ------------------------------------------------------------ stage 4
    def stage_evaluation(self) -> None:
        banner("Stage 4 / 6 - Evaluation artefacts and feature attribution", self.log)
        track_a = self.state.track_a
        X_train, X_test, y_train, y_test = self.state.split
        model = self.state.final_model
        dpi = int(self.cfg.evaluation.plots.dpi)

        section("Learning curves (training loss vs validation score per epoch)", self.log)
        history = learning_curve_history(model)
        self.state.learning_history = history
        if not history.empty:
            self.save_table(history, "learning_curve_history.csv", index=False)
            epochs = int(history["epoch"].max())
            final_loss = float(history["training_loss"].dropna().iloc[-1])
            self.log.info("  trained for %d epochs; final training loss %.5f", epochs, final_loss)
            source = "full training set"
            plot_learning_curve(
                history,
                self.paths["figures"] / "learning_curve_mlp.png",
                f"MLP learning curve ({source}, {epochs} epochs, early stopping)",
                dpi,
            )
        else:
            self.log.warning("  no epoch history available (model may not have been fitted)")

        section("Confusion matrix and ROC curve (held-out test set)", self.log)
        threshold = getattr(self.state, "threshold", 0.5)
        y_score = model.predict_proba(X_test.to_numpy())[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        if not getattr(self.state, "trainable", True):
            self.log.error(
                "Held-out test metrics are NOT ESTIMABLE for this dataset (see the trainability "
                "assessment in stage 3). The fit-on-everything model is reported for the external "
                "validation stage only; no test-set ROC curve, confusion matrix, permutation "
                "importance or SHAP attribution is produced, because none of them would mean "
                "anything with so few positive examples."
            )
            history = learning_curve_history(model)
            if not history.empty:
                self.state.learning_history = history
                self.save_table(history, "learning_curve_history.csv", index=False)
                self.log.info("  learning curve recorded anyway: %d epochs, final training loss %.5f",
                              int(history["epoch"].max()),
                              float(history["training_loss"].dropna().iloc[-1]))
            return

        cm = self.state.confusion_matrix.to_numpy()
        plot_confusion_matrix(cm, ["Undocumented (0)", "Documented (1)"],
                             self.paths["figures"] / "confusion_matrix.png",
                             f"Track A confusion matrix - held-out genera (threshold {threshold:.2f})", dpi)
        plot_roc_curve(y_test, y_score, self.state.test_metrics["roc_auc"],
                       self.paths["figures"] / "roc_curve.png",
                       "Track A ROC curve - held-out genera", dpi)

        section("Precision-recall diagnostics (imbalanced positive class)", self.log)
        pr_path = self.paths["figures"] / "precision_recall_curve.png"
        plot_precision_recall_curve(y_test, y_score,
                                    self.state.test_metrics.get("average_precision", float("nan")),
                                    pr_path, "Track A precision-recall curve - held-out genera", dpi)

        section("Permutation importance on the held-out test set", self.log)
        perm = permutation_importance_table(model, X_test, y_test, self.cfg, self.seed)
        top = perm.head(int(self.cfg.evaluation.permutation_importance.max_features))
        self.save_table(top, "permutation_importance.csv", index=False)
        for row in top.head(10).itertuples():
            self.log.info("  %-40s %.5f +/- %.5f", row.feature, row.importance_mean, row.importance_std)
        plot_ranked_importance(top, "importance_mean",
                              self.paths["figures"] / "permutation_importance.png",
                              "Permutation importance (ROC-AUC drop)", dpi=dpi,
                              error_column="importance_std")

        if bool(self.cfg.evaluation.shap.enabled):
            section("SHAP attribution", self.log)
            shap_frame = shap_importance_table(model, X_test, self.cfg, self.seed)
            if shap_frame is not None:
                top_shap = shap_frame.head(int(self.cfg.evaluation.shap.max_features))
                self.save_table(top_shap, "shap_importance.csv", index=False)
                plot_ranked_importance(top_shap, "mean_abs_shap",
                                      self.paths["figures"] / "shap_importance.png",
                                      "SHAP mean |value|", dpi=dpi)
                for row in top_shap.head(10).itertuples():
                    self.log.info("  %-40s %.5f", row.feature, row.mean_abs_shap)
            else:
                self.log.warning("  SHAP unavailable; permutation importance is the reported attribution")

    # ------------------------------------------------------------ stage 5
    def stage_external_validation(self, observed: pd.DataFrame, communities: pd.DataFrame,
                                  efficiencies: pd.DataFrame, labels: pd.DataFrame) -> None:
        banner("Stage 5 / 6 - External validation on in-house biofilms (SM, OS)", self.log)
        track_a = self.state.track_a
        X_train = self.state.split[0]

        section("Applied prediction: trained classifier + abundance weighting", self.log)
        if not getattr(self.state, "trainable", True):
            self.log.error(
                "CAVEAT: this dataset did not support cross-validation, so the classifier was fitted "
                "on every available genus and its per-genus probabilities are in-sample and therefore "
                "over-confident. Treat the model-based activity score below as illustrative only; the "
                "literature-index score (documented-active fraction) is the defensible quantity."
            )
        scores, details = predict_inhouse_samples(
            self.state.final_model, track_a.X, track_a.genera, labels, communities, self.cfg
        )
        self.state.validation_scores = scores
        self.save_table(scores, "external_validation_scores.csv", index=False)
        for sample_id, frame in details.items():
            self.save_table(frame, f"external_validation_per_genus_{sample_id}.csv", index=False)

        section("Projection of SM/OS into the training feature space", self.log)
        projected = project_inhouse(self.state.feature_space, communities, self.cfg)
        self.save_table(projected, "inhouse_clr_projection.csv")
        for sample_id in projected.index:
            row = projected.loc[sample_id]
            self.log.info("  %s projected onto %d training genera (CLR domain; %d non-zero)",
                          sample_id, len(row), int((row != 0).sum()))

        section("Predicted vs observed removal", self.log)
        comparison = compare_to_observed(observed, communities, efficiencies, self.cfg)
        self.state.validation_comparison = comparison
        self.save_table(comparison, "external_validation_comparison.csv", index=False)

        observed_only = comparison.dropna(subset=["observed_removal_pct"])
        self.save_table(
            observed_only[["sample_id", "metal", "observed_removal_pct", "observed_outcome"]],
            "observed_removal_efficiency.csv", index=False,
        )

        report = report_validation(scores, comparison, self.cfg)
        (self.paths["results"] / "external_validation_report.txt").write_text(report, encoding="utf-8")
        self.log.info("  wrote external_validation_report.txt")

    # ------------------------------------------------------------ stage 6
    def stage_save(self, labels: pd.DataFrame, observed: pd.DataFrame) -> None:
        banner("Stage 6 / 6 - Saving artifacts", self.log)
        models_dir = self.paths["models"]

        model_path = models_dir / "track_a_mlp_classifier.pkl"
        joblib.dump(self.state.final_model, model_path)
        self.state.artifacts["model"] = model_path
        self.log.info("  saved model : %s", model_path.name)

        pipeline = self.state.final_model
        scaler = pipeline.named_steps.get("scaler") if hasattr(pipeline, "named_steps") else None
        if scaler is not None:
            scaler_path = models_dir / "track_a_standard_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            self.state.artifacts["scaler"] = scaler_path
            self.log.info("  saved scaler: %s", scaler_path.name)

        feature_names = pd.DataFrame(
            {
                "feature": list(self.state.track_a.X.columns),
                "group": [
                    next((group for group, cols in self.state.track_a.feature_groups.items() if col in cols),
                         "other")
                    for col in self.state.track_a.X.columns
                ],
            }
        )
        self.save_table(feature_names, "feature_names.csv", index=False)

        self.save_table(labels, "literature_labels_used.csv", index=False)
        self.save_table(observed, "inhouse_observed_removal.csv", index=False)

        metadata = {
            "pipeline_version": __version__,
            "config_fingerprint": self.fingerprint,
            "seed": self.seed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "mlp_backend": resolve_backend(self.cfg),
            "mlp_device": getattr(self.state, "device_description", "n/a"),
            "cv_folds_used": int(getattr(self.state, "cv_folds", 0) or 0),
            "minority_class_count": int(getattr(self.state, "minority_class_count", 0) or 0),
            "trainable": bool(getattr(self.state, "trainable", True)),
            "confusion_summary": getattr(self.state, "confusion_summary", {}),
            "acquisition": self.state.acquisition,
            "track_a_shape": list(self.state.track_a.X.shape),
            "track_a_label_balance": {
                "positive": int(self.state.track_a.y.sum()),
                "negative": int((self.state.track_a.y == 0).sum()),
            },
            "best_params": self.state.best_params,
            "holdout_metrics": self.state.test_metrics,
            "cv_model_comparison": self.state.model_comparison.to_dict(),
            "feature_group_ablation": (
                self.state.ablation.to_dict() if not self.state.ablation.empty else {}
            ),
            "external_validation": self.state.validation_scores.to_dict("records"),
            "data_provenance_warning": (
                "SIMULATED DATA - results are not biologically meaningful"
                if self.state.acquisition.get("simulated") else None
            ),
        }
        self.save_json(metadata, "run_metadata.json")
        self.state.results = metadata

    # ---------------------------------------------------------------- driver
    def run(self) -> dict:
        started = time.perf_counter()
        paths = self.paths
        banner(f"{self.cfg.project.name} - reproducible ML pipeline", self.log)
        kv_table(
            [
                ("configuration", self.cfg["_config_path"]),
                ("config fingerprint", self.fingerprint),
                ("random seed", self.seed),
                ("data directory", paths["data"]),
                ("results directory", paths["results"]),
                ("python", f"{sys.version.split()[0]} ({platform.system()})"),
                ("scikit-learn", sklearn.__version__),
                ("MLP backend", f"{resolve_backend(self.cfg)} -> {self.state.device_description}"),
            ]
        )

        section("Loading in-house and literature resources", self.log)
        labels = load_genus_labels(paths["literature"] / "genus_labels.csv")
        efficiencies = load_metal_efficiency(paths["literature"] / "genus_metal_efficiency.csv")
        concentrations = load_metal_concentrations(paths["literature"] / "inhouse_metal_concentrations.csv")
        communities = load_communities(paths["literature"] / "inhouse_communities.csv")
        observed = compute_observed_removal(concentrations, self.cfg)
        describe_inhouse(concentrations, communities)
        self.save_table(observed, "observed_removal_details.csv", index=False)

        phylum_map = dict(zip(labels["genus"], labels["phylum"]))

        self.stage_acquisition(labels, phylum_map)
        self.stage_datasets(labels, efficiencies, communities)
        self.stage_training()
        self.stage_evaluation()
        self.stage_external_validation(observed, communities, efficiencies, labels)
        self.stage_save(labels, observed)

        elapsed = time.perf_counter() - started
        banner("Pipeline complete", self.log)
        self.log.info("Total runtime: %.1f s", elapsed)
        self.log.info("Artifacts written to: %s", paths["results"])
        if self.state.acquisition.get("simulated"):
            self.log.warning("REMINDER: these results used SIMULATED data and carry no biological meaning")
        return self.state.results
