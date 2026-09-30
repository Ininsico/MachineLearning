"""Fallback acquisition strategies: R-driven MGnify/DADA2, and an offline surrogate.

Order of preference (see ``config.yaml: acquisition.strategy_order``):

1. ``mgnify_r``    - Bioconductor MGnifyR via ``r/mgnify_retrieval.R`` (spec-preferred)
2. ``mgnify_rest`` - pure-Python MGnify REST equivalent (no R required)
3. ``sra_dada2``   - NCBI SRA + cutadapt + DADA2 + SILVA 138.1 via ``r/dada2_sra_pipeline.R``
4. ``simulate``    - deterministic offline surrogate

The surrogate exists so the pipeline is fully runnable and testable without
network access or R. Any artifact derived from it is tagged
``simulated=True`` in provenance metadata, written into the README results, and
refuses to be silently presented as real data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

PHYLUM_DISTRIBUTION = {
    "Proteobacteria": 0.40,
    "Firmicutes": 0.12,
    "Actinobacteria": 0.10,
    "Bacteroidetes": 0.10,
    "Patescibacteria": 0.05,
    "Acidobacteria": 0.04,
    "Chloroflexi": 0.04,
    "Planctomycetes": 0.03,
    "Verrucomicrobia": 0.03,
    "Nitrospirae": 0.02,
    "Gemmatimonadetes": 0.02,
    "Other": 0.05,
}


def rscript_available(cfg) -> bool:
    return shutil.which(str(cfg.acquisition.r_scripts.rscript_bin)) is not None


def _run_r(script: Path, args: List[str], cwd: Path, timeout: int = 7200) -> int:
    logger = get_logger()
    command = ["Rscript", str(script), *args]
    logger.info("Invoking: %s", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if completed.stdout:
        for line in completed.stdout.splitlines():
            logger.info("  [R] %s", line)
    if completed.returncode != 0:
        logger.error("R script failed (exit %d): %s", completed.returncode, completed.stderr.strip()[:2000])
    return completed.returncode


def acquire_via_r_mgnify(cfg, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    logger = get_logger()
    project_root = Path(cfg["_project_root"])
    script = project_root / str(cfg.acquisition.r_scripts.mgnify)
    outdir = paths["raw"] / "mgnify_r"
    if not script.exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "r_script_missing", "script": str(script)}
    if not rscript_available(cfg):
        logger.info("Rscript not found on PATH - skipping MGnifyR strategy")
        return pd.DataFrame(), pd.DataFrame(), {"status": "rscript_unavailable"}

    status = _run_r(
        script,
        ["--outdir", str(outdir),
         "--biome", "wastewater",
         "--max-samples", str(cfg.acquisition.mgnify.max_analyses)],
        cwd=project_root,
    )
    counts_file = outdir / "genus_counts.csv"
    meta_file = outdir / "sample_metadata.csv"
    if status != 0 or not counts_file.exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "r_mgnify_failed", "exit_code": status}

    counts = pd.read_csv(counts_file, index_col=0)
    metadata = pd.read_csv(meta_file) if meta_file.exists() else pd.DataFrame()
    info = {"status": "ok", "source": "MGnifyR (Bioconductor)",
            "n_samples": int(counts.shape[1]), "n_genera": int(counts.shape[0])}
    logger.info("MGnifyR returned %d genera x %d samples", counts.shape[0], counts.shape[1])
    return counts, metadata, info


def acquire_via_r_dada2(cfg, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    logger = get_logger()
    accessions = list(cfg.acquisition.sra.get("accessions") or [])
    if not accessions:
        logger.info("No SRA accessions configured (acquisition.sra.accessions empty) - skipping DADA2 strategy")
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_sra_accessions"}

    project_root = Path(cfg["_project_root"])
    script = project_root / str(cfg.acquisition.r_scripts.dada2)
    outdir = paths["raw"] / "sra"
    outdir.mkdir(parents=True, exist_ok=True)
    if not rscript_available(cfg):
        logger.info("Rscript not found on PATH - skipping DADA2 strategy")
        return pd.DataFrame(), pd.DataFrame(), {"status": "rscript_unavailable"}

    acc_file = outdir / "accessions.txt"
    acc_file.write_text("\n".join(str(a) for a in accessions), encoding="utf-8")
    status = _run_r(
        script,
        ["--accessions", str(acc_file),
         "--outdir", str(outdir),
         "--threads", str(cfg.acquisition.sra.threads),
         "--silva-dir", str(paths["raw"] / "silva")],
        cwd=project_root,
    )
    counts_file = outdir / "genus_counts.csv"
    if status != 0 or not counts_file.exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "dada2_failed", "exit_code": status}

    counts = pd.read_csv(counts_file, index_col=0)
    info = {"status": "ok", "source": "NCBI SRA + DADA2 + SILVA 138.1",
            "n_samples": int(counts.shape[1]), "n_genera": int(counts.shape[0]),
            "trunc_len": list(cfg.acquisition.sra.trunc_len),
            "max_ee": list(cfg.acquisition.sra.max_ee)}
    logger.info("DADA2 returned %d genera x %d samples", counts.shape[0], counts.shape[1])
    return counts, metadata_empty(), info


def metadata_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["sample_accession", "analysis_accession", "study_accession"])


def simulate_wastewater_communities(
    cfg,
    seed: int,
    curated_genera: List[str],
    curated_phylum: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deterministic genus x sample surrogate for offline runs.

    Structure mirrors real 16S survey data: overdispersed (Dirichlet-multinomial)
    compositions, heavy sparsity, uneven library sizes and study-level batch
    effects. Curated metal-active genera are given a modest abundance advantage in
    a random subset of samples purely so that downstream stages have signal to
    learn from - this encodes NO biological claim whatsoever.
    """
    logger = get_logger()
    sim = cfg.acquisition.simulate
    rng = np.random.default_rng(int(sim.seed))
    n_samples = int(sim.n_samples)
    n_genera = int(sim.n_genera)

    phyla = list(PHYLUM_DISTRIBUTION.keys())
    phylum_probs = np.array([PHYLUM_DISTRIBUTION[p] for p in phyla], dtype=float)
    phylum_probs = phylum_probs / phylum_probs.sum()

    assigned_phylum: Dict[str, str] = {}
    for genus, phylum in curated_phylum.items():
        if phylum in PHYLUM_DISTRIBUTION:
            assigned_phylum[genus] = phylum

    n_synthetic = max(1, n_genera - len(assigned_phylum))
    synthetic_phyla = rng.choice(phyla, size=n_synthetic, p=phylum_probs)
    synthetic_names = [f"{p}_genus_{i:04d}" for i, p in enumerate(synthetic_phyla, start=1)]
    for name, phylum in zip(synthetic_names, synthetic_phyla):
        assigned_phylum[name] = str(phylum)

    genera = sorted(assigned_phylum.keys())
    n_genera = len(genera)

    baseline = rng.lognormal(mean=-6.0, sigma=1.6, size=n_genera)
    for index, genus in enumerate(genera):
        if genus in curated_genera:
            baseline[index] *= rng.uniform(2.0, 6.0)

    alpha = np.clip(baseline / baseline.sum() * float(sim.dirichlet_alpha) + 1e-9, 1e-9, None)
    library_sizes = rng.integers(4_000, 60_000, size=n_samples)
    n_studies = max(4, n_samples // 24)
    study_of_sample = np.repeat(np.arange(n_studies), int(np.ceil(n_samples / n_studies)))[:n_samples]
    study_effect = rng.lognormal(mean=0.0, sigma=0.45, size=(n_studies, n_genera))

    proportions = np.zeros((n_genera, n_samples), dtype=float)
    for s in range(n_samples):
        mean = alpha * study_effect[study_of_sample[s]]
        mean = mean / mean.sum()
        gamma_draw = rng.gamma(shape=mean * 50.0, scale=1.0 / 50.0)
        proportions[:, s] = gamma_draw / gamma_draw.sum()

    counts = np.zeros((n_genera, n_samples), dtype=int)
    for s in range(n_samples):
        counts[:, s] = rng.multinomial(int(library_sizes[s]), proportions[:, s])

    zeros = rng.random(counts.shape) < float(sim.sparsity)
    counts[zeros] = 0

    samples = [f"SIM{i:05d}" for i in range(1, n_samples + 1)]
    wide = pd.DataFrame(counts, index=genera, columns=samples)
    wide = wide.loc[wide.sum(axis=1) > 0]

    metadata = pd.DataFrame(
        {
            "sample_accession": samples,
            "analysis_accession": [f"SIMAN{i:05d}" for i in range(1, n_samples + 1)],
            "study_accession": [f"SIMSTUDY{study_of_sample[s - 1]:03d}" for s in range(1, n_samples + 1)],
            "biome": "SIMULATED:root:Engineered:Wastewater",
            "pipeline_version": "simulated",
            "instrument_platform": "SIMULATED",
            "instrument_model": "SIMULATED",
            "n_genera": [int((wide[samples[s - 1]] > 0).sum()) for s in range(1, n_samples + 1)],
            "total_counts": [int(wide[samples[s - 1]].sum()) for s in range(1, n_samples + 1)],
        }
    )

    wide.attrs["phylum_map"] = assigned_phylum
    info = {
        "status": "ok",
        "source": "SIMULATED OFFLINE SURROGATE",
        "simulated": True,
        "seed": int(sim.seed),
        "n_samples": int(wide.shape[1]),
        "n_genera": int(wide.shape[0]),
        "n_studies": int(n_studies),
        "warning": (
            "These data are SYNTHETIC. They were generated because neither MGnify nor SRA "
            "acquisition succeeded in this environment. Results derived from them demonstrate "
            "pipeline mechanics only and carry no biological meaning."
        ),
    }
    logger.warning("=" * 76)
    logger.warning("USING SIMULATED DATA - public 16S acquisition did not succeed.")
    logger.warning("All downstream results reflect PIPELINE MECHANICS ONLY, not biology.")
    logger.warning("=" * 76)
    return wide, metadata, info
