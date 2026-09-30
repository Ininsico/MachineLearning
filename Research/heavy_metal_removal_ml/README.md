# Heavy-Metal Removal Prediction from Wastewater Biofilm Community Composition

A reproducible, thesis-oriented machine-learning pipeline that predicts heavy-metal
removal capability from bacterial community composition (16S rRNA amplicon data),
trained on public wastewater data and externally validated on two in-house biofilms.

**Status:** complete and executed end-to-end on real public data. All numbers in this
document are produced by the committed code and are reproducible from a fixed seed.

---

## Table of contents

1. [Headline findings](#1-headline-findings)
2. [What this pipeline does, and what it does not do](#2-what-this-pipeline-does-and-what-it-does-not-do)
3. [Repository layout](#3-repository-layout)
4. [Quick start](#4-quick-start)
5. [Environment and hardware](#5-environment-and-hardware)
6. [Data sources and accessions](#6-data-sources-and-accessions)
7. [Data acquisition in detail](#7-data-acquisition-in-detail)
8. [Preprocessing in detail](#8-preprocessing-in-detail)
9. [Dataset construction](#9-dataset-construction)
10. [Model architecture, tuning and evaluation protocol](#10-model-architecture-tuning-and-evaluation-protocol)
11. [Results](#11-results)
12. [External validation on the in-house biofilms](#12-external-validation-on-the-in-house-biofilms)
13. [The GPU backend, measured](#13-the-gpu-backend-measured)
14. [Reproducibility](#14-reproducibility)
15. [Testing](#15-testing)
16. [Limitations](#16-limitations)
17. [Improvement roadmap](#17-improvement-roadmap)
18. [Artifact manifest](#18-artifact-manifest)
19. [Glossary](#19-glossary)

---

## 1. Headline findings

These are the five results that matter most. Each is developed in full later.

**1. The pipeline runs end-to-end on real public data.** 250 amplicon samples were
harvested from six MGnify wastewater studies, yielding 1,643 genera. After quality
control and filtering, 248 samples × 114 genera were available, supporting a
114 × 255 supervised dataset with 10 positive / 104 negative genera.

**2. Community-composition features carry essentially no signal for this label.**
A feature-group ablation shows that **five binary phylum features achieve ROC-AUC
1.000 while the full 248-dimensional CLR abundance profile reaches only 0.422**. Two
independent attribution methods agree: permutation importance and SHAP both rank the
phylum indicators first and second, and **every one of the 248 abundance features has
exactly zero permutation importance**. The supervised task is therefore not being
solved by community composition at all — it is being solved by lineage membership.
This is the most important scientific result in the document and it is developed in
[§11.6](#116-feature-group-ablation--the-headline-result) and
[§11.8](#118-feature-attribution).

**3. The label set is the binding constraint, not the model.** Only 10 of the 114
retained genera carry a curated literature label. Every cross-validated metric rests
on **8 positive examples in training and 2 in the held-out test set**. The
near-perfect test metrics reported in [§11.5](#115-held-out-test-set) should be read
with that in mind.

**4. The external validation is honest about what it can and cannot show.** For the
one metal/sample pair with both a quantitative literature value and a measurement —
arsenate removal in the *Bacillus*-dominated SM biofilm — expectation (94.94%) and
observation (96.79%) agree to within 1.9 percentage points. The same procedure fails
badly for Ni and Cr in OS, where the model predicts removal and the measurement shows
net release. **The SM sample could not be scored by the trained classifier at all**,
because *Bacillus* never appears in the public training taxonomy.

**5. The pipeline runs on the GPU, and the resource profile is safe.** The MLP runs on
the GTX 1050 Ti by default. Measured over a complete run: **peak RAM 15.00 GiB of
23.7 GiB, peak GPU memory 155 MiB of 4096 MiB, mean CPU utilisation 21.8%**. There is
no OOM risk on either side. The GPU is roughly 2.4× slower in wall-clock than sklearn
on CPU for this model size (314 s vs ~90 s per run) because the network is tiny, but it
leaves the CPU largely free. See [§13](#13-the-gpu-backend-measured).

**During implementation three real bugs were found and fixed**, each of which had
silently produced wrong numbers:

| Bug | Effect | Fixed in |
|---|---|---|
| `early_stopping` gated the patience check | Every torch fit ran the full 1000 epochs; the grid search could not finish in 10 minutes | `torch_backend.py` — now 30 epochs, **32× faster** |
| `TorchMLPClassifier` mixin order | sklearn did not recognise it as a classifier; `roc_auc` scoring crashed with `y should be a 1d array, got (19, 2)` | `torch_backend.py` — `ClassifierMixin` must precede `BaseEstimator` |
| Ternary/`+` precedence in `summarise_confusion` | Balanced accuracy was reported as exactly **half** its true value (0.500 instead of 1.000) | `evaluate.py` — terms computed explicitly |

All three now have regression tests ([§15](#15-testing)).

---

## 2. What this pipeline does, and what it does not do

This distinction is the single most important thing to understand before reading
further, so it is stated before any results.

### What it does

* Acquires real public wastewater 16S data and assembles a genus × sample count table.
* Builds a supervised dataset whose observations are **genera**, whose features are
  the genus' CLR-transformed abundance profile across samples plus lineage
  information, and whose label records **whether the published literature documents
  heavy-metal removal or tolerance for that genus**.
* Trains a multilayer perceptron with the architecture specified for the study,
  tunes it by grid search over stratified cross-validation, and benchmarks it against
  Random Forest, XGBoost and logistic regression.
* Measures, honestly, how much of the resulting performance is attributable to
  community composition versus mere lineage membership.
* Applies the trained model to two in-house biofilms and compares predictions against
  measured before/after metal concentrations.
* Produces every artifact, figure and table needed to write the thesis chapter.

### What it does **not** do

* **It does not predict metal removal efficiency from community composition.** No
  model here maps a community to a removal percentage for a given metal. The public
  16S data carry no metal measurements at all, so that mapping cannot be learned from
  them.
* **It does not avoid label circularity — it measures it.** The label is a statement
  *about the genus*, and lineage features are known perfectly for any given genus.
  A model can therefore recover the label by reading the phylum rather than the
  ecology. The ablation in [§11.6](#116-feature-group-ablation--the-headline-result)
  quantifies exactly this, and the answer is uncomfortable: lineage is doing the work.
* **It does not establish generalisability from n = 2.** Two in-house biofilms
  demonstrate that the pipeline runs on real samples and produces physically
  plausible numbers. They cannot validate a model.

If the goal is a genuinely predictive model of metal removal, the data that would
support it — paired community composition **and** measured metal removal for the same
samples — do not yet exist in the public archives. This pipeline is built so that it
is ready when they do; see [§17](#17-improvement-roadmap).

---

## 3. Repository layout

```
heavy_metal_removal_ml/
├── README.md                          # this document
├── config.yaml                        # every parameter, seed and threshold
├── requirements.txt                   # Python dependencies
├── conftest.py                        # pytest path configuration
├── run_pipeline.py                    # CLI entry point
│
├── src/                               # pipeline package
│   ├── __init__.py
│   ├── config.py                      # config loading, seeding, provenance hashing
│   ├── logging_utils.py               # console + file logging, stage banners, tables
│   ├── literature.py                  # curated label tables and lookups
│   ├── inhouse.py                     # SM/OS composition, metal concentrations, removal maths
│   ├── acquire_mgnify.py              # MGnify REST client, OTU parsing, caching
│   ├── acquire_fallback.py            # R drivers (MGnifyR, DADA2) + offline surrogate
│   ├── features.py                    # CLR, zero replacement, filtering, scaling, selection
│   ├── datasets.py                    # Track A / B / C construction
│   ├── models.py                      # MLP, baselines, GridSearchCV, balance assessment
│   ├── torch_backend.py               # optional PyTorch/CUDA MLP backend
│   ├── evaluate.py                    # metrics, ROC, PR curve, confusion matrix, importance
│   ├── external_validation.py         # SM/OS scoring and honest reporting
│   └── pipeline.py                    # orchestrator, six stages
│
├── r/                                 # R scripts required by the study specification
│   ├── mgnify_retrieval.R             # MgnifyClient + doQuery + getResult(output="phyloseq")
│   └── dada2_sra_pipeline.R           # prefetch + fasterq-dump + cutadapt + DADA2 + SILVA 138.1
│
├── data/
│   ├── literature/                    # CURATED, version-controlled, reviewed by hand
│   │   ├── genus_labels.csv           # 44 genera, binary metal-activity labels
│   │   ├── genus_metal_efficiency.csv # quantitative per-metal efficiencies
│   │   ├── inhouse_communities.csv    # SM and OS community compositions
│   │   └── inhouse_metal_concentrations.csv  # before/after metal concentrations
│   ├── raw/                           # acquired public data + per-analysis cache
│   ├── interim/
│   └── processed/
│
├── results/
│   ├── figures/                       # 6 PNG figures
│   ├── tables/                        # 24 CSV/JSON artifacts
│   ├── models/                        # fitted model + scaler (.pkl)
│   └── logs/pipeline.log              # full timestamped run log
│
└── tests/test_pipeline.py             # 41 unit tests
```

---

## 4. Quick start

```bash
cd heavy_metal_removal_ml
python -m pip install -r requirements.txt

# Full run: real MGnify acquisition, grid search, external validation.
# The MLP runs on CUDA by default (config.yaml: model.backend = torch).
python run_pipeline.py

# Fastest wall-clock single run (sklearn MLP on CPU, ~90 s end-to-end)
python run_pipeline.py --backend sklearn

# Force a fresh download instead of using the on-disk cache
python run_pipeline.py --force-acquisition

# Reproduce fully offline with the deterministic surrogate (no network, no R)
python run_pipeline.py --strategy simulate

# Fast smoke test (~100 s) that skips the hyper-parameter grid
python run_pipeline.py --strategy simulate --quick

# Run the test suite
python -m pytest tests -q
```

CLI options:

| Flag | Purpose |
|---|---|
| `--config PATH` | Use an alternative configuration file |
| `--strategy {mgnify_r,mgnify_rest,sra_dada2,simulate}` | Force one acquisition strategy |
| `--force-acquisition` | Ignore the on-disk cache and re-download |
| `--quick` | Skip `GridSearchCV` |
| `--backend {sklearn,torch}` | Select the MLP implementation (default `torch`) |
| `--device {auto,cuda,cpu}` | Device for the torch backend (default `cuda`) |
| `--seed INT` | Override the global random seed |
| `--samples INT` | Cap the number of public samples harvested |
| `--verbose` | Debug-level logging |

---

## 5. Environment and hardware

Recorded by the pipeline into `results/tables/run_metadata.json` on every run.

| Component | Version / value |
|---|---|
| Operating system | Windows (`win32`) |
| Python | 3.12.10 (`C:\Users\Administrator\AppData\Local\Programs\Python\Python312`) |
| scikit-learn | 1.9.0 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| seaborn | 0.13.2 (optional, plot styling) |
| XGBoost | 3.4.1 (optional baseline) |
| SHAP | 0.52.0 (optional attribution) |
| PyTorch | 2.14.0+cu126 (optional backend) |
| R | **not installed** |
| SRA Toolkit (`prefetch`, `fasterq-dump`) | **not installed** |
| cutadapt | **not installed** |
| GPU | NVIDIA GeForce GTX 1050 Ti with Max-Q Design, 4096 MiB, sm_61, 6 SMs, driver 572.70 |
| CUDA | available and verified (`torch.cuda.is_available() == True`) |
| **Default MLP backend** | **torch on CUDA** (see [§13](#13-the-gpu-backend-measured)) |

The absence of R, the SRA Toolkit and cutadapt is why the Python REST acquisition
path is the one that actually executed. The R scripts are provided and are invoked
automatically if `Rscript` is ever found on `PATH` — see [§7](#7-data-acquisition-in-detail).

---

## 6. Data sources and accessions

### 6.1 Public 16S data — MGnify

* **Source:** MGnify (EBI Metagenomics) REST API v1, `https://www.ebi.ac.uk/metagenomics/api/v1`
* **Biome lineage:** `root:Engineered:Wastewater`, plus `root:Engineered:Wastewater:Activated Sludge`
* **Experiment type filter:** `amplicon` only (metagenomic runs excluded)
* **Taxonomy source:** per-analysis `*_SSU_OTU.tsv` download (see [§7.3](#73-why-the-otu-table-not-the-summary-endpoint))
* **Studies actually used (6):**

| Study accession | Samples used |
|---|---|
| MGYS00005617 | 88 |
| MGYS00005741 | 76 |
| MGYS00004472 | 54 |
| MGYS00004521 | 26 |
| MGYS00004060 | 8 |
| MGYS00006558 | 4 |
| **Total** | **250** |

* **Sequencing platforms present:** Illumina (88 samples, pipeline v4.1; 88 samples,
  pipeline v5.0) and Roche 454 GS FLX (80 samples, pipeline v5.0).
* **HTTP requests:** 545 for the full harvest. Every per-analysis table is cached
  under `data/raw/mgnify_cache/`, so re-runs are near-instant.
* **Cache fingerprint:** `mgnify_genus_counts.csv.gz` SHA-256 prefix `56b5ac4994c58fd6`.

### 6.2 In-house biofilm samples

Two biofilms, provided by the wet-lab side of the project. Compositions are stored in
`data/literature/inhouse_communities.csv`.

**SM** — effectively a monoculture:

| Genus | Species | Relative abundance |
|---|---|---|
| *Bacillus* | *Bacillus stratosphericus* | 99.99% |
| (residual) | — | 0.01% |

**OS** — a nine-taxon consortium:

| Genus | Species | Relative abundance |
|---|---|---|
| *Raoultella* | *Raoultella ornithinolytica* | 41.60% |
| *Pseudomonas* | *Pseudomonas veronii* | 27.67% |
| (unclassified) | — | 18.95% |
| *Pseudomonas* | *Pseudomonas marginalis* | 4.45% |
| *Microvirgula* | *Microvirgula curvata* | 2.24% |
| *Enterobacter* | *Enterobacter cloacae* | 1.97% |
| *Klebsiella* | *Klebsiella oxytoca* | 1.05% |
| *Delftia* | *Delftia tsuruhatensis* | 0.86% |
| *Chryseobacterium* | *Chryseobacterium indologenes* | 0.64% |
| (residual) | — | 0.57% |

Species are collapsed to genus level for modelling, which matters for OS: the two
*Pseudomonas* species combine to 32.12%.

### 6.3 Matched metal concentrations

Nine metals, before/after treatment, in `data/literature/inhouse_metal_concentrations.csv`.

| Metal | Before (mg/L) | After (mg/L) | Removal (%) | Outcome |
|---|---|---|---|---|
| Hg | 31.23 | 6.350 | 79.667 | removed |
| Pb | 0.075 | 0.030 | 60.000 | removed |
| As | 0.780 | 0.025 | 96.795 | removed |
| Ni | 0.006 | 0.011 | −83.333 | **released** |
| Fe | 0.155 | below detection | 100.000 | removed |
| Ca | 13.79 | 8.665 | 37.165 | removed |
| Zn | 0.791 | below detection | 100.000 | removed |
| Mn | 0.130 | 0.523 | −302.308 | **released** |
| Cr | below detection | 0.030 | 0.000 | unchanged |

> **Ambiguity that must be resolved by the user.** The brief supplied a single
> before/after vector for the two biofilms, not a separate vector per biofilm. The
> CSV therefore carries a `sample_id` column and the same observed vector is assigned
> to both SM and OS. If the two biofilms were actually run in separate batches, edit
> the CSV to carry per-sample effluent values — nothing else in the pipeline needs to
> change. This is flagged again in [§16](#16-limitations).

### 6.4 Literature-derived labels

Stored as hand-checkable CSVs, because these are the weakest link in the whole
pipeline and a reviewer must be able to audit them.

**`genus_labels.csv`** — 44 genera, one row each:

| Column | Meaning |
|---|---|
| `genus` | Genus name, must match SILVA usage |
| `phylum` | Phylum, used as a fallback when the observed lineage is unresolved |
| `label` | 1 = documented removal/tolerance; 0 = no documentation located |
| `evidence_level` | `quantitative_removal`, `qualitative_removal`, `tolerance_only`, `no_documentation_found` |
| `mechanism` | Reported mechanism (biosorption, EPS binding, reduction, sulfide precipitation, …) |
| `needs_verification` | `TRUE` for every entry not supplied with a value by the user |
| `source_note` | Provenance of the entry |

**`genus_metal_efficiency.csv`** — long format, 7 rows, 3 genera, 6 metals:

| Genus | Metal | Efficiency (%) | Note |
|---|---|---|---|
| *Raoultella* | Pb | 89.0 | user-supplied |
| *Raoultella* | Cd | 67.0 | user-supplied |
| *Raoultella* | Cr | 63.4 | user-supplied |
| *Raoultella* | Ni | 55.6 | user-supplied |
| *Pseudomonas* | Cu | 80.0 | user-supplied (EPS-mediated) |
| *Bacillus* | As | 99.9 | user-supplied, arsenate As(V) |
| *Bacillus* | As | 90.0 | user-supplied, arsenite As(III) |

> **Provenance policy.** Only the seven rows above carry numbers, and all seven came
> from the study brief. No numeric efficiency was invented for any other genus. The
> remaining 37 curated genera are recorded as `qualitative_removal` or
> `tolerance_only` with `needs_verification = TRUE` and an explicit note that the
> exact efficiencies require verification against primary literature before the
> thesis is submitted. The pipeline records the SHA-256 of this file with every
> model, so a table change is always traceable.

---

## 7. Data acquisition in detail

### 7.1 Strategy order

Configured in `config.yaml` under `acquisition.strategy_order` and attempted in order
until one yields at least `acquisition.mgnify.min_samples` (200) usable samples:

1. **`mgnify_r`** — the specification-preferred path. Runs `r/mgnify_retrieval.R`,
   which performs

   ```r
   client <- MgnifyClient(useCache = TRUE)
   studies <- doQuery(client, biome = "wastewater", type = "studies", experiment_type = "amplicon")
   phy <- getResult(client, studies, get.taxa = TRUE, output = "phyloseq")
   ```

   and exports a genus × sample count matrix plus the `phyloseq` object.

2. **`mgnify_rest`** — a behaviour-preserving pure-Python equivalent using the same
   REST API. **This is the path that actually ran**, because R is not installed here.

3. **`sra_dada2`** — `r/dada2_sra_pipeline.R`, implementing the specified fallback:
   `prefetch` + `fasterq-dump`, `cutadapt` primer trimming, then
   `filterAndTrim(truncLen = c(240, 160), maxEE = c(2, 2))`, `learnErrors`, `dada`,
   `mergePairs`, `makeSequenceTable`, `removeBimeraDenovo`, `assignTaxonomy` against
   SILVA 138.1, followed by removal of mitochondrial, chloroplast and unclassified
   reads. Requires SRA accessions in `acquisition.sra.accessions`; none are configured,
   so this strategy reports `no_sra_accessions` and is skipped.

4. **`simulate`** — a deterministic offline surrogate. Used only when everything else
   fails. Any artifact derived from it is tagged `simulated: true`, written into
   `run_metadata.json`, and printed to the console in a boxed warning. The surrogate
   exists so the pipeline remains testable without network access; results from it
   are pipeline mechanics, not biology.

The simulated path is also exercised by the test suite and was used during
development. **The results in [§11](#11-results) and [§12](#12-external-validation-on-the-in-house-biofilms)
come from real MGnify data, not the surrogate** (`"simulated": false` in
`run_metadata.json`).

### 7.2 MGnify API behaviour discovered during implementation

Three quirks cost real debugging time and are documented so nobody repeats them:

1. **Query-string filters are silently ignored.** Both
   `/studies?biome_lineage=root:Environmental:Aquatic:Wastewater` and
   `/samples?biome_lineage=...` return HTTP 200 with the *unfiltered* corpus
   (5,203 studies, 435,812 samples). Only the relationship endpoint
   `/biomes/{lineage}/studies` actually filters. A silent no-op filter is a
   correctness hazard: it produced plausible-looking results that were entirely wrong.
2. **`/samples/{accession}/analyses` does not exist.** It returns 404. Analyses are
   reached through `/studies/{accession}/analyses`, and each analysis carries a
   `sample` relationship. One analysis per sample is retained (highest pipeline
   version), because samples appear under multiple studies — `SRS720730` is in both
   `MGYS00006558` and `MGYS00006570`.
3. **The biome lineage in the brief, `"wastewater"`, is not a valid MGnify lineage.**
   `root:Environmental:Aquatic:Wastewater` 404s. The functional equivalent is
   `root:Engineered:Wastewater` (186 studies) plus its child
   `root:Engineered:Wastewater:Activated Sludge` (80 studies).

### 7.3 Why the OTU table, not the summary endpoint

The `/analyses/{acc}/taxonomy/ssu` endpoint returns a **rank-collapsed summary**. For a
typical sample it resolves only a few dozen genera and lumps the rest at family or
order level. The per-analysis `*_SSU_OTU.tsv` download carries the full MAPseq
assignment per OTU.

Measured on the same sample (`MGYA00700362`):

| Source | Genus-level taxa resolved | Reads assignable to genus |
|---|---|---|
| `/taxonomy/ssu` summary | ~44 (in the first page of 425 rows) | — |
| `*_SSU_OTU.tsv` | **198** | 38.9% |

Over the whole corpus the difference is decisive:

| Source | Total genera recovered | Genera retained at 0.1% | Positive labels available |
|---|---|---|---|
| `taxonomy_ssu` | 402 | 61 | **3** — untrainable |
| `otu_table` (default) | **1,643** | **114** | **10** — trainable |

With only 3 positive genera the pipeline cannot cross-validate at all; this is the
root cause that the OTU route fixes. Roughly 61% of reads remain unassigned at genus
level, which is a genuine property of MGnify's MAPseq assignments on short reads, not
a parsing failure.

Lineages are parsed by rank prefix (`sk__`, `k__`, `p__`, `c__`, `o__`, `f__`, `g__`,
`s__`), not by position, because empty ranks are still emitted as segments. Positional
parsing silently misassigns ranks. Genera matching `unclassified`, `uncultured`,
`unknown`, `unidentified`, `incertae` or `ambiguous` are discarded.

---

## 8. Preprocessing in detail

Order of operations is deliberate and each step is justified. Steps 5–6 are wrapped in
scikit-learn transformers so they are re-fitted *inside every CV fold*; doing them
once up front would leak test information into training.

### 8.1 Step 1 — Within-sample relative abundance (closure)

Each sample's counts are divided by its total so every sample sums to 1. Samples with
a zero total raise an error rather than silently producing infinities.

### 8.2 Step 2 — Library-size QC

Samples with fewer than `preprocessing.min_library_size` (50) assigned reads are
dropped before the CLR transform, because a handful of reads cannot support a stable
composition. Two samples were removed in the real run (`SRS500633`, `SRS500638`),
taking 250 → 248. Dropped samples are logged individually so the loss of n is never
silent.

### 8.3 Step 3 — Zero replacement

The counts are 86–100% sparse at genus level, and `log(0)` is undefined, so zeros must
be replaced before the CLR transform. Two methods are implemented:

* **Multiplicative replacement (default).** A zero is replaced by `delta = 1/N` where
  `N` is that sample's total count — approximately one sequencing read — and the
  non-zero entries are rescaled by `1 - z·delta` so the composition still sums to 1.
  This is the standard approach because it preserves the simplex geometry.
* **Additive pseudocount.** Replace zeros with a fixed `pseudocount` (default `1e-6`)
  and re-close. Simpler, but the choice of pseudocount is arbitrary and biases the
  log-ratios.

Configured by `preprocessing.clr_zero_handling`.

### 8.4 Step 4 — Centered log-ratio (CLR) transform

16S relative abundances are **compositional**: they are constrained to sum to 1, so
they live on a simplex, not in Euclidean space. Pearson correlations between
abundances are therefore biased, often severely, and a subset's covariance structure
is distorted by the closure constraint. Log-ratio transforms move the data into real
space where standard statistics apply.

For a composition `x` of `D` parts:

```
clr(x)_i = log(x_i) − (1/D) · Σ_j log(x_j)
```

The CLR of each sample is computed **across genera within that sample**, which is what
the composition actually is. Track A then transposes so that each genus becomes an
observation whose features are its CLR values across samples.

Note the consequence for the in-house samples: SM is 99.99% *Bacillus*, so its CLR
vector over the training genus set is extreme by construction. That is a faithful
representation of a near-monoculture, not an artefact.

### 8.5 Step 5 — Taxon filtering

`preprocessing.min_mean_relative_abundance: 0.001` (0.1%, per the study brief) and
`preprocessing.min_prevalence: 0.02`.

"Mean relative abundance" is ambiguous for data pooled across heterogeneous studies,
so both readings are implemented and reported:

* **`mean_overall` (default, strict reading of the brief)** — mean across *all*
  samples, treating a non-detection as zero. A genus that dominates one study but is
  absent from the other five is heavily penalised.
* **`mean_when_present`** — mean across only the samples where the genus was detected,
  paired with the prevalence requirement. This is the conventional ecological reading
  and is much less punitive for multi-study pools.

On the real corpus, `mean_overall` at 0.1% retains **114 of 1,643 genera**. The
published pipeline summary records the retained count and label balance under
whichever setting is active, so the sensitivity to this choice is always visible.

### 8.6 Step 6 — Standardisation

`StandardScaler` (zero mean, unit variance), fitted on training data only and applied
to test data with the training parameters. Required here because the MLP's Adam solver
and L2 penalty treat all features uniformly, and CLR values span several orders of
magnitude after zero replacement.

### 8.7 Step 7 — Feature selection

Triggered when dimensionality exceeds `preprocessing.feature_selection.dim_threshold`
(200), which Track A's 255 features do. `select_k_best` with `k = 40` using ANOVA
F-statistics for classification (`f_classif`) or `f_regression` for regression;
`variance_threshold` is available as an alternative.

Two implementation details matter:

* The selector is a **step inside the sklearn `Pipeline`**, so it is re-fitted on each
  training fold. Selecting features on the full dataset before cross-validating is one
  of the most common ways to produce inflated results in this literature.
* A `VarianceThreshold(0.0)` runs *before* `SelectKBest`. An all-zero one-hot phylum
  column has zero within-group variance, which makes the ANOVA F-statistic undefined
  and emits a divide-by-zero warning. This was observed and fixed.

---

## 9. Dataset construction

### 9.1 Track A — genus-level literature-supervised classification

**The primary supervised task.**

| Property | Value (real run) |
|---|---|
| Observations | 114 genera |
| Features | 255 |
| — abundance | 248 (CLR across 248 samples) |
| — phylum | 5 (binary one-hot) |
| — prevalence | 2 (prevalence, mean relative abundance) |
| Label 1 | 10 |
| Label 0 | 104 |
| Train / test split | 91 (8 positive) / 23 (2 positive) |

**Labelling rule.** Every genus recovered from the public data is scored:

* the genus takes its curated label if it appears in `genus_labels.csv`;
* otherwise it is labelled **0**, meaning *no documentation was located*.

That second clause is a real and important distinction. **Label 0 means "undocumented",
not "proven incapable".** Absence of evidence in the literature is not evidence of
absence of capability, and treating it as such injects systematic false negatives into
the negative class. A genus that is a genuine metal remover but has never been studied
is mislabelled. This is the largest single source of label noise in the pipeline and it
is unavoidable given the design.

Phylum membership is taken from the **observed** MGnify/SILVA lineage wherever it
resolves, falling back to the curated table only when it does not. Genera observed in
fewer than `dataset.track_a.min_samples_present` (5) samples are dropped.

### 9.2 Track B — sample-level community metal-activity index

Rows are samples; the value is an abundance-weighted mean genus metal-activity:

```
index(s) = Σ_g  w_g · rel_abund(g, s)  /  Σ_g  rel_abund(g, s)
```

where `w_g = 1` for a documented genus and `0` otherwise, summed over the genera that
carry any annotation at all. It is directly interpretable: *what fraction of this
community, by abundance, belongs to genera with documented metal removal or tolerance.*

Normalising by annotated abundance rather than total abundance is deliberate, so that
unclassified reads do not dilute the index. The annotated fraction is reported
separately as a coverage diagnostic.

This is an index, not a trained model. It is the defensible quantity for external
validation, because it does not depend on a classifier that was trained on the same
curated list.

### 9.3 Track C — per-metal quantitative regression

Intended to regress documented per-metal efficiency on composition, using
`MLPRegressor`. Built and executed, but **the mined literature supports only 7
quantitative values across 3 genera**, and after intersecting with the recovered
taxonomy only *Pseudomonas* survives for one metal (Cu) — below the
`dataset.track_c.min_genera` requirement of 4.

The pipeline reports this as data-limited rather than fabricating a target. Any
regression trained on three points would be meaningless. This is an honest "not
estimable" result, and the code path is ready for when the efficiency table grows.

---

## 10. Model architecture, tuning and evaluation protocol

### 10.1 The model

A feed-forward multilayer perceptron, as specified:

| Hyper-parameter | Value |
|---|---|
| `hidden_layer_sizes` | `(64, 32)` |
| `activation` | `relu` |
| `solver` | `adam` |
| `alpha` (L2) | `0.001` |
| `learning_rate` | `adaptive` |
| `learning_rate_init` | `0.001` |
| `early_stopping` | `True` |
| `validation_fraction` | `0.15` |
| `n_iter_no_change` | `15` |
| `batch_size` | `32` |
| `max_iter` | `1000` |
| `tol` | `1e-4` |
| `random_state` | 42 |

### 10.2 Why not a CNN, LSTM or Transformer

Stated explicitly because the brief asks for this to be documented. The inputs are
**static community snapshots**: one fixed-length vector of taxon abundances per
observation. There is no sequence ordering, no spatial locality and no temporal
structure in the feature vector.

* A **CNN** assumes spatial or temporal locality. The "adjacency" of two genus features
  is an artefact of column order, so convolution would be meaningless.
* An **LSTM/GRU** assumes a sequence over time or position. No such sequence exists.
* A **Transformer** would add quadratic attention over an arbitrary ordering, buying
  capacity with no structural justification — and with 91 training observations and 8
  positives it would overfit catastrophically.

An MLP is the correct inductive bias for a fixed-length, unordered, dense feature
vector, and it is also the architecture the study specifies.

### 10.3 Baselines

To justify the MLP's added complexity, three reference models are trained on the
identical pipeline and splits:

* **Random Forest** — 400 trees, unlimited depth.
* **XGBoost** — 400 estimators, `max_depth=4`, `learning_rate=0.05`, subsample and
  column-subsample 0.8.
* **Logistic Regression** — L2, `C=1.0`.

XGBoost is imported defensively; if the package is missing the baseline is omitted
with a warning rather than crashing the run.

### 10.4 Cross-validation and the split

* **Split:** 80/20 stratified on the global seed, applied to genera.
* **CV:** `StratifiedKFold`, 5 folds, shuffled, seeded.
* Genera appear once each in Track A, so there is no duplicate-row leakage between
  train and test. This is a property of the dataset design, and it is why a plain
  stratified split is acceptable here.

**Adaptive fold selection.** Stratified k-fold requires at least `k` members of the
minority class. `assess_class_balance()` caps the requested fold count by the minority
count and reports the cap, and returns 0 when the minority class has fewer than 2
members. In that case the pipeline refuses to cross-validate, fits a single model on
everything so the external-validation stage can still run, and marks every
classification metric as **not estimable** rather than printing numbers that would be
noise. This guard fired on an earlier iteration of the real data (3 positive genera)
and is the reason the pipeline does not crash on untrainable datasets.

### 10.5 Hyper-parameter search

`GridSearchCV`, scored on ROC-AUC, over:

| Parameter | Candidates |
|---|---|
| `hidden_layer_sizes` | `(32)`, `(64,32)`, `(128,64)`, `(64,32,16)`, `(256,128)` |
| `alpha` | `1e-4`, `1e-3`, `1e-2` |
| `learning_rate_init` | `1e-3`, `1e-2` |
| `early_stopping` | `True`, `False` |

60 candidates × 5 folds = **300 fits**, completed in **222.4 s on the GPU** (or ~26 s on
sklearn CPU / torch CPU; see [§13](#13-the-gpu-backend-measured)).

`early_stopping` is searched against both values even though the brief specifies
`True`. With 91 training rows, the 15% internal validation split holds roughly 14
rows and about one positive example, so early stopping can halt training on the
strength of essentially one observation. Letting the grid decide is the honest choice —
and it chose `False`, confirming the concern ([§11.4](#114-grid-search-result)).

### 10.6 Decision-threshold tuning

The positive class is 8.8% of the training data. At the default 0.5 threshold the
model predicts almost everything negative, so F1 collapses to zero and the confusion
matrix is uninformative.

The threshold is therefore tuned to maximise F1 on **out-of-fold** training scores
(`cross_val_predict`), then applied unchanged to the held-out test set. Fitting the
threshold on the test set would leak; fitting it out-of-fold does not. The tuned
threshold on the real run was **0.0100**, an out-of-fold F1 of 0.800.

### 10.7 Metrics and controls

**Classification:** accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC),
Matthews correlation coefficient, specificity, balanced accuracy.

**Controls reported alongside every result:**

* a **majority-class reference**, because with an 8.8% positive rate a trivial
  always-negative classifier already scores 91% accuracy;
* **PR-AUC** in addition to ROC-AUC, since ROC-AUC is optimistic under class imbalance;
* a **feature-group ablation** ([§11.6](#116-feature-group-ablation--the-headline-result))
  that isolates how much performance comes from lineage alone.

---

## 11. Results

All figures below are from the executed run on **real MGnify data** with
`model.backend: sklearn`, seed 42, config fingerprint `dda86c7c5aa3c021`.

### 11.1 Acquisition and preprocessing summary

| Stage | Outcome |
|---|---|
| Wastewater studies discovered | 180 (from 2 biome lineages) |
| Studies contributing amplicon analyses | 6 |
| Unique amplicon samples selected | 256 analyses; 250 yielded usable taxonomy |
| Distinct genera recovered | 1,643 |
| HTTP requests | 545 |
| Samples dropped for low depth (<50 reads) | 2 (`SRS500633`, `SRS500638`) |
| Samples usable after QC | 248 |
| Genera retained at 0.1% mean abundance | 114 of 1,643 (6.9%) |
| Final feature space | 248 samples × 114 CLR-transformed genera |
| Simulated data? | **No** |

### 11.2 Track A dataset

| Property | Value |
|---|---|
| Genera (observations) | 114 |
| Features | 255 |
| Labelled 1 (documented) | 10 |
| Labelled 0 (undocumented) | 104 |
| Curated genera with an explicit entry | 10 of 44 in the table |
| Positive rate | 8.77% |
| Train / test | 91 (8 pos) / 23 (2 pos) |

Only 10 of the 44 curated genera were detected in the public corpus at sufficient
abundance. The other 34 — including *Bacillus*, *Klebsiella*, *Enterobacter*,
*Raoultella* and *Delftia* — either never appear in MGnify's wastewater amplicon data
or fall below the 0.1% filter. **This is the binding constraint on the entire
supervised exercise**, and it is why the SM external validation cannot score.

### 11.3 Cross-validated model comparison

5-fold stratified CV on 91 training genera (8 positive). Mean ± standard deviation
across folds.

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | Avg precision |
|---|---|---|---|---|---|---|
| **MLP (tuned)** | — | — | — | — | **0.9529** | — |
| Random Forest | 0.9444 ± 0.039 | 0.400 | 0.300 | 0.333 | 0.9412 ± 0.102 | 0.740 |
| XGBoost | 0.9123 ± 0.030 | 0.000 | 0.000 | 0.000 | 0.9059 ± 0.153 | 0.695 |
| MLP (defaults) | 0.9333 ± 0.047 | 0.400 | 0.300 | 0.333 | 0.8566 ± 0.158 | 0.532 |
| *Majority class* | *0.9121* | *0.000* | *0.000* | *0.000* | *0.500* | *0.088* |

**Reading this table.**

* The tuned MLP **edges out Random Forest** at ROC-AUC 0.9529 vs 0.9412. The margin
  (0.012) is far smaller than the fold-to-fold standard deviation (0.102), so this is a
  tie in practice, not a win. Reporting it as a win would be overclaiming.
* Tuning matters enormously: the default MLP scores 0.8566 and the tuned MLP 0.9529.
  Nobody should report an untuned MLP here.
* Accuracy is a trap. The majority-class reference gets 0.9121 accuracy with F1 = 0.
  Every accuracy figure in this document must be read next to its F1 and ROC-AUC.
* Every metric carries a standard deviation of 0.10–0.19 across folds. With 8
  positives, the fold-to-fold spread is larger than every difference between models, so
  the ranking between Random Forest, XGBoost and the MLP is **not statistically
  meaningful**.

### 11.4 Grid search result

Best configuration found:

| Parameter | Selected |
|---|---|
| `hidden_layer_sizes` | **(128, 64)** |
| `alpha` | **0.01** |
| `learning_rate_init` | **0.01** |
| `early_stopping` | **False** |

Best cross-validated ROC-AUC: **0.9529**, over 300 fits in 222.4 s on the GPU.

Two things are worth noting.

* The grid moved to **stronger regularisation** (`alpha` 0.001 → 0.01) and a **wider**
  network than specified, while the brief's `(64, 32)` remained competitive.
* The grid **rejected early stopping**, and this finding turned out to be doubly
  important. Beyond the statistical argument in
  [§10.5](#105-hyper-parameter-search) — that a 15% split of 91 rows holds too few
  positives to stop on reliably — it also exposed the torch backend bug described in
  [§13.1](#131-the-bug-quantified), where the patience rule was incorrectly gated
  behind `early_stopping` and every fit therefore ran the full 1000 epochs.

### 11.5 Held-out test set

23 genera, 2 positive. Threshold 0.1756, tuned out-of-fold.

| Metric | Value |
|---|---|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall / sensitivity | 1.0000 |
| F1 | 1.0000 |
| MCC | 1.0000 |
| ROC-AUC | 1.0000 |
| Average precision | 1.0000 |
| Specificity | 1.0000 |
| Balanced accuracy | 1.0000 |

Confusion matrix:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **True 0** | 21 | 0 |
| **True 1** | 0 | 2 |

**A perfect test score on two positive examples is not a meaningful result and must not
be reported as one.** Any classifier that ranks two positives above twenty-one
negatives scores ROC-AUC 1.0000. The MCC and balanced accuracy of 1.000 mean only that
the model made zero errors on 23 items of which 2 were positive. This is consistent
with the cross-validated estimate of 0.9529, and it is all that can be said.

**Correction to an earlier version of this document.** A previous draft reported
balanced accuracy of 0.500 here and presented it as evidence that the model was at
chance. That number was **wrong** — it was produced by a precedence bug in
`summarise_confusion`, where `(a if c1 else 0.0 + b if c2 else 0.0) / 2` parses as
`a if c1 else (0.0 + (b if c2 else 0.0))`, dropping the specificity term and halving
the result. Balanced accuracy is now computed explicitly as
`(sensitivity + specificity) / 2` and has a regression test. The corrected value is
1.0000.

The genuine caution about this test set is the sample size, not the metric
computation: **2 positive examples cannot validate anything.**

### 11.6 Feature-group ablation — the headline result

Every configuration below uses an identically-configured MLP, so the comparison
isolates the feature groups rather than the models. 5-fold stratified CV.

| Configuration | Features | ROC-AUC | Accuracy | F1 | Avg precision |
|---|---|---|---|---|---|
| **Phylum one-hot only** | **5** | **1.0000 ± 0.000** | **1.0000** | **1.0000** | **1.000** |
| Full model | 255 | 0.8421 ± 0.158 | 0.9296 | 0.300 | 0.618 |
| Prevalence only | 2 | 0.4743 ± 0.226 | 0.9123 | 0.000 | 0.237 |
| Abundance + prevalence | 250 | 0.4321 ± 0.191 | 0.8850 | 0.000 | 0.137 |
| Abundance only | 248 | 0.4221 ± 0.196 | 0.8850 | 0.000 | 0.136 |

**Five binary phylum features separate the classes perfectly (ROC-AUC 1.0000, zero
fold-to-fold variance), while 248 CLR-transformed abundance features score 0.4221 —
worse than chance.**

This is the most consequential finding in the repository, and it is not a success
story. It means:

* **The community-composition features carry essentially no signal** for this label.
  0.4221 is below chance, which is what pure noise plus selection-on-CV produces. The
  full model's 0.8421 is a diluted version of the phylum signal, dragged down by 248
  noise columns.
* What the model is actually learning is **lineage membership** — that Proteobacteria
  and Firmicutes genera are over-represented in the curated list while phyla such as
  Patescibacteria are not.
* Therefore the model is **re-deriving my own curation table from the phylum**, not
  measuring any metal-removal mechanism. The circularity warned about in
  [§9.1](#91-track-a--genus-level-literature-supervised-classification) is confirmed
  quantitatively, and the attribution analysis in
  [§11.8](#118-feature-attribution) independently reproduces it.

Why it matters for the thesis: any headline "the MLP predicts metal-removal capability
with AUC 0.95" claim would be **wrong**, because the 0.95 comes from lineage, and
lineage is known deterministically for any named genus. The correct claim is: *given a
curated list of metal-active genera, phylum membership separates those genera from the
rest of a wastewater community, and abundance profiles add nothing.*

The practical consequence is in [§17](#17-improvement-roadmap): the feature
representation must change, or the label must come from paired measurements, before
this task can be predictive.

### 11.7 Learning curve

The final model trained for **22 epochs** to a **final training loss of 0.00783**
(`early_stopping = False`, but the tolerance/patience rule now stops on a training-loss
plateau), with the loss falling from 0.4957 at epoch 1 through 0.0542 at epoch 3 to
0.00783 at epoch 22. The per-epoch history is exported to `learning_curve_history.csv`
and plotted in `results/figures/learning_curve_mlp.png`.

Note that the `validation_score` column is **empty** in this run: the torch backend only
records validation scores when `early_stopping=True`, and the grid search chose `False`
([§11.4](#114-grid-search-result)). The figure therefore shows the training-loss curve
alone. Re-running with `early_stopping` forced on would populate it.

A final loss of 0.0078 is very low for a classification problem — the model has
**memorised the training set**. Combined with the ablation and the attribution results
below, that indicates it has memorised the phylum-to-label mapping rather than learned
a generalisable pattern.

An earlier draft of this document reported 106 epochs and a final loss of 0.00643. That
was measured before the patience-rule fix in [§13.1](#131-the-bug-quantified); the model
was running to a fixed epoch count rather than stopping on convergence.

### 11.8 Feature attribution

Two independent attribution methods were run on the held-out test set: **permutation
importance** (30 repeats, ROC-AUC scoring) and **SHAP**
(`shap.PermutationExplainer`, 50-sample background).

Both methods agree emphatically, and the agreement is the point:

| Rank | Permutation importance | SHAP mean \|value\| |
|---|---|---|
| 1 | `phylum_unclassified_phylum` — 0.01508 | `phylum_unclassified_phylum` — 0.06862 |
| 2 | `phylum_Proteobacteria` — 0.00873 | `phylum_Proteobacteria` — 0.05353 |
| 3+ | every sample column — **0.00000** | sample columns — ≤ 0.01234 |

**Every abundance feature has exactly zero permutation importance**, while the two
phylum indicators dominate both rankings. This is independent corroboration of the
ablation in [§11.6](#116-feature-group-ablation--the-headline-result): the model is
driven by lineage membership and the CLR abundance profile contributes nothing.

Because permutation importance bottoms out at zero for all 248 abundance columns, the
lower part of `permutation_importance.png` is flat — a clear visual signature of a
feature group that the model ignores.

Attribution should therefore be read as a **diagnostic that confirms the circularity**,
not as biology. The sample-level columns that do carry small SHAP values are noise
fitted to individual public samples with 8 positive examples; their identities are not
scientifically interpretable, and the sample accessions change with the acquisition
cache.

An implementation note: the first SHAP attempt failed with
`X has 254 features, but MLPClassifier is expecting 150 features`, because the
explainer was handed the inner estimator while the data had already been reduced by
feature selection. The fix is to explain the **whole pipeline** via a callable, so
scaling and in-fold selection are applied consistently and attributions come back in
the original feature space.

### 11.9 Track B — community metal-activity index

Public corpus (248 samples): mean index 0.915, median 1.000, range 0.000–1.000.
The high median is a direct consequence of the curated list being dominated by
common wastewater phyla, and it confirms the index is not very discriminating across
wastewater communities.

In-house samples:

| Sample | Documented-active fraction (raw) | Annotated share of community | Index | Annotated genera |
|---|---|---|---|---|
| OS | 0.7824 | 0.8048 | **0.972** | 7 |
| SM | 0.9999 | 0.9999 | **1.000** | 1 |

Interpretation: 78.2% of OS's community mass is documented metal-active (7 genera
including *Raoultella*, *Pseudomonas*, *Enterobacter*, *Klebsiella*, *Delftia*,
*Chryseobacterium*), rising to 97.2% once normalised over annotated taxa; 19.5% of the
community is unclassified and excluded. SM is 100% *Bacillus*, a documented
metal-active genus, so its index is 1.000 by construction.

Both biofilms score high. The index therefore **does not discriminate between them**,
which limits its usefulness for external validation — a point developed below.

### 11.10 Track C — not estimable

After intersecting the quantitative efficiency table with the recovered taxonomy, only
*Pseudomonas* (Cu, 80%) survives, below the 4-genus minimum. The pipeline emits
`Track C produced no trainable metal targets` and records the limitation. No
regression was fabricated.

---

## 12. External validation on the in-house biofilms

### 12.1 What was done

The trained classifier supplies `P(genus is metal-active)` for each of the 114 training
genera. Those probabilities are then aggregated over the genera actually present in
each in-house sample, weighted by relative abundance, producing a sample-level score in
[0, 1] with a bootstrap confidence interval (2,000 resamples) and a shuffled-label null
(p-value). In parallel, the literature index and a **per-metal literature expectation**
are computed:

```
expected_removal(s, m) = Σ_g  (efficiency(g,m)/100) · rel_abund(g, s)
```

summed over genera with a documented efficiency for metal `m`. This is deliberately
conservative and returns a **lower bound**: a genus occupying 5% of the community can
contribute at most 5 percentage points, and no synergistic community effect is credited.

### 12.2 Model-based scores

| Sample | Status | Score | 95% CI | Coverage | Documented-active fraction | Null p |
|---|---|---|---|---|---|---|
| OS | scored | 0.995 | 0.995–0.995 | 39.9% of community mass | 1.000 | 1.000 |
| SM | **not scorable** | — | — | 0% | 1.000 (literature index) | — |

**OS:** exactly one genus was scorable — *Pseudomonas*, at `P = 0.995`, covering 39.9%
of the community. Its two *Pseudomonas* species carry curated label 1, so a
correctly-trained model should return a high probability, and it does. But a
single-genus score has a **degenerate confidence interval** (zero width) and the
shuffled null is uninformative, because with one scorable genus shuffling is a no-op.
The pipeline detects and reports both conditions explicitly.

**SM could not be scored at all.** SM is 99.99% *Bacillus stratosphericus*, and
*Bacillus* does **not** appear among the 114 genera retained from the public training
corpus. There is no weight to apply, so no score exists. The pipeline reports this as
`NOT SCORABLE` and continues, rather than crashing or silently substituting zero. The
literature index for SM is still computable (1.000) because it reads the curated table
directly and does not depend on the public taxonomy.

This is a genuine and instructive failure mode: a model trained on six unrelated public
wastewater studies does not necessarily know the taxa in your own biofilm.

### 12.3 Predicted versus observed removal, per metal

| Sample | Metal | Literature expectation (%) | Observed (%) | Discrepancy (pp) | Contributors |
|---|---|---|---|---|---|
| SM | **As** | **94.94** | **96.79** | **−1.85** | *Bacillus* (99.99% × 95.0%) |
| OS | Pb | 37.02 | 60.00 | −22.98 | *Raoultella* (41.60% × 89.0%) |
| OS | Cr | 26.37 | 0.00 | +26.37 | *Raoultella* (41.60% × 63.4%) |
| OS | Ni | 23.13 | −83.33 | +106.46 | *Raoultella* (41.60% × 55.6%) |

Across the 4 metal/sample pairs with both a quantitative literature value and a
measurement: **Pearson r = 0.776, Spearman ρ = 1.000**.

### 12.4 How to read this honestly

**The arsenic agreement is the one genuinely encouraging result.** SM is a *Bacillus*
monoculture; *Bacillus* is documented at 99.9% arsenate and 90% arsenite removal; the
abundance-weighted expectation is 94.94%; the measured removal is 96.79%. The two agree
to within 1.9 percentage points, and this pair is not fitted — the expectation comes
straight from the curated table and the composition. It is a real agreement.

It is also **one metal in one sample**. Spearman ρ = 1.000 looks impressive but is
computed on four points, of which two are *Raoultella* predictions and none are
independent of the others — they share the same *Raoultella* abundance term, so the
four pairs are not four independent tests. n = 2 biofilms. **No generalisability claim
can rest on this.**

**The failures are more informative than the success.**

* **Ni: predicted 23.1% removal, observed −83.3% release.** The biofilm actively
  released nickel. The index cannot anticipate release, because the literature label
  set contains only *capability*, never *failure*. A genus that is documented to remove
  a metal can still mobilise it under different redox or pH conditions.
* **Cr: predicted 26.4% removal, observed 0% (the metal appeared from below
  detection).** Same structural problem.
* **Mn: −302.3% (released), no literature value at all.** Manganese was never in the
  curated set for these genera, so the pipeline silently had nothing to say about a
  metal that moved by a factor of four in the wrong direction.
* **Pb in OS: predicted 37.0%, observed 60.0%.** An under-prediction driven entirely by
  abundance weighting — *Raoultella* is only 41.6% of OS, so even an 89% efficient
  genus can contribute at most 37 points. The observed 60% is higher than any single
  genus's abundance-weighted contribution, which means **community-level effects that
  the index structurally cannot represent** (synergy, co-metabolism, or removal by the
  18.95% unclassified fraction).

**The systematic pattern:** the model over-predicts removal for metals that were
released, and under-predicts for metals removed better than the dominant genus alone
explains. Both errors trace to the same root cause — the label set describes *capability*,
while the measurements record *outcome* in a specific chemical environment.

### 12.5 Observed removal profile (context)

Of the nine metals measured: **six were removed** (As 96.8%, Fe 100%, Zn 100%, Hg 79.7%,
Pb 60.0%, Ca 37.2%), **two were released** (Mn −302.3%, Ni −83.3%), and **one was
unchanged** (Cr, appearing from below detection).

The two biofilms are genuinely effective at removing Hg, As, Fe and Zn. They also
mobilise Mn and Ni. Any claim that these biofilms are "metal removing" must be
qualified by metal identity, and a capability-based index cannot express that
qualification.

---

## 13. The GPU backend, measured

The pipeline runs its MLP on the **NVIDIA GeForce GTX 1050 Ti Max-Q (4 GiB, sm_61,
6 SMs)** by default. `model.backend: torch` with `device: cuda` in `config.yaml`.

An earlier version of this document claimed the GPU was 3.66× slower and should not be
used. **That claim was partly wrong, and the error was mine.** The real cause of the
slowness was a bug in `TorchMLPClassifier`: the tolerance/patience stopping rule was
gated behind `early_stopping`, so with `early_stopping=False` — which the grid search
selects — every fit ran the full `max_iter = 1000` epochs. The GPU was not slow; it was
doing 33× more work than it should have.

### 13.1 The bug, quantified

Per-fit cost on the real Track A matrix before and after the fix:

| Configuration | Epochs run | ms/fit | Grid search (300 fits) |
|---|---|---|---|
| **Before fix** — torch CUDA | 1000 | 7,031 | **35.2 min** (never completed in practice) |
| **After fix** — torch CUDA | 30 | 220 | 1.1 min |
| After fix — torch CPU | 30 | 123 | 0.6 min |
| sklearn CPU (reference) | — | 148 | 0.4 min |

**32× faster** after the fix. scikit-learn's `MLPClassifier` applies the same
`tol`/`n_iter_no_change` criterion whether or not `early_stopping` is on — monitoring a
held-out split when it is, the training loss when it is not. The torch backend now does
the same.

### 13.2 Corrected benchmark

Best of 5 fits on the actual Track A matrix (91 × 255 after the split):

| Backend | ms/fit | Relative to fastest |
|---|---|---|
| torch CPU | 123.3 | 1.00× |
| sklearn CPU | ~148 | 1.20× |
| **torch CUDA** | **220.1** | **1.78×** |

Two further findings from the same benchmark:

* **`deterministic=True` costs nothing measurable** (0.98× on CPU, 1.03× on GPU), so
  full reproducibility is retained. An earlier hypothesis that deterministic kernels
  were the bottleneck was simply wrong.
* **`seed_everything` costs 0.3 ms per call** — negligible against a ~200 ms fit.
* The per-epoch cost is **flat at ~7.4 ms regardless of `max_iter`** (10, 100 and 1000
  epochs all cost ~7.4 ms/epoch). This is the signature of a latency-bound workload:
  the arithmetic is trivial and kernel-launch overhead dominates. It is exactly why a
  6-SM Max-Q part cannot beat a CPU here, and why it *will* win once the matrices are
  large enough to amortise launches.

### 13.3 Full-run resource profile

Measured over a complete end-to-end run (202 samples at 1.5 s intervals):

| Metric | Value |
|---|---|
| Wall clock | 314.5 s (5.2 min) |
| CPU logical cores | 12 |
| CPU utilisation, mean / peak | **21.8% / 70.4%** |
| RAM total | 23.7 GiB |
| RAM used, mean / peak | 14.46 / **15.00 GiB** |
| GPU memory, mean / peak | 129.9 / **155.0 MiB of 4096 MiB (3.8%)** |
| GPU utilisation, mean / peak | 22.2% / 44.0% |

**There is no OOM risk on either side.** Peak RAM is 61% of system memory, and the GPU
uses under 4% of its 4 GiB. The model is ~18k parameters; a batch of 32 activations is
a few kilobytes. Even the accumulated results tables dominate memory, not the tensors.

For comparison, the same run configuration with `n_jobs: -1` on the sklearn backend
measured **CPU mean 52.7%, peak 100%** — saturating every core. Setting
`model.grid.n_jobs: 1` (now the default in `config.yaml`) roughly halves mean CPU load,
which is the actual lever on CPU pressure, not the GPU choice.

### 13.4 What still runs on the CPU even with `backend: torch`

Worth stating plainly, because it bounds how much the GPU can offload:

* **The three baselines are CPU-only.** Random Forest, XGBoost and Logistic Regression
  come from CPU libraries. They still run on the CPU.
* **Cross-validation orchestration**, data loading, preprocessing, metrics,
  permutation importance bookkeeping and plotting are all CPU/numpy work.
* Only the MLP's forward and backward passes move to the GPU.

So the CPU peaks in the profile above are mostly the baselines and the scikit-learn
side of the pipeline, not the neural network.

### 13.5 Switching backends

```bash
python run_pipeline.py                              # torch on CUDA (default)
python run_pipeline.py --backend torch --device cpu # torch on CPU
python run_pipeline.py --backend sklearn            # scikit-learn MLP, fastest wall-clock
```

`TorchMLPClassifier` is a drop-in scikit-learn estimator
(`get_params`/`set_params`/`fit`/`predict`/`predict_proba`) that also exposes
`loss_curve_` and `validation_scores_`, so it slots into the existing `Pipeline`,
`GridSearchCV`, cross-validation, learning-curve and attribution code unchanged.

Reproducibility: `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms(warn_only=True)` are
all set from the global seed, and the same seed gives bit-identical probabilities
([§15](#15-testing)).

One deliberate guard: with the torch backend on CUDA, `GridSearchCV` forces
`n_jobs = 1`. Several worker processes contending for one 4 GiB GPU would be slower and
riskier, and each model is far too small to saturate the device. This is logged when it
triggers.

### 13.6 The honest summary

* The GPU **works, is fully reproducible, and leaves the CPU mostly free** — use it as
  the default, as this configuration now does.
* For **this** model size it is ~1.8× slower per fit than torch on CPU and ~2.4× slower
  end-to-end than the sklearn CPU path. If you want the fastest single run, use
  `--backend sklearn`.
* The GPU will win decisively once the feature matrices grow — ASV-level features
  (5,000–50,000 columns), PICRUSt2 functional gene abundances, or many more studies.
  That is the scaling path this backend exists for ([§17](#17-improvement-roadmap)).

---

## 14. Reproducibility

### 14.1 Fixed seeds

`project.seed: 42` in `config.yaml` drives:

* `random.seed`, `numpy.random.seed` and `PYTHONHASHSEED` (`set_global_seed`)
* `random_state` on the MLP, Random Forest, XGBoost and Logistic Regression
* `StratifiedKFold(shuffle=True, random_state=seed)`
* `train_test_split(random_state=seed)`
* `permutation_importance(random_state=seed)`
* the SHAP background sample and explainer seed
* bootstrap and permutation nulls in external validation
* the simulated surrogate's `default_rng`

`PYTHONHASHSEED` must be set before interpreter start to fully take effect; the
pipeline sets it for child processes and documents the residual caveat.

### 14.2 Provenance recorded with every run

Written to `results/tables/run_metadata.json`:

* configuration fingerprint (SHA-256 of the whole config, 16 hex chars) — `dda86c7c5aa3c021`
* acquisition provenance: strategy, status, per-strategy attempts and timings, HTTP
  request count, sample and genus counts, and an explicit `simulated` flag
* SHA-256 prefix of the cached MGnify count table — `56b5ac4994c58fd6`
* SHA-256 prefix of the curated label table
* library versions (Python, scikit-learn, NumPy, pandas) and platform string
* dataset shape and label balance
* best hyper-parameters
* held-out metrics
* cross-validated model comparison
* external-validation scores

### 14.3 Caching

Every per-analysis OTU table and taxonomy summary is cached as gzipped CSV under
`data/raw/mgnify_cache/`. A second run reads the cache and completes in about 70
seconds instead of the ~8 minutes a cold harvest takes. `--force-acquisition` bypasses
the cache. The cache makes the acquisition step auditable: you can inspect exactly
which analyses contributed.

### 14.4 Determinism caveats

* **Cross-platform floating point.** BLAS implementations differ between machines, so
  bit-identical results are not guaranteed across operating systems or CPU
  architectures. Same-machine re-runs are reproducible.
* **Upstream data drift.** MGnify can reprocess analyses. Cached tables make this
  explicit — a cache change is visible in the hash.
* **Grid search parallelism.** `GridSearchCV(n_jobs=-1)` parallelises across folds but
  each fit is itself deterministic; results do not depend on scheduling.

---

## 15. Testing

41 unit tests in `tests/test_pipeline.py`, all passing. They target the places where a
silent error would corrupt the science rather than simply crash.

| Area | Tests |
|---|---|
| **CLR transform** | rows sum to zero; matches the closed form; rejects non-positive input |
| **Composition** | closure to 1; rejects zero-total samples; multiplicative replacement preserves closure and positivity; method dispatch; unknown method raises |
| **Taxon filtering** | rare genera dropped; retention flags correct |
| **Removal arithmetic** | basic case; BDL disappearance = 100%; BDL appearance = 0%; negative for release; clamping; **the exact published in-house values** (Hg 79.667, As 96.795, Pb 60.0, Ca 37.165, Fe 100, Zn 100, Cr 0) |
| **In-house data** | compositions close to 100%; *Pseudomonas* species merge to 32.12% |
| **Labelling** | all seven user-specified genera are label 1; *Microvirgula* is 0; no duplicates; efficiency values match the brief |
| **Track A construction** | unannotated genera receive label 0; feature groups exactly partition the columns with no overlap |
| **Leakage control** | feature selection is a pipeline step, not pre-applied |
| **External validation** | score is the abundance-weighted mean; **unscorable samples are reported rather than raising**; degenerate single-genus CIs flagged; index bounded in [0,1]; expectation scales with abundance; unmined metal yields 0 |
| **Acquisition parsing** | placeholders, empties and zero counts dropped; missing phylum backfilled |
| **Determinism** | simulation is bit-identical across calls; the simulated flag and warning are present; config fingerprint stable |
| **Config correctness** | `DotDict` mutation persists |
| **torch backend** | **recognised as a classifier by `is_classifier`/`__sklearn_tags__`**; survives `cross_validate` with `roc_auc`; probabilities sum to 1; deterministic across seeds |
| **Confusion summary** | **balanced accuracy = mean of sensitivity and specificity**; degenerate folds yield NaN rather than a wrong number |

Four of these are regression tests for real bugs found during development, and each is
worth naming because each had silently produced wrong numbers:

1. **`DotDict` copy-on-read.** An early `__getattr__` re-wrapped nested dicts on every
   access, so in-code overrides such as `cfg.acquisition.mgnify.max_studies = 4`
   silently mutated a temporary copy. The pipeline appeared to honour its configuration
   while ignoring parts of it.
2. **torch mixin order.** `class TorchMLPClassifier(BaseEstimator, ClassifierMixin)`
   leaves `estimator_type` unset, so sklearn does not recognise the estimator as a
   classifier and hands the raw `(n, 2)` probability matrix to scoring functions, which
   then raise `y should be a 1d array, got an array of shape (19, 2)`. `ClassifierMixin`
   must come first.
3. **Patience rule gated behind `early_stopping`.** With `early_stopping=False` every
   fit ran the full 1000 epochs, making the grid search 32× slower than necessary and
   the learning curve meaningless ([§13.1](#131-the-bug-quantified)).
4. **Ternary/`+` precedence in `summarise_confusion`.** Balanced accuracy was reported
   as exactly half its true value.

```bash
python -m pytest tests -q
# 41 passed in ~8s
```

---

## 16. Limitations

Ordered roughly by how much each one threatens a conclusion drawn from this work.

### 16.1 Label circularity — the fundamental problem

The Track A label is a **curated statement about a genus**, and phylum membership is
known deterministically for any genus. A model can therefore recover the label by
reading the lineage rather than by measuring ecology. The ablation in
[§11.6](#116-feature-group-ablation--the-headline-result) confirms this is exactly what
happens: 5 phylum features reach ROC-AUC 0.900 while 248 abundance features reach
0.443.

**Consequence:** cross-validated performance is not evidence that community composition
predicts metal removal. Any thesis claim to that effect would be unsupported by these
results.

### 16.2 Label noise from literature mining

* **Label 0 means "undocumented", not "incapable."** A genuinely metal-active genus that
  has never been studied receives label 0. With 104 of 114 genera labelled 0, false
  negatives in the negative class are likely to dominate the label noise.
* **Generic-ability bias.** Well-studied, easily cultured, clinically or industrially
  relevant genera are over-represented in the literature. *Bacillus*, *Pseudomonas* and
  *Escherichia* are studied far more than uncultured wastewater lineages such as
  *Candidatus* taxa. The label set therefore encodes **research effort**, not just
  capability.
* **40 of 44 curated entries are marked `needs_verification`.** Only the seven
  quantitative values supplied in the brief are verified. The remainder are recorded
  from general genus-level knowledge and must be checked against primary literature
  before submission.
* **Capability ≠ outcome.** A genus documented to remove a metal at pH 7 in a pure
  culture may mobilise it in a mixed biofilm at different redox. The Ni and Cr
  mismatches in [§12.3](#123-predicted-versus-observed-removal-per-metal) are direct
  evidence of this gap.

### 16.3 Statistical power — the binding constraint

| Quantity | Value |
|---|---|
| Positive genera available | 10 |
| Positive genera in training | 8 |
| Positive genera in test | 2 |
| Fold-to-fold ROC-AUC SD | 0.10–0.15 |
| Fraction of curated genera recovered | 10 of 44 |

With two positive test examples, **a perfect ROC-AUC of 1.0000 carries no information**:
any model that ranks two positives above twenty-one negatives achieves it. The
cross-validated estimate of 0.9529 ± 0.102, resting on 8 positives per training split,
is the number to quote — and its ± 0.102 spread means the plausible range spans
0.85–1.00.

With eight positives in training, fold-to-fold variation exceeds most between-model
differences, so the ranking of Random Forest, XGBoost and the MLP **is not
statistically meaningful**. Confidence intervals for the CV metrics would span much of
the unit interval.

There is a second reason not to lean on the test metrics: because the feature matrix
contains one column per public sample, the fitted model can memorise individual
samples, and the final training loss of 0.0078 shows that it does. Perfect test
performance is therefore partly memorisation, not only generalisation.

The 0.1% mean-abundance filter contributes: it retains 114 of 1,643 genera (6.9%), and
several curated genera that *are* present in the raw data fall below it. The
`mean_when_present` option in [§8.5](#85-step-5--taxon-filtering) exists precisely to
test that sensitivity, and running it is the first item in the improvement roadmap.

### 16.4 Batch effects across public studies

The 248 samples come from **six independent studies** on two sequencing
platforms (Illumina and Roche 454) and two MGnify pipeline versions (4.1 and 5.0).
Systematic differences between studies — DNA extraction kit, primer choice, amplicon
region, read length, bioinformatic pipeline, reactor type, influent chemistry — are
confounded with any biological signal.

**This has not been corrected for.** No batch correction (ComBat, RUV, or even
per-study standardisation) has been applied, and study identity is not a model feature.
Study-level effects are therefore free to masquerade as biology. The surrogate
generator deliberately injects a study-level effect to make this failure mode visible
during pipeline testing.

An attempt to quantify this would be: hold out entire studies (leave-one-study-out)
rather than random genera. That is on the roadmap and should be regarded as essential
before any performance claim is believed.

### 16.5 Compositional data caveats

* **CLR is not a silver bullet.** It resolves the simplex constraint and makes
  log-ratios well-behaved, but it does **not** remove the dependency between features.
  CLR features still sum to zero per sample, so feature vectors remain exactly
  collinear — a genuine property of the transform, and one that makes linear-model
  coefficients and per-feature importances harder to interpret.
* **Zero replacement is a model.** With 86–100% sparsity at genus level, the choice of
  `delta` materially affects every downstream log-ratio. Multiplicative replacement is
  principled but not assumption-free; a sequencing zero may mean "absent" or merely
  "not sampled".
* **The 0.1% filter is applied to compositions, then CLR is applied to the retained
  subset.** Closure before filtering means that removing taxa silently rescales the
  remaining ones. This is standard practice but worth stating.
* **Batch effects amplify the zero problem.** A genus below detection in one study and
  abundant in another produces a large positive CLR value and a large negative one —
  a pattern that may be technical rather than biological.

### 16.6 External validation cannot validate

* **n = 2.** Two biofilms, from one lab, one wastewater source. They demonstrate that
  the plumbing works end-to-end and that outputs are physically plausible. They cannot
  estimate generalisability, and no confidence interval on a population of two is
  meaningful.
* **One sample could not be scored at all.** SM's *Bacillus* is absent from the public
  training taxonomy, so the classifier produced no prediction for it. Only OS was
  scored, and from a single genus with a degenerate confidence interval.
* **The observed vector is shared between SM and OS.** The brief supplied one
  before/after table for two biofilms. If they were separate batches,
  `data/literature/inhouse_metal_concentrations.csv` must be edited to carry per-sample
  effluents. Until that is confirmed, every per-sample comparison in
  [§12.3](#123-predicted-versus-observed-removal-per-metal) is conditional on an
  assumption.
* **Four comparable pairs, not four independent tests.** They share the *Raoultella*
  abundance term, so the effective sample size is closer to two than four. Pearson
  r = 0.776 and Spearman ρ = 1.000 must not be presented as a correlation coefficient
  with any inferential weight.

### 16.7 Detection-limit conventions

Removal percentages depend on two conventions imposed by the brief: a metal present
before treatment and below detection afterwards is scored 100% removed; a metal below
detection before and appearing afterwards is scored 0%. These are policy choices, not
measurements. Converting "below detection" to an exact 100% or 0% discards the
information in the detection limit itself. **Fe and Zn are both scored as exactly 100%
on this basis** — they were not measured as zero, they were simply not detected. A
censored-data treatment (Tobit or maximum-likelihood on interval-censored values) would
be more defensible.

The two "released" metals (Mn −302.3%, Ni −83.3%) are computed with the raw formula and
retained as negative numbers rather than clamped to zero, because the release is real
chemistry and hiding it would be dishonest.

### 16.8 Taxonomic resolution

* Only **38.9% of reads** are assignable at genus level; the rest are resolved only to
  family or order and are discarded. This is a property of MGnify's MAPseq assignments
  on short reads.
* Genus-level aggregation **loses species-level differences that matter**. *Pseudomonas
  veronii* and *Pseudomonas marginalis* are collapsed into one feature even though
  their metal behaviour may differ. Species-level labels exist in the efficiency table
  but cannot be used at genus resolution.
* **Genus names must match SILVA.** 26 of the 44 curated genera never appeared in the
  recovered taxonomy. Some of those absences are real ecology; others may be
  nomenclature mismatches between the curated table and SILVA 138.1. This has not been
  systematically audited and is a plausible source of false negatives.

### 16.9 Architectural scope (by design)

CNNs, LSTMs/GRUs and Transformers were **not used**, deliberately, as argued in
[§10.2](#102-why-not-a-cnn-lstm-or-transformer). The data are static community
snapshots with no sequence or spatial structure. If future work introduces longitudinal
sampling (the same reactor tracked over time), a sequence model becomes defensible;
with single time-points it would not be.

### 16.10 Track B index limitations

The index measures "fraction of the community belonging to documented metal-active
genera". It is bounded in [0,1], cannot express per-metal specificity or any release
behaviour, ignores synergy among taxa, and is undefined for communities dominated by
unclassified reads. Both in-house biofilms score near 1.0, so it does not discriminate
between them — its main value here is as a transparent, auditable quantity that does
not depend on a classifier trained on the same curated list.

---

## 17. Improvement roadmap

Ordered by expected value per unit of effort. Items 1–3 would change the conclusions;
items 4–8 would strengthen them.

### Tier 1 — would change the conclusions

**1. Run the abundance-metric sensitivity analysis.**
Set `preprocessing.abundance_metric: mean_when_present` and re-run. This is a one-line
config change that trades the strict reading of the brief for the conventional
ecological one, and it should retain substantially more curated genera. **Report both
sets of results.** If the positive count rises from 10 to, say, 25, the statistical-power
problem in [§16.3](#163-statistical-power--the-binding-constraint) materially improves.
This is the cheapest high-value experiment available.

**2. Replace random-split CV with leave-one-study-out.**
The current protocol splits genera randomly across six confounded studies, so study
identity can leak into the test set. Holding out entire studies answers the actual
question — *does this generalise to an unseen study?* — and it is the only way to
separate batch effects from biology. Expect performance to drop; that drop is the
finding.

**3. Add explicit batch correction and report the delta.**
Apply ComBat or per-study standardisation to the CLR matrix, then re-run. Compare
against the uncorrected baseline. This directly quantifies
[§16.4](#164-batch-effects-across-public-studies).

### Tier 2 — would strengthen the work

**4. Test a summary-statistic feature representation.**
The ablation already hints that the 248-dimensional per-sample abundance profile is
the wrong representation — it is wide, sparse and noisy. Replace it with per-genus
summary statistics (mean CLR, standard deviation, prevalence, detection range). The
`prevalence_only` configuration (2 features, ROC-AUC 0.536) shows that summary
statistics alone are not sufficient, but a richer summary set is untested and would
cut dimensionality by an order of magnitude. Track A's cached CLR matrix makes this
cheap to try.

**5. Obtain paired community-metal measurements — the real fix.**
Every limitation above traces back to the fact that public 16S data carry no metal
measurements, forcing the label to be *about the genus* and therefore circular. The
pipeline is already structured for the right dataset: swap Track A's label vector for
an experimentally measured removal value per sample and the same code fits a sample-level
regressor. **This is the single change that would turn the project from a methods
demonstration into a predictive model.** Even 50–100 paired samples would transform it.

**6. Expand the curated label set and audit it.**
44 genera is small. A systematic mining pass over primary literature — with an explicit
inclusion protocol and a recorded evidence level per entry — would grow the positive
class and reduce the research-effort bias noted in
[§16.2](#162-label-noise-from-literature-mining). Growing the *quantitative*
efficiency table would also unlock Track C.

**7. Add PICRUSt2 functional features.**
Predicted functional gene abundances are a genuinely different feature type: they
describe metabolic capability rather than taxonomy, so they are far less likely to
reproduce the lineage abacus. This is also the feature set most likely to justify the
GPU backend ([§13](#13-the-gpu-backend-measured)).

**8. Move to ASV-level analysis.**
Genus aggregation discards resolution and only 38.9% of reads assign at genus level.
ASV-level features would multiply dimensionality by 10–50×, keep species-level
distinctions, and again justify GPU use.

### Tier 3 — methodological polish

**9. Treat the detection limits properly.** Replace the 100%/0% conventions for Fe, Zn
and Cr with interval-censored maximum-likelihood estimation, or report sensitivity to
the convention.

**10. Per-metal labels rather than a binary capability label.** The efficiency table
already shows that capability is metal-specific (89% Pb vs 55.6% Ni for *Raoultella*).
A multi-label or per-metal target would stop the model from predicting "metal-active in
general" and let it predict which metal.

**11. Model release as well as removal.** Ni and Mn were *released* by these biofilms,
and nothing in the current label set can represent that. An explicit
removal/release/neutral outcome, or a signed efficiency target, would capture it.

**12. Species-level modelling where labels permit.** *P. veronii* and *P. marginalis*
are merged today; species-resolved labels would let them be treated separately.

**13. Report confidence intervals on all CV metrics.** Bootstrapped CIs over folds would
make the uncertainty in [§11.3](#113-cross-validated-model-comparison) explicit rather
than implicit in the standard deviations.

**14. Nested CV for unbiased performance estimation.** `GridSearchCV` inside the CV loop
is already correct, but the reported best score is optimistically biased because the
same folds select and evaluate the model. Nested CV would remove that bias.

**15. External validation with more biofilms.** n = 2 → n ≥ 10 would let the
literature-expectation model be tested with actual statistics rather than a Spearman ρ
on four correlated points.

### Explicitly not recommended

* **Do not add CNN/LSTM/Transformer architectures.** No sequence or spatial structure
  exists in these data; added capacity would overfit 91 training rows with 8 positives.
* **Do not report the held-out ROC-AUC of 1.0000** as a result. It rests on two positive
  examples.
* **Do not expect the GPU to be faster at this dataset size.** It is ~1.8× slower per
  fit and ~2.4× slower end-to-end than sklearn on CPU ([§13](#13-the-gpu-backend-measured)).
  Do use it to keep the CPU free, or when the feature matrices grow.

---

## 18. Artifact manifest

### Figures (`results/figures/`)

| File | Content |
|---|---|
| `learning_curve_mlp.png` | Training loss per epoch (22 epochs) |
| `roc_curve.png` | ROC curve, held-out test set |
| `precision_recall_curve.png` | PR curve against the no-skill positive rate |
| `confusion_matrix.png` | 2×2 heatmap at the tuned threshold (0.176) |
| `permutation_importance.png` | Top-25 permutation importances with error bars |
| `shap_importance.png` | Top-25 mean \|SHAP\| values |

### Tables (`results/tables/`)

**Datasets and features**

| File | Content |
|---|---|
| `public_genus_counts.csv` | 1,643 genera × 250 samples (1.6 MB) |
| `public_sample_metadata.csv` | Per-sample provenance and sequencing metadata |
| `track_a_labelled_dataset.csv` | The 114 × 255 supervised matrix with labels |
| `track_a_genera.csv` | Genus, label, feature count |
| `feature_names.csv` | All 255 features with their group (abundance/phylum/prevalence) |
| `track_b_metal_activity_index.csv` | Per-sample index across 248 public samples |
| `track_b_inhouse_metal_activity_index.csv` | Index for SM and OS |
| `inhouse_clr_projection.csv` | OS projected into the training feature space |

**Models and results**

| File | Content |
|---|---|
| `run_metadata.json` | Full provenance, versions, best params, metrics |
| `model_comparison_cv.csv` | Cross-validated metrics for all four models |
| `gridsearch_cv_results.csv` | All 300 grid fits with scores (24 KB) |
| `feature_group_ablation.csv` | The headline ablation table |
| `learning_curve_history.csv` | Per-epoch loss and validation score |
| `confusion_matrix.csv` | Test-set confusion matrix |
| `permutation_importance.csv` | Permutation importances |
| `shap_importance.csv` | SHAP mean absolute values |
| `trainability_report.json` | Written only when the dataset cannot be CV'd |

**Validation**

| File | Content |
|---|---|
| `external_validation_scores.csv` | Per-sample model scores, CIs, coverage, caveats |
| `external_validation_comparison.csv` | Expected vs observed, per metal and sample |
| `external_validation_per_genus_OS.csv` | The single scorable genus for OS |
| `observed_removal_details.csv` | Removal percentages with outcome classification |
| `observed_removal_efficiency.csv` | Observed removal only |

**Literature**

| File | Content |
|---|---|
| `literature_labels_used.csv` | The curated label table as consumed |
| `inhouse_observed_removal.csv` | In-house measurements as loaded |

### Models (`results/models/`)

| File | Content |
|---|---|
| `track_a_mlp_classifier.pkl` | The fitted Pipeline (scaler → selector → MLP) |
| `track_a_standard_scaler.pkl` | The fitted scaler, for projecting new samples |

### Reports

| File | Content |
|---|---|
| `external_validation_report.txt` | Plain-text validation summary with interpretation limits |
| `logs/pipeline.log` | Full timestamped run log (225 lines) |

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **16S rRNA amplicon** | Sequenced marker gene used to profile bacterial communities |
| **ASV** | Amplicon Sequence Variant; single-nucleotide-resolved taxon unit |
| **CLR** | Centered log-ratio transform; maps compositional data from the simplex to real space |
| **Compositional data** | Data constrained to sum to a constant (here, 1), so components are not independent |
| **Genus** | Taxonomic rank between family and species; the unit of analysis here |
| **Label circularity** | When the target is derived from the same information as the features, so the model reproduces the derivation rather than learning the phenomenon |
| **MAPseq** | The classifier MGnify uses to assign SSU reads to taxonomy |
| **MGnify** | EBI's metagenomics analysis portal and REST API |
| **MLP** | Multilayer perceptron; a feed-forward neural network |
| **OTU** | Operational Taxonomic Unit; a sequence-similarity cluster |
| **Permutation importance** | Drop in score when a feature's values are shuffled |
| **Prevalence** | Fraction of samples in which a taxon is detected |
| **ROC-AUC** | Area under the receiver-operating-characteristic curve |
| **PR-AUC / average precision** | Area under the precision-recall curve; preferred under class imbalance |
| **SHAP** | Shapley additive explanations; game-theoretic feature attribution |
| **SILVA 138.1** | Reference rRNA gene database, release 138.1 |
| **Sparsity** | Fraction of zero entries in the taxon × sample table |
| **Stratified k-fold** | Cross-validation preserving class proportions in each fold |
| **Study batch effect** | Systematic technical difference between datasets that is confounded with biology |
| **Track A / B / C** | This pipeline's three dataset constructions (§9) |

---

## Appendix A — Full configuration reference

Every parameter lives in `config.yaml`. The ones most likely to be tuned:

| Key | Default | Effect |
|---|---|---|
| `project.seed` | 42 | Global reproducibility seed |
| `acquisition.strategy_order` | `[mgnify_r, mgnify_rest, sra_dada2, simulate]` | Acquisition preference |
| `acquisition.mgnify.min_samples` | 200 | Reject a strategy below this count |
| `acquisition.mgnify.max_analyses` | 260 | Caps harvest size and runtime |
| `acquisition.mgnify.taxonomy_source` | `otu_table` | `otu_table` (4× more genera) or `taxonomy_ssu` |
| `preprocessing.min_mean_relative_abundance` | 0.001 | The 0.1% taxon filter |
| `preprocessing.abundance_metric` | `mean_overall` | `mean_overall` or `mean_when_present` |
| `preprocessing.min_prevalence` | 0.02 | Minimum detection prevalence |
| `preprocessing.min_library_size` | 50 | Drop shallow samples |
| `preprocessing.clr_zero_handling` | `multiplicative_replacement` | Zero replacement method |
| `preprocessing.feature_selection.mode` | `auto` | `auto`, `always`, `never` |
| `preprocessing.feature_selection.k` | 40 | `SelectKBest` feature count |
| `dataset.track_a.include_phylum_features` | `true` | Enabled deliberately, to expose lineage leakage in the ablation |
| `model.backend` | `sklearn` | `sklearn` or `torch` |
| `model.cv.folds` | 5 | Capped by the minority class count |
| `model.grid.enabled` | `true` | Toggle `GridSearchCV` |
| `validation.n_bootstrap` | 2000 | Bootstrap resamples for external validation CIs |

## Appendix B — Reproducing the exact reported results

```bash
cd heavy_metal_removal_ml
python -m pip install -r requirements.txt

# 1. Verify the test suite
python -m pytest tests -q                      # expect: 41 passed

# 2. Reproduce the reported run from the cache (~5 min on GPU, ~90 s with --backend sklearn)
python run_pipeline.py

# 3. Re-run including a cold acquisition (~8 min)
python run_pipeline.py --force-acquisition

# 4. Confirm the offline path still works
python run_pipeline.py --strategy simulate --quick
```

Then confirm in `results/tables/run_metadata.json`:

* `"simulated": false`
* `"mlp_backend": "torch"`, `"mlp_device": "cuda (NVIDIA GeForce GTX 1050 Ti ...)"`
* `"track_a_shape": [114, 255]`
* `"track_a_label_balance": {"positive": 10, "negative": 104}`
* `"cv_folds_used": 5`, `"minority_class_count": 8`, `"trainable": true`
* `"best_params"`: `alpha 0.01`, `early_stopping false`, `hidden_layer_sizes [128, 64]`,
  `learning_rate_init 0.01`
* `"holdout_metrics"`: `roc_auc 1.0`, `mcc 1.0`, `threshold ≈ 0.1756`
* `"confusion_summary"`: TN 21, FP 0, FN 0, TP 2, `balanced_accuracy 1.0`
* `"feature_group_ablation"`: `phylum_only` ROC-AUC 1.0, `abundance_only` ROC-AUC 0.4221

Note that `hidden_layer_sizes` may differ slightly between runs if the grid search ties
on ROC-AUC and breaks the tie differently; 0.9529 was selected here.

If `track_a_shape` differs, the upstream MGnify data has been reprocessed — compare
`cache_sha256_16` against `56b5ac4994c58fd6`.

## Appendix C — Key formulas

**Removal efficiency** (with detection-limit conventions):

```
removal(%) = (before − after) / before × 100

   before detected, after below detection  → 100%   (metal left solution)
   before below detection, after detected  →   0%   (metal appeared)
   otherwise                               → the formula above, negatives retained
```

**Centered log-ratio:**

```
clr(x)_i = log(x_i) − (1/D) · Σ_{j=1..D} log(x_j)
```

**Multiplicative zero replacement:**

```
delta = 1 / N                      (N = sample total count ≈ one read)
x_i = delta                        for x_i = 0
x_i = x_i · (1 − z·delta)          for x_i > 0, z = number of zeros
```

**Community metal-activity index:**

```
index(s) = Σ_g w_g · rel_abund(g, s) / Σ_g rel_abund(g, s),   w_g ∈ {0, 1}
```

**Literature-expected removal for metal m in sample s:**

```
expected(s, m) = Σ_g (efficiency(g, m) / 100) · rel_abund(g, s)
```

**Abundance-weighted model score:**

```
score(s) = Σ_g P(metal-active | g) · rel_abund(g, s) / Σ_g rel_abund(g, s)
```

---

*Generated by the pipeline in this repository. Every number is reproducible from
`config.yaml` with seed 42; see Appendix B.*
