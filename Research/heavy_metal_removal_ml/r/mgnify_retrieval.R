#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Public wastewater 16S retrieval via the Bioconductor MGnifyR package.
#
# This is the study-specification-preferred acquisition path:
#     MgnifyClient() |> doQuery(biome = "wastewater") |> getResult(get.taxa = TRUE, output = "phyloseq")
#
# Usage:
#   Rscript r/mgnify_retrieval.R --outdir data/raw/mgnify_r --biome wastewater --max-samples 260
#
# Requires: MGnifyR, phyloseq, optparse (Bioconductor / CRAN).
#   BiocManager::install(c("MGnifyR", "phyloseq"))
#
# On success this writes into <outdir>:
#   genus_counts.csv         genus x sample count matrix (portable to the Python pipeline)
#   sample_metadata.csv      per-sample provenance
#   phyloseq_object.rds      the phyloseq object as returned by getResult()
#   acquisition_info.json    run provenance
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(MGnifyR)
  library(phyloseq)
})

parse_args <- function(args) {
  opts <- list(outdir = "data/raw/mgnify_r", biome = "wastewater", max_samples = 260)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--outdir", "--biome", "--max-samples") && i < length(args)) {
      val <- args[[i + 1]]
      if (key == "--outdir") opts$outdir <- val
      if (key == "--biome") opts$biome <- val
      if (key == "--max-samples") opts$max_samples <- as.integer(val)
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  opts
}

fail <- function(...) {
  message("[ERROR] ", ...)
  quit(status = 1)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)

  message("[1/5] Creating MgnifyClient (biome = '", opts$biome, "')")
  client <- MgnifyClient(useCache = TRUE)

  message("[2/5] doQuery(): searching amplicon studies for biome")
  studies <- tryCatch(
    doQuery(client, biome = opts$biome, type = "studies", experiment_type = "amplicon"),
    error = function(e) fail("doQuery() failed: ", conditionMessage(e))
  )
  if (is.null(studies) || length(studies) == 0) {
    fail("doQuery() returned no studies for biome '", opts$biome, "'")
  }
  message("      retrieved ", length(studies), " study records")

  message("[3/5] getResult(get.taxa = TRUE, output = 'phyloseq')")
  phy <- tryCatch(
    getResult(client, studies,
              get.taxa = TRUE,
              output = "phyloseq",
              get.analysis = TRUE),
    error = function(e) fail("getResult() failed: ", conditionMessage(e))
  )
  if (is.null(phy)) fail("getResult() returned NULL")

  n_samples <- nsamples(phy)
  message("      phyloseq object holds ", n_samples, " samples and ", ntaxa(phy), " taxa")
  if (n_samples < 200) {
    message("[WARN] fewer than 200 samples recovered; the Python pipeline will ",
            "fall back to the next configured acquisition strategy.")
  }

  message("[4/5] Aggregating to genus level and exporting")
  otu <- as(otu_table(phy), "matrix")
  if (taxa_are_rows(phy)) otu <- t(otu)
  tax <- as.data.frame(as(tax_table(phy), "matrix"))

  genus <- if ("Genus" %in% colnames(tax)) {
    tax[["Genus"]]
  } else if ("genus" %in% colnames(tax)) {
    tax[["genus"]]
  } else {
    rep(NA_character_, nrow(tax))
  }
  genus <- sub("^g__", "", as.character(genus))
  keep <- !is.na(genus) & nzchar(genus) &
    !grepl("unclassified|uncultured|unknown|incertae", genus, ignore.case = TRUE)
  if (!any(keep)) fail("no genus-rank assignments survived filtering")

  otu_keep <- otu[, keep, drop = FALSE]
  genus_keep <- genus[keep]
  aggregated <- rowsum(t(otu_keep), group = genus_keep, reorder = TRUE)
  aggregated <- aggregated[rowSums(aggregated) > 0, , drop = FALSE]

  counts_path <- file.path(opts$outdir, "genus_counts.csv")
  write.csv(as.data.frame(aggregated), counts_path, quote = TRUE)
  message("      wrote ", counts_path, " (", nrow(aggregated), " genera x ",
          ncol(aggregated), " samples)")

  meta <- data.frame(
    sample_accession = colnames(aggregated),
    stringsAsFactors = FALSE
  )
  meta_path <- file.path(opts$outdir, "sample_metadata.csv")
  write.csv(meta, meta_path, row.names = FALSE)

  message("[5/5] Saving phyloseq object and provenance")
  saveRDS(phy, file.path(opts$outdir, "phyloseq_object.rds"))
  info <- sprintf(
    '{"status":"ok","source":"MGnifyR","biome":"%s","n_samples":%d,"n_genera":%d,"n_studies":%d}',
    opts$biome, ncol(aggregated), nrow(aggregated), length(studies)
  )
  writeLines(info, file.path(opts$outdir, "acquisition_info.json"))
  message("[DONE] MGnifyR retrieval complete")
}

main()
