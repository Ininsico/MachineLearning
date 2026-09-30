#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Fallback acquisition: raw FASTQ from NCBI SRA -> cutadapt -> DADA2 -> SILVA 138.1
#
# Used only when the MGnify paths fail or return fewer than 200 samples.
#
# Usage:
#   Rscript r/dada2_sra_pipeline.R --accessions accessions.txt --outdir data/raw/sra \
#           --threads 8 --silva-dir data/raw/silva
#
# Pipeline (as specified for this study):
#   1. prefetch + fasterq-dump to retrieve raw paired-end reads
#   2. cutadapt removes the 16S primers still present in the reads
#   3. filterAndTrim(truncLen = c(240, 160), maxEE = c(2, 2))
#   4. learnErrors -> dada -> mergePairs -> makeSequenceTable -> removeBimeraDenovo
#   5. assignTaxonomy against SILVA 138.1
#   6. drop mitochondria, chloroplast and unclassified reads
#
# Requires: DADA2, Biostrings, ShortRead; and on PATH: prefetch, fasterq-dump, cutadapt.
#   BiocManager::install("dada2")
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dada2)
  library(Biostrings)
})

DEFAULTS <- list(
  outdir = "data/raw/sra",
  accessions = NULL,
  threads = 8,
  trunc_len = c(240, 160),
  max_ee = c(2, 2),
  trunc_q = 2,
  min_len = 50,
  silva_dir = "data/raw/silva",
  silva_train = "silva_nr99_v138.1_train_set.fa.gz",
  silva_species = "silva_species_assignment_v138.1.fa.gz",
  fwd_primer = "GTGYCAGCMGCCGCGGTAA",
  rev_primer = "GGACTACNVGGGTWTCTAAT"
)

parse_args <- function(args) {
  opts <- DEFAULTS
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i < length(args)) args[[i + 1]] else NA
    if (key == "--outdir") opts$outdir <- val
    else if (key == "--accessions") opts$accessions <- val
    else if (key == "--threads") opts$threads <- as.integer(val)
    else if (key == "--silva-dir") opts$silva_dir <- val
    i <- i + 2
  }
  opts
}

run <- function(cmd) {
  message("      $ ", paste(cmd, collapse = " "))
  status <- system2(cmd[[1]], cmd[-1])
  if (status != 0) stop("command failed: ", paste(cmd, collapse = " "), call. = FALSE)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.null(opts$accessions) || is.na(opts$accessions)) {
    stop("--accessions <file> is required (one SRA run accession per line)", call. = FALSE)
  }
  accs <- readLines(opts$accessions)
  accs <- accs[nzchar(trimws(accs))]
  message("[DADA2] processing ", length(accs), " SRA runs")

  raw_dir <- file.path(opts$outdir, "raw_fastq")
  trim_dir <- file.path(opts$outdir, "trimmed")
  filt_dir <- file.path(opts$outdir, "filtered")
  for (d in c(raw_dir, trim_dir, filt_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

  # ---- 1. retrieve -------------------------------------------------------
  for (acc in accs) {
    message("[1/6] prefetch/fasterq-dump: ", acc)
    run(c("prefetch", "--max-size", "50g", "--output-directory", raw_dir, acc))
    run(c("fasterq-dump", "--split-files", "--threads", opts$threads,
          "--outdir", raw_dir, file.path(raw_dir, acc)))
  }

  fwd <- sort(list.files(raw_dir, pattern = "_1\\.fastq(\\.gz)?$", full.names = TRUE, recursive = TRUE))
  rev <- sort(list.files(raw_dir, pattern = "_2\\.fastq(\\.gz)?$", full.names = TRUE, recursive = TRUE))
  if (length(fwd) == 0) stop("no paired FASTQ files found after retrieval", call. = FALSE)
  sample_names <- basename(sub("_1\\.fastq(\\.gz)?$", "", fwd))

  # ---- 2. primer trimming (cutadapt) ------------------------------------
  fwd_trim <- file.path(trim_dir, paste0(sample_names, "_1.fastq.gz"))
  rev_trim <- file.path(trim_dir, paste0(sample_names, "_2.fastq.gz"))
  for (i in seq_along(fwd)) {
    message("[2/6] cutadapt primer trimming: ", sample_names[i])
    run(c("cutadapt", "-j", opts$threads,
          "-g", opts$fwd_primer, "-G", opts$rev_primer,
          "--discard-untrimmed", "--minimum-length", opts$min_len,
          "-o", fwd_trim[i], "-p", rev_trim[i], fwd[i], rev[i]))
  }

  # ---- 3. filter + trim -------------------------------------------------
  message("[3/6] filterAndTrim(truncLen = c(", opts$trunc_len[1], ", ",
          opts$trunc_len[2], "), maxEE = c(", opts$max_ee[1], ", ", opts$max_ee[2], "))")
  filt_fwd <- file.path(filt_dir, paste0(sample_names, "_1.filt.fastq.gz"))
  filt_rev <- file.path(filt_dir, paste0(sample_names, "_2.filt.fastq.gz"))
  filterAndTrim(fwd_trim, filt_fwd, rev_trim, filt_rev,
                truncLen = opts$trunc_len, maxEE = opts$max_ee, truncQ = opts$trunc_q,
                maxN = 0, rm.phix = TRUE, compress = TRUE, multithread = opts$threads)
  keep <- file.exists(filt_fwd) & file.size(filt_fwd) > 0
  filt_fwd <- filt_fwd[keep]; filt_rev <- filt_rev[keep]; sample_names <- sample_names[keep]
  if (length(filt_fwd) == 0) stop("all samples were discarded during filtering", call. = FALSE)

  # ---- 4. denoise -------------------------------------------------------
  message("[4/6] learnErrors -> dada -> mergePairs")
  err_fwd <- learnErrors(filt_fwd, multithread = opts$threads, randomize = TRUE)
  err_rev <- learnErrors(filt_rev, multithread = opts$threads, randomize = TRUE)
  dada_fwd <- dada(filt_fwd, err = err_fwd, multithread = opts$threads)
  dada_rev <- dada(filt_rev, err = err_rev, multithread = opts$threads)
  merged <- mergePairs(dada_fwd, filt_fwd, dada_rev, filt_rev, verbose = TRUE)

  seqtab <- makeSequenceTable(merged)
  message("      sequence table: ", nrow(seqtab), " samples x ", ncol(seqtab), " ASVs")
  seqtab_nochim <- removeBimeraDenovo(seqtab, method = "consensus",
                                      multithread = opts$threads, verbose = TRUE)
  message("      after chimera removal: ", ncol(seqtab_nochim), " ASVs (",
          sprintf("%.1f%%", 100 * sum(seqtab_nochim) / sum(seqtab)), " of reads retained)")

  # ---- 5. taxonomy (SILVA 138.1) ----------------------------------------
  silva_train <- file.path(opts$silva_dir, opts$silva_train)
  silva_species <- file.path(opts$silva_dir, opts$silva_species)
  if (!file.exists(silva_train)) stop("SILVA training set not found: ", silva_train, call. = FALSE)
  message("[5/6] assignTaxonomy against SILVA 138.1")
  taxa <- assignTaxonomy(seqtab_nochim, silva_train, multithread = opts$threads, tryRC = TRUE)
  if (file.exists(silva_species)) {
    taxa <- addSpecies(taxa, silva_species, tryRC = TRUE)
  }

  # ---- 6. remove mitochondria / chloroplast / unclassified --------------
  message("[6/6] removing mitochondria, chloroplast and unclassified reads")
  is_mito <- grepl("Mitochondria", taxa[, "Family"], ignore.case = TRUE)
  is_chloro <- grepl("Chloroplast", taxa[, "Order"], ignore.case = TRUE) |
    grepl("Chloroplast", taxa[, "Class"], ignore.case = TRUE)
  is_unclassified <- is.na(taxa[, "Phylum"]) | taxa[, "Phylum"] == "" |
    is.na(taxa[, "Genus"]) | taxa[, "Genus"] == ""
  drop <- is_mito | is_chloro | is_unclassified
  message("      dropping ", sum(drop), " ASVs (mito=", sum(is_mito),
          ", chloro=", sum(is_chloro), ", unclassified=", sum(is_unclassified), ")")

  seqtab_final <- seqtab_nochim[, !drop, drop = FALSE]
  taxa_final <- taxa[!drop, , drop = FALSE]

  # ---- export genus-level counts ---------------------------------------
  genus <- sub("^g__", "", taxa_final[, "Genus"])
  aggregated <- rowsum(t(seqtab_final), group = genus, reorder = TRUE)
  aggregated <- aggregated[rowSums(aggregated) > 0, , drop = FALSE]

  dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)
  write.csv(as.data.frame(aggregated), file.path(opts$outdir, "genus_counts.csv"))
  write.csv(as.data.frame(taxa_final), file.path(opts$outdir, "asv_taxonomy.csv"))
  saveRDS(seqtab_final, file.path(opts$outdir, "seqtab_nochim.rds"))
  message("[DONE] ", nrow(aggregated), " genera x ", ncol(aggregated), " samples -> ",
          file.path(opts$outdir, "genus_counts.csv"))
}

main()
