# 19.	Identify the delimiter/structure of each file: csv = comma, tsv = tab,
# 	txt = fixed width (columns occupy character positions 1-12, 13-28, 29-36, 37-42),
# 	xlsx = workbook whose records live in the "Data" sheet only.
# 20.	Acquire each file in R.
# 21.	Compare row counts and column names.
# 22.	Check whether the four files represent the same number of records.
# 23.	Calculate the average Marks field from each imported object.
# 24.	Show what an incorrect separator or sheet selection does to the results.

args <- commandArgs(FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
lab <- if (length(file_arg) > 0) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
data_dir <- file.path(lab, "data")

csv_df <- read.csv(file.path(data_dir, "students.csv"))
tsv_df <- read.delim(file.path(data_dir, "students.tsv"))
txt_df <- read.fwf(file.path(data_dir, "students.txt"), widths = c(12, 16, 8, 6),
                   col.names = c("Student_ID", "Name", "Gender", "Marks"), strip.white = TRUE)
txt_df <- txt_df[txt_df$Student_ID != "Student_ID", ]
txt_df$Student_ID <- as.integer(txt_df$Student_ID)
txt_df$Marks <- as.integer(txt_df$Marks)
xlsx_df <- readxl::read_excel(file.path(data_dir, "students.xlsx"), sheet = "Data")

frames <- list(csv = csv_df, tsv = tsv_df, txt = txt_df, xlsx = xlsx_df)

cat("21/22. row counts and column names\n")
for (name in names(frames)) {
  df <- frames[[name]]
  cat(sprintf("  %-5s rows=%-3d columns=%s\n", name, nrow(df), paste(names(df), collapse = ", ")))
}
cat("  same row count:", length(unique(sapply(frames, nrow))) == 1, "\n")
cat("  same columns:  ", all(sapply(frames, function(df) identical(sort(names(df)), sort(names(csv_df))))), "\n")

cat("23. average Marks\n")
for (name in names(frames)) {
  cat(sprintf("  %-5s avg Marks = %.2f\n", name, mean(frames[[name]]$Marks)))
}

cat("24. incorrect separator / sheet selection\n")
wrong <- list(
  "tsv read with comma" = read.csv(file.path(data_dir, "students.tsv")),
  "txt read with comma" = read.csv(file.path(data_dir, "students.txt")),
  "xlsx wrong sheet ('Notes')" = readxl::read_excel(file.path(data_dir, "students.xlsx"), sheet = "Notes")
)
for (name in names(wrong)) {
  df <- wrong[[name]]
  cat(sprintf("  %-28s rows=%-3d columns=%s\n", name, nrow(df), paste(names(df), collapse = ", ")))
  if ("Marks" %in% names(df)) {
    cat(sprintf("  %-28s avg Marks = %.2f\n", "", mean(df$Marks)))
  } else {
    cat(sprintf("  %-28s avg Marks -> error: no 'Marks' column\n", ""))
  }
}
