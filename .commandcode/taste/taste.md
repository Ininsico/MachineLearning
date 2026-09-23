# Taste

- Organizes project folders into a consistent `src/` (scripts), `data/` (inputs), `results/` (generated outputs) layout, and expects this structure applied consistently across projects. Confidence: 0.6
- Wants data-science exercises delivered in both R and Python (parallel `foo.py` / `foo.R` solutions answering the same questions with matching results). Confidence: 0.6
- Signals urgency explicitly ("do this asap") and expects the work completed and verified in one pass rather than drip-fed. Confidence: 0.5
- Wants scripts and notebooks kept in separate folders (e.g. `py/` for `.py`, `ipynb/` for `.ipynb`) rather than co-located, with any sub-grouping (homework, case_studies) preserved inside each, and READMEs updated to match. Confidence: 0.7
- Runs Python lab work in Google Colab ("collab") and wants `.py` scripts delivered as matching Colab-ready `.ipynb` notebooks (markdown task cells, an upload cell for datasets, no local `__file__` paths). Confidence: 0.55
