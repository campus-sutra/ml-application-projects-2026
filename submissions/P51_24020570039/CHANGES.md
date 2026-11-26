# Response to feedback

| # | Feedback point (short) | What changed / remaining limitation | Where |
|---|---|---|---|
| 1 | Alpha leakage | Selected ridge and quadratic ridge alphas separately within each rolling origin using earlier years only; refitted on all training years. Documented the earliest-origin limitation and two-origin sensitivity. | Sections 4–5, Tables 2–3; `code/run_experiments.py`, `outputs/backtest_folds.csv`, `outputs/backtest_common_origins.csv` |
| 2 | Backtest spread | Reported each origin and mean ± sample SD of RMSE, relative MAE and pooled training-scaled MASE. | Tables 2–3; `outputs/backtest_summary.csv` |
| 3 | Related work | Added a section on crime forecasting, panel models, benchmarking and cyber-crime reporting with numbered citations and a statement of the contribution. | Section 2; references |
| 4 | Experimental setup | Specified training/testing years, rolling origins, alpha grid, k range, random seed 42 and k-means restarts. | Section 4 |
| 5 | Variables, missingness and skew | Added variables/coverage/units/source/missingness table, documented Ladakh's two missing years and added a log-count distribution figure. | Section 3, Table 1, Fig. 1; `outputs/missing_data.csv` |
| 6 | AI and repository declarations | Disclosed ChatGPT and OpenCode assistance and linked the public project repository. | Declarations; `metadata.yml` |
| 7 | IEEE-style references and DOI details | Converted author-year citations to numbered references; expanded bibliography to 25 entries, with DOI identifiers where available. Nine dated 2020–2025 research/review articles are included. Full independent verification of all references is still outstanding. | References and in-text citations |
| 8 | Dataset URL, licence and access date | Included OGD URL and GODL-India licence; documented the supplied dataset access date (26 November 2026). Original OGD download date is unrecorded, and live URL still needs confirmation. | Section 3; Declarations; ref. 2 |
| 9 | Equations and evaluation definitions | Added ridge objective, inverse transformation, RMSE, MAE, R², relative MAE and pooled MASE. Distinguished the original persistence-relative ratio from training-scaled MASE. | Section 4 |
| 10 | Terminology | Standardised use of registrations and distinguished recorded counts from true incidence. | Abstract; Sections 1, 3, 7–8 |
| 11 | Figure and abstract checks | Retained numbered/captioned figures, updated in-text references and kept abstract within 150–250 words; seven keywords. | Abstract; Figs. 1–6 and A1 |
| 12 | Raw data and dependencies | Retained original XLS in the personal project repository and added an XLSX conversion; use OpenPyXL instead of Calamine. Documented dependencies, Python version and run command. | Project `data/`, `requirements.txt`, `README.md` |
| 13 | Reproducibility | Corrected script ran in a fresh project directory without pre-existing outputs. A fully isolated dependency installation remains unverified. | Project `README.md`, `REVISION_CHECKLIST.md` |
| 14 | AI and originality checklist | AI declaration added. Institutional Turnitin/DrillBit similarity check has not been performed; no claim of a score below 10% is made. | Declarations |
| 15 | Template and placeholders | Converted the paper to the course's Springer LNCS article layout, with editable LaTeX tables and source figures; compiled using pdfLaTeX. | `paper.pdf`, `source/` |
| 16 | External venue screening | Preliminary venue screening documented in the personal repository; no external venue selected yet. Deadline, scope, indexing and fee checks must be completed before external submission. | Project `REVISION_CHECKLIST.md` |
| 17 | Extra years or indicators | Retained the original five-year dataset for the required revision. Extending the time series and testing lagged indicators are future work. | Section 8 |

## Other changes

- Labelled tele-density results as a conditional ablation, since observed 2022 tele-density was used.
- Defined high-volume States using 2021 registrations rather than 2022 outcomes.
- Preserved Telangana sensitivity analysis and descriptive four-tier clustering, without treating the clusters as safety rankings or causal evidence.
- Funding and competing-interest statements require author confirmation; none are asserted without that confirmation.

## Reproducibility

- Repository: https://github.com/abhirupchak/cyber-crime-forecasting-51
- Repository commit used for this version: `3f5d62983ef242626bb561dfc09e3df0bab1643b` (corrected analysis script and requirements match the local revision by Git blob hash).
- Command that regenerates the experimental results, from the project root: `python code/run_experiments.py` (after `pip install -r requirements.txt`).
- The submission folder contains the article only; code and raw datasets remain in the personal project repository.
