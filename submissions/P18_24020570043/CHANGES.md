# Response to feedback

| # | Feedback point (short) | What I changed | Where (section / table / figure / file) |
|---|---|---|---|
| 1 | State data licence, access date, and OGD page URL | Stated the GODL-India licence, retrieval date (`25 October 2026`), and full `data.gov.in` resource URL | Section 3.1, Credits (`Data and code availability`), `paper/references.bib` (`ogd2025`), `README.md` |
| 2 | Fix dangling citations `[26]` and `[27]` | Replaced plain bracket numbers with `\cite{angrist2009}` and `\cite{cameron2015}` and verified all 25 keys resolve | Section 2.2 (`paper/sections/02_related.tex`), `paper/references.bib` |
| 3 | Shorten abstract to 150–250 words; define GPI at first use | Tightened abstract to 190 words retaining the 3 questions, GPI definition, baseline vs OLS LOOCV RMSE, density coefficient + CI, and 2035 result | Abstract (`paper/paper.tex`) |
| 4 | Make decision-maker explicit and cite SDG 4 facts | Named State higher-education departments and central NEP monitoring cell; cited SDG target 4.3 with UNESCO GEM 2020 global tertiary GER (`39%`) | Section 1.1 (`paper/sections/01_introduction.tex`) |
| 5 | Justify skew decision / test log transform | Added skewness & log-transform OLS comparison showing logs do not improve LOOCV RMSE (`6.91`–`6.94` vs `6.86`), justifying raw scale + cap at 100 | Section 3.5, Table `tab:log`; `src/checks.py`, `outputs/tables/table12_log_transform.csv`, `table12b_skewness.csv` |
| 6 | Add missingness table for Ladakh and Lakshadweep | Added structured missingness table documenting affected years, variables, reason, and treatment | Section 3.2, Table `tab:missing`; `src/checks.py`, `outputs/tables/table13_missingness.csv` |
| 7 | Clarify what cross-section predicts and 2035 assumptions | Explicitly contrasted same-year prediction vs 4 projection assumptions and placed S1–S3 scenarios side by side | Section 4.9 (`sec:projection`), Table `tab:scenarios` (`paper/sections/04_methods.tex`) |
| 8 | Confirm full pipeline (`statsmodels`) and pin `requirements.txt` | Pinned exact tested versions (`==`) in `requirements.txt`; integrated `src/checks.py` into `src/run_all.py`; expanded `src/verify_outputs.py` (72/72 checks pass) | `requirements.txt`, `src/run_all.py`, `src/verify_outputs.py` |
| 9 | Bibliography size (~20–25), DOIs, and >=8 recent papers | Curated 25 references: all 21 scholarly works have Crossref-verified DOIs (9 from 2019–2026) and 4 official reports cite URL + access date | `paper/references.bib` |
| 10 | Code availability URL, terminology, captions, citations | Added GitHub repository URL; aligned `State GER`, `college density`, `PTR`; verified all tables/figures have self-contained captions and are cited | Credits (`paper/sections/10_declarations.tex`), all `paper/sections/*.tex` |
| 11 | Number only referred equations; align Section 4 & 6 notation | Cited `\eqref{eq:required}` in Section 6.6; aligned $\beta_D,\beta_P,\beta_G$, $\lambda$, $P_{s,2021}$, $G_{s,2021}$, $D_{s,2021}$, $D_{s,2035}$, $D_s^{\text{req}}$, and $P(\GER \ge 50\%)$ | Sections 4, 6, Appendix (`04_methods.tex`, `06_results.tex`, `11_appendix.tex`) |
| 12 | 6–8 keywords with SDG 4; remove Contributions/Abstract overlap | Kept 7 keywords including `SDG 4`; refocused Contributions on methodological and reproducibility artefacts | `paper/paper.tex`, Section 1.3 (`paper/sections/01_introduction.tex`) |
| 13 | Use Overleaf / Springer LNCS template; remove artefacts; specific AI statement | Converted manuscript to Springer LNCS (`\documentclass[runningheads]{llncs}`, `splncs04`), added workflow Figure 1, updated AI-use credits, removed TeX build artefacts | `paper/paper.tex`, `paper/llncs.cls`, `paper/splncs04.bst`, `paper/sections/10_declarations.tex` |

## Other changes
- Updated `README.md` with GODL-India licence, retrieval date, `src/checks.py`, Tables 12–13, `paper/` layout, and author citation (removing template placeholders).
- Note on optional enhancement: adding a second year of AISHE data or district-level sub-analysis remains noted under Limitations and Future Work (Sections 8–9).

## Reproducibility
- Repository commit used for this version: `49385a29474b8f62542765d00125f69f5b69cf54`
- Command that regenerates all results: `python src/run_all.py --skip-download` (or `python src/run_all.py` for full download) followed by `python src/verify_outputs.py` (72/72 checks passing).
