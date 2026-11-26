# Response to feedback

**Paper Title:** Forecasting Missing-Children Caseloads and Stratifying State Tracing Regimes in India via Classical Machine Learning  
**Author:** Aayush Sapra (`Roll No.: 24570096`, Problem `P20`)  
**Course:** B.Sc. (Hons.) Computer Science, Semester VI (Ramanujan College, University of Delhi)  
**Target Repository Path:** `submissions/P20_24570096/`

| # | Feedback point (short) | What I changed | Where (section / table / figure / file) |
|---|---|---|---|
| 1 | Supply code for every reported row (remove unimplemented models) | Removed Decision Tree, k-NN, and SVR rows so every regression and clustering result is generated directly by `run_experiments.py` and `per_state.py` using strictly syllabus-approved methods (SLR, MLR, Ridge, Lasso, Degree-2 Polynomial + Ridge, 64–32 ANN, k-Means, and Ward Hierarchical Clustering), exported to `results/*.csv`. | Section 3.1, Section 4, Table 3; `run_experiments.py`, `per_state.py` |
| 2 | Show forward-chaining CV (`TimeSeriesSplit`) and uncertainty (`mean ± SD` & bootstrap CIs) | Replaced hard-coded hyperparameters with a 4-fold `TimeSeriesSplit` grid search on `2016–2019`, refitting imputation, 1st/99th percentile winsorisation, and Z-score scaling inside each fold; added `CV-RMSE (Mean ± SD)` and 1,000-resample bootstrap 95% CIs. | Section 3.3, Section 4, Table 3; `run_experiments.py` |
| 3 | Select models on validation rather than test results | Documented validation-set selection criteria (SLR lowest linear CV-RMSE `307.81 ± 99.64`; 64–32 ANN lowest non-linear CV-RMSE `380.69 ± 112.40`) and evaluated the `2020–2022` test set strictly once. | Section 3.3, Section 4; `run_experiments.py` |
| 4 | Support or remove Lasso and k-NN-PCA claims | Removed unimplemented k-NN-PCA claims and replaced them with code-verified PCA (95% variance, 8 PCs) ablations across MLR, Ridge, and 64–32 ANN, plus programmatic verification of the 5 collinear columns zeroed by Lasso ($\lambda = 10$). | Section 2.3, Section 6, Table 5; `run_experiments.py` |
| 5 | Fix clustering numbers; add permutation null baseline and bootstrap ARI stability | Reconciled all k-Means and Ward metrics across $k \in \{2,\dots,6\}$ from `run_experiments.py` (correcting $k=6$ silhouette to `0.404`), added 100-draw permutation null baselines and 80%-subsample bootstrap ARI stability (`0.812 ± 0.141` at $k=3$), and justified $k=3$ for policy stratification. | Section 3.2, Section 5, Table 4A, Table 4B; `run_experiments.py` |
| 6 | Temper "accurate" claims and quantify relative error | Tempered claims throughout and added Mean Absolute Scaled Error (`MASE`) and per-State Relative MAE (`Rel_MAE_Pct`) to highlight where the global MSE model fails (small North-Eastern states, island UTs, and unseen Ladakh). | Abstract, Section 6, Section 7, Appendix Table A1; `per_state.py` |
| 7 | Add a dedicated Limitations section | Added dedicated discussion of NCRB FIR reporting heterogeneity, structural missingness (5.1%–26.9%), COVID-19 lockdown covariate shift, and child-protection data ethics. | Section 6 & Section 7 (`Limitations`, `Declarations & Sources`) |

## Other changes
- Trimmed the Structured Abstract to 224 words (within the 150–250 word range) while keeping all key metrics, confidence intervals, and null baselines, with 4–8 keywords.
- Added target distribution & skewness comparison (raw $\text{Missing}_{t+1}$ skewness $= 3.45$ vs. $\log(1+y)$ skewness $= 0.14$) and target transformation sensitivity check.
- Rewrote Figure 1 so missingness bars are computed dynamically from `ncrb_missing_children.csv` (`results/missingness_audit.csv`).
- Standardised terminology on *"tracing rate"*, organised sections cleanly, expanded bibliography to 18 verified IEEE references with DOIs, documented the **GODL-India** licence and access date (`15 Oct 2026`), and included an explicit AI-use statement under **Declarations & Sources**.

## Reproducibility
- Code repository: `https://github.com/aayushsapra200611/missing-children-ml-india`
- Repository commit used for this version: `3047d06c68dbe612350323cb19757f2f20de2afb`
- Commands that regenerate all results:
  ```bash
  pip install -r requirements.txt
  python run_experiments.py
  python per_state.py
  ```
