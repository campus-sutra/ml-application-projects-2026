# Response to Feedback

| # | Feedback point | Change made | Location |
|---|---|---|---|
| 1 | Stakeholder and practical use | The study identifies State Planning Departments and SDG policy analysts as potential users. The results are exploratory and are not intended as official rankings. | `paper_source/main.tex`, Introduction and Ethics |
| 2 | Stability and baseline analysis | Added 50-seed K-Means stability, 100 bootstrap resamples, a shuffled-feature null baseline, and State/UT administrative comparison. The manuscript still needs to be updated to report these additional results. | `src/revision_analysis.py`, `results/` |
| 3 | Outliers and State-only analysis | Added Isolation Forest outlier screening and a separate analysis of the 28 States. These results should be interpreted cautiously. | `src/cluster_interpretation.py`, `src/revision_analysis.py`, `results/` |
| 4 | Cluster interpretation | Cluster assignments and goal-wise cluster profiles are available in the results. | `results/k-means_cluster_profiles.csv`, cluster assignment files |
| 5 | Feature weighting | Added a PCA-based sensitivity comparison, retaining approximately 90% of variance. | `src/feature_weighting_comparison.py`, `results/` |
| 6 | Algorithm settings and formulas | The implementation uses StandardScaler, K-Means with `n_init=20` and `random_state=42`, and Ward hierarchical linkage. The paper includes the principal evaluation metric formulas. | `src/clustering.py`, `src/hierarchical_clustering.py`, `paper_source/main.tex` |
| 7 | Figure and caption consistency | Figures and captions should be checked against the final compiled manuscript before submission. | `paper_source/main.tex`, `results/` |
| 8 | Related work and references | Existing references are included. The bibliography still requires further verification and expansion to fully meet the reviewer's recommendation. | `paper_source/references.bib` |
| 9 | Dataset and reproducibility | Included the dataset, source scripts, results, README, and requirements file in the submission package. | `source/data/`, `source/src/`, `source/results/`, `source/README.md` |
| 10 | Remaining manuscript revisions | Some requested manuscript updates and reference checks remain incomplete; no claim is made that all reviewer suggestions have been addressed. | `source/paper_source/main.tex` |

## Other Changes
- Added the revised analysis script and its generated result files.
- Included the project source code and supporting data in the submission folder.

## Reproducibility
- Project repository: https://github.com/Vidhimaheshwari013/ML-Project
- Main analysis commands:
  - `python src/preprocessing.py`
  - `python src/clustering.py`
  - `python src/hierarchical_clustering.py`
  - `python src/revision_analysis.py`
  - `python src/feature_weighting_comparison.py`
- The commands should be run from the project root. Run other analysis scripts as needed to regenerate their corresponding outputs.
- Submission commit: to be recorded after the final commit.
