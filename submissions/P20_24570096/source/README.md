# Forecasting Missing-Children Caseloads and Stratifying State Tracing Regimes in India via Classical Machine Learning

**Author:** Aayush Sapra (`Roll No.: 24570096`)  
**Course:** B.Sc. (Hons.) Computer Science, Semester VI (Ramanujan College, University of Delhi)  
**Subject:** Machine Learning | **Supervisor:** Nikhil Sir  
**Submission Folder:** `submissions/P20_24570096/` (`https://github.com/campus-sutra/ml-application-projects-2026`)  
**UN SDGs:** SDG 16 (Target 16.2) & SDG 5 (Target 5.2)

---

## 📁 Repository Layout

```text
submissions/P20_24570096/
├── paper.pdf                                      # Compiled 5-page camera-ready Short Communication
├── CHANGES.md                                     # Point-by-point response to reviewer feedback
├── metadata.yml                                   # Structured submission metadata
└── source/
    ├── Missing_Children_ML_Assignment_Aayush_Sapra.docx
    ├── run_experiments.py                         # Generates Tables 1-5, all 7 PDF figures, and results/*.csv
    ├── per_state.py                               # Generates Appendix Table A1 (results/tableA1_per_state_audit.csv)
    ├── ncrb_missing_children.csv                  # Compiled NCRB / Dataful dataset (6,048 rows, 36 States/UTs)
    ├── requirements.txt                           # Pinned Python dependencies
    ├── figures/                                   # Generated publication figures (fig1 to fig7)
    └── results/                                   # Exported CSV tables matching every number in paper.pdf
```

---

## Mapping of Scripts to Paper Tables & Figures

| Script Command | Exported Output File | Corresponding Paper Element |
| :--- | :--- | :--- |
| `python run_experiments.py` | `results/missingness_audit.csv` & `figures/fig1_missingness.pdf` | **Table 1** & **Figure 1** |
| `python run_experiments.py` | `figures/fig2_trends.pdf` – `figures/fig5_pca_variance.pdf` | **Figures 2–5** |
| `python run_experiments.py` | `results/table3_regression_results.csv` & `figures/fig6_poly_scatter.pdf` | **Table 3** & **Figure 6** |
| `python run_experiments.py` | `results/table4_clustering_sweep.csv` & `results/table4_tier_profiles.csv` & `figures/fig7_state_clusters.pdf` | **Table 4** & **Figure 7** |
| `python run_experiments.py` | `results/table5_ablation_lockdown.csv` | **Table 5** |
| `python per_state.py` | `results/tableA1_per_state_audit.csv` | **Appendix Table A1** |

---

## Reproducing All Results from a Fresh Clone

```bash
pip install -r requirements.txt
python run_experiments.py
python per_state.py
```

---

## Data Provenance & Licence

* **Source:** National Crime Records Bureau (NCRB), *Crime in India (2016–2022), Chapter 15: Missing and Traced Persons*, Ministry of Home Affairs, Government of India, compiled via Dataful ([Dataset ID 18468](https://dataful.in/datasets/18468/), accessed 15 Oct 2026) and verified against [Lok Sabha Unstarred Question No. 4003](https://www.mha.gov.in/MHA1/Par2017/pdfs/par2025-pdfs/LS25032025/4003.pdf).
* **Licence:** Released under the **Government Open Data Licence -- India (GODL)**.