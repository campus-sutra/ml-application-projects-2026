
# Response to Feedback

| # | Feedback point (short) | What I changed | Where (section / table / figure / file) |
|---|---|---|---|
| 1 | Add dataset and document its source and licence | Added the dataset to the project repository and documented its source, licence, and the discrepancy between the UCI catalogue count (1,013) and the 1,014 records used in the experiments. | Dataset and Data Description; README.md |
| 2 | Generate confusion matrix from actual predictions | Generated the confusion matrix using actual out-of-fold predictions instead of manually reconstructing it. | Model Evaluation; Confusion Matrix figure |
| 3 | Ensure notebook runs from a fresh kernel | Executed the research notebook sequentially and checked that all code cells complete without errors. | ML_project.ipynb |
| 4 | Include simple baseline classifiers | Added majority-class and stratified DummyClassifier baselines using grouped cross-validation. | Experimental Results; Baseline Comparison table; src/experiments.py |
| 5 | Clarify model screening and nested CV | Distinguished five-fold grouped model screening from the nested grouped cross-validation evaluation protocol. | Methodology; Model Comparison and Nested CV tables |
| 6 | Investigate the HeartRate = 7 anomaly | Retained the observations in the primary dataset and performed sensitivity analysis by comparing results with and without the two anomalous observations. | Data Quality; HeartRate Sensitivity Analysis |
| 7 | Add feature relevance and ablation analysis | Added feature-ablation experiments, including blood-pressure-only and individual feature-removal configurations. | Feature Analysis; Feature Ablation table and figure |
| 8 | Evaluate high-risk recall against the predefined target | Reported nested-CV high-risk recall of 0.7906 ± 0.0874 and explicitly acknowledged that it falls slightly below the 0.80 target. | Results; Discussion |
| 9 | Report mean and standard deviation across folds | Reported Macro-F1, balanced accuracy, and high-risk recall using mean and standard deviation for grouped screening and nested cross-validation. | Model Comparison tables; Results |
| 10 | Clarify intended prediction context | Described the intended use as experimental decision support during antenatal assessment when all six measurements are available. | Application Context; Discussion |
| 11 | Discuss dataset transferability | Added the limitation that the dataset originates from Bangladesh and requires independent validation before application to rural antenatal care in India. | Limitations and Future Work |
| 12 | Improve references, figures, tables and declarations | Revised the manuscript structure, added result visualisations and tables, and included an AI-use declaration. | References; Figures; Tables; Declarations |

## Other changes

- Expanded the discussion of class-wise performance, particularly the low recall for the medium-risk class.
- Added discussion of repeated feature profiles and conflicting labels.
- Clarified that feature importance represents predictive contribution and does not establish clinical causation.
- Emphasised that the application is an experimental research prototype and not a substitute for clinical diagnosis or professional medical advice.
- Updated project documentation, experiment scripts, and automated tests.

## Reproducibility

- Repository commit used for this version: [ADD FINAL COMMIT HASH]
- Command that regenerates all results: [ADD THE EXACT COMMAND FROM THE PROJECT README]
