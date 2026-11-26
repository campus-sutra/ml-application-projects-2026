#!/usr/bin/env python3
"""
per_state.py — Per-State/UT Test Error & Relative Error Audit (Appendix A, Table A1)
Subject: Machine Learning | Course: B.Sc. (Hons.) Computer Science, 3rd Year (Sem VI)
Author: Aayush Sapra | Roll No.: 24570096 | Supervisor: Nikhil Sir
"""

import sys
import pathlib
import numpy as np
import pandas as pd

try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    BASE_DIR = pathlib.Path.cwd()

sys.path.insert(0, str(BASE_DIR))
from run_experiments import (
    train_df, test_df, NUMERIC_COLS, Preprocessor, Poly2RidgeModel,
    best_configs, df_clust, RESULTS_DIR
)

prep = Preprocessor(use_pca=False).fit(train_df)
X_tr = prep.transform(train_df)
X_te = prep.transform(test_df)
y_tr = train_df["target_missing_next"].values

poly_alpha = best_configs["Poly2"]["params"]["alpha"]
m_poly = Poly2RidgeModel(alpha=poly_alpha, n_num=len(NUMERIC_COLS)).fit(X_tr, y_tr)

test_eval = test_df.copy()
test_eval["pred_poly"] = np.maximum(0.0, m_poly.predict(X_te))
test_eval["pred_pers"] = test_eval["curr_missing_t"]

tier_map = dict(zip(df_clust["state"], df_clust["cluster"].map({0: "Tier 1", 1: "Tier 2", 2: "Tier 3"})))

rows = []
for st in sorted(test_eval["state"].unique()):
    n_tr = int((train_df["state"] == st).sum())
    sub = test_eval[test_eval["state"] == st]
    n_te = len(sub)
    mean_miss = float(sub["target_missing_next"].mean())
    rmse_poly = float(np.sqrt(np.mean((sub["target_missing_next"] - sub["pred_poly"]) ** 2)))
    rmse_pers = float(np.sqrt(np.mean((sub["target_missing_next"] - sub["pred_pers"]) ** 2)))
    mae_poly = float(np.mean(np.abs(sub["target_missing_next"] - sub["pred_poly"])))
    rel_err_pct = round(100.0 * mae_poly / mean_miss, 1) if mean_miss > 0 else np.nan
    rows.append({
        "State_UT": st,
        "n_train": n_tr,
        "n_test": n_te,
        "Mean_Miss": round(mean_miss, 1),
        "RMSE_Poly2": round(rmse_poly, 1),
        "RMSE_Pers": round(rmse_pers, 1),
        "Rel_MAE_Pct": rel_err_pct,
        "Tier": tier_map.get(st, "--"),
        "Worse_Than_Pers": rmse_poly > rmse_pers
    })

df_out = pd.DataFrame(rows).sort_values("Mean_Miss", ascending=False)
df_out.to_csv(RESULTS_DIR / "tableA1_per_state_audit.csv", index=False)

if __name__ == "__main__":
    print(df_out.to_string(index=False))