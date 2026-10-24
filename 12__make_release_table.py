"""Build the release table for all 1623 BCUs (seven-feature noSky run).

Run from the files_v2 folder:
    python3 12__make_release_table.py

Takes confidence, assignment and tier for ALL BCUs from data/oof_scores_noSky_bcu.csv
(conf_fsrq / conf_bll are the noSky values there), and the per-classifier columns
from results_candidates.csv (noSky values, candidates only; blank for the rest),
because the per-classifier columns in oof_scores_noSky_bcu.csv are still the old
nine-feature ones.
"""
import os
import numpy as np
import pandas as pd

IN_CSV = os.environ.get("IN_CSV", "data/oof_scores_noSky_bcu.csv")
CAND_CSV = os.environ.get("CAND_CSV", "results_candidates.csv")
OUT_CSV = os.environ.get("OUT_CSV", "bcu_classification_all.csv")
THR = 0.1
CLF = ["CE", "WE", "BC", "MLR", "PUET"]
FEAT = ["PL_Index", "LP_beta", "LP_SigCurv", "log_Pivot_Energy",
        "log_Flux1000", "log_Variability_Index", "log_Flux_Peak"]
CLFCOLS = ([f"fsrq_{c}" for c in CLF] + [f"bll_{c}" for c in CLF])
FROM_CAND = ["conf_assigned_ge4", "n_models_agree"] + CLFCOLS

d = pd.read_csv(IN_CSV)
b = d[d["is_bcu"].astype(bool)].copy()
print("BCU rows:", len(b))
assert len(b) == 1623, "expected 1623 BCUs"

f_hit = b["conf_fsrq"] >= THR
b_hit = b["conf_bll"] >= THR
b["assignment"] = np.select(
    [f_hit & ~b_hit, b_hit & ~f_hit, f_hit & b_hit],
    ["FSRQ-like", "BL Lac-like", "ambiguous"], default="no class")
is_f = b["assignment"] == "FSRQ-like"
assigned = b["assignment"].isin(["FSRQ-like", "BL Lac-like"])
b["conf_assigned"] = np.where(is_f, b["conf_fsrq"], b["conf_bll"])
b.loc[~assigned, "conf_assigned"] = np.nan
ca = b["conf_assigned"]
b["tier"] = np.select([ca >= 0.9, ca >= 0.5, ca >= THR], ["A", "B", "C"], default="")
b.loc[b["assignment"] == "ambiguous", "tier"] = "ambiguous"
b.loc[b["assignment"] == "no class", "tier"] = "none"

c = pd.read_csv(CAND_CSV).set_index("Source_Name")
b = b.set_index("Source_Name")
for col in FROM_CAND:
    b[col] = c[col].reindex(b.index)
b = b.reset_index()

# consistency with results_candidates.csv on the overlapping rows
cc = c.reindex(b["Source_Name"])
for col in ["assignment", "tier"]:
    n = int((b[col].values != cc[col].values)[cc[col].notna().values].sum())
    print(f"mismatch {col}: {n}")
for col in ["conf_fsrq", "conf_bll"]:
    m = cc[col].notna().values
    n = int((~np.isclose(b[col].values[m], cc[col].values[m], atol=0.011)).sum())
    print(f"mismatch {col}: {n}")
print("candidate-file sources not found among BCUs:",
      len(c.index.difference(b["Source_Name"])))

cols = (["Source_Name", "assignment", "tier", "conf_fsrq", "conf_bll", "conf_assigned"]
        + FROM_CAND + FEAT)
out = b[cols].copy()
rc = ["conf_fsrq", "conf_bll", "conf_assigned", "conf_assigned_ge4"] + CLFCOLS
out[rc] = out[rc].round(2)
order = {"FSRQ-like": 0, "BL Lac-like": 1, "ambiguous": 2, "no class": 3}
out = out.sort_values(["assignment", "Source_Name"],
                      key=lambda s: s.map(order) if s.name == "assignment" else s)
out.to_csv(OUT_CSV, index=False)

print(out["assignment"].value_counts().to_string())
print(out["tier"].value_counts().to_string())
print("Expected: FSRQ-like 437, BL Lac-like 597, ambiguous 12, no class 577")
print("Wrote", OUT_CSV, out.shape)
