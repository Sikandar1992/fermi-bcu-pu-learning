#!/usr/bin/env python
# coding: utf-8
"""
Step 11: noSky candidates by tier, and a noSky score file for Step 9 (T1 only)
==============================================================================
Reads tables/ablation_candidates_noSky.csv (written by 7__ablation_v2.py) and
reports, for the noSky variant against the main run:
  1. Tier counts (A >= 0.9, B >= 0.5, C >= 0.1) per assigned class, main vs noSky.
  2. Tier of the 'gained' candidates (are they mostly tier C?).
  3. Tier shift of the sources that are candidates in both runs.
  4. Share of candidates in the Galactic plane (|b| < 10), main vs noSky.
It also writes data/oof_scores_noSky_bcu.csv: a copy of data/oof_scores_all.csv
in which conf_fsrq / conf_bll of the BCUs are replaced by the noSky values.
Known sources keep their Step 4B scores, because the ablation does not export
noSky scores for them. Use this file with Step 9 for the T1 (BCU) check only.
T2 (known sources) from it is NOT a noSky result.
T1 = BCUs with an external class; T2 = known FSRQ/BL Lac with an external class.
Only conf_fsrq, conf_bll and bcu_assignment carry noSky values. The per-classifier
columns (fsrq_CE ...), conf_*_ge4 and votes_* keep the nine-feature Step 4B values.

Environment variables
  OOF_CSV   (default data/oof_scores_all.csv)
  CAND_CSV  (default tables/ablation_candidates_noSky.csv)
  CONF      (default 0.1)
  NAME_COL  (name column of OOF_CSV; default: first text column)
Output: tables/nosky_tiers.csv, tables/nosky_gained_tiers.csv,
        data/oof_scores_noSky_bcu.csv
"""
import os
import numpy as np
import pandas as pd

CONF = float(os.environ.get("CONF", 0.1))
OOF = os.environ.get("OOF_CSV", "data/oof_scores_all.csv")
CAND = os.environ.get("CAND_CSV", "tables/ablation_candidates_noSky.csv")
for d in ("tables", "data"):
    os.makedirs(d, exist_ok=True)


def clean(s):
    return (s.astype(str).str.strip().str.strip("b'\"").str.strip()
            .str.replace(" ", "", regex=False).str.upper())


def tier(conf):
    return np.where(conf >= 0.9, "A", np.where(conf >= 0.5, "B",
                    np.where(conf >= CONF, "C", "-")))


c = pd.read_csv(CAND)
need = ["Source_Name", "conf_fsrq_main", "conf_fsrq_variant", "conf_bll_main",
        "conf_bll_variant", "assignment_main", "assignment_variant", "GLAT", "status"]
miss = [x for x in need if x not in c.columns]
if miss:
    raise SystemExit(f"Missing columns in {CAND}: {miss}")

for v in ("main", "variant"):
    cf, cb = c[f"conf_fsrq_{v}"].values, c[f"conf_bll_{v}"].values
    c[f"tier_fsrq_{v}"] = tier(cf)
    c[f"tier_bll_{v}"] = tier(cb)
    # tier of the assigned class (ambiguous 'both' sources are reported separately)
    a = c[f"assignment_{v}"]
    c[f"tier_{v}"] = np.where(a == "FSRQ-like", c[f"tier_fsrq_{v}"],
                      np.where(a == "BL Lac-like", c[f"tier_bll_{v}"], "-"))

rows = []
for v in ("main", "variant"):
    for cls in ("FSRQ-like", "BL Lac-like"):
        sub = c[c[f"assignment_{v}"] == cls]
        for t in ("A", "B", "C"):
            rows.append({"run": "main" if v == "main" else "noSky", "class": cls,
                         "tier": t, "n": int((sub[f"tier_{v}"] == t).sum())})
tt = pd.DataFrame(rows)
tt.to_csv("tables/nosky_tiers.csv", index=False)
print("Candidates per class and tier (all BCUs in the ablation file):")
print(tt.pivot_table(index=["class", "tier"], columns="run", values="n").to_string())

g = c[c["status"] == "gained"]
print(f"\nGained candidates: {len(g)}")
gt = g.groupby(["assignment_variant", "tier_variant"]).size().rename("n").reset_index()
gt.to_csv("tables/nosky_gained_tiers.csv", index=False)
print(gt.to_string(index=False))
print("Share of gained that are tier C: "
      f"{(g['tier_variant'] == 'C').mean():.2f}; tier A: {(g['tier_variant'] == 'A').mean():.2f}")

both = c[(c["assignment_main"] == c["assignment_variant"]) &
         c["assignment_main"].isin(["FSRQ-like", "BL Lac-like"])]
print(f"\nCandidates with the same class in both runs: {len(both)}; tier shift "
      "(rows: main tier, columns: noSky tier):")
print(pd.crosstab(both["tier_main"], both["tier_variant"]).to_string())

pl = c["GLAT"].abs() < 10
for v, lab in (("main", "main"), ("variant", "noSky")):
    cand = c[f"assignment_{v}"].isin(["FSRQ-like", "BL Lac-like", "both"])
    print(f"{lab}: candidate share |b|<10 = {cand[pl].mean():.3f}, "
          f"|b|>=10 = {cand[~pl].mean():.3f}")

# ---- hybrid score file for Step 9 (T1 only) ----
if os.path.exists(OOF):
    o = pd.read_csv(OOF)
    txt = [x for x in o.columns if not pd.api.types.is_numeric_dtype(o[x])
           and x not in ("class_label", "bcu_assignment")]
    name = os.environ.get("NAME_COL") or (txt[0] if txt else None)
    if name is None:
        raise SystemExit("No name column in OOF_CSV; set NAME_COL.")
    o["_k"] = clean(o[name])
    c["_k"] = clean(c["Source_Name"])
    m = c.drop_duplicates("_k").set_index("_k")
    hit = o["_k"].isin(m.index)
    o["conf_fsrq_Step4B"] = o["conf_fsrq"]
    o["conf_bll_Step4B"] = o["conf_bll"]
    o.loc[hit, "conf_fsrq"] = o.loc[hit, "_k"].map(m["conf_fsrq_variant"]).values
    o.loc[hit, "conf_bll"] = o.loc[hit, "_k"].map(m["conf_bll_variant"]).values
    if "bcu_assignment" in o.columns:
        o.loc[hit, "bcu_assignment"] = o.loc[hit, "_k"].map(m["assignment_variant"]).values
    o.drop(columns="_k").to_csv("data/oof_scores_noSky_bcu.csv", index=False)
    print(f"\nWrote data/oof_scores_noSky_bcu.csv: {int(hit.sum())} BCU rows now carry "
          f"noSky scores; {int((~hit).sum())} rows (known sources) keep Step 4B scores.")
else:
    print(f"\n{OOF} not found; hybrid score file not written.")
print("Step 11 done.")
