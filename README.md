# Fermi 4FGL-DR4 BCU classification with PU learning

Companion data for "Classification of Fermi Blazars of Uncertain Type Using
Positive-Unlabelled Machine Learning".

`bcu_classification_all.csv`: 1623 BCUs from 4FGL-DR4, seven-feature run
(100 repeated five-fold splits, five PU classifiers).

- assignment: FSRQ-like, BL Lac-like, ambiguous (candidate in both modes), no class
- tier: A (confidence >= 0.9), B (>= 0.5), C (>= 0.1) for the assigned class
- conf_fsrq, conf_bll: fraction of repeats in which at least 3 of 5 classifiers
  flag the source in FSRQ mode / BL Lac mode (a stability measure, not a probability)
- conf_assigned: confidence of the assigned class; conf_assigned_ge4: same with a
  vote threshold of 4
- n_models_agree and the fsrq_* / bll_* columns (per-classifier fractions) are
  given for single-class candidates only and are blank otherwise
- the last seven columns are the input features; log_Flux_Peak is blank where
  4FGL-DR4 gives no peak flux

Scripts: `11__nosky_tiers_and_scores.py` (tiers and scores for the seven-feature
run), `12__make_release_table.py` (builds the table). `pu_params.json` holds the
classifier hyperparameters.
