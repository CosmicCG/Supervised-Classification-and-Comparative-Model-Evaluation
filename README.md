# Supervised Classification and Comparative Model Evaluation

**Predicting arthritis and inflammatory back pain in young adults.** This project builds and compares machine learning classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest) for arthritis and inflammatory back-pain indicators.

Team: Mamoudou Balde, Donovan Okwonna

## Repository layout

```
src/        download, cleaning, models, evaluation, experiment runner
tests/      synthetic-data checks for the cleaning logic
results/    metric tables (CSV)
figures/    PR curves, bias-variance curves, feature importance
notebooks/  exploration, one owner per notebook
report/     final report + slides
data/       NOT committed; created by src/download_data.py
```

## Quick start

```bash
pip install -r requirements.txt
python src/download_data.py          # ~110 MB download; BRFSS unzips to ~1 GB
python src/prep_nhanes.py            # -> data/processed/nhanes_young.csv
python src/prep_vertebral.py         # -> data/processed/vertebral.csv
python src/prep_brfss.py             # -> data/processed/brfss2024_young.csv (takes a few minutes)
python tests/test_prep_synthetic.py  # optional: checks the cleaning logic on fake data
python src/run_experiments.py        # all models x all tasks -> results/ + figures/ (~5-10 min)
```

While developing, `python src/run_experiments.py --task nhanes_ibp` runs one task, and
`--brfss-sample 40000` makes BRFSS runs faster.

Each prep script prints row counts and class balance. Save that output, because it is the first table in your report.
If a script prints `warning: column X not found`, CDC named that variable differently. Check the codebook link below and fix the name in the script.

Add `data/` to `.gitignore`. The raw files are too big for GitHub.

## Data sources

| Dataset | What we use it for | Files | Docs |
|---|---|---|---|
| **NHANES 2009–2010** | Inflammatory back pain (IBP) and doctor-diagnosed arthritis, ages **20–44**. This was the only NHANES cycle with the IBP questionnaire, and it was asked only of ages 20–69. | `DEMO_F`, `ARQ_F` (back pain/IBP), `MCQ_F` (arthritis dx), `BMX_F` (BMI), `ARX_F` (spinal mobility exam), `SMQ_F` (smoking) | [ARQ_F codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/ARQ_F.htm) · [questionnaire PDF](https://wwwn.cdc.gov/nchs/data/nhanes/public/2009/questionnaires/arq_f.pdf) · [MCQ_F](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/MCQ_F.htm) · [ARX_F](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/ARX_F.htm) |
| **BRFSS 2024** | Large-scale arthritis prediction, ages **18–44** (`_AGEG5YR` 1–5) | `LLCP2024XPT.zip` (457,670 respondents, all ages) | [2024 data page](https://www.cdc.gov/brfss/annual_data/annual_2024.html) · [codebook zip](https://www.cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip) · [variable layout](https://www.cdc.gov/brfss/annual_data/2024/llcp_varlayout_24_onecolumn.html) |
| **UCI Vertebral Column** | Small-sample (n=310) comparison | `column_2C.dat`, `column_3C.dat` | [UCI page](https://archive.ics.uci.edu/dataset/212/vertebral+column) |

BRFSS 2025 also uses `HAVARTH4`. To use it, run `--brfss-year 2025` / `--year 2025` once CDC posts the file.

## Targets

| File | Target | Definition |
|---|---|---|
| `nhanes_young.csv` | `arthritis_dx` | `MCQ160A == 1`: a doctor ever said you had arthritis |
| `nhanes_young.csv` | `ibp` | Approximate **Calin criteria**: chronic back pain ≥3 months, plus at least 4 of 5: onset before 40, gradual onset, duration ≥3 months, morning stiffness >30 min, improves with exercise. Built from ARQ022/024/025/040/080 at back sites. |
| `brfss2024_young.csv` | `arthritis_dx` | `HAVARTH4 == 1`: told you have arthritis, RA, gout, lupus, or fibromyalgia |
| `vertebral.csv` | `abnormal` / `label_3c` | Abnormal (hernia or spondylolisthesis) vs normal |

## Feature sets: avoid leakage

- **NHANES `ibp`:** use `prep_nhanes.FEATURES_IBP` only. The target is built from the ARQ back-pain answers, so using those answers as features would just let the model rebuild the rule. The test script checks this.
- **NHANES `arthritis_dx`:** use `prep_nhanes.FEATURES_ARTHRITIS`. Back-pain symptoms are allowed here, but never `MCQ180A` (age at diagnosis) or `MCQ191` (arthritis type).
- **BRFSS:** use `prep_brfss.FEATURES`. Never use `ARTHEXER` or other arthritis-module items, because only people who already said yes were asked them. `difficulty_walking` is kept apart in `OPTIONAL_FEATURES`. It is often a *result* of arthritis, so run models with and without it.

## Things to mention in the report

1. **Small positive counts in NHANES.** Ages 20–44 give a few thousand people, and IBP and arthritis are both rare at those ages. Expect a few hundred positives at most. Use stratified splits, stratified CV, and PR-AUC.
2. **The IBP label is an approximation.** `ARQ025` (speed of onset) was skipped for people whose pain is still ongoing, so missing answers count as "criterion not met". Compare your IBP rate with published NHANES 2009–10 IBP prevalence (Weisman et al., 2013).
3. **Survey weights.** Both surveys oversample some groups. Use the weights (`weight`, `weight_interview`/`weight_mec`) for prevalence numbers. For model training, unweighted is standard. Passing `sample_weight` is an optional extra experiment.
4. **The vertebral dataset is not an arthritis dataset.** It has no age and the labels are hernia/spondylolisthesis. Frame it as "how do the models behave with n=310?"
5. **Arthritis in BRFSS is self-reported** and includes gout, lupus, and fibromyalgia, not just inflammatory arthritis.

## Modelling (src/models.py, src/evaluate.py, src/run_experiments.py)

- The same preprocessing for every model: median impute + scale for numeric features, mode impute + one-hot for categorical ones.
- Models: LogReg, DecisionTree (max_depth 6), KNN (k=25), GaussianNB, RandomForest (300 trees).
- Imbalance strategies: `none`, `class_weight="balanced"` (LogReg/Tree/RF only), and `smote` (needs imbalanced-learn; applied to training data only).
- Outputs per task:
  - `results/metrics_<task>.csv`: test precision/recall/F1 at a 0.5 threshold, PR-AUC, ROC-AUC, train PR-AUC, the train-test gap, and 5-fold CV for the small tasks
  - `figures/pr_<task>.png`: precision-recall curves
  - `figures/vc_tree_depth_<task>.png`, `figures/vc_knn_k_<task>.png`: bias-variance curves
  - `results/importance_<task>.csv`, `figures/importance_<task>.png`: RF permutation importance plus LogReg coefficients
