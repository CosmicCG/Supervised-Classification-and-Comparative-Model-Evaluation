"""
Run every model on every task and write results/ + figures/.

    python src/run_experiments.py                 # all tasks
    python src/run_experiments.py --task nhanes_ibp
    python src/run_experiments.py --brfss-sample 40000   # faster BRFSS runs while developing

For each task:
  1. stratified 75/25 train/test split (seed 42)
  2. 5 models x 3 imbalance strategies (none / class_weight / smote)
     -> results/metrics_<task>.csv  (test precision/recall/F1/PR-AUC/ROC-AUC + train scores)
  3. 5-fold CV PR-AUC on the training set for small tasks (stability check)
  4. PR curves                         -> figures/pr_<task>.png
  5. validation curves (tree depth, k)  -> figures/vc_tree_depth_<task>.png, vc_knn_k_<task>.png
  6. permutation importance (RF) + LogReg coefficients
                                       -> results/importance_<task>.csv, figures/importance_<task>.png
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import prep_brfss  # noqa: E402
import prep_nhanes  # noqa: E402
import prep_vertebral  # noqa: E402
from models import HAS_IMBLEARN, MODEL_NAMES, SEED, make_pipeline  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"

TASKS = {
    "nhanes_arthritis": dict(file="nhanes_young.csv", target="arthritis_dx",
                             features=prep_nhanes.FEATURES_ARTHRITIS, categorical=["race_eth"],
                             label="NHANES 20-44: doctor-diagnosed arthritis"),
    "nhanes_ibp": dict(file="nhanes_young.csv", target="ibp",
                       features=prep_nhanes.FEATURES_IBP, categorical=["race_eth"],
                       label="NHANES 20-44: inflammatory back pain (Calin)"),
    "brfss_arthritis": dict(file="brfss2024_young.csv", target="arthritis_dx",
                            features=prep_brfss.FEATURES,
                            categorical=prep_brfss.CATEGORICAL_FEATURES,
                            label="BRFSS 2024, 18-44: arthritis"),
    "vertebral_abnormal": dict(file="vertebral.csv", target="abnormal",
                               features=prep_vertebral.FEATURES, categorical=[],
                               label="UCI Vertebral (n=310): abnormal spine"),
}
STRATEGIES = ["none", "class_weight", "smote"]
SMALL = 20_000  # tasks below this size also get 5-fold CV


def run_task(key: str, cfg: dict, brfss_sample: int | None) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(PROC / cfg["file"]).dropna(subset=[cfg["target"]])
    if brfss_sample and key.startswith("brfss") and len(df) > brfss_sample:
        df = df.sample(brfss_sample, random_state=SEED)
    X, y = df[cfg["features"]], df[cfg["target"]].astype(int)
    cat = cfg["categorical"]
    num = [c for c in cfg["features"] if c not in cat]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"\n=== {key}: n={len(df):,}  positives={y.sum():,} ({y.mean():.1%}) ===")

    rows, probas = [], {}
    for strat in STRATEGIES:
        for name in MODEL_NAMES:
            pipe = make_pipeline(name, num, cat, strat)
            if pipe is None:
                continue
            row, p_te = ev.evaluate(pipe, X_tr, y_tr, X_te, y_te)
            if len(df) < SMALL:
                cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                s = cross_val_score(make_pipeline(name, num, cat, strat), X_tr, y_tr, cv=cv,
                                    scoring="average_precision", n_jobs=-1)
                row.update(cv_pr_auc_mean=s.mean(), cv_pr_auc_sd=s.std())
            rows.append({"task": key, "model": name, "strategy": strat, **row})
            if strat == "none":
                probas[name] = p_te
            print(f"  {strat:12s} {name:12s} PR-AUC={row['test_pr_auc']:.3f}  "
                  f"recall={row['test_recall']:.2f}  F1={row['test_f1']:.2f}  "
                  f"gap={row['pr_auc_gap']:+.3f}")
    res = pd.DataFrame(rows)
    res.insert(3, "prevalence_test", y_te.mean())
    res.round(4).to_csv(RES / f"metrics_{key}.csv", index=False)
    ev.plot_pr_curves(y_te, probas, cfg["label"], FIG / f"pr_{key}.png")

    # --- bias-variance: complexity knobs (on a training subsample for speed)
    Xv, yv = X_tr, y_tr
    if len(Xv) > 30_000:
        Xv, _, yv, _ = train_test_split(X_tr, y_tr, train_size=30_000, stratify=y_tr, random_state=SEED)
    ev.plot_validation_curve(make_pipeline("DecisionTree", num, cat), Xv, yv,
                             "model__max_depth", list(range(1, 16)), "Tree max_depth",
                             f"Decision tree complexity\n{cfg['label']}",
                             FIG / f"vc_tree_depth_{key}.png").round(4).to_csv(
        RES / f"vc_tree_depth_{key}.csv", index=False)
    ks = [k for k in [1, 3, 5, 9, 15, 25, 41, 65, 101, 151, 251] if k < 0.5 * len(Xv)]
    ev.plot_validation_curve(make_pipeline("KNN", num, cat), Xv, yv, "model__n_neighbors", ks,
                             "k (neighbours)", f"KNN complexity\n{cfg['label']}",
                             FIG / f"vc_knn_k_{key}.png").round(4).to_csv(
        RES / f"vc_knn_k_{key}.csv", index=False)

    # --- feature importance
    rf = make_pipeline("RandomForest", num, cat).fit(X_tr, y_tr)
    Xi, yi = X_te, y_te
    if len(Xi) > 10_000:
        Xi, _, yi, _ = train_test_split(X_te, y_te, train_size=10_000, stratify=y_te, random_state=SEED)
    imp = ev.permutation_table(rf, Xi, yi)
    lr = make_pipeline("LogReg", num, cat).fit(X_tr, y_tr)
    coef = pd.Series(lr.named_steps["model"].coef_[0],
                     index=lr.named_steps["pre"].get_feature_names_out()).rename("logreg_coef")
    coef.index = coef.index.str.replace(r"^(num|cat)__", "", regex=True)
    imp = imp.merge(coef, left_on="feature", right_index=True, how="left")
    imp.round(4).to_csv(RES / f"importance_{key}.csv", index=False)
    ev.plot_importance(imp, f"Random forest permutation importance\n{cfg['label']}",
                       FIG / f"importance_{key}.png")
    print(f"  top features: {', '.join(imp.feature.head(5))}   ({time.time() - t0:.0f}s)")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS))
    p.add_argument("--brfss-sample", type=int, default=None)
    a = p.parse_args()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    if not HAS_IMBLEARN:
        print("imbalanced-learn not installed -> SMOTE rows skipped")
    keys = [a.task] if a.task else [k for k in TASKS if (PROC / TASKS[k]["file"]).exists()]
    allres = pd.concat([run_task(k, TASKS[k], a.brfss_sample) for k in keys])
    if not a.task:
        allres.round(4).to_csv(RES / "all_metrics.csv", index=False)
    print("\nsaved results/ and figures/")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
