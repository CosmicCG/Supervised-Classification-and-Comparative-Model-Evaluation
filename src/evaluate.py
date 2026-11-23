"""Metrics + figures shared by all experiments."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, validation_curve  # noqa: E402

# Validated categorical palette (fixed order, one colour per model, never cycled)
MODEL_COLORS = {"LogReg": "#2a78d6", "DecisionTree": "#eb6834", "KNN": "#1baf7a",
                "NaiveBayes": "#eda100", "RandomForest": "#e87ba4"}
MODEL_STYLES = {"LogReg": "-", "DecisionTree": "--", "KNN": "-.",
                "NaiveBayes": ":", "RandomForest": "-"}
TRAIN_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 130,
})


def scores(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def evaluate(pipe, X_tr, y_tr, X_te, y_te) -> tuple[dict, np.ndarray]:
    """Fit, then score on train AND test (the gap is the overfitting signal)."""
    pipe.fit(X_tr, y_tr)
    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_te = pipe.predict_proba(X_te)[:, 1]
    tr, te = scores(y_tr, p_tr), scores(y_te, p_te)
    row = {f"test_{k}": v for k, v in te.items()}
    row.update({"train_f1": tr["f1"], "train_pr_auc": tr["pr_auc"],
                "pr_auc_gap": tr["pr_auc"] - te["pr_auc"]})
    return row, p_te


def plot_pr_curves(y_te, probas: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_te, p)
        ap = average_precision_score(y_te, p)
        ax.plot(rec, prec, color=MODEL_COLORS[name], ls=MODEL_STYLES[name],
                label=f"{name}  (PR-AUC {ap:.2f})")
    base = np.mean(y_te)
    ax.axhline(base, color=MUTED, lw=1, ls="--")
    ax.text(0.98, base - 0.015, f"no-skill = prevalence {base:.2f}", color=MUTED,
            ha="right", va="top", fontsize=8)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(loc="lower left" if base > 0.5 else "upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_validation_curve(pipe, X, y, param, values, xlabel, title, path: Path) -> pd.DataFrame:
    """Train vs cross-validated PR-AUC while one complexity knob changes."""
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    tr, va = validation_curve(pipe, X, y, param_name=param, param_range=values, cv=cv,
                              scoring="average_precision", n_jobs=-1)
    df = pd.DataFrame({"value": values, "train_pr_auc": tr.mean(1), "cv_pr_auc": va.mean(1),
                       "cv_sd": va.std(1)})
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(values, df.train_pr_auc, color=TRAIN_COLOR, marker="o", ms=4, label="Train")
    ax.plot(values, df.cv_pr_auc, color=TEST_COLOR, marker="s", ms=4, ls="--", label="Cross-validated")
    ax.fill_between(values, df.cv_pr_auc - df.cv_sd, df.cv_pr_auc + df.cv_sd,
                    color=TEST_COLOR, alpha=0.12, lw=0)
    best = df.loc[df.cv_pr_auc.idxmax()]
    ax.axvline(best.value, color=MUTED, lw=1, ls=":")
    ax.text(best.value, ax.get_ylim()[0], f" best = {best.value:g} ", color=MUTED, fontsize=8, va="bottom",
            ha="right" if best.value > np.median(values) else "left")
    ax.set(xlabel=xlabel, ylabel="PR-AUC", title=title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return df


def permutation_table(pipe, X_te, y_te, n_repeats=10) -> pd.DataFrame:
    r = permutation_importance(pipe, X_te, y_te, scoring="average_precision",
                               n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return (pd.DataFrame({"feature": X_te.columns, "importance": r.importances_mean,
                          "sd": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_importance(df: pd.DataFrame, title: str, path: Path, top=15) -> None:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 0.32 * len(d) + 1.2))
    ax.barh(d.feature, d.importance, xerr=d.sd, color=MODEL_COLORS["RandomForest"],
            height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Drop in PR-AUC when feature is shuffled", title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
