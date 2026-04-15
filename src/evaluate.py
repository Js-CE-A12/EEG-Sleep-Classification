"""
evaluate.py
Phase 6: Statistical evaluation — Bootstrap CI, McNemar test, AUC-ROC.

Re-runs RF and XGBoost with the same subject-wise 5-fold split (same seed)
to collect per-epoch predictions, then applies statistical tests required
by the project instructions.

Tests performed:
  1. Bootstrap confidence intervals (95%) on accuracy, macro-F1, kappa
     for every model — answers "how certain are we about these numbers?"

  2. McNemar test between every pair of models — answers "is the difference
     between model A and model B statistically significant, or just noise?"

  3. AUC-ROC curves (one-vs-rest) for RF and XGBoost — answers
     "how well does each model separate each sleep stage from the rest?"

Outputs (data/results/):
  - predictions.csv             per-epoch: subject, fold, y_true, pred per model
  - bootstrap_ci.csv            95% CI for each model x metric
  - mcnemar_results.csv         test statistic and p-value for each model pair
  - auc_roc.csv                 per-class AUC for RF and XGBoost
  - bootstrap_ci.png            CI bar chart
  - roc_curves_<model>.png      ROC curve per model (one curve per class)

Usage:
    python src/evaluate.py
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_curve, auc
)
from scipy.stats import chi2
from xgboost import XGBClassifier

# ── config (must match train_models.py) ───────────────────────────────────────
SEED         = 42
N_FOLDS      = 5
N_BOOTSTRAP  = 1000     # number of bootstrap resamples for CI
CI_LEVEL     = 0.95     # confidence level

np.random.seed(SEED)

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SRC_DIR)
FEATURE_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features")
RESULTS_DIR = os.path.join(_ROOT_DIR, "data", "results")

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES   = len(CLASS_NAMES)


# ── models ─────────────────────────────────────────────────────────────────────
def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt",
            min_samples_leaf=1, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=SEED, verbosity=0,
        ),
    }


# ── data loading ───────────────────────────────────────────────────────────────
def load_data():
    csv_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.csv")))
    frames = []
    for path in csv_files:
        subject_id = os.path.basename(path).replace(".csv", "")
        df = pd.read_csv(path)
        df["subject"] = subject_id
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in data.columns if c not in ("label", "subject")]
    X      = data[feature_cols].values.astype(np.float32)
    y      = data["label"].values.astype(int)
    groups = data["subject"].values
    print(f"Loaded {len(csv_files)} subjects | {len(X):,} epochs | {len(feature_cols)} features")
    return X, y, groups


# ── collect per-epoch predictions ─────────────────────────────────────────────
def collect_predictions(X, y, groups):
    """
    Re-run RF and XGBoost with the exact same splits (same SEED).
    Returns a DataFrame with one row per test epoch:
      subject, fold, y_true, pred_RF, pred_XGBoost,
      proba_RF_0..4, proba_XGBoost_0..4
    """
    models = get_models()
    gkf    = GroupKFold(n_splits=N_FOLDS)
    rows   = []

    print(f"\nCollecting per-epoch predictions ({N_FOLDS} folds)...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_test          = groups[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        preds  = {}
        probas = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds[name]  = model.predict(X_test)
            probas[name] = model.predict_proba(X_test)   # (n_test, 5)

        for i in range(len(y_test)):
            row = {
                "subject" : g_test[i],
                "fold"    : fold + 1,
                "y_true"  : y_test[i],
            }
            for name in models:
                row[f"pred_{name}"] = preds[name][i]
                for c in range(N_CLASSES):
                    row[f"proba_{name}_{CLASS_NAMES[c]}"] = probas[name][i, c]
            rows.append(row)

        print("done.")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "predictions.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    return df


# ── bootstrap confidence intervals ────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """
    Estimate CI for a scalar metric using bootstrap resampling.

    Procedure:
      1. Draw n_boot samples of size n WITH replacement from (y_true, y_pred)
      2. Compute the metric on each resample
      3. Return the (1-ci)/2 and (1+ci)/2 percentiles

    This is the percentile bootstrap — simple and robust for classification metrics.
    """
    rng    = np.random.RandomState(SEED)
    n      = len(y_true)
    scores = np.zeros(n_boot)
    for b in range(n_boot):
        idx      = rng.choice(n, size=n, replace=True)
        scores[b] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    return np.percentile(scores, [100 * alpha, 100 * (1 - alpha)])


def run_bootstrap(df_preds, model_names):
    """Compute 95% CI for accuracy, macro-F1, kappa for each model."""
    print(f"\nBootstrap CI ({N_BOOTSTRAP} resamples, {int(CI_LEVEL*100)}% CI)...")

    metric_fns = {
        "accuracy" : accuracy_score,
        "macro_f1" : lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "kappa"    : cohen_kappa_score,
    }

    rows = []
    for name in model_names:
        y_true = df_preds["y_true"].values
        y_pred = df_preds[f"pred_{name}"].values
        for metric, fn in metric_fns.items():
            point_est = fn(y_true, y_pred)
            lo, hi    = bootstrap_ci(y_true, y_pred, fn)
            rows.append({
                "model"      : name,
                "metric"     : metric,
                "estimate"   : round(point_est, 4),
                "ci_lower"   : round(lo, 4),
                "ci_upper"   : round(hi, 4),
                "ci_width"   : round(hi - lo, 4),
            })
            print(f"  {name:<15} {metric:<12}: {point_est:.4f}  "
                  f"[{lo:.4f}, {hi:.4f}]")

    df_ci = pd.DataFrame(rows)
    path  = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    df_ci.to_csv(path, index=False)
    print(f"Saved {path}")
    return df_ci


# ── McNemar test ───────────────────────────────────────────────────────────────
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar test: are two classifiers making DIFFERENT errors?

    Contingency table:
      b = model A correct, model B wrong   (A wins)
      c = model A wrong,   model B correct (B wins)

    H0: b == c  (models make the same types of errors)
    Statistic = (|b - c| - 1)^2 / (b + c)  — with continuity correction
    Under H0 this follows chi-squared with df=1.

    p < 0.05: the difference in errors is statistically significant.
    p >= 0.05: the difference could be random.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum( correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a &  correct_b)   # A wrong, B right

    if (b + c) == 0:
        return 0.0, 1.0   # identical predictions

    stat  = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - chi2.cdf(stat, df=1)
    return round(stat, 4), round(p_val, 6)


def run_mcnemar(df_preds, model_names):
    """Run McNemar test for every pair of models."""
    print("\nMcNemar test (all model pairs)...")

    y_true = df_preds["y_true"].values
    rows   = []
    pairs  = [(a, b) for i, a in enumerate(model_names)
                     for b in model_names[i+1:]]

    for a, b in pairs:
        stat, p = mcnemar_test(
            y_true,
            df_preds[f"pred_{a}"].values,
            df_preds[f"pred_{b}"].values,
        )
        sig = "YES" if p < 0.05 else "NO"
        rows.append({
            "model_A"    : a,
            "model_B"    : b,
            "statistic"  : stat,
            "p_value"    : p,
            "significant": sig,
        })
        print(f"  {a} vs {b}: chi2={stat:.4f}  p={p:.6f}  significant={sig}")

    df_mc = pd.DataFrame(rows)
    path  = os.path.join(RESULTS_DIR, "mcnemar_results.csv")
    df_mc.to_csv(path, index=False)
    print(f"Saved {path}")
    return df_mc


# ── AUC-ROC curves ─────────────────────────────────────────────────────────────
def run_auc_roc(df_preds, model_names):
    """
    One-vs-rest AUC-ROC for each class and each model.
    Uses the predicted probabilities (not just the hard predictions).

    For each class c:
      - Treat it as a binary problem: class c vs all others
      - Use the model's predicted probability for class c
      - Compute the ROC curve and the AUC
    """
    print("\nAUC-ROC (one-vs-rest)...")

    y_true_bin = label_binarize(df_preds["y_true"].values,
                                classes=list(range(N_CLASSES)))
    auc_rows = []

    for name in model_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        model_aucs = {}

        for i, cls_name in enumerate(CLASS_NAMES):
            proba_col = f"proba_{name}_{cls_name}"
            y_score   = df_preds[proba_col].values
            y_bin     = y_true_bin[:, i]

            fpr, tpr, _ = roc_curve(y_bin, y_score)
            auc_val     = auc(fpr, tpr)
            model_aucs[cls_name] = round(auc_val, 4)

            ax.plot(fpr, tpr, label=f"{cls_name}  (AUC={auc_val:.3f})")
            auc_rows.append({"model": name, "class": cls_name, "auc": round(auc_val, 4)})

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {name} (one-vs-rest, all folds)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"roc_curves_{name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  {name}: {model_aucs}  -> saved {path}")

    df_auc = pd.DataFrame(auc_rows)
    path   = os.path.join(RESULTS_DIR, "auc_roc.csv")
    df_auc.to_csv(path, index=False)
    print(f"Saved {path}")
    return df_auc


# ── CI bar chart ───────────────────────────────────────────────────────────────
def save_ci_chart(df_ci):
    """Visualize bootstrap CI for macro-F1 per model."""
    sub = df_ci[df_ci["metric"] == "macro_f1"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    for idx, (_, row) in enumerate(sub.iterrows()):
        mid   = row["estimate"]
        lo    = row["ci_lower"]
        hi    = row["ci_upper"]
        color = colors[idx % len(colors)]
        ax.bar(row["model"], mid, color=color, alpha=0.8,
               label=row["model"])
        ax.errorbar(row["model"], mid,
                    yerr=[[mid - lo], [hi - mid]],
                    fmt="none", color="black", capsize=6, linewidth=2)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-score")
    ax.set_title(f"Macro F1 with {int(CI_LEVEL*100)}% Bootstrap CI\n"
                 f"({N_BOOTSTRAP} resamples, all test epochs pooled)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bootstrap_ci.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── entry point ────────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. load data
    X, y, groups = load_data()

    model_names = list(get_models().keys())

    # 2. collect per-epoch predictions (re-runs models with same seed/splits)
    df_preds = collect_predictions(X, y, groups)

    # 3. bootstrap CI
    df_ci = run_bootstrap(df_preds, model_names)

    # 4. McNemar test
    df_mc = run_mcnemar(df_preds, model_names)

    # 5. AUC-ROC
    df_auc = run_auc_roc(df_preds, model_names)

    # 6. CI chart
    print("\nGenerating CI chart...")
    save_ci_chart(df_ci)

    # 7. final summary
    print("\n" + "=" * 60)
    print("STATISTICAL EVALUATION SUMMARY")
    print("=" * 60)

    print("\nBootstrap 95% CI (macro-F1):")
    for _, row in df_ci[df_ci["metric"] == "macro_f1"].iterrows():
        print(f"  {row['model']:<15}: {row['estimate']:.4f}  "
              f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

    print("\nMcNemar test:")
    for _, row in df_mc.iterrows():
        print(f"  {row['model_A']} vs {row['model_B']}: "
              f"p={row['p_value']:.6f}  significant={row['significant']}")

    print("\nAUC-ROC per class:")
    pivot = df_auc.pivot(index="model", columns="class", values="auc")
    print(pivot.to_string())

    print("\nDone.")


if __name__ == "__main__":
    run()
