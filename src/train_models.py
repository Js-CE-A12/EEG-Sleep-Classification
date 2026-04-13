"""
train_models.py
Phase 4: Subject-wise k-fold training for sleep stage classification.

Models:
  1. SVM with RBF kernel        (subsample 15K epochs/fold — SVM is O(n²))
  2. Random Forest / bagging
  3. XGBoost (gradient boosting)

Protocol:
  - Subject-wise 5-fold CV: all epochs from a subject go to train OR test, never both
  - StandardScaler fit on training fold only  →  no data leakage
  - Class imbalance: class_weight='balanced' for SVM/RF; scale_pos_weight for XGBoost
  - Fixed seed everywhere for reproducibility

Outputs (data/results/):
  - fold_metrics.csv           per-fold metrics for every model
  - summary_metrics.csv        mean ± std across folds
  - confusion_<model>.png      normalized confusion matrix (sum over folds)
  - feature_importance_<m>.png top-20 features for tree-based models
  - run_config.json            hyperparameters, seed, partitions used

Usage:
    python src/train_models.py
"""

import os
import glob
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
)
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] xgboost not found — using GradientBoostingClassifier instead.")

# ── config ─────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
SVM_MAX_N  = 15_000      # SVM is O(n²): subsample training data per fold

np.random.seed(SEED)

# resolve paths relative to this file so the script works regardless of
# which directory it is launched from (terminal, Code Runner, Docker, etc.)
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SRC_DIR)

FEATURE_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features")
RESULTS_DIR = os.path.join(_ROOT_DIR, "data", "results")

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES   = len(CLASS_NAMES)


# ── model definitions ──────────────────────────────────────────────────────────
def get_models():
    """
    Returns {name: model}.
    Hyperparameters chosen based on EEG sleep-staging literature (see report).
    All values are logged to run_config.json for reproducibility.
    """
    models = {}

    # ── SVM-RBF ────────────────────────────────────────────────────────────────
    # C=10        : moderate margin penalty; lets a few points be misclassified
    #               for a cleaner hyperplane. Too high → overfit; too low → underfit.
    # gamma=scale : 1 / (n_features * X.var()) — adapts to the feature scale
    # class_weight=balanced : minority classes (N1 = 1%) get higher penalty,
    #               forcing the model to not just predict the majority class
    # ovr         : one-vs-rest multiclass strategy (5 binary classifiers)
    models["SVM_RBF"] = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=SEED,
    )

    # ── Random Forest ──────────────────────────────────────────────────────────
    # n_estimators=200          : enough trees for stable variance reduction
    # max_features=sqrt         : standard for classification (reduces correlation
    #                             between trees → better ensemble diversity)
    # class_weight=balanced_subsample : recomputes weights per bootstrap sample
    #                             (more robust than global 'balanced' for RF)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )

    # ── XGBoost / GradientBoosting ─────────────────────────────────────────────
    # learning_rate=0.1  : shrinkage — each tree contributes 10% of its prediction.
    #                      Smaller = more trees needed but less overfitting.
    # subsample=0.8      : use 80% of data per tree (adds randomness like RF)
    # colsample=0.8      : use 80% of features per tree
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
        )
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
        )

    return models


# ── data loading ───────────────────────────────────────────────────────────────
def load_data():
    """
    Load all feature CSVs. Returns:
      X       : (n_epochs, n_features) float32
      y       : (n_epochs,)            int  — sleep stage label 0-4
      groups  : (n_epochs,)            str  — subject ID per epoch (for GroupKFold)
      feature_names : list of str
    """
    csv_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs in {FEATURE_DIR}. Run feature_extraction.py first."
        )

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

    print(f"Subjects : {len(csv_files)}")
    print(f"Epochs   : {len(X):,}")
    print(f"Features : {len(feature_cols)}")
    print("Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        n = (y == i).sum()
        print(f"  {name} : {n:6,}  ({100*n/len(y):.1f}%)")

    return X, y, groups, feature_cols


# ── stratified subsample for SVM ──────────────────────────────────────────────
def subsample_for_svm(X_train, y_train, max_n=SVM_MAX_N, seed=SEED):
    """
    SVM does not scale to hundreds of thousands of samples (O(n²) kernel matrix).
    Take a class-stratified subsample of the training fold so each class is
    represented proportionally but capped at max_n total epochs.
    """
    if len(X_train) <= max_n:
        return X_train, y_train

    classes, counts = np.unique(y_train, return_counts=True)
    fractions = counts / counts.sum()
    n_per_class = np.maximum(1, (fractions * max_n).astype(int))

    idx_list = []
    rng = np.random.RandomState(seed)
    for cls, n in zip(classes, n_per_class):
        cls_idx = np.where(y_train == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False)
        idx_list.append(chosen)

    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name, fold):
    row = {
        "model"      : model_name,
        "fold"       : fold,
        "accuracy"   : accuracy_score(y_true, y_pred),
        "macro_f1"   : f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "kappa"      : cohen_kappa_score(y_true, y_pred),
    }
    for i, name in enumerate(CLASS_NAMES):
        row[f"f1_{name}"] = f1_score(
            y_true, y_pred, labels=[i], average="macro", zero_division=0
        )
    return row


# ── plots ──────────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, model_name):
    """Normalized confusion matrix summed over all folds."""
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}\n(normalized, summed over {N_FOLDS} folds)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_feature_importance(importances, feature_names, model_name):
    """Top-20 feature importance bar chart (averaged over folds)."""
    avg_imp = importances / N_FOLDS
    idx     = np.argsort(avg_imp)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        avg_imp[idx[::-1]],
        color="steelblue"
    )
    ax.set_xlabel("Mean Importance (over folds)")
    ax.set_title(f"Top-20 Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_metrics_comparison(df_folds):
    """Bar chart comparing macro-F1 across models."""
    summary = df_folds.groupby("model")["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary["model"], summary["mean"], yerr=summary["std"],
                  capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-score")
    ax.set_title(f"Model Comparison — Macro F1 (mean ± std, {N_FOLDS}-fold subject-wise CV)")
    for bar, val in zip(bars, summary["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── main training loop ─────────────────────────────────────────────────────────
def train_subject_wise(X, y, groups, feature_names):
    models      = get_models()
    gkf         = GroupKFold(n_splits=N_FOLDS)

    all_results   = []
    conf_matrices = {name: np.zeros((N_CLASSES, N_CLASSES), int) for name in models}
    importances   = {name: np.zeros(len(feature_names))          for name in models}
    partitions    = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        test_subjects  = np.unique(groups[test_idx]).tolist()
        train_subjects = np.unique(groups[train_idx]).tolist()
        partitions.append({
            "fold"           : fold + 1,
            "n_train_subjects": len(train_subjects),
            "n_test_subjects" : len(test_subjects),
            "test_subjects"  : test_subjects,
        })
        print(f"\n── Fold {fold+1}/{N_FOLDS} "
              f"| train={len(train_subjects)} subj  test={len(test_subjects)} subj"
              f"  ({len(test_idx):,} test epochs)")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fit scaler on TRAINING fold only — prevent data leakage
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        for name, model in models.items():
            # SVM needs a subsample — kernel matrix would be ~298K × 298K otherwise
            if name == "SVM_RBF":
                Xtr, ytr = subsample_for_svm(X_train, y_train)
                print(f"  {name:<18} (subsample {len(Xtr):,}/{len(X_train):,}) ...",
                      end=" ", flush=True)
            else:
                Xtr, ytr = X_train, y_train
                print(f"  {name:<18} ...", end=" ", flush=True)

            t0 = time.time()
            model.fit(Xtr, ytr)
            elapsed = time.time() - t0

            y_pred  = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, name, fold + 1)
            metrics["train_time_s"] = round(elapsed, 2)
            all_results.append(metrics)

            conf_matrices[name] += confusion_matrix(
                y_test, y_pred, labels=list(range(N_CLASSES))
            )

            if hasattr(model, "feature_importances_"):
                importances[name] += model.feature_importances_

            print(
                f"acc={metrics['accuracy']:.3f}  "
                f"macro-F1={metrics['macro_f1']:.3f}  "
                f"kappa={metrics['kappa']:.3f}  "
                f"N1-F1={metrics['f1_N1']:.3f}  "
                f"time={elapsed:.1f}s"
            )

    return all_results, conf_matrices, importances, partitions, list(models.keys())


# ── summary table ──────────────────────────────────────────────────────────────
def print_summary(df_folds, model_names):
    print("\n" + "=" * 65)
    print("SUMMARY  —  mean ± std across folds")
    print("=" * 65)
    metric_cols = ["accuracy", "macro_f1", "kappa",
                   "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    for name in model_names:
        sub = df_folds[df_folds["model"] == name]
        print(f"\n{name}")
        for col in metric_cols:
            print(f"  {col:<12}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")


# ── entry point ────────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. load
    print("Loading feature data...\n")
    X, y, groups, feature_names = load_data()

    # 2. train
    print(f"\nStarting {N_FOLDS}-fold subject-wise cross-validation  (seed={SEED})\n")
    all_results, conf_matrices, importances, partitions, model_names = \
        train_subject_wise(X, y, groups, feature_names)

    # 3. save fold-level CSV
    df_folds = pd.DataFrame(all_results)
    folds_path = os.path.join(RESULTS_DIR, "fold_metrics.csv")
    df_folds.to_csv(folds_path, index=False)
    print(f"\nSaved fold metrics      → {folds_path}")

    # 4. save summary CSV
    metric_cols = ["accuracy", "macro_f1", "kappa",
                   "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM", "train_time_s"]
    summary = df_folds.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary metrics   → {summary_path}")

    # 5. print summary
    print_summary(df_folds, model_names)

    # 6. plots
    print("\nGenerating plots...")
    for name in model_names:
        save_confusion_matrix(conf_matrices[name], name)
    for name in model_names:
        if importances[name].sum() > 0:
            save_feature_importance(importances[name], feature_names, name)
    save_metrics_comparison(df_folds)

    # 7. reproducibility log
    config = {
        "seed"              : SEED,
        "n_folds"           : N_FOLDS,
        "svm_subsample_max" : SVM_MAX_N,
        "feature_dir"       : FEATURE_DIR,
        "n_subjects"        : int(len(np.unique(groups))),
        "n_epochs"          : int(len(y)),
        "class_distribution": {
            CLASS_NAMES[i]: int((y == i).sum()) for i in range(N_CLASSES)
        },
        "models": {
            "SVM_RBF"         : {"C": 10, "gamma": "scale",
                                  "class_weight": "balanced", "kernel": "rbf"},
            "RandomForest"    : {"n_estimators": 200, "max_features": "sqrt",
                                  "class_weight": "balanced_subsample"},
            "XGBoost"         : {"n_estimators": 200, "max_depth": 6,
                                  "learning_rate": 0.1, "subsample": 0.8},
        },
        "partitions": partitions,
    }
    config_path = os.path.join(RESULTS_DIR, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run config        → {config_path}")
    print("\nDone.")


if __name__ == "__main__":
    run()
