"""
hmm_smoother.py
Phase 5: HMM post-processing smoother for sleep stage sequences.

The HMM is NOT a standalone classifier here — it is a smoother.
It takes the per-epoch predictions from XGBoost (the best model) and corrects
biologically impossible transitions, e.g. N3 → W in a single step.

How it works:
  1. Re-run XGBoost with the same 5-fold subject-wise split (same seed → same folds)
  2. Per fold, estimate HMM parameters from the TRAINING data:
       π  (initial probs) : how often does each stage start a recording?
       A  (transition)    : P(stage j at t+1 | stage i at t) — from true labels
       B  (emission)      : P(XGBoost predicted j | true stage is i) — from train confusion
  3. Per test subject, run Viterbi decoding on XGBoost's prediction sequence
     → produces a smoother, temporally coherent sequence
  4. Compare metrics before vs after smoothing
  5. Save transition matrix heatmap and updated confusion matrix

Outputs (data/results/):
  - hmm_fold_metrics.csv        per-fold accuracy/F1/kappa before & after HMM
  - hmm_summary_metrics.csv     mean ± std comparison
  - transition_matrix.png       learned A matrix heatmap
  - confusion_HMM.png           confusion matrix after HMM smoothing

Usage:
    python src/hmm_smoother.py
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from xgboost import XGBClassifier

# ── config (must match train_models.py exactly) ────────────────────────────────
SEED     = 42
N_FOLDS  = 5
np.random.seed(SEED)

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SRC_DIR)
FEATURE_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features")
RESULTS_DIR = os.path.join(_ROOT_DIR, "data", "results")

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES   = len(CLASS_NAMES)

# XGBoost config — identical to train_models.py
XGBOOST_PARAMS = dict(
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
    return X, y, groups


# ── HMM parameter estimation ───────────────────────────────────────────────────
def estimate_hmm_params(y_true, y_pred, groups, smoothing=1e-6):
    """
    Estimate π, A, B from the TRAINING fold only.

    π[i]   = P(recording starts in stage i)
             → look at the first epoch of each subject
    A[i,j] = P(next stage is j | current stage is i)
             → count consecutive true-label pairs within each subject
    B[i,j] = P(model predicts j | true stage is i)
             → normalized confusion matrix of XGBoost on training data

    smoothing: small value added to all counts before normalizing
               prevents zero probabilities that would make Viterbi fail
    """
    # ── π: initial state probabilities ────────────────────────────────────────
    pi = np.zeros(N_CLASSES) + smoothing
    for subj in np.unique(groups):
        mask = groups == subj
        first_label = y_true[mask][0]   # first epoch of this subject
        pi[first_label] += 1
    pi /= pi.sum()

    # ── A: transition matrix ───────────────────────────────────────────────────
    # Count transitions from true labels, within each subject only
    # (no transition counted between last epoch of S1 and first of S2)
    A = np.zeros((N_CLASSES, N_CLASSES)) + smoothing
    for subj in np.unique(groups):
        mask  = groups == subj
        labels = y_true[mask]
        for t in range(len(labels) - 1):
            A[labels[t], labels[t + 1]] += 1
    # normalize each row → probability distribution over next states
    A /= A.sum(axis=1, keepdims=True)

    # ── B: emission matrix (confusion matrix of the model) ────────────────────
    # B[true_stage, predicted_class] = P(model says j | truth is i)
    B = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).astype(float)
    B += smoothing
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B


# ── Viterbi algorithm ──────────────────────────────────────────────────────────
def viterbi(observations, pi, A, B):
    """
    Find the most likely sequence of hidden states given observations.

    observations : (T,) array of integers — model's predictions per epoch
    pi           : (N,) initial state probabilities
    A            : (N, N) transition matrix
    B            : (N, N) emission matrix  B[hidden, observed]

    Returns:
      path : (T,) array — smoothed sleep stage sequence

    Algorithm (log-space to avoid underflow):
      For each time step t and each state s:
        delta[t,s] = max probability of any path ending in state s at time t
        psi[t,s]   = which previous state gave that max probability

      Backtrack from the final best state to recover the full path.
    """
    T = len(observations)
    N = len(pi)

    # work in log-space: log(a*b) = log(a)+log(b) → no underflow
    log_pi = np.log(pi + 1e-300)
    log_A  = np.log(A  + 1e-300)
    log_B  = np.log(B  + 1e-300)

    delta = np.full((T, N), -np.inf)   # best log-prob path ending at (t, s)
    psi   = np.zeros((T, N), dtype=int)  # backpointer

    # initialization: t=0
    delta[0] = log_pi + log_B[:, observations[0]]

    # recursion: t=1..T-1
    for t in range(1, T):
        for s in range(N):
            # score of coming from each previous state
            scores = delta[t - 1] + log_A[:, s]
            best_prev      = np.argmax(scores)
            psi[t, s]      = best_prev
            delta[t, s]    = scores[best_prev] + log_B[s, observations[t]]

    # backtrack
    path    = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


# ── metrics helper ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    row = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "macro_f1" : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa"    : cohen_kappa_score(y_true, y_pred),
    }
    for i, name in enumerate(CLASS_NAMES):
        row[f"f1_{name}"] = f1_score(
            y_true, y_pred, labels=[i], average="macro", zero_division=0
        )
    return row


# ── plots ──────────────────────────────────────────────────────────────────────
def save_transition_matrix(A_avg, title="Learned Transition Matrix (avg over folds)"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        A_avg, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Next stage")
    ax.set_ylabel("Current stage")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "transition_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_confusion_matrix(cm, model_name):
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} (normalized, all folds)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_comparison_chart(df):
    """Bar chart: macro-F1 before vs after HMM, per class."""
    classes = ["macro_f1"] + [f"f1_{n}" for n in CLASS_NAMES]
    labels  = ["Overall"] + CLASS_NAMES

    before = [df[df["stage"] == "before"][c].mean() for c in classes]
    after  = [df[df["stage"] == "after"][c].mean()  for c in classes]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, before, w, label="XGBoost raw",      color="#4C72B0")
    ax.bar(x + w/2, after,  w, label="XGBoost + HMM",    color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score Before vs After HMM Smoothing")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "hmm_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── main loop ──────────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    X, y, groups = load_data()

    gkf      = GroupKFold(n_splits=N_FOLDS)
    model    = XGBClassifier(**XGBOOST_PARAMS)

    all_results  = []
    cm_before    = np.zeros((N_CLASSES, N_CLASSES), int)
    cm_after     = np.zeros((N_CLASSES, N_CLASSES), int)
    A_accumulate = np.zeros((N_CLASSES, N_CLASSES))

    print(f"\nRunning {N_FOLDS}-fold subject-wise CV with HMM smoothing...\n")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"-- Fold {fold+1}/{N_FOLDS}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train         = groups[train_idx]
        g_test          = groups[test_idx]

        # scale — fit on train only
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # train XGBoost
        print("  Training XGBoost...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        print("done.")

        # estimate HMM parameters from training fold
        pi, A, B = estimate_hmm_params(y_train, y_pred_train, g_train)
        A_accumulate += A

        # apply Viterbi per test subject (temporal order matters)
        y_smoothed = np.empty_like(y_pred_test)
        for subj in np.unique(g_test):
            mask = g_test == subj
            raw_seq      = y_pred_test[mask]
            smoothed_seq = viterbi(raw_seq, pi, A, B)
            y_smoothed[mask] = smoothed_seq

        # metrics before HMM
        m_before = compute_metrics(y_test, y_pred_test)
        m_before.update({"fold": fold + 1, "stage": "before"})
        all_results.append(m_before)

        # metrics after HMM
        m_after = compute_metrics(y_test, y_smoothed)
        m_after.update({"fold": fold + 1, "stage": "after"})
        all_results.append(m_after)

        cm_before += confusion_matrix(y_test, y_pred_test,  labels=list(range(N_CLASSES)))
        cm_after  += confusion_matrix(y_test, y_smoothed,   labels=list(range(N_CLASSES)))

        print(
            f"  Before HMM → acc={m_before['accuracy']:.3f}  "
            f"macro-F1={m_before['macro_f1']:.3f}  N1={m_before['f1_N1']:.3f}"
        )
        print(
            f"  After  HMM → acc={m_after['accuracy']:.3f}  "
            f"macro-F1={m_after['macro_f1']:.3f}  N1={m_after['f1_N1']:.3f}"
        )

    # ── save results ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)

    fold_path = os.path.join(RESULTS_DIR, "hmm_fold_metrics.csv")
    df.to_csv(fold_path, index=False)
    print(f"\nSaved {fold_path}")

    metric_cols = ["accuracy", "macro_f1", "kappa",
                   "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    summary = df.groupby("stage")[metric_cols].agg(["mean", "std"]).round(4)
    summary_path = os.path.join(RESULTS_DIR, "hmm_summary_metrics.csv")
    summary.to_csv(summary_path)
    print(f"Saved {summary_path}")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — XGBoost  before vs after HMM smoothing")
    print("=" * 60)
    for stage in ["before", "after"]:
        sub = df[df["stage"] == stage]
        print(f"\n{stage.upper()} HMM:")
        for col in metric_cols:
            print(f"  {col:<12}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    A_avg = A_accumulate / N_FOLDS
    save_transition_matrix(A_avg)
    save_confusion_matrix(cm_before, "XGBoost_raw")
    save_confusion_matrix(cm_after,  "HMM_smoothed")
    save_comparison_chart(df)

    # ── print transition matrix interpretation ─────────────────────────────────
    print("\nTransition matrix (avg over folds):")
    print(f"{'':>6}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>8}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>6}", end="")
        for j in range(N_CLASSES):
            print(f"{A_avg[i,j]:>8.3f}", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    run()
