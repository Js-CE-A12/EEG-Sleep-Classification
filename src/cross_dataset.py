"""
cross_dataset.py
Phase 7: Cross-dataset generalization experiment.

Setup:
  SOURCE (train) : Sleep-EDF Cassette  (SC) — 153 healthy subjects, at-home recordings
  TARGET (test)  : Sleep-EDF Telemetry (ST) — 44 clinical patients, temazepam users

Why this is a real cross-dataset test:
  - Different recording environment (home vs. hospital)
  - Different population (healthy vs. medicated patients)
  - Different EEG characteristics (temazepam suppresses N3, alters spindle activity)
  - No subject overlap between SC and ST

Protocol:
  1. If ST features don't exist yet, run the full pipeline on ST raw EDFs automatically.
  2. Train XGBoost and RandomForest on ALL 153 SC subjects (no fold, no leakage).
  3. Evaluate on ALL 44 ST subjects.
  4. Compare cross-dataset performance against within-dataset (from summary_metrics.csv).
  5. Save results and plots.

Outputs (data/results/):
  - cross_dataset_metrics.csv       accuracy/F1/kappa on ST test set
  - cross_dataset_comparison.png    bar chart: within-dataset vs cross-dataset
  - confusion_cross_<model>.png     confusion matrix on ST subjects

Usage:
    python src/cross_dataset.py
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

import mne
mne.set_log_level("WARNING")

from scipy.signal import butter, filtfilt
from scipy.signal import welch
from scipy.stats import skew, kurtosis as sp_kurtosis
from collections import Counter
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
)
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

# source dataset (SC) — already processed
SC_FEATURES_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features")

# target dataset (ST) — will be processed if not present
ST_RAW_DIR      = os.path.join(_ROOT_DIR, "data", "sleep-edf",
                               "sleep-edf-database-expanded-1.0.0",
                               "sleep-telemetry")
ST_CSV_DIR      = os.path.join(_ROOT_DIR, "data", "processed", "sleep-telemetry")
ST_CLEAN_DIR    = os.path.join(_ROOT_DIR, "data", "processed", "sleep-telemetry-clean")
ST_FEATURES_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features-st")

RESULTS_DIR = os.path.join(_ROOT_DIR, "data", "results")

# ── shared config ──────────────────────────────────────────────────────────────
SEED        = 42
SFREQ       = 100
EPOCH_SAMP  = 3000
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES   = len(CLASS_NAMES)
np.random.seed(SEED)

STAGE_MAP = {
    "Sleep stage W" : 0,
    "Sleep stage 1" : 1,
    "Sleep stage 2" : 2,
    "Sleep stage 3" : 3,
    "Sleep stage 4" : 3,
    "Sleep stage R" : 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}
EEG_CHANNEL = "EEG Fpz-Cz"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta" : (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# ST PIPELINE  (mirrors edf_to_csv → preprocess → feature_extraction)
# Only runs if ST features are not already present.
# ══════════════════════════════════════════════════════════════════════════════

# ── Step A: EDF → CSV ──────────────────────────────────────────────────────────
def edf_to_csv_st():
    os.makedirs(ST_CSV_DIR, exist_ok=True)
    psg_files = sorted(glob.glob(os.path.join(ST_RAW_DIR, "*-PSG.edf")))
    print(f"  Found {len(psg_files)} ST subjects.")

    for psg_path in psg_files:
        base       = os.path.basename(psg_path)
        subject_id = base[:6]
        out_path   = os.path.join(ST_CSV_DIR, f"{subject_id}.csv")
        if os.path.exists(out_path):
            continue

        hyp_candidates = glob.glob(
            os.path.join(ST_RAW_DIR, f"{subject_id}*Hypnogram.edf")
        )
        if not hyp_candidates:
            print(f"  [WARN] No hypnogram for {subject_id}, skipping.")
            continue

        print(f"  EDF->CSV  {subject_id}...", end=" ", flush=True)
        try:
            raw    = mne.io.read_raw_edf(psg_path, include=[EEG_CHANNEL], preload=True)
            if raw.info["sfreq"] != SFREQ:
                raw.resample(SFREQ)
            signal = raw.get_data()[0]

            ann    = mne.read_annotations(hyp_candidates[0])
            labels = []
            for desc, dur in zip(ann.description, ann.duration):
                stage = STAGE_MAP.get(desc, -1)
                labels.extend([stage] * int(round(dur / 30)))
            labels = np.array(labels)

            n_epochs = min(len(signal) // EPOCH_SAMP, len(labels))
            rows = []
            for i in range(n_epochs):
                lbl = labels[i]
                if lbl == -1:
                    continue
                epoch = signal[i * EPOCH_SAMP:(i + 1) * EPOCH_SAMP]
                rows.append((*epoch, lbl))

            cols = [f"sample_{i}" for i in range(EPOCH_SAMP)] + ["label"]
            pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
            print(f"{len(rows)} epochs.")
        except Exception as e:
            print(f"ERROR: {e}")


# ── Step B: preprocess ────────────────────────────────────────────────────────
def preprocess_st():
    os.makedirs(ST_CLEAN_DIR, exist_ok=True)
    nyq  = SFREQ / 2
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")

    for csv_path in sorted(glob.glob(os.path.join(ST_CSV_DIR, "*.csv"))):
        subject_id = os.path.basename(csv_path).replace(".csv", "")
        out_path   = os.path.join(ST_CLEAN_DIR, f"{subject_id}.csv")
        if os.path.exists(out_path):
            continue

        print(f"  Preprocess {subject_id}...", end=" ", flush=True)
        df     = pd.read_csv(csv_path)
        labels = df["label"].values
        epochs = df.drop(columns=["label"]).values.astype(np.float32)

        # bandpass
        filtered = filtfilt(b, a, epochs.flatten()).reshape(epochs.shape)

        # artifact removal
        global_std = filtered.std()
        mask       = np.abs(filtered).max(axis=1) < 6 * global_std
        filtered   = filtered[mask]
        labels     = labels[mask]

        # z-score per subject
        m, s = filtered.mean(), filtered.std()
        if s > 0:
            filtered = (filtered - m) / s

        cols = [f"sample_{i}" for i in range(EPOCH_SAMP)]
        out  = pd.DataFrame(filtered, columns=cols)
        out["label"] = labels
        out.to_csv(out_path, index=False)
        print(f"{len(out)} epochs.")


# ── Step C: feature extraction ────────────────────────────────────────────────
def _time_features(epochs):
    feats = {}
    feats["mean"]     = epochs.mean(axis=1)
    feats["std"]      = epochs.std(axis=1)
    feats["variance"] = epochs.var(axis=1)
    feats["skewness"] = skew(epochs, axis=1)
    feats["kurtosis"] = sp_kurtosis(epochs, axis=1)
    signs = np.sign(epochs)
    feats["zero_crossing_rate"] = (np.diff(signs, axis=1) != 0).sum(axis=1) / EPOCH_SAMP
    feats["line_length"] = np.abs(np.diff(epochs, axis=1)).mean(axis=1)
    d1, d2 = np.diff(epochs, n=1, axis=1), np.diff(epochs, n=2, axis=1)
    vx, vd1, vd2 = epochs.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mob = np.sqrt(vd1 / np.clip(vx, 1e-10, None))
    feats["hjorth_mobility"]   = mob
    feats["hjorth_complexity"] = (
        np.sqrt(vd2 / np.clip(vd1, 1e-10, None)) / np.clip(mob, 1e-10, None)
    )
    return pd.DataFrame(feats)


def _freq_features(epochs):
    freqs, psd = welch(epochs, fs=SFREQ, nperseg=256, axis=1)
    freq_res   = freqs[1] - freqs[0]
    total_pw   = psd.sum(axis=1) * freq_res
    feats, bpw = {}, {}
    for name, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bp  = psd[:, idx].sum(axis=1) * freq_res
        feats[f"{name}_power"]          = bp
        feats[f"{name}_relative_power"] = bp / np.clip(total_pw, 1e-10, None)
        bpw[name] = bp
    feats["delta_beta_ratio"]  = bpw["delta"] / np.clip(bpw["beta"],  1e-10, None)
    feats["theta_alpha_ratio"] = bpw["theta"] / np.clip(bpw["alpha"], 1e-10, None)
    psd_norm = psd / np.clip(psd.sum(axis=1, keepdims=True), 1e-10, None)
    feats["spectral_entropy"] = -(psd_norm * np.log2(psd_norm + 1e-10)).sum(axis=1)
    feats["peak_frequency"]   = freqs[psd.argmax(axis=1)]
    return pd.DataFrame(feats)


def _sample_entropy(epochs, m=2, r_factor=0.2):
    results = np.zeros(len(epochs))
    for i, epoch in enumerate(epochs):
        r = r_factor * epoch.std()
        if r == 0:
            continue
        x  = epoch[::10]
        n  = len(x)
        im  = np.arange(m)[None, :]   + np.arange(n - m)[:, None]
        im1 = np.arange(m+1)[None, :] + np.arange(n - m)[:, None]
        dm  = np.abs(x[im][:, None, :]  - x[im][None, :, :]).max(axis=2)
        dm1 = np.abs(x[im1][:, None, :] - x[im1][None, :, :]).max(axis=2)
        cm  = (dm  < r).sum() - (n - m)
        cm1 = (dm1 < r).sum() - (n - m)
        if cm > 0 and cm1 > 0:
            results[i] = -np.log(cm1 / cm)
    return results


def _permutation_entropy(epochs, order=3, delay=1):
    max_ent = np.log2(math.factorial(order))
    results = np.zeros(len(epochs))
    for i, epoch in enumerate(epochs):
        x = epoch[::10]
        n = len(x)
        if n < order:
            continue
        emb      = np.array([x[j:j + order] for j in range(0, n - order + 1, delay)])
        patterns = np.argsort(emb, axis=1)
        counts   = Counter(map(tuple, patterns))
        total    = sum(counts.values())
        probs    = np.array(list(counts.values())) / total
        pe       = -np.sum(probs * np.log2(probs + 1e-10))
        results[i] = pe / max_ent
    return results


def extract_features_st():
    os.makedirs(ST_FEATURES_DIR, exist_ok=True)
    for csv_path in sorted(glob.glob(os.path.join(ST_CLEAN_DIR, "*.csv"))):
        subject_id = os.path.basename(csv_path).replace(".csv", "")
        out_path   = os.path.join(ST_FEATURES_DIR, f"{subject_id}.csv")
        if os.path.exists(out_path):
            continue

        print(f"  Features  {subject_id}...", end=" ", flush=True)
        df     = pd.read_csv(csv_path)
        labels = df["label"].values
        epochs = df.drop(columns=["label"]).values.astype(np.float64)

        tf = _time_features(epochs)
        ff = _freq_features(epochs)
        se = _sample_entropy(epochs)
        pe = _permutation_entropy(epochs)

        result = pd.concat([tf, ff], axis=1)
        result["sample_entropy"]      = se
        result["permutation_entropy"] = pe
        result["label"]               = labels
        result.to_csv(out_path, index=False)
        print(f"{len(result)} epochs, {result.shape[1]-1} features.")


def prepare_st_pipeline():
    """Run the full ST pipeline if ST features don't exist yet."""
    st_csvs = glob.glob(os.path.join(ST_FEATURES_DIR, "*.csv"))
    if st_csvs:
        print(f"ST features already exist ({len(st_csvs)} subjects). Skipping pipeline.")
        return

    print("\nST features not found. Running full pipeline on Sleep Telemetry data...")
    print("\n[A] EDF -> CSV")
    edf_to_csv_st()
    print("\n[B] Preprocessing")
    preprocess_st()
    print("\n[C] Feature extraction")
    extract_features_st()
    print("\nST pipeline complete.")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-DATASET EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(features_dir, dataset_name):
    csv_files = sorted(glob.glob(os.path.join(features_dir, "*.csv")))
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
    print(f"  {dataset_name}: {len(csv_files)} subjects | {len(X):,} epochs")
    return X, y, groups, feature_cols


def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", n_jobs=-1,
            random_state=SEED, verbosity=0,
        ),
    }


def compute_metrics(y_true, y_pred, model_name, split):
    row = {
        "model"    : model_name,
        "split"    : split,
        "accuracy" : accuracy_score(y_true, y_pred),
        "macro_f1" : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa"    : cohen_kappa_score(y_true, y_pred),
    }
    for i, name in enumerate(CLASS_NAMES):
        row[f"f1_{name}"] = f1_score(
            y_true, y_pred, labels=[i], average="macro", zero_division=0
        )
    return row


def save_confusion_matrix(cm, title, filename):
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_comparison_chart(df_results, df_within):
    """
    Bar chart comparing macro-F1:
      within-dataset (from summary_metrics.csv) vs cross-dataset (ST test set)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    models  = df_results["model"].unique()
    x       = np.arange(len(models))
    w       = 0.35

    within_f1 = []
    cross_f1  = []
    for m in models:
        # within: mean macro_f1 from the SC 5-fold experiment
        row_w = df_within[df_within.index.get_level_values(0) == m]
        wf1   = float(row_w[("macro_f1", "mean")].values[0]) if len(row_w) else 0
        within_f1.append(wf1)
        cross_f1.append(
            float(df_results[df_results["model"] == m]["macro_f1"].values[0])
        )

    ax.bar(x - w/2, within_f1, w, label="Within-dataset (SC 5-fold)", color="#4C72B0")
    ax.bar(x + w/2, cross_f1,  w, label="Cross-dataset  (train SC, test ST)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Within-dataset vs Cross-dataset Performance\n"
                 "(SC = Cassette/healthy, ST = Telemetry/medicated)")
    ax.legend()

    for i, (w_val, c_val) in enumerate(zip(within_f1, cross_f1)):
        drop = w_val - c_val
        ax.text(i, max(w_val, c_val) + 0.02,
                f"drop={drop:+.3f}", ha="center", fontsize=9, color="darkred")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cross_dataset_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def run_cross_dataset(X_sc, y_sc, X_st, y_st):
    """Train on all SC, test on all ST. No k-fold — this is a single generalization test."""
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_sc)
    X_test  = scaler.transform(X_st)

    models  = get_models()
    results = []
    cms     = {}

    for name, model in models.items():
        print(f"  Training {name} on SC...", end=" ", flush=True)
        model.fit(X_train, y_sc)
        y_pred = model.predict(X_test)
        print("done.")

        row = compute_metrics(y_st, y_pred, name, "cross_dataset")
        results.append(row)
        cms[name] = confusion_matrix(y_st, y_pred, labels=list(range(N_CLASSES)))

        print(f"    acc={row['accuracy']:.3f}  macro-F1={row['macro_f1']:.3f}  "
              f"kappa={row['kappa']:.3f}  N1-F1={row['f1_N1']:.3f}")

    return pd.DataFrame(results), cms


# ── entry point ────────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. prepare ST features (runs full pipeline if not already done)
    prepare_st_pipeline()

    # 2. load both datasets
    print("\nLoading datasets...")
    X_sc, y_sc, _, _ = load_dataset(SC_FEATURES_DIR, "SC (Cassette)")
    X_st, y_st, _, _ = load_dataset(ST_FEATURES_DIR, "ST (Telemetry)")

    # 3. run cross-dataset experiment
    print("\nCross-dataset experiment: train on SC, test on ST")
    print("(No hyperparameter tuning on ST — strict generalization test)\n")
    df_results, cms = run_cross_dataset(X_sc, y_sc, X_st, y_st)

    # 4. save results CSV
    path = os.path.join(RESULTS_DIR, "cross_dataset_metrics.csv")
    df_results.to_csv(path, index=False)
    print(f"\nSaved {path}")

    # 5. load within-dataset results for comparison
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    df_within = pd.read_csv(summary_path, header=[0, 1], index_col=0)

    # 6. plots
    print("\nGenerating plots...")
    for name, cm in cms.items():
        save_confusion_matrix(
            cm,
            title=f"{name} — Cross-dataset (train SC, test ST)",
            filename=f"confusion_cross_{name}.png"
        )
    save_comparison_chart(df_results, df_within)

    # 7. print summary
    print("\n" + "=" * 65)
    print("CROSS-DATASET RESULTS  (train=SC cassette, test=ST telemetry)")
    print("=" * 65)
    cols = ["model", "accuracy", "macro_f1", "kappa",
            "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    print(df_results[cols].to_string(index=False))

    print("\nWithin-dataset macro-F1 (SC 5-fold) for reference:")
    for m in df_results["model"]:
        row = df_within[df_within.index.get_level_values(0) == m]
        if len(row):
            wf1 = float(row[("macro_f1", "mean")].values[0])
            cf1 = float(df_results[df_results["model"] == m]["macro_f1"].values[0])
            print(f"  {m:<15}: within={wf1:.4f}  cross={cf1:.4f}  "
                  f"drop={wf1-cf1:+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    run()
