"""
edf_to_csv.py
Converts Sleep-EDF Expanded EDF files to per-subject CSV files.

Each row = one 30-second epoch (3000 samples at 100 Hz).
Columns: sample_0 ... sample_2999, label
Labels:  0=W, 1=N1, 2=N2, 3=N3, 4=REM  (-1 = unknown/movement, dropped)

Usage:
    python src/edf_to_csv.py --subset cassette
    python src/edf_to_csv.py --subset telemetry
    python src/edf_to_csv.py --subset all
"""

import argparse
import os
import glob

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(
    "data", "sleep-edf",
    "sleep-edf-database-expanded-1.0.0"
)
SUBSETS = {
    "cassette": os.path.join(BASE, "sleep-cassette"),
    "telemetry": os.path.join(BASE, "sleep-telemetry"),
}
OUTPUT_DIR = os.path.join("data", "processed", "sleep-edf")

# ── label mapping ─────────────────────────────────────────────────────────────
STAGE_MAP = {
    "Sleep stage W":  0,
    "Sleep stage 1":  1,
    "Sleep stage 2":  2,
    "Sleep stage 3":  3,
    "Sleep stage 4":  3,   # merge N3/N4 as per AASM
    "Sleep stage R":  4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

EEG_CHANNEL = "EEG Fpz-Cz"
SFREQ       = 100          # Hz
EPOCH_SEC   = 30
EPOCH_SAMP  = SFREQ * EPOCH_SEC   # 3000


def load_hypnogram(hyp_path: str) -> np.ndarray:
    """Returns array of integer labels, one per 30-second epoch."""
    ann = mne.read_annotations(hyp_path)
    labels = []
    for desc, duration in zip(ann.description, ann.duration):
        stage = STAGE_MAP.get(desc, -1)
        n_epochs = int(round(duration / EPOCH_SEC))
        labels.extend([stage] * n_epochs)
    return np.array(labels)


def load_eeg(psg_path: str) -> np.ndarray:
    """Returns raw EEG signal from Fpz-Cz channel, shape (n_samples,)."""
    raw = mne.io.read_raw_edf(psg_path, include=[EEG_CHANNEL], preload=True)
    # resample if needed
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ)
    signal = raw.get_data()[0]   # shape (n_samples,)
    return signal


def process_subject(psg_path: str, hyp_path: str, subject_id: str) -> pd.DataFrame:
    signal = load_eeg(psg_path)
    labels = load_hypnogram(hyp_path)

    n_epochs_signal = len(signal) // EPOCH_SAMP
    n_epochs = min(n_epochs_signal, len(labels))

    rows = []
    for i in range(n_epochs):
        label = labels[i]
        if label == -1:          # skip unknown/movement
            continue
        epoch = signal[i * EPOCH_SAMP : (i + 1) * EPOCH_SAMP]
        rows.append((*epoch, label))

    cols = [f"sample_{i}" for i in range(EPOCH_SAMP)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    return df


def get_pairs(folder: str):
    """Yield (psg_path, hyp_path, subject_id) for each subject in folder."""
    psg_files = sorted(glob.glob(os.path.join(folder, "*-PSG.edf")))
    for psg_path in psg_files:
        base = os.path.basename(psg_path)
        subject_id = base[:6]   # e.g. SC4001
        # hypnogram has same prefix but different suffix
        hyp_candidates = glob.glob(
            os.path.join(folder, f"{subject_id}*Hypnogram.edf")
        )
        if not hyp_candidates:
            print(f"[WARN] No hypnogram found for {subject_id}, skipping.")
            continue
        yield psg_path, hyp_candidates[0], subject_id


def run(subsets: list[str]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for subset in subsets:
        folder = SUBSETS[subset]
        print(f"\n-- Processing subset: {subset} ({folder})")

        for psg_path, hyp_path, subject_id in get_pairs(folder):
            out_path = os.path.join(OUTPUT_DIR, f"{subject_id}.csv")
            if os.path.exists(out_path):
                print(f"  [SKIP] {subject_id} already processed.")
                continue

            print(f"  Processing {subject_id} ...", end=" ", flush=True)
            try:
                df = process_subject(psg_path, hyp_path, subject_id)
                df.to_csv(out_path, index=False)
                print(f"{len(df)} epochs saved -> {out_path}")
            except Exception as e:
                print(f"ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["cassette", "telemetry", "all"],
        default="cassette",
        help="Which subset to process (default: cassette)",
    )
    args = parser.parse_args()

    subsets = ["cassette", "telemetry"] if args.subset == "all" else [args.subset]
    run(subsets)
