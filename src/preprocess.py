"""
preprocess.py
Phase 2: Filter, normalize, and clean the raw epoch CSVs.

For each subject CSV in data/processed/sleep-edf/:
  1. Bandpass filter  : keep 0.5 - 40 Hz (real brain signal)
  2. Artifact removal : drop epochs where amplitude is too large (noise/movement)
  3. Normalize        : z-score per subject (mean=0, std=1)
  4. Save to          : data/processed/sleep-edf-clean/

Usage:
    python src/preprocess.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ── config ────────────────────────────────────────────────────────────────────
SFREQ         = 100          # Hz
EPOCH_SAMP    = 3000         # samples per epoch (30s * 100Hz)
LOWCUT        = 0.5          # Hz - removes slow drift
HIGHCUT       = 40.0         # Hz - removes muscle noise and powerline
ARTIFACT_STD  = 6            # epochs where any sample > 6 std are artifacts

INPUT_DIR  = os.path.join("data", "processed", "sleep-edf")
OUTPUT_DIR = os.path.join("data", "processed", "sleep-edf-clean")


# ── filter ────────────────────────────────────────────────────────────────────
def make_bandpass_filter(lowcut: float, highcut: float, sfreq: float, order: int = 4):
    """
    Creates a bandpass Butterworth filter.
    Butterworth = smooth filter, no ripple in the frequency response.
    order=4 means strong enough to cut noise but not distort the signal.
    """
    nyq = sfreq / 2                        # Nyquist frequency (max detectable = half of sfreq)
    low  = lowcut  / nyq                   # normalize to 0-1 range
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_filter(signal: np.ndarray, b, a) -> np.ndarray:
    """
    Applies the bandpass filter to a 1D signal.
    filtfilt = applies the filter twice (forward + backward) to avoid phase shift.
    Phase shift would time-shift the signal, which would misalign labels.
    """
    return filtfilt(b, a, signal)


# ── artifact detection ────────────────────────────────────────────────────────
def remove_artifacts(epochs: np.ndarray, threshold_std: float = ARTIFACT_STD) -> np.ndarray:
    """
    Returns a boolean mask. True = keep epoch, False = artifact.

    Logic: calculate the global std of the entire recording.
    If any single sample in an epoch exceeds threshold * global_std,
    the epoch is likely an artifact (patient moved, electrode popped off).
    """
    global_std = epochs.std()
    max_per_epoch = np.abs(epochs).max(axis=1)   # max absolute value per row
    mask = max_per_epoch < (threshold_std * global_std)
    return mask


# ── normalization ─────────────────────────────────────────────────────────────
def normalize_subject(epochs: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per subject.
    Subtracts the mean and divides by std across the entire subject's recording.

    Why per subject? Because each person has different baseline EEG amplitude.
    One person might have naturally stronger signals than another.
    Normalizing per subject removes that personal bias so the model
    learns brain patterns, not individual amplitude differences.

    Result: mean=0, std=1 for each subject.
    """
    mean = epochs.mean()
    std  = epochs.std()
    if std == 0:
        return epochs
    return (epochs - mean) / std


# ── main processing ───────────────────────────────────────────────────────────
def process_subject(csv_path: str, b, a) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    labels  = df["label"].values
    epochs  = df.drop(columns=["label"]).values.astype(np.float32)  # shape (n_epochs, 3000)

    # 1. flatten to 1D signal, filter, reshape back
    signal_1d   = epochs.flatten()
    filtered_1d = apply_filter(signal_1d, b, a)
    epochs      = filtered_1d.reshape(epochs.shape)

    # 2. artifact removal
    mask   = remove_artifacts(epochs)
    epochs = epochs[mask]
    labels = labels[mask]

    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"    Artifacts removed: {n_removed} epochs")

    # 3. normalize per subject
    epochs = normalize_subject(epochs)

    # 4. rebuild dataframe
    cols = [f"sample_{i}" for i in range(EPOCH_SAMP)]
    out  = pd.DataFrame(epochs, columns=cols)
    out["label"] = labels

    return out


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    b, a = make_bandpass_filter(LOWCUT, HIGHCUT, SFREQ)

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    print(f"Found {len(csv_files)} subjects to process.\n")

    total_before = 0
    total_after  = 0

    for csv_path in csv_files:
        subject_id = os.path.basename(csv_path).replace(".csv", "")
        out_path   = os.path.join(OUTPUT_DIR, f"{subject_id}.csv")

        if os.path.exists(out_path):
            print(f"  [SKIP] {subject_id} already processed.")
            continue

        print(f"  Processing {subject_id} ...", end=" ", flush=True)
        try:
            df_in  = pd.read_csv(csv_path)
            df_out = process_subject(csv_path, b, a)
            df_out.to_csv(out_path, index=False)

            before = len(df_in)
            after  = len(df_out)
            total_before += before
            total_after  += after
            print(f"{before} -> {after} epochs saved.")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. Total epochs: {total_before} -> {total_after}")
    removed = total_before - total_after
    print(f"Total artifacts removed: {removed} ({100*removed/max(total_before,1):.1f}%)")


if __name__ == "__main__":
    run()
