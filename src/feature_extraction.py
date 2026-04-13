"""
feature_extraction.py
Phase 3: Extract meaningful features from clean epoch CSVs.

Each epoch goes from 3000 raw voltage samples → 25 features the ML model can learn from.

Feature groups:
  1. Time-domain (9)     : mean, std, variance, skewness, kurtosis, zero_crossing_rate,
                           line_length, hjorth_mobility, hjorth_complexity
  2. Frequency-domain (14): absolute band power (delta/theta/alpha/beta/gamma),
                             relative band power (same 5 bands),
                             delta_beta_ratio, theta_alpha_ratio,
                             spectral_entropy, peak_frequency
  3. Non-linear (2)      : sample_entropy, permutation_entropy

Input : data/processed/sleep-edf-clean/   (one CSV per subject, 3000 cols + label)
Output: data/processed/features/          (one CSV per subject, 25 features + label)

Usage:
    python src/feature_extraction.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ── config ─────────────────────────────────────────────────────────────────────
SFREQ      = 100     # Hz
EPOCH_SAMP = 3000    # samples per epoch

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta" : (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

INPUT_DIR  = os.path.join(_ROOT_DIR, "data", "processed", "sleep-edf-clean")
OUTPUT_DIR = os.path.join(_ROOT_DIR, "data", "processed", "features")


# ── time-domain features ───────────────────────────────────────────────────────
def time_features(epochs: np.ndarray) -> pd.DataFrame:
    """
    Compute statistical features directly from the raw voltage values.
    epochs shape: (n_epochs, 3000)
    Returns DataFrame with one row per epoch.
    """
    feats = {}

    # mean: average voltage level across the epoch
    feats["mean"] = epochs.mean(axis=1)

    # std: how much the signal fluctuates (active brain = high std)
    feats["std"] = epochs.std(axis=1)

    # variance = std^2, captures signal power in time domain
    feats["variance"] = epochs.var(axis=1)

    # skewness: is the distribution of voltages symmetric?
    # REM/wake tend to have more asymmetric distributions
    feats["skewness"] = skew(epochs, axis=1)

    # kurtosis: are there extreme spikes in the signal?
    # high kurtosis = sharp peaks (spindles in N2, artifacts)
    feats["kurtosis"] = kurtosis(epochs, axis=1)

    # zero crossing rate: how many times does signal cross zero per second
    # high ZCR = high frequency activity (awake/REM)
    # low ZCR  = slow waves (N3)
    signs       = np.sign(epochs)
    crossings   = np.diff(signs, axis=1)
    feats["zero_crossing_rate"] = (crossings != 0).sum(axis=1) / EPOCH_SAMP

    # line length: mean absolute step size between consecutive samples
    # high = rapid fluctuations (wake/REM), low = slow waves (N3)
    feats["line_length"] = np.abs(np.diff(epochs, axis=1)).mean(axis=1)

    # ── Hjorth parameters ──────────────────────────────────────────────────────
    # Invented by Bo Hjorth in 1970, widely used in EEG analysis.
    # They describe the signal using just the first and second derivatives.

    first_diff  = np.diff(epochs, n=1, axis=1)   # velocity of signal
    second_diff = np.diff(epochs, n=2, axis=1)   # acceleration of signal

    var_x  = epochs.var(axis=1)
    var_d1 = first_diff.var(axis=1)
    var_d2 = second_diff.var(axis=1)

    # Mobility: mean frequency of the signal
    # sqrt(var of first derivative / var of signal)
    # high mobility = faster signal = higher frequency brain activity
    feats["hjorth_mobility"] = np.sqrt(var_d1 / np.clip(var_x, 1e-10, None))

    # Complexity: how complex/irregular is the signal?
    # ratio of mobility of first derivative vs mobility of signal
    # 1.0 = pure sine wave, higher = more complex/irregular
    mobility_d1 = np.sqrt(var_d2 / np.clip(var_d1, 1e-10, None))
    feats["hjorth_complexity"] = mobility_d1 / np.clip(feats["hjorth_mobility"], 1e-10, None)

    return pd.DataFrame(feats)


# ── frequency-domain features ──────────────────────────────────────────────────
def frequency_features(epochs: np.ndarray) -> pd.DataFrame:
    """
    Compute power in each EEG frequency band using Welch's method.
    epochs shape: (n_epochs, 3000)

    Welch's method:
      - Splits each epoch into overlapping windows
      - Computes FFT on each window
      - Averages the results → Power Spectral Density (PSD)
      - More stable than a single FFT on the full epoch

    Returns DataFrame with absolute and relative band powers.
    """
    # nperseg=256: each Welch window is 256 samples (2.56 seconds)
    # This gives us good frequency resolution (0.39 Hz per bin)
    freqs, psd = welch(epochs, fs=SFREQ, nperseg=256, axis=1)
    # freqs shape: (n_freqs,)
    # psd   shape: (n_epochs, n_freqs)

    freq_res    = freqs[1] - freqs[0]   # Hz per bin
    total_power = psd.sum(axis=1) * freq_res   # total power across all frequencies

    feats = {}
    band_powers = {}

    for band_name, (low, high) in BANDS.items():
        # find which frequency bins fall inside this band
        idx = np.where((freqs >= low) & (freqs < high))[0]

        # integrate (sum) the power in those bins
        # multiply by freq_res to convert to proper power units (uV^2)
        band_power = psd[:, idx].sum(axis=1) * freq_res

        feats[f"{band_name}_power"]          = band_power
        feats[f"{band_name}_relative_power"] = band_power / np.clip(total_power, 1e-10, None)
        band_powers[band_name]               = band_power

    # delta/beta ratio: classic sleep depth indicator
    # high ratio = lots of slow waves = deep sleep (N3)
    # low ratio  = lots of fast waves = awake or REM
    feats["delta_beta_ratio"] = (
        band_powers["delta"] / np.clip(band_powers["beta"], 1e-10, None)
    )

    # theta/alpha ratio: useful for distinguishing N1 from Wake
    feats["theta_alpha_ratio"] = (
        band_powers["theta"] / np.clip(band_powers["alpha"], 1e-10, None)
    )

    # spectral entropy: measures how spread-out (uniform) the PSD is
    # low = signal energy concentrated in narrow bands (N3 = mostly delta)
    # high = energy spread across many bands (wake/REM = broadband)
    psd_norm = psd / np.clip(psd.sum(axis=1, keepdims=True), 1e-10, None)
    feats["spectral_entropy"] = -(psd_norm * np.log2(psd_norm + 1e-10)).sum(axis=1)

    # peak frequency: the frequency bin with maximum power
    # N3 → ~1-2 Hz (delta), N2 → ~12-14 Hz (spindles), wake → ~10 Hz (alpha)
    feats["peak_frequency"] = freqs[psd.argmax(axis=1)]

    return pd.DataFrame(feats)


# ── non-linear features ────────────────────────────────────────────────────────
def sample_entropy(epochs: np.ndarray, m: int = 2, r_factor: float = 0.2) -> np.ndarray:
    """
    Sample Entropy: measures the complexity/unpredictability of the signal.

    Theory:
      - A very regular signal (pure sine wave) has LOW entropy → predictable
      - A very irregular signal (noise, REM) has HIGH entropy → unpredictable
      - Deep sleep (N3) has LOW entropy (regular delta waves)
      - REM and Wake have HIGH entropy (complex, irregular)

    Vectorized implementation:
      - Downsample each epoch to 300 points for speed
      - Build all template pairs at once using numpy broadcasting (no inner loop)
      - Compute pairwise Chebyshev distances in one matrix operation
      - Much faster than nested Python loops
    """
    results = np.zeros(len(epochs))

    for i, epoch in enumerate(epochs):
        r = r_factor * epoch.std()
        if r == 0:
            results[i] = 0
            continue

        # downsample to 300 points (keep every 10th sample)
        x = epoch[::10]
        n = len(x)

        # build all templates at once: shape (n-m, m) and (n-m, m+1)
        idx_m  = np.arange(m)[None, :]  + np.arange(n - m)[:, None]
        idx_m1 = np.arange(m+1)[None,:] + np.arange(n - m)[:, None]
        tm  = x[idx_m]    # (n-m, m)
        tm1 = x[idx_m1]   # (n-m, m+1)

        # pairwise Chebyshev distances via broadcasting (no inner loop)
        dist_m  = np.abs(tm[:, None, :]  - tm[None, :, :]).max(axis=2)   # (n-m, n-m)
        dist_m1 = np.abs(tm1[:, None, :] - tm1[None, :, :]).max(axis=2)  # (n-m, n-m)

        # count matches, subtract diagonal (self-matches)
        count_m  = (dist_m  < r).sum() - (n - m)
        count_m1 = (dist_m1 < r).sum() - (n - m)

        if count_m <= 0 or count_m1 <= 0:
            results[i] = 0
        else:
            results[i] = -np.log(count_m1 / count_m)

    return results


def permutation_entropy(epochs: np.ndarray, order: int = 3, delay: int = 1) -> np.ndarray:
    """
    Permutation Entropy: captures temporal complexity via ordinal patterns.

    Theory:
      - Look at all subsequences of length `order` in the signal.
      - Replace each subsequence with its rank order (e.g. [0.3, 0.1, 0.5] → [1,0,2]).
      - Count how often each rank pattern appears → probability distribution.
      - High entropy = many different patterns = complex/irregular (wake, REM).
      - Low entropy  = same pattern repeating = regular signal (N3 delta waves).

    order=3 → only 3! = 6 possible patterns, very fast to compute.
    Result is normalized to [0, 1] by dividing by log2(order!).
    """
    from collections import Counter
    import math

    max_entropy = np.log2(math.factorial(order))
    results = np.zeros(len(epochs))

    for i, epoch in enumerate(epochs):
        x = epoch[::10]   # downsample to 300 points for speed
        n = len(x)
        if n < order:
            continue

        # build embedding: each row is a window of length `order`
        embedded = np.array([x[j:j + order] for j in range(0, n - order + 1, delay)])

        # ordinal pattern = argsort of each window
        patterns = np.argsort(embedded, axis=1)

        # count pattern frequencies (tuples are hashable, no collision risk)
        counts = Counter(map(tuple, patterns))
        total  = sum(counts.values())
        probs  = np.array(list(counts.values())) / total

        pe = -np.sum(probs * np.log2(probs + 1e-10))
        results[i] = pe / max_entropy   # normalize to [0, 1]

    return results


# ── main processing ────────────────────────────────────────────────────────────
def extract_features(csv_path: str) -> pd.DataFrame:
    df     = pd.read_csv(csv_path)
    labels = df["label"].values
    epochs = df.drop(columns=["label"]).values.astype(np.float64)

    print("    time features...",      end=" ", flush=True)
    tf = time_features(epochs)

    print("freq features...",          end=" ", flush=True)
    ff = frequency_features(epochs)

    print("sample entropy...",          end=" ", flush=True)
    se = sample_entropy(epochs)

    print("permutation entropy...",     end=" ", flush=True)
    pe = permutation_entropy(epochs)

    result = pd.concat([tf, ff], axis=1)
    result["sample_entropy"]      = se
    result["permutation_entropy"] = pe
    result["label"]               = labels

    return result


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    print(f"Found {len(csv_files)} subjects.\n")

    for i, csv_path in enumerate(csv_files):
        subject_id = os.path.basename(csv_path).replace(".csv", "")
        out_path   = os.path.join(OUTPUT_DIR, f"{subject_id}.csv")

        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(csv_files)}] [SKIP] {subject_id}")
            continue

        print(f"  [{i+1}/{len(csv_files)}] {subject_id} ...", end=" ", flush=True)
        try:
            result = extract_features(csv_path)
            result.to_csv(out_path, index=False)
            print(f"done. {len(result)} epochs, {result.shape[1]-1} features.")
        except Exception as e:
            print(f"ERROR: {e}")

    # show feature list from last subject
    sample = pd.read_csv(out_path)
    features = [c for c in sample.columns if c != "label"]
    print(f"\nFeatures extracted ({len(features)}):")
    for f in features:
        print(f"  - {f}")


if __name__ == "__main__":
    run()
