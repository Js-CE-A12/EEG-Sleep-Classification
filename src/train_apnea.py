"""
train_apnea.py
Phase 8b: Entrenar y evaluar modelos de detección de apnea.

¿Qué hace este archivo?
  Tarea diferente a las etapas del sueño:
    - Etapas del sueño: 5 clases (W, N1, N2, N3, REM)
    - Detección de apnea: 2 clases (apnea=1, no-apnea=0)

  Usamos los MISMOS modelos (SVM, RF, XGBoost) pero:
    - Clasificación binaria en vez de multiclase
    - Métricas distintas: sensibilidad, especificidad, AUC-ROC
    - El desbalance es diferente: ~30-40% apnea vs ~5% N1

¿Por qué métricas distintas?
  En medicina, errar de un lado es peor que del otro:
    - Falso negativo (no detectar apnea real): PELIGROSO para el paciente
    - Falso positivo (decir apnea cuando no hay): molesto pero no peligroso
  Por eso reportamos sensibilidad (cuántas apneas reales capturamos)
  y especificidad (cuántos no-apnea clasificamos bien).

Protocolo:
  - Subject-wise 5-fold (igual que las etapas) para evitar fuga de información
  - Las mismas 25 características (reutilizamos feature_extraction si ya existe)
    O extraemos directamente aquí con el pipeline simplificado

Outputs (data/results/):
  - apnea_fold_metrics.csv         métricas por fold
  - apnea_summary_metrics.csv      resumen mean ± std
  - apnea_confusion_<model>.png    matrices de confusión
  - apnea_roc_<model>.png          curvas ROC
  - apnea_feature_importance.png   importancia de features

Uso:
    python src/train_apnea.py
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

from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis as sp_kurtosis

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, cohen_kappa_score
)
from xgboost import XGBClassifier

# ── configuración ──────────────────────────────────────────────────────────────
SEED    = 42   # misma semilla que el resto del proyecto → reproducibilidad
N_FOLDS = 5    # mismo protocolo subject-wise

# MIT-BIH PSG graba a 250 Hz (distinto a Sleep-EDF que era 100 Hz)
SFREQ      = 250
EPOCH_SAMP = SFREQ * 30  # 7500 muestras por época de 30s

# Submuestreo para SVM (igual lógica que train_models.py)
# SVM es O(n²) — con muchas épocas tarda demasiado
SVM_MAX_SAMPLES = 10000

np.random.seed(SEED)

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

APNEA_DIR  = os.path.join(_ROOT_DIR, "data", "processed", "apnea")
RESULTS_DIR = os.path.join(_ROOT_DIR, "data", "results")

CLASS_NAMES = ["No-Apnea", "Apnea"]


# ── extracción de características ──────────────────────────────────────────────
# Reutilizamos las mismas 25 características del pipeline de etapas del sueño.
# Tiene sentido fisiológico: la apnea interrumpe la respiración → cambia
# la frecuencia y complejidad de la señal EEG (más delta, menos actividad rápida).

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta" : (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

def extract_features_from_epochs(epochs):
    """
    ¿Por qué extraemos features aquí y no usamos los de feature_extraction.py?
    Porque el MIT-BIH PSG tiene 7500 muestras por época (250Hz × 30s)
    en vez de 3000 (100Hz × 30s). Las features son las mismas pero
    aplicadas a épocas de diferente longitud.

    epochs: array de shape (n_epochs, 7500)
    returns: DataFrame de shape (n_epochs, 25)
    """
    feats = {}

    # ── dominio temporal ──
    feats["mean"]     = epochs.mean(axis=1)
    feats["std"]      = epochs.std(axis=1)
    feats["variance"] = epochs.var(axis=1)
    feats["skewness"] = skew(epochs, axis=1)
    feats["kurtosis"] = sp_kurtosis(epochs, axis=1)

    # Zero crossing rate: cuántas veces la señal cruza el cero por segundo
    # Durante apnea: menos cruces (señal más lenta y regular)
    signs     = np.sign(epochs)
    crossings = np.diff(signs, axis=1)
    feats["zero_crossing_rate"] = (crossings != 0).sum(axis=1) / EPOCH_SAMP

    feats["line_length"] = np.abs(np.diff(epochs, axis=1)).mean(axis=1)

    # Parámetros de Hjorth — describen la señal en términos de frecuencia
    d1     = np.diff(epochs, n=1, axis=1)
    d2     = np.diff(epochs, n=2, axis=1)
    var_x  = epochs.var(axis=1)
    var_d1 = d1.var(axis=1)
    var_d2 = d2.var(axis=1)
    feats["hjorth_mobility"]   = np.sqrt(var_d1 / np.clip(var_x, 1e-10, None))
    mob_d1 = np.sqrt(var_d2 / np.clip(var_d1, 1e-10, None))
    feats["hjorth_complexity"] = mob_d1 / np.clip(feats["hjorth_mobility"], 1e-10, None)

    # ── dominio frecuencial ──
    # Usamos nperseg=512 en vez de 256 porque las épocas son más largas (7500 vs 3000)
    freqs, psd = welch(epochs, fs=SFREQ, nperseg=512, axis=1)
    freq_res   = freqs[1] - freqs[0]
    total_pwr  = psd.sum(axis=1) * freq_res

    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bp  = psd[:, idx].sum(axis=1) * freq_res
        feats[f"{band}_power"]          = bp
        feats[f"{band}_relative_power"] = bp / np.clip(total_pwr, 1e-10, None)
        band_powers[band] = bp

    feats["delta_beta_ratio"]  = band_powers["delta"] / np.clip(band_powers["beta"],  1e-10, None)
    feats["theta_alpha_ratio"] = band_powers["theta"] / np.clip(band_powers["alpha"], 1e-10, None)

    psd_norm = psd / np.clip(psd.sum(axis=1, keepdims=True), 1e-10, None)
    feats["spectral_entropy"] = -(psd_norm * np.log2(psd_norm + 1e-10)).sum(axis=1)
    feats["peak_frequency"]   = freqs[psd.argmax(axis=1)]

    # ── no-lineales simplificados ──
    # Usamos permutation entropy solamente (sample entropy es muy lento con 7500 muestras)
    from collections import Counter
    import math
    order = 3
    max_pe = np.log2(math.factorial(order))
    pe_vals = np.zeros(len(epochs))
    for i, epoch in enumerate(epochs):
        x = epoch[::25]  # downsample a 300 puntos
        n = len(x)
        embedded = np.array([x[j:j+order] for j in range(n - order + 1)])
        patterns = np.argsort(embedded, axis=1)
        counts   = Counter(map(tuple, patterns))
        total    = sum(counts.values())
        probs    = np.array(list(counts.values())) / total
        pe_vals[i] = -np.sum(probs * np.log2(probs + 1e-10)) / max_pe
    feats["permutation_entropy"] = pe_vals
    feats["sample_entropy"]      = np.zeros(len(epochs))  # placeholder por velocidad

    return pd.DataFrame(feats)


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, order=4):
    """Filtro pasa-banda Butterworth — igual que preprocess.py pero adaptado a 250 Hz."""
    nyq  = SFREQ / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


# ── carga de datos ─────────────────────────────────────────────────────────────

def load_apnea_data():
    """
    Carga todos los CSVs de apnea, extrae features y devuelve X, y, groups.
    
    ¿Por qué groups?
    Para el GroupKFold subject-wise — asegura que ningún sujeto
    esté en entrenamiento y prueba al mismo tiempo.
    """
    csv_files = sorted(glob.glob(os.path.join(APNEA_DIR, "*.csv")))

    if not csv_files:
        print(f"\n[ERROR] No se encontraron CSVs en: {APNEA_DIR}")
        print("Ejecuta primero: python src/apnea_to_csv.py")
        return None, None, None

    print(f"Cargando {len(csv_files)} sujetos de MIT-BIH PSG...")

    all_features = []
    all_labels   = []
    all_groups   = []

    for path in csv_files:
        subject_id = os.path.basename(path).replace(".csv", "")
        df         = pd.read_csv(path)

        labels  = df["apnea"].values
        epochs  = df.drop(columns=["apnea"]).values.astype(np.float64)

        # Filtrado pasa-banda (igual que preprocess.py)
        signal_1d   = epochs.flatten()
        filtered_1d = bandpass_filter(signal_1d)
        epochs      = filtered_1d.reshape(epochs.shape)

        # Normalización z-score por sujeto
        mean = epochs.mean()
        std  = epochs.std()
        if std > 0:
            epochs = (epochs - mean) / std

        # Extraer features
        feat_df = extract_features_from_epochs(epochs)

        all_features.append(feat_df)
        all_labels.extend(labels)
        all_groups.extend([subject_id] * len(labels))

        n_apnea = labels.sum()
        print(f"  {subject_id}: {len(labels)} épocas, {n_apnea} apnea ({100*n_apnea/len(labels):.0f}%)")

    X      = pd.concat(all_features, ignore_index=True).values.astype(np.float32)
    y      = np.array(all_labels, dtype=int)
    groups = np.array(all_groups)

    print(f"\nTotal: {len(X):,} épocas | {y.sum():,} apnea ({100*y.mean():.1f}%) | {len(np.unique(groups))} sujetos")
    return X, y, groups


# ── modelos ────────────────────────────────────────────────────────────────────

def get_models():
    """
    Los mismos 3 modelos que en train_models.py pero configurados para
    clasificación binaria. class_weight='balanced' es especialmente
    importante aquí porque la apnea puede ser ~30-40% de las épocas.
    """
    return {
        "SVM_RBF": SVC(
            C=10, gamma="scale", kernel="rbf",
            class_weight="balanced",
            probability=True,   # necesario para AUC-ROC
            random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1,
            random_state=SEED, verbosity=0
        ),
    }


# ── métricas de apnea ──────────────────────────────────────────────────────────

def apnea_metrics(y_true, y_pred, y_proba=None):
    """
    ¿Por qué estas métricas específicas?

    Sensibilidad (Recall para clase 1):
      De todos los casos reales de apnea, ¿cuántos detectamos?
      Alta sensibilidad = pocos falsos negativos (no perder apneas reales)
      CRÍTICO en aplicación clínica.

    Especificidad (Recall para clase 0):
      De todos los casos sin apnea, ¿cuántos clasificamos bien?
      Alta especificidad = pocos falsos positivos (no alarmar sin razón)

    AUC-ROC:
      Área bajo la curva ROC — resumen del balance sensibilidad/especificidad
      en todos los umbrales posibles. AUC=1.0 es perfecto, AUC=0.5 es aleatorio.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])

    sensitivity  = tp / max(tp + fn, 1)   # recall para apnea
    specificity  = tn / max(tn + fp, 1)   # recall para no-apnea
    accuracy     = accuracy_score(y_true, y_pred)
    f1_apnea     = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa        = cohen_kappa_score(y_true, y_pred)

    auc_val = 0.0
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc_val     = auc(fpr, tpr)

    return {
        "accuracy"   : round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_apnea"   : round(f1_apnea, 4),
        "macro_f1"   : round(f1_macro, 4),
        "kappa"      : round(kappa, 4),
        "auc_roc"    : round(auc_val, 4),
    }


# ── gráficas ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm, model_name):
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Apnea — Matriz de Confusión — {model_name}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"apnea_confusion_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado {path}")


def save_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc_val     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#028090", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.set_title(f"Curva ROC — Detección de Apnea — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"apnea_roc_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado {path}")


def save_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx[::-1]], importances[idx[::-1]], color="#028090")
    ax.set_xlabel("Importancia media (Gini)")
    ax.set_title("Top-20 Features — Detección de Apnea (Random Forest)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "apnea_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Guardado {path}")


# ── entrenamiento principal ────────────────────────────────────────────────────

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Cargar datos
    X, y, groups = load_apnea_data()
    if X is None:
        return

    feature_names = [
        "mean", "std", "variance", "skewness", "kurtosis",
        "zero_crossing_rate", "line_length", "hjorth_mobility", "hjorth_complexity",
        "delta_power", "delta_relative_power",
        "theta_power", "theta_relative_power",
        "alpha_power", "alpha_relative_power",
        "beta_power", "beta_relative_power",
        "gamma_power", "gamma_relative_power",
        "delta_beta_ratio", "theta_alpha_ratio",
        "spectral_entropy", "peak_frequency",
        "permutation_entropy", "sample_entropy"
    ]

    models  = get_models()
    gkf     = GroupKFold(n_splits=N_FOLDS)
    results = []

    # 2. Validación subject-wise 5-fold
    print(f"\nEntrenando con {N_FOLDS}-fold subject-wise CV...\n")

    all_y_true  = {name: [] for name in models}
    all_y_pred  = {name: [] for name in models}
    all_y_proba = {name: [] for name in models}
    all_cm      = {name: np.zeros((2, 2), dtype=int) for name in models}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"── Fold {fold+1}/{N_FOLDS} ──")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler ajustado SOLO con entrenamiento
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        for name, model in models.items():
            print(f"  {name}...", end=" ", flush=True)

            # Submuestreo para SVM (O(n²))
            if name == "SVM_RBF" and len(X_train) > SVM_MAX_SAMPLES:
                rng   = np.random.RandomState(SEED)
                idx   = rng.choice(len(X_train), SVM_MAX_SAMPLES, replace=False)
                Xtr   = X_train[idx]
                ytr   = y_train[idx]
            else:
                Xtr, ytr = X_train, y_train

            model.fit(Xtr, ytr)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = apnea_metrics(y_test, y_pred, y_proba)
            metrics.update({"fold": fold + 1, "model": name})
            results.append(metrics)

            all_y_true[name].extend(y_test)
            all_y_pred[name].extend(y_pred)
            all_y_proba[name].append(y_proba)
            all_cm[name] += confusion_matrix(y_test, y_pred, labels=[0, 1])

            print(f"acc={metrics['accuracy']:.3f}  sens={metrics['sensitivity']:.3f}  "
                  f"spec={metrics['specificity']:.3f}  AUC={metrics['auc_roc']:.3f}")

    # 3. Guardar resultados
    df_results = pd.DataFrame(results)
    fold_path  = os.path.join(RESULTS_DIR, "apnea_fold_metrics.csv")
    df_results.to_csv(fold_path, index=False)
    print(f"\nGuardado {fold_path}")

    metric_cols = ["accuracy", "sensitivity", "specificity", "f1_apnea", "macro_f1", "kappa", "auc_roc"]
    summary = df_results.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    summary_path = os.path.join(RESULTS_DIR, "apnea_summary_metrics.csv")
    summary.to_csv(summary_path)
    print(f"Guardado {summary_path}")

    # 4. Gráficas
    print("\nGenerando gráficas...")
    rf_model = None
    for name in models:
        yt    = np.array(all_y_true[name])
        yp    = np.array(all_y_pred[name])
        yprob = np.vstack(all_y_proba[name])

        save_confusion_matrix(all_cm[name], name)
        save_roc_curve(yt, yprob, name)

        if name == "RandomForest":
            # Re-entrenar RF completo para feature importance
            scaler_full = StandardScaler()
            Xf = scaler_full.fit_transform(X)
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=SEED)
            rf.fit(Xf, y)
            save_feature_importance(rf, feature_names)

    # 5. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN — Detección de Apnea (MIT-BIH PSG, 5-fold subject-wise)")
    print("=" * 60)
    for model_name in models:
        sub = df_results[df_results["model"] == model_name]
        print(f"\n{model_name}:")
        for col in metric_cols:
            print(f"  {col:<14}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
