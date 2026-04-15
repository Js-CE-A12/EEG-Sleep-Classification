# Sleep Stage Classification — EEG Machine Learning Pipeline

Proyecto 1: Clasificación de etapas del sueño con EEG de un solo canal usando SVM, Random Forest, XGBoost y HMM.  
Universidad Interamericana PR – Bayamón | Data Science

---

## Objetivo

Clasificar 5 etapas del sueño (W, N1, N2, N3, REM) a partir de señales EEG de un solo canal (Fpz-Cz), usando el dataset Sleep-EDF Expanded. El pipeline cubre extracción de características, entrenamiento de modelos, post-procesamiento con HMM, evaluación estadística y experimento cross-dataset.

---

## Estructura del proyecto

```
.
├── data/
│   ├── sleep-edf/                          # Raw EDF files (Sleep-EDF Expanded)
│   │   └── sleep-edf-database-expanded-1.0.0/
│   │       ├── sleep-cassette/             # 153 healthy subjects (SC)
│   │       └── sleep-telemetry/            # 44 clinical patients (ST)
│   ├── processed/
│   │   ├── features/                       # SC feature CSVs (one per subject)
│   │   └── features-st/                    # ST feature CSVs (auto-generated)
│   └── results/                            # All output CSVs and figures
├── src/
│   ├── edf_to_csv.py                       # Phase 1: EDF -> epoch CSV
│   ├── preprocess.py                       # Phase 2: bandpass, artifact removal, z-score
│   ├── feature_extraction.py               # Phase 3: 25 features per epoch
│   ├── train_models.py                     # Phase 4: SVM, RF, XGBoost (5-fold CV)
│   ├── hmm_smoother.py                     # Phase 5: HMM post-processing smoother
│   ├── evaluate.py                         # Phase 6: bootstrap CI, McNemar, AUC-ROC
│   └── cross_dataset.py                    # Phase 7: cross-dataset generalization (SC->ST)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_scripts.txt                         # Step-by-step execution guide
```

---

## Dataset

**Sleep-EDF Expanded** (PhysioNet)  
- Sleep-Cassette (SC): 153 healthy subjects, at-home recordings, unmedicated  
- Sleep-Telemetry (ST): 44 clinical patients, hospital recordings, temazepam users  
- Single-channel EEG: Fpz-Cz at 100 Hz  
- Labels: W=0, N1=1, N2=2, N3=3, REM=4 (30-second epochs)

---

## Features extraidas (25 por epoca)

**Time domain (9):** mean, std, variance, skewness, kurtosis, zero_crossing_rate, line_length, hjorth_mobility, hjorth_complexity

**Frequency domain (14):** delta/theta/alpha/beta/gamma absolute and relative power, delta_beta_ratio, theta_alpha_ratio, spectral_entropy, peak_frequency

**Non-linear (2):** sample_entropy, permutation_entropy

---

## Modelos

| Model | Config |
|---|---|
| SVM-RBF | C=10, gamma=scale, class_weight=balanced, subsample 15K/fold |
| Random Forest | 200 trees, balanced_subsample |
| XGBoost | 200 estimators, lr=0.1, subsample=0.8 |
| HMM (smoother) | Viterbi post-processing on XGBoost predictions |

Validation: subject-wise 5-fold GroupKFold (SEED=42, no data leakage)

---

## Resultados (resumen)

| Model | Accuracy | Macro-F1 | Kappa |
|---|---|---|---|
| SVM-RBF | ~0.72 | ~0.56 | ~0.62 |
| Random Forest | ~0.79 | ~0.62 | ~0.72 |
| XGBoost | ~0.80 | ~0.63 | ~0.73 |
| XGBoost + HMM | see hmm_summary_metrics.csv | | |

Cross-dataset (train SC, test ST): XGBoost macro-F1 = 0.446 (drop of 0.185 vs within-dataset)

All detailed results in `data/results/`.

---

## Requisitos

- Python 3.11+
- Ver `requirements.txt`

---

## Ejecucion local

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Ver `run_scripts.txt` para la secuencia completa de comandos.

---

## Ejecucion con Docker

```bash
docker compose build
docker compose run --rm app python src/edf_to_csv.py
docker compose run --rm app python src/preprocess.py
docker compose run --rm app python src/feature_extraction.py
docker compose run --rm app python src/train_models.py
docker compose run --rm app python src/hmm_smoother.py
docker compose run --rm app python src/evaluate.py
docker compose run --rm app python src/cross_dataset.py
```

---

## Outputs (data/results/)

| File | Description |
|---|---|
| fold_metrics.csv | Per-fold metrics for all models |
| summary_metrics.csv | Mean ± std across folds |
| confusion_*.png | Confusion matrices (normalized) |
| feature_importance_*.png | Top features by model |
| model_comparison_f1.png | F1 comparison bar chart |
| hmm_fold_metrics.csv | Before/after HMM per fold |
| hmm_summary_metrics.csv | HMM smoothing summary |
| transition_matrix.png | Learned HMM transition matrix |
| bootstrap_ci.csv | 95% bootstrap CI (accuracy, F1, kappa) |
| mcnemar_results.csv | McNemar pairwise significance tests |
| auc_roc.csv | Per-class AUC for RF and XGBoost |
| roc_curves_*.png | ROC curves (one-vs-rest) |
| cross_dataset_metrics.csv | Cross-dataset generalization results |
| cross_dataset_comparison.png | Within vs cross-dataset F1 chart |

---

## Pruebas

```bash
pytest -q
```
