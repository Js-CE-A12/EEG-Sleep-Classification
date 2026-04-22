"""
apnea_to_csv.py
Phase 8a: Cargar el dataset MIT-BIH PSG y convertirlo a CSV con etiquetas apnea/no-apnea.

¿Qué hace este archivo?
  El dataset MIT-BIH PSG contiene grabaciones de sueño de pacientes con apnea.
  Cada grabación tiene una señal EEG + anotaciones de cuándo ocurre un evento de apnea.
  Este script:
    1. Lee la señal EEG del canal elegido
    2. La corta en ventanas de 30 segundos (igual que las etapas del sueño)
    3. Le pone etiqueta 1 si hubo apnea en esa ventana, 0 si no
    4. Guarda un CSV por sujeto

¿Por qué MIT-BIH PSG?
  Es el dataset público más usado para detección de apnea con EEG monocanal.
  Tiene 18 sujetos con distintos grados de severidad (leve, moderada, severa).
  Lo usa la literatura como benchmark → podemos comparar nuestros resultados.

Descarga:
  https://physionet.org/content/slpdb/1.0.0/
  Carpeta destino: data/mit-bih-psg/

Output:
  data/processed/apnea/   (un CSV por sujeto)
  Columnas: sample_0 ... sample_2999, apnea
  apnea = 1 si hubo evento de apnea en esa ventana, 0 si no

Uso:
    python src/apnea_to_csv.py
"""

import os
import glob
import argparse

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

# ── configuración ──────────────────────────────────────────────────────────────
# El dataset MIT-BIH PSG usa distintos nombres de canal según el sujeto.
# Intentamos estos en orden hasta encontrar uno que exista.
EEG_CANDIDATES = ["EEG", "EEG1", "EEG2", "C3-A2", "C4-A1"]

SFREQ      = 250    # Hz — MIT-BIH PSG graba a 250 Hz (distinto a Sleep-EDF que era 100 Hz)
EPOCH_SEC  = 30     # segundos por ventana
EPOCH_SAMP = SFREQ * EPOCH_SEC  # 250 * 30 = 7500 muestras por época

# Etiquetas de apnea en las anotaciones del dataset MIT-BIH PSG
# 'A' = Apnea Obstructiva, 'H' = Hipopnea, 'OA' = Obstructiva, 'CA' = Central
APNEA_LABELS = {"A", "H", "OA", "CA", "OAA", "CAA", "MA", "HAA"}

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

RAW_DIR    = os.path.join(_ROOT_DIR, "data", "mit-bih-psg")
OUTPUT_DIR = os.path.join(_ROOT_DIR, "data", "processed", "apnea")


# ── funciones ──────────────────────────────────────────────────────────────────

def find_eeg_channel(raw):
    """
    ¿Por qué esto?
    El MIT-BIH PSG no usa un nombre de canal estándar.
    Algunos sujetos tienen 'EEG', otros 'EEG1', otros 'C3-A2'.
    Esta función prueba los candidatos y devuelve el primero que encuentre.
    """
    available = [ch.upper() for ch in raw.ch_names]
    for candidate in EEG_CANDIDATES:
        if candidate.upper() in available:
            idx = available.index(candidate.upper())
            return raw.ch_names[idx]
    # Si no encontramos ninguno conocido, usamos el primero disponible
    print(f"  [WARN] No se encontró canal EEG estándar. Canales: {raw.ch_names}")
    print(f"  [WARN] Usando primer canal: {raw.ch_names[0]}")
    return raw.ch_names[0]


def load_apnea_annotations(raw):
    """
    ¿Qué hace esto?
    Las anotaciones son como un log de eventos:
      [tiempo=120.0s, duración=25s, descripción='A']  → apnea a los 2 minutos
    
    Esta función convierte eso en un array de booleanos, uno por época:
      [False, False, True, True, False, ...]
    
    Una época se marca como apnea=1 si CUALQUIER parte de ella
    estuvo dentro de un evento de apnea.
    """
    n_samples = len(raw.times)
    n_epochs  = n_samples // EPOCH_SAMP

    # Array de False por defecto (sin apnea)
    apnea_mask = np.zeros(n_epochs, dtype=int)

    annotations = raw.annotations
    for ann in annotations:
        if ann["description"] in APNEA_LABELS:
            onset_sec    = ann["onset"]
            duration_sec = ann["duration"]

            # Convertir tiempos a índice de época
            start_epoch = int(onset_sec // EPOCH_SEC)
            end_epoch   = int((onset_sec + duration_sec) // EPOCH_SEC) + 1

            # Marcar todas las épocas que se superponen con el evento
            for ep in range(start_epoch, min(end_epoch, n_epochs)):
                apnea_mask[ep] = 1

    n_apnea = apnea_mask.sum()
    pct = 100 * n_apnea / max(len(apnea_mask), 1)
    print(f"    Apnea: {n_apnea}/{len(apnea_mask)} épocas ({pct:.1f}%)")

    return apnea_mask


def process_subject(edf_path, subject_id):
    """
    ¿Qué hace esto paso a paso?
    1. Lee el archivo EDF del sujeto
    2. Selecciona el canal EEG correcto
    3. Remuestrea a 250 Hz si es necesario (para consistencia)
    4. Obtiene las etiquetas de apnea de las anotaciones
    5. Corta la señal en épocas de 30s
    6. Retorna un DataFrame listo para guardar

    Nota importante: a diferencia del Sleep-EDF que tenía archivos separados
    para la señal y el hipnograma, aquí las anotaciones de apnea VIENEN DENTRO
    del mismo archivo EDF.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Seleccionar canal EEG
    eeg_ch = find_eeg_channel(raw)
    raw.pick_channels([eeg_ch])

    # Remuestrear si es necesario
    if abs(raw.info["sfreq"] - SFREQ) > 1:
        print(f"    Remuestreando {raw.info['sfreq']} Hz → {SFREQ} Hz")
        raw.resample(SFREQ)

    signal = raw.get_data()[0]           # shape (n_samples,)
    apnea  = load_apnea_annotations(raw) # shape (n_epochs,)

    n_epochs = len(signal) // EPOCH_SAMP
    n_epochs = min(n_epochs, len(apnea))

    rows = []
    for i in range(n_epochs):
        epoch = signal[i * EPOCH_SAMP : (i + 1) * EPOCH_SAMP]
        rows.append((*epoch, apnea[i]))

    cols = [f"sample_{i}" for i in range(EPOCH_SAMP)] + ["apnea"]
    df   = pd.DataFrame(rows, columns=cols)
    return df


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buscar archivos EDF en la carpeta del dataset
    edf_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.edf")))

    if not edf_files:
        print(f"\n[ERROR] No se encontraron archivos .edf en: {RAW_DIR}")
        print("Descarga el dataset desde: https://physionet.org/content/slpdb/1.0.0/")
        print(f"Y colócalo en: {RAW_DIR}")
        return

    print(f"Encontrados {len(edf_files)} archivos EDF en MIT-BIH PSG\n")

    total_apnea    = 0
    total_no_apnea = 0

    for edf_path in edf_files:
        subject_id = os.path.basename(edf_path).replace(".edf", "")
        out_path   = os.path.join(OUTPUT_DIR, f"{subject_id}.csv")

        if os.path.exists(out_path):
            print(f"  [SKIP] {subject_id} ya procesado.")
            continue

        print(f"  Procesando {subject_id} ...", end=" ", flush=True)
        try:
            df = process_subject(edf_path, subject_id)
            df.to_csv(out_path, index=False)

            n_apnea    = df["apnea"].sum()
            n_no_apnea = len(df) - n_apnea
            total_apnea    += n_apnea
            total_no_apnea += n_no_apnea
            print(f"  {len(df)} épocas guardadas → {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nResumen del dataset de apnea:")
    print(f"  Total épocas sin apnea : {total_no_apnea}")
    print(f"  Total épocas con apnea : {total_apnea}")
    total = total_apnea + total_no_apnea
    if total > 0:
        print(f"  Desbalance             : {100*total_apnea/total:.1f}% apnea")
        print(f"  (esperado ~30-40% para dataset clínico)")
    print("\nDone.")


if __name__ == "__main__":
    run()
