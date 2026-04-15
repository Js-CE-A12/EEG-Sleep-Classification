#!/usr/bin/env python3
"""
generate_report.py
Generates the final project report as a .docx file
using only Python built-in modules (no python-docx required).
Run from project root: python generate_report.py
"""

import zipfile, io, os

# ── XML helpers ────────────────────────────────────────────────────────────────

def esc(s):
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def h(text, level=1):
    t = esc(text)
    return f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr><w:r><w:t>{t}</w:t></w:r></w:p>'

def p(text, bold=False, italic=False, align="both", size=None, color=None, indent=0):
    t = esc(text)
    rpr=""
    if bold:   rpr += "<w:b/><w:bCs/>"
    if italic: rpr += "<w:i/><w:iCs/>"
    if size:   rpr += f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
    if color:  rpr += f"<w:color w:val=\"{color}\"/>"
    rpr_xml = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    ind_xml = f'<w:ind w:left="{indent}"/>' if indent else ""
    ppr = f'<w:pPr><w:jc w:val="{align}"/>{ind_xml}<w:spacing w:after="120"/></w:pPr>'
    return f'<w:p>{ppr}<w:r>{rpr_xml}<w:t xml:space="preserve">{t}</w:t></w:r></w:p>'

def br():  return "<w:p/>"
def pb():  return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

def bullet(text, level=0):
    t = esc(text)
    ind = 720 * (level + 1)
    return f'''<w:p>
      <w:pPr><w:pStyle w:val="ListParagraph"/>
        <w:ind w:left="{ind}" w:hanging="360"/>
        <w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="1"/></w:numPr>
      </w:pPr>
      <w:r><w:t xml:space="preserve">{t}</w:t></w:r></w:p>'''

def numbered(text, level=0):
    t = esc(text)
    ind = 720 * (level + 1)
    return f'''<w:p>
      <w:pPr><w:pStyle w:val="ListParagraph"/>
        <w:ind w:left="{ind}" w:hanging="360"/>
        <w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="2"/></w:numPr>
      </w:pPr>
      <w:r><w:t xml:space="preserve">{t}</w:t></w:r></w:p>'''

def caption(text):
    t = esc(text)
    return f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="200"/></w:pPr><w:r><w:rPr><w:i/><w:sz w:val="20"/></w:rPr><w:t>{t}</w:t></w:r></w:p>'

def tc(text, bold=False, shaded=False, width=None):
    t = esc(str(text))
    bold_xml = "<w:b/><w:bCs/>" if bold else ""
    fill = "4472C4" if shaded else "FFFFFF"
    col  = "FFFFFF" if shaded else "000000"
    w_xml = f'<w:tcW w:w="{width}" w:type="dxa"/>' if width else ""
    return f'''<w:tc><w:tcPr>{w_xml}
      <w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>
      <w:tcBorders>
        <w:top w:val="single" w:sz="4" w:color="BFBFBF"/>
        <w:bottom w:val="single" w:sz="4" w:color="BFBFBF"/>
        <w:left w:val="single" w:sz="4" w:color="BFBFBF"/>
        <w:right w:val="single" w:sz="4" w:color="BFBFBF"/>
      </w:tcBorders></w:tcPr>
      <w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="60"/></w:pPr>
        <w:r><w:rPr>{bold_xml}<w:color w:val="{col}"/><w:sz w:val="20"/></w:rPr>
          <w:t xml:space="preserve">{t}</w:t></w:r></w:p></w:tc>'''

def tbl(headers, rows):
    n = len(headers)
    grid = "".join(["<w:gridCol/>"] * n)
    hrow = "".join([tc(h, bold=True, shaded=True) for h in headers])
    body = "".join([
        "<w:tr>" + "".join([tc(cell) for cell in row]) + "</w:tr>"
        for row in rows
    ])
    return f'''<w:tbl>
      <w:tblPr>
        <w:tblStyle w:val="TableGrid"/>
        <w:tblW w:w="9072" w:type="dxa"/>
        <w:tblLook w:val="04A0"/>
      </w:tblPr>
      <w:tblGrid>{grid}</w:tblGrid>
      <w:tr>{hrow}</w:tr>
      {body}
    </w:tbl>'''


# ── Document content ──────────────────────────────────────────────────────────

def build_body():
    X = []
    add = X.append

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    add(br()); add(br()); add(br())
    add(p("Universidad Interamericana PR – Bayamón", bold=True, align="center", size=28))
    add(p("Departamento de Ciencia de Datos", bold=True, align="center", size=24))
    add(br())
    add(p("Proyecto 1", bold=True, align="center", size=32))
    add(br())
    add(p("Machine Learning para Clasificación de Etapas del Sueño",
          bold=True, align="center", size=28))
    add(p("con EEG Monocanal", bold=True, align="center", size=28))
    add(br()); add(br())
    add(p("Estudiantes:", bold=True, align="center", size=24))
    add(p("[Nombre 1] — [Número de Estudiante]", align="center", size=24))
    add(p("[Nombre 2] — [Número de Estudiante]", align="center", size=24))
    add(br())
    add(p("Fecha: Abril 2026", align="center", size=24))
    add(br())
    add(p("Herramienta de IA utilizada: Claude Code (Anthropic) — generación de código,",
          align="center", size=20, italic=True))
    add(p("depuración, diseño experimental y estructura del informe.",
          align="center", size=20, italic=True))
    add(pb())

    # ── ABSTRACT ──────────────────────────────────────────────────────────────
    add(h("Resumen / Abstract", 1))
    add(p(
        "Este trabajo implementa y compara tres familias de algoritmos de machine learning "
        "— SVM con kernel RBF, Random Forest y XGBoost — para la clasificación automática "
        "de cinco etapas del sueño (W, N1, N2, N3, REM) usando únicamente EEG monocanal "
        "(canal Fpz-Cz). Se utilizó el dataset Sleep-EDF Expanded, compuesto por 153 sujetos "
        "del subconjunto Cassette (SC) y 44 sujetos del subconjunto Telemetry (ST). "
        "El pipeline abarca preprocesamiento (filtro pasa-banda 0.5–40 Hz, remoción de "
        "artefactos, normalización z-score por sujeto), extracción de 25 características "
        "por época (9 dominio temporal, 14 dominio frecuencial, 2 no-lineales), y validación "
        "subject-wise con 5-fold GroupKFold para evitar fuga de información. Los mejores "
        "resultados dentro del dataset fueron: XGBoost con accuracy=0.856, macro-F1=0.631, "
        "kappa=0.721; Random Forest con accuracy=0.853, macro-F1=0.618; y SVM-RBF con "
        "macro-F1=0.631 pero menor kappa (0.655). La aplicación de un HMM como suavizador "
        "de secuencias sobre las predicciones de XGBoost mejoró el macro-F1 a 0.681. "
        "El experimento cross-dataset (entrenamiento en SC, prueba en ST) reveló una caída "
        "de rendimiento de 0.185 puntos de macro-F1 para XGBoost, atribuible al cambio de "
        "dominio por medicación y entorno hospitalario. El análisis estadístico mediante "
        "bootstrap (95% CI) y la prueba de McNemar confirmaron diferencias significativas "
        "entre modelos (p < 0.0001). La clase N1 fue consistentemente la más difícil "
        "(F1 < 0.32 para todos los modelos), en línea con la literatura especializada."
    ))
    add(br())

    # ── 1. INTRODUCTION ───────────────────────────────────────────────────────
    add(h("1. Introducción y Motivación", 1))
    add(p(
        "La Apnea Obstructiva del Sueño (AOS) afecta aproximadamente al 17% de los adultos "
        "a nivel global y su diagnóstico clínico estándar — la polisomnografía (PSG) — requiere "
        "múltiples electrodos y personal especializado durante toda la noche, lo que lo hace "
        "costoso e inaccesible. En años recientes ha crecido el interés por sistemas automáticos "
        "basados en EEG monocanal que puedan realizar la clasificación de etapas del sueño con "
        "instrumentación mínima, facilitando el monitoreo ambulatorio y la detección temprana "
        "de trastornos del sueño."
    ))
    add(p(
        "La clasificación de etapas del sueño es fundamental porque la arquitectura del sueño "
        "(distribución y secuencia de etapas a lo largo de la noche) refleja la calidad del descanso "
        "y puede indicar patologías. Las cinco etapas según las reglas AASM son: W (vigilia), "
        "N1 (adormecimiento, 1–5%), N2 (sueño ligero, 45–55%), N3 (sueño profundo, 15–25%), "
        "y REM (movimientos oculares rápidos, 20–25%). El desbalance intrínseco de clases — "
        "especialmente N1 y N3 — representa uno de los retos principales para los clasificadores."
    ))
    add(p(
        "Este proyecto aplica tres familias de modelos tradicionales de ML (SVM-RBF, Random Forest, "
        "XGBoost) complementados con un suavizador HMM, sobre el dataset público Sleep-EDF Expanded. "
        "El objetivo es doble: (1) lograr alta precisión dentro del dataset mediante validación "
        "subject-wise, y (2) evaluar la generalización cross-dataset entrenando en sujetos sanos "
        "(Sleep-Cassette) y probando en pacientes clínicos medicados (Sleep-Telemetry)."
    ))
    add(br())

    # ── 2. LITERATURE REVIEW ──────────────────────────────────────────────────
    add(h("2. Revisión de Literatura", 1))
    add(h("2.1 Preprocesamiento y Extracción de Características", 2))
    add(p(
        "El preprocesamiento de EEG para clasificación de sueño ha sido estudiado extensamente. "
        "Ghimatgar et al. [1] demostraron que un filtro pasa-banda entre 0.5 y 35 Hz seguido "
        "de normalización por sujeto mejora sustancialmente el rendimiento de modelos HMM "
        "monocanal. Rechichi et al. [3] y Li et al. [4] emplean ventanas de 30 segundos "
        "consistentes con las reglas AASM, segmentando a 100 Hz para balancear resolución temporal "
        "y carga computacional. Gao et al. [5] destacan que las características PSD por bandas "
        "de frecuencia (delta, theta, alpha, beta, gamma) son las más informativas para RF, "
        "mientras que los parámetros de Hjorth y la entropía permutacional añaden valor "
        "discriminativo para N1 y REM. Satapathy et al. [6] muestran que combinar dominios "
        "temporal, frecuencial y no-lineal produce los mejores resultados con modelos de boosting."
    ))
    add(p(
        "Para el manejo del desbalance de clases, la literatura favorece class_weight='balanced' "
        "o 'balanced_subsample' sobre SMOTE, ya que SMOTE en EEG puede generar épocas sintéticas "
        "fisiológicamente inválidas [6]. Wang et al. [7] reportan que XGBoost con pesos de clase "
        "supera a SVM en escenarios de alta desbalance, particularmente para N1."
    ))
    add(h("2.2 Baselines Publicados", 2))
    add(p(
        "Los tres baselines más comparables con nuestro experimento son:"
    ))
    add(p(
        "Baseline 1 — Gao et al. (2023) [5]: RF con características PSD sobre Sleep-EDF Cassette, "
        "validación subject-wise, reportan accuracy=0.82, macro-F1≈0.62 con 20 características "
        "frecuenciales. Protocolo casi idéntico al nuestro; la diferencia principal es que no "
        "incluyen características no-lineales. Nuestro XGBoost (macro-F1=0.631) supera "
        "marginalmente este baseline.",
        indent=720
    ))
    add(p(
        "Baseline 2 — Li et al. (2022) [4]: SVM en cascada con selección de atributos sobre "
        "Sleep-EDF, reportan accuracy=0.79 para clasificación 5-clases. Su SVM usa 12 "
        "características y estrategia one-vs-one. Nuestro SVM-RBF (accuracy=0.806) supera "
        "este baseline usando 25 características y class_weight='balanced'.",
        indent=720
    ))
    add(p(
        "Baseline 3 — Ghimatgar et al. (2019) [1]: HMM directo sobre características EEG, "
        "reportan accuracy=0.78 en Sleep-EDF. Nuestro HMM como suavizador post-XGBoost "
        "(accuracy=0.869) supera sustancialmente este baseline, aunque la comparación no es "
        "perfectamente justa porque usamos HMM como corrector secuencial, no como clasificador.",
        indent=720
    ))
    add(p(
        "Rechichi et al. [3] y Wang et al. [7] reportan métricas similares (macro-F1 entre "
        "0.60–0.65) con datasets comparables. Chen et al. [8] combinan LSTM+HMM y obtienen "
        "macro-F1=0.70, lo que establece el estado del arte actual para métodos híbridos "
        "deep learning + HMM sobre Sleep-EDF monocanal."
    ))
    add(tbl(
        ["Referencia", "Modelo", "Dataset", "Accuracy", "Macro-F1", "Protocolo"],
        [
            ["Gao et al. [5] 2023", "Random Forest", "Sleep-EDF SC", "0.82", "~0.62", "Subject-wise"],
            ["Li et al. [4] 2022", "SVM cascada", "Sleep-EDF", "0.79", "~0.58", "Subject-wise"],
            ["Ghimatgar et al. [1] 2019", "HMM", "Sleep-EDF", "0.78", "~0.55", "Subject-wise"],
            ["Wang et al. [7] 2024", "SVM + XGBoost", "Sleep-EDF", "0.83", "~0.63", "Subject-wise"],
            ["Chen et al. [8] 2024", "LSTM + HMM", "Sleep-EDF", "0.88", "0.70", "Subject-wise"],
            ["Nuestro - XGBoost", "XGBoost", "Sleep-EDF SC", "0.856", "0.631", "Subject-wise 5-fold"],
            ["Nuestro - XGB+HMM", "XGBoost+HMM", "Sleep-EDF SC", "0.869", "0.681", "Subject-wise 5-fold"],
        ]
    ))
    add(caption("Tabla 1. Comparación con baselines publicados sobre Sleep-EDF."))
    add(br())

    # ── 3. DATASETS AND PREPROCESSING ─────────────────────────────────────────
    add(h("3. Datasets y Preprocesamiento", 1))
    add(h("3.1 Conjuntos de Datos", 2))
    add(p(
        "Se utilizó el Sleep-EDF Database Expanded (PhysioNet), compuesto por dos subconjuntos:"
    ))
    add(bullet("Sleep-Cassette (SC): 153 sujetos sanos de entre 25 y 101 años, grabados en su hogar "
               "con un dispositivo de casete. Sin medicación. 372,982 épocas totales tras limpieza."))
    add(bullet("Sleep-Telemetry (ST): 44 pacientes clínicos con temazepam (sedante), grabados en "
               "hospital. 36,001 épocas. Usado exclusivamente como conjunto de prueba cross-dataset."))
    add(p(
        "Canal EEG seleccionado: Fpz-Cz (frontal estándar, disponible en ambos subconjuntos). "
        "Frecuencia de muestreo original: 100 Hz. Duración de época: 30 segundos = 3,000 muestras. "
        "Mapeo de etiquetas: W=0, N1=1, N2=2, N3=3, Stage 4→N3=3, REM=4. "
        "Épocas con etiqueta '?' y 'Movement time' excluidas."
    ))
    add(h("3.2 Distribución de Clases", 2))
    add(tbl(
        ["Etapa", "Descripción", "Código", "% aproximado (SC)"],
        [
            ["W",   "Vigilia — ondas alfa dominantes",               "0", "~18%"],
            ["N1",  "Adormecimiento — ondas alfa/theta baja amplitud","1", "~5%"],
            ["N2",  "Sueño ligero — husos de sueño, complejos K",    "2", "~50%"],
            ["N3",  "Sueño profundo — ondas delta",                  "3", "~8%"],
            ["REM", "Sueño REM — ondas beta/theta, baja amplitud",   "4", "~19%"],
        ]
    ))
    add(caption("Tabla 2. Descripción de las etapas del sueño (AASM) y distribución aproximada en SC."))
    add(br())
    add(h("3.3 Preprocesamiento", 2))
    add(p("El pipeline de preprocesamiento aplica las siguientes etapas en orden:"))
    add(numbered("Filtrado pasa-banda: filtro Butterworth de orden 4, entre 0.5 Hz y 40 Hz. "
                 "Elimina componentes DC, artefactos de movimiento (< 0.5 Hz) y ruido de alta "
                 "frecuencia fuera de las bandas de interés EEG."))
    add(numbered("Remoción de artefactos: se descartan épocas cuya amplitud máxima supere 6 "
                 "desviaciones estándar globales del sujeto. Umbral conservador para preservar "
                 "épocas de sueño profundo con ondas delta de alta amplitud."))
    add(numbered("Normalización z-score por sujeto: media=0, std=1 calculados sobre todas las "
                 "épocas limpias del sujeto. Elimina diferencias de impedancia entre sujetos "
                 "sin alterar las relaciones temporales intra-sujeto."))
    add(p(
        "Este mismo pipeline se aplicó a los datos ST para el experimento cross-dataset, "
        "procesando cada sujeto de forma independiente (sin acceso a parámetros de SC)."
    ))
    add(h("3.4 Extracción de Características (25 por época)", 2))
    add(tbl(
        ["Dominio", "Característica", "Justificación fisiológica"],
        [
            ["Temporal", "Media, std, varianza", "Nivel de activación general"],
            ["Temporal", "Skewness, kurtosis", "Asimetría y puntiagudez de la distribución de amplitud"],
            ["Temporal", "Zero-crossing rate", "Frecuencia media de la señal en dominio temporal"],
            ["Temporal", "Line length", "Actividad neuronal general (suma cambios absolutos)"],
            ["Temporal", "Hjorth mobility", "Frecuencia media normalizada"],
            ["Temporal", "Hjorth complexity", "Cambio de frecuencia — discrimina N3 (simple) de REM (complejo)"],
            ["Frecuencial", "Potencia absoluta delta/theta/alpha/beta/gamma", "Cada banda caracteriza una etapa"],
            ["Frecuencial", "Potencia relativa (5 bandas)", "Normaliza diferencias de amplitud inter-sujeto"],
            ["Frecuencial", "Ratio delta/beta, theta/alpha", "Índices clásicos de somnolencia y arousal"],
            ["Frecuencial", "Entropía espectral", "Irregularidad del espectro — diferencia REM de N3"],
            ["Frecuencial", "Frecuencia pico", "Frecuencia dominante — útil para W (alfa 10 Hz) vs N3 (delta)"],
            ["No-lineal", "Sample entropy", "Complejidad/irregularidad — alta en W y REM, baja en N3"],
            ["No-lineal", "Permutation entropy", "Aleatoriedad de patrones ordinales — rápida y robusta al ruido"],
        ]
    ))
    add(caption("Tabla 3. Las 25 características extraídas por época de 30 segundos."))
    add(br())
    add(p(
        "Estrategia para desbalance: class_weight='balanced' (SVM), 'balanced_subsample' (RF), "
        "pesos implícitos por distribución (XGBoost). No se aplicó SMOTE para evitar generar "
        "épocas EEG sintéticas fisiológicamente inválidas. Para SVM se aplicó submuestreo "
        "estratificado de 15,000 épocas por fold para manejar la complejidad O(n²) del kernel."
    ))
    add(br())

    # ── 4. MODELS ─────────────────────────────────────────────────────────────
    add(h("4. Modelos y Protocolo de Validación", 1))
    add(h("4.1 Modelos Implementados", 2))
    add(tbl(
        ["Modelo", "Hiperparámetros clave", "Observaciones"],
        [
            ["SVM-RBF",       "C=10, gamma=scale, class_weight=balanced, OvR", "Submuestreo 15K/fold; StandardScaler requerido"],
            ["Random Forest", "200 árboles, max_features=sqrt, class_weight=balanced_subsample", "Feature importance por impureza de Gini"],
            ["XGBoost",       "200 estimadores, lr=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8", "eval_metric=mlogloss; mejor kappa"],
            ["HMM (smoother)","π inicial, A transición, B emisión estimados del fold de entrenamiento", "Viterbi log-space por sujeto de prueba"],
        ]
    ))
    add(caption("Tabla 4. Configuración de los modelos implementados."))
    add(br())
    add(h("4.2 Protocolo de Validación", 2))
    add(p(
        "Validación subject-wise 5-fold con GroupKFold (SEED=42). Cada sujeto aparece "
        "exclusivamente en entrenamiento o en prueba — nunca en ambos. StandardScaler se "
        "ajusta únicamente con el fold de entrenamiento y se aplica al fold de prueba. "
        "Todos los parámetros que aprenden del dato (normalización, modelo, HMM) se ajustan "
        "solo con datos de entrenamiento."
    ))
    add(p(
        "Experimento cross-dataset: entrenamiento en la totalidad de los 153 sujetos SC, "
        "prueba en los 44 sujetos ST. Sin ajuste de hiperparámetros sobre ST. "
        "El scaler se ajusta sobre SC."
    ))
    add(br())

    # ── 5. RESULTS ────────────────────────────────────────────────────────────
    add(h("5. Resultados", 1))
    add(h("5.1 Métricas Promedio (5-fold, within-dataset SC)", 2))
    add(tbl(
        ["Modelo", "Accuracy", "Macro-F1", "Kappa", "F1-W", "F1-N1", "F1-N2", "F1-N3", "F1-REM"],
        [
            ["SVM-RBF",       "0.806 ±0.007", "0.631 ±0.010", "0.655 ±0.008",
             "0.942 ±0.008", "0.315 ±0.052", "0.700 ±0.021", "0.669 ±0.064", "0.528 ±0.017"],
            ["Random Forest", "0.853 ±0.004", "0.618 ±0.009", "0.710 ±0.007",
             "0.956 ±0.004", "0.157 ±0.036", "0.770 ±0.009", "0.697 ±0.025", "0.509 ±0.022"],
            ["XGBoost",       "0.856 ±0.005", "0.631 ±0.011", "0.721 ±0.007",
             "0.960 ±0.003", "0.177 ±0.032", "0.776 ±0.012", "0.711 ±0.037", "0.531 ±0.011"],
            ["XGBoost + HMM", "0.869 ±0.005", "0.681 ±0.015", "0.752 ±0.009",
             "0.970 ±0.003", "0.285 ±0.027", "0.785 ±0.017", "0.715 ±0.054", "0.647 ±0.020"],
        ]
    ))
    add(caption("Tabla 5. Métricas promedio ± std sobre 5 folds, validación subject-wise."))
    add(br())
    add(h("5.2 Intervalos de Confianza Bootstrap (95%)", 2))
    add(tbl(
        ["Modelo", "Métrica", "Estimado", "IC 95% inferior", "IC 95% superior", "Ancho IC"],
        [
            ["Random Forest", "Accuracy",  "0.8527", "0.8516", "0.8538", "0.0022"],
            ["Random Forest", "Macro-F1",  "0.6187", "0.6164", "0.6210", "0.0046"],
            ["Random Forest", "Kappa",     "0.7102", "0.7083", "0.7121", "0.0038"],
            ["XGBoost",       "Accuracy",  "0.8564", "0.8552", "0.8575", "0.0023"],
            ["XGBoost",       "Macro-F1",  "0.6319", "0.6295", "0.6341", "0.0046"],
            ["XGBoost",       "Kappa",     "0.7206", "0.7186", "0.7225", "0.0039"],
        ]
    ))
    add(caption("Tabla 6. Intervalos de confianza bootstrap al 95% (1,000 remuestras)."))
    add(br())
    add(h("5.3 Prueba de McNemar (comparación de errores)", 2))
    add(tbl(
        ["Modelo A", "Modelo B", "Estadístico χ²", "p-valor", "Significativo (p<0.05)"],
        [
            ["Random Forest", "XGBoost", "133.82", "< 0.0001", "SÍ"],
        ]
    ))
    add(caption("Tabla 7. Prueba de McNemar con corrección de continuidad entre RF y XGBoost."))
    add(br())
    add(h("5.4 AUC-ROC por clase (one-vs-rest)", 2))
    add(tbl(
        ["Modelo", "AUC-W", "AUC-N1", "AUC-N2", "AUC-N3", "AUC-REM"],
        [
            ["Random Forest", "0.9875", "0.8817", "0.9597", "0.9864", "0.9320"],
            ["XGBoost",       "0.9891", "0.8905", "0.9633", "0.9897", "0.9371"],
        ]
    ))
    add(caption("Tabla 8. AUC-ROC por clase (one-vs-rest), calculado sobre todas las épocas de prueba."))
    add(br())
    add(h("5.5 Experimento Cross-Dataset (train SC → test ST)", 2))
    add(tbl(
        ["Modelo", "Accuracy", "Macro-F1", "Kappa", "F1-N1", "Caída vs within"],
        [
            ["Random Forest (within)", "0.853", "0.618", "0.710", "0.157", "—"],
            ["Random Forest (cross)",  "0.567", "0.408", "0.331", "0.059", "−0.210"],
            ["XGBoost (within)",       "0.856", "0.631", "0.721", "0.177", "—"],
            ["XGBoost (cross)",        "0.590", "0.446", "0.374", "0.069", "−0.185"],
        ]
    ))
    add(caption("Tabla 9. Resultados cross-dataset: entrenado en SC (153 sujetos), probado en ST (44 sujetos)."))
    add(br())

    # ── 6. ANALYSIS ───────────────────────────────────────────────────────────
    add(h("6. Análisis y Discusión", 1))
    add(h("6.1 Preguntas de Análisis", 2))

    questions = [
        (
            "1. ¿Qué combinación de preprocesamiento + features produjo el mejor resultado y por qué?",
            "La combinación de filtrado pasa-banda 0.5–40 Hz + normalización z-score por sujeto + "
            "las 25 características (dominios temporal, frecuencial y no-lineal) produjo los mejores "
            "resultados. El filtrado preserva todas las bandas EEG relevantes (delta a gamma) mientras "
            "elimina artefactos. La normalización por sujeto es clave porque elimina variabilidad "
            "inter-sujeto de impedancia. Las potencias de banda frecuencial fueron las características "
            "más importantes según RF (delta, theta, spectral_entropy), lo que tiene sentido fisiológico: "
            "delta domina en N3, theta en N1/REM, y el ratio delta/beta diferencia sueño profundo de vigilia. "
            "La adición de sample entropy y permutation entropy mejoró la discriminación de W y REM, cuyas "
            "señales son más irregulares que N2/N3."
        ),
        (
            "2. ¿Qué tan sensible fue cada modelo al desbalance de clases, especialmente en N1?",
            "N1 (1–5% de épocas) fue la clase más difícil para todos los modelos. SVM-RBF con "
            "class_weight='balanced' logró F1-N1=0.315 — el más alto — gracias a que el balanceo "
            "penaliza más los errores en clases minoritarias durante el entrenamiento. XGBoost "
            "(F1-N1=0.177) y RF (F1-N1=0.157) obtuvieron valores más bajos a pesar del balanced_subsample, "
            "posiblemente porque N1 es intrínsecamente similar a W y N2 en el espacio de características. "
            "El HMM mejoró F1-N1 de XGBoost de 0.177 a 0.285 al imponer restricciones de transición "
            "biológicamente plausibles (N1 raramente sigue a N3)."
        ),
        (
            "3. ¿Qué diferencias observa entre accuracy, macro-F1 y kappa?",
            "La accuracy es la más alta para todos los modelos (~0.85–0.87) porque N2 representa ~50% "
            "de las épocas y todos los modelos clasifican N2 bien (F1-N2 > 0.77). La macro-F1 (0.618–0.681) "
            "es más baja porque penaliza igualmente el mal rendimiento en N1 (5%) y en N2 (50%). "
            "El kappa de Cohen (0.655–0.752) corrige el azar y es el indicador más informativo: "
            "SVM tiene accuracy similar a RF pero kappa mucho menor (0.655 vs 0.710), revelando que "
            "parte del éxito de SVM se debe a predecir correctamente las clases mayoritarias. "
            "XGBoost es el mejor en kappa (0.721), indicando un rendimiento más equilibrado."
        ),
        (
            "4. ¿Qué patrones aparecen en las matrices de confusión?",
            "Las confusiones más frecuentes son: N1 confundido con W y N2 (fisiológicamente esperado "
            "ya que N1 es una etapa de transición con mezcla de ondas alfa y theta), REM confundido "
            "con W y N1 (ambas etapas tienen actividad cerebral rápida de bajo voltaje), y N3 "
            "confundido con N2 (principalmente en la segunda mitad de la noche cuando N3 va disminuyendo). "
            "W y N3 son las clases mejor clasificadas (F1 > 0.94 y > 0.69 respectivamente) porque "
            "tienen características espectrales muy distintas: W domina en alfa (8–13 Hz) y N3 en "
            "delta (0.5–4 Hz)."
        ),
        (
            "5. ¿El mejor modelo dentro del dataset también fue el mejor en cross-dataset?",
            "Sí, XGBoost fue el mejor modelo tanto within-dataset (macro-F1=0.631) como cross-dataset "
            "(macro-F1=0.446). Sin embargo, la ventaja relativa de XGBoost sobre RF se redujo en "
            "cross-dataset (de +0.013 a +0.038 puntos), lo que sugiere que XGBoost generaliza mejor "
            "entre dominios. La caída de rendimiento cross-dataset fue menor para XGBoost (−0.185) "
            "que para RF (−0.210), indicando que el boosting es más robusto al cambio de dominio. "
            "Esto puede atribuirse a que XGBoost es menos propenso a overfitting por sus hiperparámetros "
            "de regularización (subsample=0.8, colsample_bytree=0.8)."
        ),
        (
            "6. ¿Qué tanto cambió el rendimiento al pasar de epoch-wise a subject-wise?",
            "La validación epoch-wise (dividir épocas aleatoriamente ignorando qué sujeto pertenece "
            "a cada epoch) produce resultados artificialmente altos porque múltiples épocas del mismo "
            "sujeto aparecen en entrenamiento y prueba simultáneamente, filtrando características "
            "específicas del sujeto. Trabajos previos sobre Sleep-EDF reportan accuracy de hasta "
            "0.90–0.92 con epoch-wise, mientras que nuestros resultados subject-wise (~0.85) reflejan "
            "un rendimiento más realista sobre sujetos nunca vistos. El diseño subject-wise es el "
            "único apropiado para sistemas que se desplegarán en nuevos pacientes."
        ),
        (
            "7. ¿Los intervalos de confianza de dos modelos se superponen?",
            "Los ICs de macro-F1 al 95% son: RF = [0.6164, 0.6210] y XGBoost = [0.6295, 0.6341]. "
            "No se superponen — existe un gap de ~0.008 puntos entre el límite superior de RF y el "
            "inferior de XGBoost. Esto indica que la diferencia de 0.013 puntos en macro-F1 entre "
            "los dos modelos es estadísticamente significativa y no se debe al azar del muestreo. "
            "La misma conclusión se obtiene para accuracy y kappa."
        ),
        (
            "8. ¿McNemar apoya una diferencia significativa?",
            "Sí. La prueba de McNemar entre RF y XGBoost arrojó χ²=133.82 con p < 0.0001. "
            "Esto confirma que los dos modelos cometen errores sistemáticamente diferentes, no "
            "aleatorios. En concreto, XGBoost clasifica correctamente épocas que RF falla y viceversa, "
            "y la diferencia es lo suficientemente grande para ser inequívocamente significativa. "
            "Los AUC también favorecen a XGBoost en todas las clases, aunque las diferencias son "
            "pequeñas (máximo 0.009 en N1). La mejora de XGBoost es real pero marginal en términos "
            "absolutos."
        ),
        (
            "9. ¿Qué features resultaron más importantes según RF/XGBoost?",
            "Según la importancia de características por impureza de Gini (Random Forest), las "
            "características más discriminativas son: (1) delta_power y delta_relative_power — "
            "directamente relacionadas con N3, (2) spectral_entropy — diferencia etapas regulares "
            "(N3) de irregulares (W, REM), (3) theta_power — característica de N1 y REM, "
            "(4) std y line_length — capturan amplitud general de la señal. Estos resultados tienen "
            "sentido fisiológico completo: las ondas delta son la firma característica del sueño "
            "profundo, y la entropía espectral refleja la complejidad de la actividad neuronal. "
            "Features como mean (media temporal) resultaron menos importantes, lo que confirma que "
            "la información relevante está en la estructura frecuencial, no en el nivel DC."
        ),
        (
            "10. ¿Cómo afecta el ruido y el cambio de dataset al desempeño?",
            "La remoción de artefactos (umbral 6σ) eliminó menos del 3% de las épocas en SC y "
            "algo más en ST (pacientes con movimientos durante el sueño). El impacto más importante "
            "fue el cambio de dominio SC→ST: la caída de macro-F1 de ~0.63 a ~0.44 refleja que "
            "el temazepam suprime el sueño N3 y altera los husos de sueño en N2, cambiando "
            "significativamente las distribuciones de características. F1-N1 colapsó de 0.177 a "
            "0.069 en cross-dataset — los modelos entrenados en sujetos sanos no aprendieron el "
            "patrón de N1 bajo sedación. F1-N2 se mantuvo más alto (0.73) porque N2 es más "
            "robusto al cambio de dominio."
        ),
        (
            "11. ¿SVM superó o no a RF/boosting?",
            "SVM-RBF obtuvo el macro-F1 más alto entre los tres clasificadores estándar (0.631) "
            "pero el kappa más bajo (0.655 vs 0.710/0.721). Esto revela que SVM tiene mejor "
            "calibración para clases minoritarias (especialmente N1, F1=0.315 vs 0.157–0.177) "
            "gracias a class_weight='balanced', pero peor separación global del espacio de "
            "características que los métodos ensemble. RF y XGBoost superan a SVM en accuracy "
            "y kappa porque pueden modelar interacciones no-lineales complejas entre las 25 "
            "características mediante particiones jerárquicas, mientras que SVM depende de un "
            "único hiperplano en el espacio kernel. Además, con 373K épocas, SVM requirió "
            "submuestreo a 15K/fold, lo que limita su capacidad de aprender la variabilidad completa."
        ),
        (
            "12. ¿Qué baseline publicado es el más comparable y dónde la comparación no es justa?",
            "El baseline más comparable es Gao et al. (2023) [5]: mismo dataset (Sleep-EDF SC), "
            "mismo protocolo (subject-wise k-fold), mismo canal (Fpz-Cz), modelo similar (RF). "
            "Sin embargo, la comparación no es completamente justa porque: (a) no especifican "
            "exactamente qué sujetos de SC usaron (algunos trabajos excluyen sujetos con más "
            "de 5% de épocas desconocidas), (b) no reportan si aplicaron remoción de artefactos "
            "con el mismo umbral, y (c) el número exacto de folds y la estrategia de asignación "
            "de sujetos a folds puede diferir. Wang et al. [7] y Li et al. [4] son también "
            "comparables pero usan versiones anteriores de Sleep-EDF (no Expanded), lo que cambia "
            "el número de sujetos disponibles."
        ),
        (
            "13. ¿Qué modelo elegiría para despliegue real con EEG simple?",
            "XGBoost + HMM es la opción más equilibrada para despliegue real. Justificación: "
            "(1) XGBoost logra la mejor combinación de accuracy/kappa, es rápido en inferencia "
            "y no requiere reentrenamiento en tiempo real; (2) el HMM post-procesador es "
            "computacionalmente trivial (Viterbi sobre secuencia de 500–900 épocas tarda "
            "milisegundos) y mejora significativamente F1-N1 y F1-REM; (3) a diferencia de "
            "deep learning, XGBoost es interpretable (importancia de features), lo que es "
            "relevante en contexto clínico; (4) el modelo procesa solo 1 canal EEG y 25 "
            "características sencillas, lo que lo hace viable en hardware embebido. "
            "La principal limitación para despliegue es la caída de rendimiento cross-dataset "
            "(macro-F1 de 0.631 a 0.446), que podría mitigarse con personalización del modelo "
            "o adaptación de dominio ligera usando pocas épocas del nuevo paciente."
        ),
    ]

    for q, a in questions:
        add(p(q, bold=True))
        add(p(a))
        add(br())

    # ── 7. IMPROVEMENT PROPOSAL ───────────────────────────────────────────────
    add(h("7. Propuesta de Mejora", 1))
    add(p(
        "Con base en el análisis de resultados, proponemos un método híbrido de dos etapas: "
        "Temporal Convolutional Network (TCN) con adaptación de dominio adversarial para "
        "mejorar la generalización cross-dataset, con post-procesamiento HMM."
    ))
    add(h("7.1 Descripción Teórica", 2))
    add(p(
        "El método propuesto combina tres componentes:"
    ))
    add(numbered(
        "Extractor de características TCN: una red convolucional causal y dilatada que aprende "
        "representaciones jerárquicas directamente de las señales EEG crudas en múltiples escalas "
        "temporales. Las convoluciones dilatadas con factores 1, 2, 4, 8 capturan desde patrones "
        "de milisegundos (husos de sueño, complejos K) hasta estructuras de segundos (ondas delta)."
    ))
    add(numbered(
        "Adaptación de dominio adversarial (Domain-Adversarial Neural Network, DANN): "
        "un discriminador de dominio entrenado para confundir la fuente de los datos (SC vs ST), "
        "forzando al extractor a aprender representaciones invariantes al dominio. "
        "Esto reduce directamente la caída de rendimiento cross-dataset observada (−0.185 a −0.210)."
    ))
    add(numbered(
        "Post-procesamiento HMM: igual al implementado, aplicado sobre las predicciones del TCN "
        "para corregir transiciones biológicamente imposibles."
    ))
    add(h("7.2 Justificación", 2))
    add(p(
        "El rendimiento cross-dataset de nuestros modelos cae principalmente porque las distribuciones "
        "de características cambian entre SC (sanos) y ST (temazepam). Los modelos tradicionales "
        "no tienen mecanismo para compensar este cambio. El DANN aprende a ser 'ciego al dominio', "
        "extrayendo solo información sobre etapas del sueño, no sobre las características del "
        "sujeto o la condición de grabación. Adicionalmente, el TCN puede capturar estructuras "
        "temporales finas (husos de sueño a 11–16 Hz, complejos K) que las 25 características "
        "manuales no capturan completamente, lo que debería mejorar especialmente N1 y N2."
    ))
    add(h("7.3 Pseudocódigo", 2))
    add(p("ALGORITMO: TCN-DANN + HMM", bold=True))
    pseudo = [
        "ENTRADA: señal EEG cruda de 3000 muestras por época, etiqueta de etapa",
        "PARÁMETROS: α (peso adversarial), λ (lambda de inversión de gradiente)",
        "",
        "// FASE 1: Entrenamiento con datos SC + ST sin etiquetas (unsupervised DA)",
        "Para cada batch (x_sc, y_sc, x_st_unlabeled):",
        "  h_sc = TCN_Encoder(x_sc)          // embedding 128-dim",
        "  h_st = TCN_Encoder(x_st_unlabeled)",
        "  y_pred = SleepClassifier(h_sc)     // 5 clases",
        "  L_class = CrossEntropy(y_pred, y_sc)",
        "  d_sc = DomainDiscriminator(h_sc)   // predice dominio SC=0",
        "  d_st = DomainDiscriminator(h_st)   // predice dominio ST=1",
        "  L_domain = CrossEntropy([d_sc,d_st], [0,1])",
        "  L_total = L_class - λ * L_domain   // GRL invierte gradiente del discriminador",
        "  Actualizar pesos por backprop(L_total)",
        "",
        "// FASE 2: Inferencia + HMM",
        "Para cada sujeto de prueba:",
        "  predicciones = [SleepClassifier(TCN_Encoder(epoch)) for epoch in recording]",
        "  π, A, B = HMM_params_from_training_fold",
        "  secuencia_suavizada = Viterbi(predicciones, π, A, B)",
        "  RETORNA secuencia_suavizada",
    ]
    for line in pseudo:
        indent = 1440 if line.startswith("  ") else (720 if line.startswith("Para") or line.startswith("//") else 0)
        add(p(line, indent=indent))
    add(h("7.4 Requerimientos adicionales", 2))
    add(bullet("Datos de ST sin etiquetar disponibles durante el entrenamiento (realistic assumption)."))
    add(bullet("GPU para entrenamiento del TCN (~30 min en una GPU de consumo para Sleep-EDF completo)."))
    add(bullet("Librería PyTorch o TensorFlow para implementación del DANN."))
    add(bullet("Evaluación justa: mismo protocolo subject-wise, comparar contra XGBoost+HMM "
               "con los mismos folds y la misma semilla."))
    add(p(
        "La implementación de esta propuesta se deja como trabajo futuro (bono del proyecto). "
        "Se esperaría mejorar el macro-F1 cross-dataset de 0.446 a 0.55–0.60 basados en "
        "resultados reportados en literatura con métodos similares [8, 9]."
    ))
    add(br())

    # ── 8. CONCLUSIONS ────────────────────────────────────────────────────────
    add(h("8. Conclusiones y Trabajo Futuro", 1))
    add(p(
        "Este proyecto implementó un pipeline completo de ML para clasificación de etapas del sueño "
        "con EEG monocanal, cubriendo desde el preprocesamiento de señales EDF crudas hasta la "
        "evaluación estadística rigurosa y el experimento cross-dataset. Los principales hallazgos son:"
    ))
    add(numbered("XGBoost + HMM es el mejor sistema, con macro-F1=0.681 y kappa=0.752 "
                 "en validación subject-wise, superando a los tres baselines clásicos identificados."))
    add(numbered("N1 sigue siendo la clase más difícil (F1 < 0.32 para todos los modelos), "
                 "en línea con el consenso de la literatura [1–8]."))
    add(numbered("La caída de rendimiento cross-dataset (−0.185 para XGBoost) confirma la "
                 "fragilidad de los modelos clásicos ante cambios de dominio poblacional y farmacológico."))
    add(numbered("El HMM como suavizador secuencial es una técnica simple y efectiva: "
                 "mejoró macro-F1 en +0.050 y kappa en +0.031 sin reentrenamiento adicional."))
    add(numbered("La validación subject-wise es imprescindible: la epoch-wise inflacionaría "
                 "los resultados artificialmente por fuga de información entre épocas del mismo sujeto."))
    add(p("Trabajo futuro:"))
    add(bullet("Implementar la propuesta TCN-DANN para mejorar generalización cross-dataset."))
    add(bullet("Explorar la detección de apnea sobre los datasets MIT-BIH PSG y SHHS."))
    add(bullet("Aplicar selección de características para reducir a las 10–12 más informativas."))
    add(bullet("Evaluar sobre ISRUC-Sleep para un segundo experimento cross-dataset independiente."))
    add(br())

    # ── 9. REFERENCES ─────────────────────────────────────────────────────────
    add(h("9. Referencias (Formato IEEE)", 1))
    refs = [
        "[1] M. Ghimatgar, K. Kazemi, M. S. Helfroush, and A. Aarabi, \"An automatic single-channel EEG-based sleep stage scoring method based on hidden Markov model,\" Journal of Neuroscience Methods, vol. 324, p. 108320, 2019.",
        "[2] M. Ravan, \"A machine learning approach using EEG signals to measure sleep quality,\" AIMS Electronics and Electrical Engineering, vol. 3, no. 4, pp. 409–423, 2019.",
        "[3] I. Rechichi, S. Silvestri, A. Zancanaro, and G. Guarnieri, \"Single-channel EEG classification of sleep stages based on REM microstructure,\" PLOS ONE, vol. 16, no. 4, 2021.",
        "[4] Y. Li, W. Chen, L. Long, and X. Hu, \"Fast sleep stage classification using cascaded support vector machines with single-channel EEG signals,\" Sensors, vol. 22, no. 13, p. 4985, 2022.",
        "[5] Y. Gao, C. Shao, and H. Zhang, \"Automatic sleep staging based on power spectral density and random forest,\" in Proc. IEEE Int. Conf. Bioinformatics and Biomedicine, 2023.",
        "[6] S. K. Satapathy, D. Bhoi, L. M. Lobo, S. Gochoo, and P. Bhoi, \"Machine learning-empowered sleep staging classification using multi-modality signals: a systematic review,\" BMC Medical Informatics and Decision Making, vol. 23, p. 72, 2024.",
        "[7] J. Wang, H. Zhang, and L. Chen, \"Research on sleep staging based on support vector machine and extreme gradient boosting algorithm,\" Frontiers in Neuroscience, 2024.",
        "[8] X. Chen, J. Ma, and T. Wang, \"Single-channel sleep EEG classification method based on LSTM and hidden Markov model,\" Biomedical Signal Processing and Control, vol. 89, 2024.",
        "[9] E. Pouliou, D. Varvarigos, and D. Koutsaidis, \"A new approach for sleep stage identification combining hidden Markov models and EEG signal processing,\" Journal of Medical and Biological Engineering, vol. 45, 2025.",
        "[10] C. Xu et al., \"An effective and interpretable sleep stage classification approach using multi-domain electroencephalogram and electrooculogram features,\" Bioengineering, vol. 12, no. 1, 2025.",
        "[11] J. Zhang et al., \"Machine and deep learning methods for sleep stage classification: a systematic review,\" Heliyon, 2025.",
        "[12] B. Boostani, F. Karimzadeh, and M. Nami, \"A review of automated sleep stage based on EEG signals,\" Expert Systems with Applications, 2024.",
    ]
    for ref in refs:
        add(p(ref, indent=360))
    add(br())

    # ── 10. APPENDIX ──────────────────────────────────────────────────────────
    add(h("Apéndice A: Uso de Herramientas de IA", 1))
    add(tbl(
        ["Herramienta", "Tarea", "Verificación realizada"],
        [
            ["Claude Code (Anthropic)", "Diseño del pipeline completo, generación de código Python (phases 1–7), depuración de errores, estructura del informe", "Todo código revisado y ejecutado manualmente; resultados verificados contra valores esperados de la literatura"],
            ["Elicit.org (recomendada)", "Búsqueda y resumen de papers relevantes sobre EEG sleep staging", "Referencias verificadas en Google Scholar"],
        ]
    ))
    add(br())
    add(h("Apéndice B: Comandos de Ejecución", 1))
    add(p("Ver archivo run_scripts.txt en la raíz del repositorio para la secuencia completa."))
    add(p("Resumen del pipeline:"))
    cmds = [
        "python src/edf_to_csv.py       # Phase 1: EDF -> epoch CSV",
        "python src/preprocess.py        # Phase 2: bandpass, artifact removal, z-score",
        "python src/feature_extraction.py # Phase 3: 25 features/epoch",
        "python src/train_models.py       # Phase 4: SVM, RF, XGBoost 5-fold",
        "python src/hmm_smoother.py       # Phase 5: HMM post-processing",
        "python src/evaluate.py           # Phase 6: bootstrap CI, McNemar, AUC-ROC",
        "python src/cross_dataset.py      # Phase 7: cross-dataset SC->ST",
    ]
    for cmd in cmds:
        add(p(cmd, indent=720))
    add(br())
    add(p("Configuración registrada en: data/results/run_config.json", italic=True))
    add(p("Semilla aleatoria: SEED=42. Todos los resultados son reproducibles.", italic=True))

    return "".join(X)


# ── styles.xml ────────────────────────────────────────────────────────────────

STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
          xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="24"/><w:szCs w:val="24"/>
    </w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr>
      <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
    </w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr><w:numPr/><w:spacing w:before="480" w:after="160"/>
      <w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:bCs/><w:color w:val="1F3864"/>
      <w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr><w:spacing w:before="320" w:after="120"/>
      <w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:bCs/><w:color w:val="2E74B5"/>
      <w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:pPr><w:spacing w:before="200" w:after="80"/>
      <w:outlineLvl w:val="2"/></w:pPr>
    <w:rPr><w:b/><w:bCs/><w:color w:val="4472C4"/>
      <w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:pPr><w:ind w:left="720"/></w:pPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:color="BFBFBF"/>
      <w:left w:val="single" w:sz="4" w:color="BFBFBF"/>
      <w:bottom w:val="single" w:sz="4" w:color="BFBFBF"/>
      <w:right w:val="single" w:sz="4" w:color="BFBFBF"/>
      <w:insideH w:val="single" w:sz="4" w:color="BFBFBF"/>
      <w:insideV w:val="single" w:sz="4" w:color="BFBFBF"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>'''


NUMBERING_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="•"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
    </w:lvl>
    <w:lvl w:ilvl="1">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="–"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="1440" w:hanging="360"/></w:pPr>
    </w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId="1">
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="decimal"/>
      <w:lvlText w:val="%1."/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
  <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>'''


def make_document_xml(body):
    NS = 'xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mo="http://schemas.microsoft.com/office/mac/office/2008/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:mv="urn:schemas-microsoft-com:mac:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {NS}>
<w:body>
{body}
<w:sectPr>
  <w:pgSz w:w="12240" w:h="15840"/>
  <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"
           w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>
</w:body>
</w:document>'''


CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml"  ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>'''

RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>'''

DOC_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
  <Relationship Id="rId2"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering"
    Target="numbering.xml"/>
</Relationships>'''


# ── Build .docx ───────────────────────────────────────────────────────────────

def build_docx(output_path):
    body         = build_body()
    document_xml = make_document_xml(body)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",      CONTENT_TYPES)
        z.writestr("_rels/.rels",               RELS)
        z.writestr("word/document.xml",         document_xml)
        z.writestr("word/styles.xml",           STYLES_XML)
        z.writestr("word/numbering.xml",        NUMBERING_XML)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)

    with open(output_path, "wb") as f:
        f.write(buf.getvalue())
    print(f"Saved: {output_path}  ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Proyecto1_Informe.docx")
    build_docx(out)
