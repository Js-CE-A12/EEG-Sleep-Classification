# Data Science Project Template (Python + Docker)

Plantilla base para cursos/proyectos de Data Science donde el dataset final lo entrega el profesor.

## Objetivo

Este template permite:
- Cargar un dataset (`.csv`) desde `data/raw/`
- Ejecutar un pipeline de preprocesamiento
- Guardar datos limpios en `data/processed/`
- Ejecutar todo localmente o dentro de Docker

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/            # datasets originales (no versionar archivos grandes)
│   └── processed/      # datasets procesados
├── notebooks/          # notebooks de exploración
├── reports/            # salidas, métricas, figuras
├── src/
│   ├── main.py         # punto de entrada del pipeline
│   └── pipeline.py     # funciones de procesamiento
├── tests/              # pruebas automáticas
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Requisitos

- Docker Desktop (recomendado para todos los estudiantes)
- Opcional: Python 3.11+ para ejecutar sin Docker

## Uso con Docker (recomendado)

1. Copia tu dataset a `data/raw/`, por ejemplo: `data/raw/profesor_dataset.csv`
2. Construye la imagen:

```bash
docker compose build
```

3. Ejecuta el pipeline:

```bash
docker compose run --rm app python src/main.py --input data/raw/profesor_dataset.csv --output data/processed/dataset_limpio.csv
```

## Uso local (sin Docker)

1. Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Ejecutar:

```bash
python src/main.py --input data/raw/profesor_dataset.csv --output data/processed/dataset_limpio.csv
```

## Qué hace el pipeline de ejemplo

- Elimina filas completamente vacías
- Elimina duplicados
- Estandariza nombres de columnas (`snake_case`)
- Rellena valores faltantes:
  - Numéricos -> mediana
  - Categóricos -> moda
- Guarda un reporte simple en consola

## Pruebas

```bash
pytest -q
```

## Sugerencias para estudiantes

- Mantener notebooks para exploración y mover lógica final a `src/`
- Separar etapas: limpieza, features, entrenamiento, evaluación
- Versionar código, no datasets pesados
- Documentar decisiones y supuestos

