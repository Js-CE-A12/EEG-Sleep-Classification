import glob
import pandas as pd
import os

dataset_name = os.environ.get("DATASET")
if not dataset_name:
    raise ValueError("Debes definir la variable de entorno DATASET con el nombre de la carpeta del dataset")

input_folder = f"data/{dataset_name}"
output_folder = f"data/processed/{dataset_name}"

os.makedirs(output_folder, exist_ok=True)

# Tomamos todos los CSV dentro del dataset
input_files = glob.glob(f"{input_folder}/*.csv")

if not input_files:
    print(f"No se encontraron CSV en {input_folder}. El contenedor está listo para procesar datos más tarde.")
else:
    for file_path in input_files:
        print(f"Procesando {file_path}")
        df = pd.read_csv(file_path)
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(output_path, index=False)

    print(f"Procesamiento terminado para el dataset: {dataset_name}")

