from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class PipelineReport:
    input_rows: int
    output_rows: int
    removed_empty_rows: int
    removed_duplicates: int


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, PipelineReport]:
    input_rows = len(df)

    clean = df.copy()
    clean.columns = [to_snake_case(col) for col in clean.columns]

    before_dropna = len(clean)
    clean = clean.dropna(how="all")
    removed_empty_rows = before_dropna - len(clean)

    before_dedup = len(clean)
    clean = clean.drop_duplicates()
    removed_duplicates = before_dedup - len(clean)

    numeric_columns = clean.select_dtypes(include=[np.number]).columns
    object_columns = clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_columns:
        median_value = clean[col].median()
        clean[col] = clean[col].fillna(median_value)

    for col in object_columns:
        mode = clean[col].mode(dropna=True)
        if not mode.empty:
            clean[col] = clean[col].fillna(mode.iloc[0])

    report = PipelineReport(
        input_rows=input_rows,
        output_rows=len(clean),
        removed_empty_rows=removed_empty_rows,
        removed_duplicates=removed_duplicates,
    )
    return clean, report

