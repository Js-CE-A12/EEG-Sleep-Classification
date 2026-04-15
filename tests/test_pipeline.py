import pandas as pd

from src.pipeline import preprocess_dataframe


def test_preprocess_dataframe_basic_cleanup() -> None:
    df = pd.DataFrame(
        {
            "Edad ": [20, 20, None, None],
            "Ciudad": ["Lima", "Lima", "Quito", None],
        }
    )

    clean_df, report = preprocess_dataframe(df)

    assert "edad" in clean_df.columns
    assert "ciudad" in clean_df.columns
    assert report.input_rows == 4
    assert report.output_rows == 2
    assert report.removed_duplicates == 1
    assert report.removed_empty_rows == 1
    assert clean_df["edad"].isna().sum() == 0

