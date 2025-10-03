import pytest
import pandas as pd
from project_name.processing import preprocess_data

def test_data_preprocess_data_1():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Moscow", "Berlin", "Moscow"],
        "income": [1000, 2000, 1500]
    })

    processed = preprocess_data(df)
    assert len(processed.columns) == 4 # age, income, is_berlin, is_moscow
    assert not processed.isna().any().any()