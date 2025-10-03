import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns

    processed_parts = []

    if len(numeric_cols) > 0:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data[numeric_cols])
        scaled_df = pd.DataFrame(scaled, columns=numeric_cols, index=data.index)
        processed_parts.append(scaled_df)

    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(data[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)
        processed_parts.append(encoded_df)

    if processed_parts:
        return pd.concat(processed_parts, axis=1)
    else:
        return data.copy()

def postprocess_predictions(predictions):
    return predictions
