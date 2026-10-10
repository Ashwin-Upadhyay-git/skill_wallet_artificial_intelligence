"""Prepare model inputs and preprocessing pipelines."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from anemia_sense.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and rows without a usable target; retain feature gaps for imputation."""
    cleaned = frame.drop_duplicates().copy()
    return cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}.")
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(int).copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def preprocessing_configuration() -> dict:
    return {
        "features": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one-hot; unknown categories ignored",
        "numeric_scaling": "standard scaling",
        "missing_age_source_behavior": "age defaults to 35 when the source CSV has no age column",
    }
