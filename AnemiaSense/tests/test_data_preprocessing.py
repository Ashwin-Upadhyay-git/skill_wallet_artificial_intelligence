import pandas as pd
import pytest

from anemia_sense.data_loader import normalize_dataset
from anemia_sense.data_preprocessing import clean_dataset, split_features_target


def test_normalize_common_source_column_names_and_text_labels():
    source = pd.DataFrame(
        {
            "Gender": ["Female", "Male"],
            "Hemoglobin": [10.5, 14.2],
            "MCH": [23.0, 29.0],
            "MCHC": [30.0, 34.0],
            "MCV": [72.0, 90.0],
            "Results": ["anemic", "normal"],
        }
    )

    normalized = normalize_dataset(source)
    features, target = split_features_target(clean_dataset(normalized))

    assert list(features.columns) == ["gender", "age", "hemoglobin", "mch", "mchc", "mcv"]
    assert features["age"].tolist() == [35, 35]
    assert target.tolist() == [1, 0]


def test_normalize_rejects_missing_blood_parameters():
    source = pd.DataFrame({"Results": [0, 1], "Hemoglobin": [13, 10]})

    with pytest.raises(ValueError, match="missing required blood parameter"):
        normalize_dataset(source)
