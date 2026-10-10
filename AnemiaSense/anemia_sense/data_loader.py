"""Load anemia data or create a clearly labeled synthetic demo dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from anemia_sense.config import DEMO_DATA_PATH, SOURCE_DATA_PATH

ALIASES = {
    "gender": "gender",
    "sex": "gender",
    "age": "age",
    "hemoglobin": "hemoglobin",
    "hb": "hemoglobin",
    "mch": "mch",
    "mchc": "mchc",
    "mcv": "mcv",
    "anemia": "anemia",
    "results": "anemia",
    "result": "anemia",
    "target": "anemia",
    "label": "anemia",
}


def make_demo_dataset(rows: int = 800, seed: int = 41) -> pd.DataFrame:
    """Generate non-clinical example data for exercising the application."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["female", "male"], size=rows)
    age = np.clip(rng.normal(38, 16, size=rows), 18, 85)
    hemoglobin = rng.normal(np.where(gender == "female", 12.4, 13.5), 1.8)
    mch = np.clip(hemoglobin * 2.35 + rng.normal(0, 1.8, rows), 17, 36)
    mchc = np.clip(rng.normal(33, 2.4, rows), 25, 40)
    mcv = np.clip(hemoglobin * 5.8 + rng.normal(0, 8, rows), 55, 115)

    # This artificial rule exists only to give the demo models learnable labels.
    threshold = np.where(gender == "female", 12.0, 13.0)
    anemia = ((hemoglobin < threshold) | ((mch < 25) & (mcv < 80))).astype(int)
    flip = rng.random(rows) < 0.04
    anemia[flip] = 1 - anemia[flip]

    return pd.DataFrame(
        {
            "gender": gender,
            "age": age.round(1),
            "hemoglobin": hemoglobin.round(2),
            "mch": mch.round(2),
            "mchc": mchc.round(2),
            "mcv": mcv.round(2),
            "anemia": anemia,
        }
    )


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common source column names to the project's canonical schema."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    normalized = normalized.rename(columns={column: ALIASES.get(column, column) for column in normalized.columns})

    if "anemia" not in normalized:
        raise ValueError("The dataset must include an anemia target column (or Results/label).")

    if "age" not in normalized:
        normalized["age"] = 35
    if "gender" not in normalized:
        normalized["gender"] = "unspecified"

    required = {"hemoglobin", "mch", "mchc", "mcv"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"Dataset is missing required blood parameter columns: {', '.join(missing)}.")

    for column in ("age", "hemoglobin", "mch", "mchc", "mcv"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["gender"] = normalized["gender"].fillna("unspecified").astype(str).str.strip().str.lower()

    target = normalized["anemia"]
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.notna().all() and set(numeric_target.unique()).issubset({0, 1}):
        normalized["anemia"] = numeric_target.astype(int)
    else:
        positive = {"1", "true", "yes", "positive", "anemia", "anemic"}
        negative = {"0", "false", "no", "negative", "normal", "non-anemic", "non_anemic"}
        labels = target.astype(str).str.strip().str.lower()
        unknown = sorted(set(labels) - positive - negative)
        if unknown:
            raise ValueError(f"Unrecognized anemia labels: {', '.join(unknown[:5])}.")
        normalized["anemia"] = labels.isin(positive).astype(int)

    return normalized


def load_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load a supplied CSV, otherwise persist and use the synthetic demo."""
    source_path = Path(path) if path else SOURCE_DATA_PATH
    if source_path.exists():
        return normalize_dataset(pd.read_csv(source_path)), "Your CSV dataset"

    if path is not None:
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEMO_DATA_PATH.exists():
        make_demo_dataset().to_csv(DEMO_DATA_PATH, index=False)
    return normalize_dataset(pd.read_csv(DEMO_DATA_PATH)), "Synthetic demo dataset"
