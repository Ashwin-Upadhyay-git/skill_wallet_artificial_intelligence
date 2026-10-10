from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
GENERATED_ASSETS_DIR = ROOT_DIR / "static" / "generated"

SOURCE_DATA_PATH = ROOT_DIR / "anemia.csv"
DEMO_DATA_PATH = DATA_DIR / "demo_anemia.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_anemia.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
PREPROCESSING_CONFIG_PATH = REPORTS_DIR / "preprocessing_config.json"

FEATURE_COLUMNS = ["gender", "age", "hemoglobin", "mch", "mchc", "mcv"]
NUMERIC_FEATURES = ["age", "hemoglobin", "mch", "mchc", "mcv"]
CATEGORICAL_FEATURES = ["gender"]
TARGET_COLUMN = "anemia"
