"""Train, compare, and persist AnemiaSense classification models."""

import json
import pickle
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from anemia_sense.config import (
    DATA_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSING_CONFIG_PATH,
    PROCESSED_DATA_PATH,
    REPORTS_DIR,
)
from anemia_sense.data_loader import load_dataset
from anemia_sense.data_preprocessing import (
    build_preprocessor,
    clean_dataset,
    preprocessing_configuration,
    split_features_target,
)
from anemia_sense.eda_analysis import run_eda


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=41),
        "Random Forest": RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=41),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=41),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(class_weight="balanced", probability=True, random_state=41),
    }


def train_and_evaluate(dataset_path=None) -> dict:
    frame, data_source = load_dataset(dataset_path)
    frame = clean_dataset(frame)
    features, target = split_features_target(frame)
    class_counts = target.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 2:
        raise ValueError("Training requires at least two examples from each anemia class.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DATA_PATH, index=False)
    data_summary = run_eda(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=41, stratify=target
    )

    model_results = {}
    fitted_models = {}
    for name, estimator in candidate_models().items():
        model = Pipeline([("preprocessing", build_preprocessor()), ("classifier", estimator)])
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        model_results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test, predicted, labels=[0, 1], target_names=["not flagged", "flagged"], output_dict=True, zero_division=0
            ),
        }
        fitted_models[name] = model

    best_name = max(model_results, key=lambda name: model_results[name]["f1"])
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(fitted_models[best_name], model_file)

    report = {
        "data_source": data_source,
        "evaluation": "stratified 80/20 holdout; not clinical validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "selection_metric": "f1",
        "data_summary": data_summary,
        "models": model_results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PREPROCESSING_CONFIG_PATH.write_text(
        json.dumps(preprocessing_configuration(), indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = train_and_evaluate()
    print(f"Best model: {result['best_model']} (F1 {result['models'][result['best_model']]['f1']:.3f})")
    print(f"Data source: {result['data_source']}")
