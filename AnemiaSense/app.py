"""Flask application for the AnemiaSense educational dashboard."""

import json
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

from anemia_sense.config import METRICS_PATH, MODEL_PATH
from anemia_sense.workflow import train_and_evaluate

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024


def load_model_and_metrics():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_evaluate()
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return model, metrics


@app.get("/")
def index():
    _, metrics = load_model_and_metrics()
    best = metrics["models"][metrics["best_model"]]
    return render_template("index.html", metrics=metrics, best=best)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    try:
        values = {
            "gender": str(payload.get("gender", "")).strip().lower(),
            "age": 35.0,
            "hemoglobin": float(payload.get("hemoglobin", "")),
            "mch": float(payload.get("mch", "")),
            "mchc": float(payload.get("mchc", "")),
            "mcv": float(payload.get("mcv", "")),
        }
        if values["gender"] not in {"0", "1"}:
            raise ValueError("Choose a gender code from the dataset (0 or 1).")
        bounds = {
            "age": (1, 120),
            "hemoglobin": (1, 30),
            "mch": (1, 60),
            "mchc": (1, 60),
            "mcv": (1, 200),
        }
        for field, (minimum, maximum) in bounds.items():
            if not minimum <= values[field] <= maximum:
                raise ValueError(f"Enter a plausible numeric value for {field.replace('_', ' ')}.")
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    model, metrics = load_model_and_metrics()
    probability = float(model.predict_proba(pd.DataFrame([values]))[0][1])
    flagged = bool(model.predict(pd.DataFrame([values]))[0])
    return jsonify(
        {
            "label": "Model flagged anemia pattern" if flagged else "Model did not flag anemia pattern",
            "class": int(flagged),
            "probability": round(probability, 3),
            "model": metrics["best_model"],
            "notice": "Educational estimate only. It is not a diagnosis or a substitute for a clinician.",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
