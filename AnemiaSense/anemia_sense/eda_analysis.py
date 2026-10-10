"""Generate dataset quality summaries and exploratory charts."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from anemia_sense.config import GENERATED_ASSETS_DIR, REPORTS_DIR


def run_eda(frame: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    missing = frame.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = (missing["missing_count"] / max(len(frame), 1) * 100).round(2)
    missing.to_csv(REPORTS_DIR / "data_quality.csv")
    frame.describe(include="all").transpose().to_csv(REPORTS_DIR / "descriptive_statistics.csv")
    frame["anemia"].value_counts().sort_index().rename_axis("anemia").to_csv(REPORTS_DIR / "class_distribution.csv")

    sns.set_theme(style="whitegrid", palette=["#247c73", "#e08361"])
    figure, axis = plt.subplots(figsize=(8, 3.6))
    sns.histplot(data=frame, x="hemoglobin", hue="anemia", bins=24, element="step", stat="count", common_norm=False, ax=axis)
    axis.set(title="Hemoglobin distribution by anemia label", xlabel="Hemoglobin (g/dL)", ylabel="Records")
    figure.tight_layout()
    figure.savefig(GENERATED_ASSETS_DIR / "hemoglobin_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    numeric = frame.select_dtypes(include="number")
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Numeric feature correlation")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "rows": int(len(frame)),
        "features": int(frame.shape[1] - 1),
        "missing_cells": int(frame.isna().sum().sum()),
        "positive_cases": int(frame["anemia"].sum()),
    }
