# AnemiaSense

AnemiaSense is an educational machine-learning application for exploring anemia classification from the provided blood-test dataset. It is not a medical device: predictions are not diagnoses, and model scores do not represent clinical performance.

## Requirements

- Python 3.10 or newer
- A virtual environment

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. On first launch, AnemiaSense loads the project-root `anemia.csv`, runs analysis and model comparison, then saves the processed CSV, model, metrics, preprocessing configuration, and charts locally.

## Use your own dataset

The included `anemia.csv` has `Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`, and `Result` columns. The loader accepts common capitalization and maps `Result` to the anemia target. `Gender` is retained as source codes 0 and 1 for prediction inputs. This dataset has no age column, so age is filled with 35 during preprocessing and recorded in the preprocessing configuration. Exact duplicate records are removed before training. Do not use identifiable patient information.

Run the workflow explicitly with:

```bash
python -m anemia_sense.workflow
```

The workflow compares Logistic Regression, Random Forest, Decision Tree, Gaussian Naive Bayes, and Support Vector Machine models using a stratified holdout split. It selects the highest-F1 model on that split. This is a dataset comparison, not clinical validation or hyperparameter optimization.

## Project layout

```text
anemia_sense/       Data loading, preprocessing, EDA, and training workflow
data/               Source and processed CSV files
models/             Serialized best model
reports/            Metrics, confusion matrices, and analysis summaries
static/              Dashboard styles, scripts, and generated chart
templates/           Flask page template
tests/               Focused workflow tests
```

## Checks

```bash
pytest
```
