from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Expected placement: <project-root>/backend/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# The saved sklearn Pipeline references this function by its module name.
# Importing it before joblib.load registers the preprocessing module.
try:
    from preprocessing import clean_text  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "Could not import src/preprocessing.py. Keep backend/ beside src/ "
        "and confirm src/preprocessing.py defines clean_text()."
    ) from exc

MODEL_PATH = PROJECT_ROOT / "models" / "review_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.png"

app = FastAPI(
    title="ReviewGuard API",
    description="NLP review classification API for ReviewGuard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any = None


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Trained model not found: {MODEL_PATH}. Run python src/train.py first.",
            )
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Unable to load the saved model. Run the API with the same Python "
                    "environment and compatible scikit-learn version used for training. "
                    f"Details: {exc}"
                ),
            ) from exc
    return _model


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def class_details(prediction: int) -> tuple[str, str, str]:
    if prediction == 1:
        return (
            "Likely computer-generated",
            "CG",
            "Patterns associated with the computer-generated class were detected.",
        )
    return (
        "Likely original",
        "OR",
        "Patterns associated with the original class were detected.",
    )


def classify_texts(texts: list[str]) -> list[dict[str, Any]]:
    model = get_model()
    predictions = model.predict(texts)
    probability_rows = model.predict_proba(texts)
    model_classes = [int(value) for value in model.classes_]
    output: list[dict[str, Any]] = []

    for index, raw_prediction in enumerate(predictions):
        prediction = int(raw_prediction)
        label, label_code, explanation = class_details(prediction)
        row_scores = probability_rows[index]
        class_probabilities = {
            str(class_code): float(row_scores[model_classes.index(class_code)])
            for class_code in model_classes
        }
        predicted_score = class_probabilities[str(prediction)] * 100

        output.append(
            {
                "prediction": label,
                "label_code": label_code,
                "predicted_class_score": round(predicted_score, 2),
                "cg_score": round(class_probabilities.get("1", 0.0) * 100, 2),
                "or_score": round(class_probabilities.get("0", 0.0) * 100, 2),
                "explanation": explanation,
            }
        )
    return output


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "ReviewGuard API",
        "model_file_exists": MODEL_PATH.exists(),
    }


@app.get("/api/metrics")
def metrics():
    saved = load_metrics()
    return {
        **saved,
        "model_name": "TF-IDF + Logistic Regression",
        "labels": {"OR": "Original", "CG": "Computer-generated"},
        "confusion_matrix_available": CONFUSION_MATRIX_PATH.exists(),
    }


@app.get("/api/confusion-matrix")
def confusion_matrix():
    if not CONFUSION_MATRIX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Confusion matrix not found. Run python src/train.py first.",
        )
    return FileResponse(
        CONFUSION_MATRIX_PATH,
        media_type="image/png",
        filename="confusion_matrix.png",
    )


@app.post("/api/predict")
def predict(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Review text cannot be blank.")
    result = classify_texts([request.text.strip()])[0]
    return {
        **result,
        "note": (
            "This predicts the training dataset's CG/OR labels. It is not proof of fraud "
            "or authenticity, and the model score is not a verified fraud probability."
        ),
    }


@app.post("/api/batch")
async def batch_predict(
    file: UploadFile = File(...),
    text_column: str = Form("auto"),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw_bytes = await file.read()

    # Allow CSV files up to 25 MB
    if len(raw_bytes) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="CSV must be smaller than 25 MB.",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")

    # Allow up to 50,000 rows
    if len(df) > 50000:
        raise HTTPException(
            status_code=413,
            detail="Please upload at most 50,000 rows at a time.",
        )

    if text_column == "auto":
        candidates = ["text_", "Review_Text", "review_text", "review", "text"]
        selected_column = next(
            (name for name in candidates if name in df.columns),
            None,
        )

        if selected_column is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not detect a review-text column. "
                    "Use text_ or Review_Text, or select a column."
                ),
            )
    else:
        selected_column = text_column

        if selected_column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{selected_column}' was not found in the CSV.",
            )

    text_series = df[selected_column].fillna("").astype(str).str.strip()
    valid_mask = text_series.ne("")
    valid_texts = text_series[valid_mask].tolist()

    if not valid_texts:
        raise HTTPException(
            status_code=422,
            detail="No non-empty reviews found in that column.",
        )

    predictions = classify_texts(valid_texts)

    results = df.copy().astype(object).where(pd.notna(df), None)
    results["prediction"] = "No text provided"
    results["label_code"] = ""
    results["model_score_percent"] = None
    results["cg_score_percent"] = None
    results["or_score_percent"] = None

    valid_indices = list(results.index[valid_mask])

    for row_index, result in zip(valid_indices, predictions):
        results.at[row_index, "prediction"] = result["prediction"]
        results.at[row_index, "label_code"] = result["label_code"]
        results.at[row_index, "model_score_percent"] = (
            result["predicted_class_score"]
        )
        results.at[row_index, "cg_score_percent"] = result["cg_score"]
        results.at[row_index, "or_score_percent"] = result["or_score"]

    records = json.loads(results.to_json(orient="records", force_ascii=False))
    cg_count = sum(1 for item in predictions if item["label_code"] == "CG")
    or_count = sum(1 for item in predictions if item["label_code"] == "OR")

    return {
        "filename": file.filename,
        "text_column": selected_column,
        "total_rows": int(len(df)),
        "analyzed_rows": len(valid_texts),
        "skipped_rows": int(len(df) - len(valid_texts)),
        "cg_count": cg_count,
        "or_count": or_count,
        "rows": records,
        "note": (
            "Predictions reflect CG/OR training labels and do not verify "
            "fraud or authenticity."
        ),
    }
