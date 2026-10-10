
from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# Ensure Python can find src/preprocessing.py when loading the model
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# Required because the saved model references clean_text
from preprocessing import clean_text  # noqa: F401

ROOT = SRC_DIR.parent
DATA_PATH = ROOT / "data" / "raw" / "fake_reviews.csv"
MODEL_PATH = ROOT / "models" / "review_classifier.joblib"
REPORT_DIR = ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Reproduce the same data filtering used in train.py
    df = pd.read_csv(DATA_PATH)

    required = {"text_", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain {required}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[["text_", "label"]].copy()
    df["text_"] = df["text_"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    df = df[df["label"].isin(["CG", "OR"])]
    df = df[df["text_"] != ""]
    df = df.drop_duplicates(subset=["text_", "label"])

    # Exclude identical review text with conflicting labels
    label_counts = df.groupby("text_")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    df = df[~df["text_"].isin(conflicting_texts)].copy()

    X = df["text_"]
    y = df["label"].map({"OR": 0, "CG": 1})

    # Must match the train.py test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    metrics = {
        "dataset_rows_used": int(len(df)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_cg": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall_cg": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1_cg": float(
            f1_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "label_mapping": {
            "OR": "Original",
            "CG": "Computer-generated",
        },
    }

    print("\n--- Saved Model Evaluation ---")
    print(f"Usable reviews: {len(df):,}")
    print(f"Test reviews:   {len(X_test):,}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision (CG): {metrics['precision_cg']:.4f}")
    print(f"Recall (CG):    {metrics['recall_cg']:.4f}")
    print(f"F1-score (CG):  {metrics['f1_cg']:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Original (OR)", "Computer-generated (CG)"],
            zero_division=0,
        )
    )

    # Save evaluation metrics
    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Original (OR)", "Computer-generated (CG)"],
        values_format="d",
    )
    display.ax_.set_title("Review Classifier — Test Set")
    plt.tight_layout()
    image_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {image_path}")


if __name__ == "__main__":
    main()
