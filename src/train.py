
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import clean_text

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "fake_reviews.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place fake_reviews.csv inside data/raw/"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"text_", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_columns}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[["text_", "label"]].copy()
    df["text_"] = df["text_"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    # Keep only known labels and non-empty reviews
    df = df[df["label"].isin(["CG", "OR"])]
    df = df[df["text_"] != ""]

    # Remove exact duplicate rows
    df = df.drop_duplicates(subset=["text_", "label"])

    # Remove identical texts carrying conflicting labels
    conflicting = df.groupby("text_")["label"].nunique()
    conflicting_texts = conflicting[conflicting > 1].index
    df = df[~df["text_"].isin(conflicting_texts)].copy()

    if df["label"].nunique() != 2:
        raise ValueError("Training requires both CG and OR labels.")

    print(f"Usable reviews: {len(df):,}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # OR = 0 (original), CG = 1 (computer-generated)
    X = df["text_"]
    y = df["label"].map({"OR": 0, "CG": 1})

    # 2. Split BEFORE learning TF-IDF features
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. NLP + Machine Learning pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])

    # 4. Train
    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    # 5. Evaluate on unseen test data
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (CG class)")
    print(f"Recall   : {recall:.4f} (CG class)")
    print(f"F1-score : {f1:.4f} (CG class)")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Original (OR)", "Computer-generated (CG)"],
        zero_division=0,
    ))

    # 6. Save the complete pipeline (preprocessing + model)
    model_path = MODEL_DIR / "review_classifier.joblib"
    joblib.dump(model, model_path)

    # Save numeric evaluation results
    metrics = {
        "dataset_rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision_cg": float(precision),
        "recall_cg": float(recall),
        "f1_cg": float(f1),
        "label_mapping": {
            "OR": "Original",
            "CG": "Computer-generated",
        },
    }

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save confusion matrix image
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        display_labels=["Original", "Computer-generated"],
    )
    display.plot(values_format="d")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {REPORT_DIR / 'metrics.json'}")
    print(f"Saved confusion matrix: {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
