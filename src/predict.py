
from pathlib import Path
import sys
import argparse

import joblib

# Make the preprocessing function available when loading the saved pipeline
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from preprocessing import clean_text  # noqa: F401

ROOT = SRC_DIR.parent
MODEL_PATH = ROOT / "models" / "review_classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train it first using: python src/train.py"
        )

    return joblib.load(MODEL_PATH)


def predict_review(model, review_text):
    if not review_text or not review_text.strip():
        raise ValueError("Review text cannot be empty.")

    prediction = int(model.predict([review_text])[0])
    probabilities = model.predict_proba([review_text])[0]

    # The training labels are OR=0 and CG=1
    label_names = {
        0: "Likely Original (OR)",
        1: "Likely Computer-Generated (CG)",
    }

    predicted_class_score = float(max(probabilities)) * 100

    return label_names[prediction], predicted_class_score


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether review text is CG or OR."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Review text to classify. Omit this to use interactive mode.",
    )
    args = parser.parse_args()

    model = load_model()
    print("ReviewGuard — Terminal Prediction")
    print("Labels: CG = computer-generated, OR = original")

    if args.text is not None:
        reviews = [args.text]
    else:
        reviews = []
        while True:
            text = input("\nEnter a review (or type 'exit' to quit): ").strip()

            if text.lower() in {"exit", "quit"}:
                break

            if not text:
                print("Please enter some review text.")
                continue

            reviews.append(text)
            label, score = predict_review(model, text)
            print(f"Prediction: {label}")
            print(f"Predicted-class score: {score:.2f}%")
            reviews.clear()

        return

    for review in reviews:
        label, score = predict_review(model, review)
        print(f"\nReview: {review}")
        print(f"Prediction: {label}")
        print(f"Predicted-class score: {score:.2f}%")

    print(
        "\nNote: This model predicts dataset labels, not verified fraud. "
        "The score is not a verified probability of authenticity."
    )


if __name__ == "__main__":
    main()
