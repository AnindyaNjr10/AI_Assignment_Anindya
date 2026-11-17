"""
Task 5 + CLI interface (Task 6 Option A) + bonus suggested reply.

  python src/predict.py          -> interactive command-line app
  python src/predict.py --test   -> run the documented test cases on new tickets
"""
import sys
from pathlib import Path

import joblib

from preprocessing import ROOT, clean_text
from responses import suggested_response

MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"

# 10 NEW tickets written after training; none of them appear in data/tickets.csv.
TEST_CASES = [
    ("I forgot my password again and the reset mail is not coming", "Login Issue"),
    ("The software crashed and showed an error when I pressed save", "Application Error"),
    ("Kindly send me last week's sales report", "Report"),
    ("Please change the phone number linked with my profile", "Account Update"),
    ("Website is taking ages to open this afternoon", "Performance"),
    ("You have deducted money twice from my card", "Billing Issue"),
    ("Cannot sign in, it keeps saying my credentials are wrong", "Login Issue"),
    ("Need an Excel report of all customers who joined this year", "Report"),
    ("Please update the address on my account", "Account Update"),
    ("Everything is lagging and the screen hangs frequently", "Performance"),
]


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_ticket(model, text: str):
    """Return (category, confidence_percent) for one ticket description."""
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    idx = proba.argmax()
    return model.classes_[idx], float(proba[idx] * 100)


def run_tests(model):
    print(f"{'Test input':<65}{'Expected':<19}{'Predicted':<19}{'Conf.':>7}  OK")
    print("-" * 115)
    correct = 0
    for text, expected in TEST_CASES:
        pred, conf = predict_ticket(model, text)
        ok = pred == expected
        correct += ok
        print(f"{text:<65}{expected:<19}{pred:<19}{conf:6.1f}%  {'Y' if ok else 'N'}")
    print(f"\nCorrect: {correct}/{len(TEST_CASES)}")


def cli(model):
    print("=" * 40)
    print(" Customer Support Classifier")
    print("=" * 40)
    print("Type a ticket description (or 'quit' to exit).\n")
    while True:
        text = input("Enter Ticket Description:\n> ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            print("Please enter some text.\n")
            continue
        category, conf = predict_ticket(model, text)
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {conf:.1f}%")
        print(f"Suggested Response: {suggested_response(category)}\n")


if __name__ == "__main__":
    mdl = load_model()
    run_tests(mdl) if "--test" in sys.argv else cli(mdl)
