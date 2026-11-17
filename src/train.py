"""
Training pipeline: data prep -> EDA -> model comparison -> final model -> evaluation.

Run from the project root:   python src/train.py
Outputs: model/ticket_classifier.joblib, model/metrics.json, screenshots/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import ROOT, clean_dataframe, load_raw

SHOTS = ROOT / "screenshots"
MODEL_DIR = ROOT / "model"
SHOTS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def save_text_image(text, filename, title):
    """Save console-style output as an image (used for the screenshots folder)."""
    lines = text.split("\n")
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(lines) + 1))
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0, 1, text, family="monospace", fontsize=8.5, va="top")
    fig.savefig(SHOTS / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Task 1
section("TASK 1 - DATA PREPARATION")
raw = load_raw()
print(raw.head(10).to_string())
print("\nShape:", raw.shape)
print("\nMissing values per column:\n", raw.isna().sum())
print("\nExact duplicate rows (ignoring ticket_id):",
      raw.drop(columns="ticket_id").duplicated().sum())
save_text_image(raw.head(15).to_string(), "dataset.png", "Raw dataset (first 15 rows)")

df = clean_dataframe(raw)
print("\nDataset info after cleaning:")
df.info()
print("\nSample cleaned text:")
print(df[["ticket_description", "clean_description"]].head(5).to_string())

# ------------------------------------------------------------------ Task 2
section("TASK 2 - EXPLORATORY DATA ANALYSIS")
print("Total tickets:", len(df))
for col in ["category", "priority", "status"]:
    print(f"\nTickets by {col}:\n{df[col].value_counts().to_string()}")
df["word_count"] = df["clean_description"].str.split().str.len()
print("\nWords per ticket:\n", df["word_count"].describe().round(1).to_string())
print("\nPriority by category (counts):")
print(pd.crosstab(df["category"], df["priority"]).to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
order = df["category"].value_counts().index
sns.countplot(data=df, y="category", order=order, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Chart 1: Category distribution")
sns.countplot(data=df, x="priority", order=["High", "Medium", "Low", "Unknown"], ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Chart 2: Priority distribution")
sns.countplot(data=df, x="status", ax=axes[1, 0], color="#C44E52")
axes[1, 0].set_title("Chart 3: Status distribution")
sns.countplot(data=df, y="category", hue="priority", order=order,
              hue_order=["High", "Medium", "Low", "Unknown"], ax=axes[1, 1])
axes[1, 1].set_title("Chart 4: Priority within each category")
plt.tight_layout()
plt.savefig(SHOTS / "analysis.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ Task 3 / 4
section("TASK 3 & 4 - MODEL SELECTION, TRAINING AND TESTING")
X, y = df["clean_description"], df["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")


def make_pipeline(clf):
    # TF-IDF with unigrams + bigrams ("not working", "too slow" are useful phrases)
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", clf),
    ])


candidates = {
    "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.5)),
    "Linear SVM": make_pipeline(LinearSVC(random_state=RANDOM_STATE)),
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, C=10, random_state=RANDOM_STATE)),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
print("\n5-fold cross-validation on the TRAINING set only (macro F1):")
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"  {name:<20} {scores.mean():.3f} (+/- {scores.std():.3f})")

# Logistic Regression is chosen: it is simple, explainable and gives probabilities
# (needed for the confidence score). Choice is made from CV on training data, the
# test set is touched only once, below.
model = candidates["Logistic Regression"]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
pw, rw, fw, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
print(f"\nTEST RESULTS (Logistic Regression)")
print(f"Accuracy           : {acc:.3f}")
print(f"Precision (macro)  : {p:.3f}   (weighted {pw:.3f})")
print(f"Recall    (macro)  : {r:.3f}   (weighted {rw:.3f})")
print(f"F1 score  (macro)  : {f1:.3f}   (weighted {fw:.3f})")
report_text = classification_report(y_test, y_pred, zero_division=0)
print("\nPer-class report:\n" + report_text)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

fig, ax = plt.subplots(figsize=(8, 6.5))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
ax.set_title("Confusion matrix - test set")
plt.tight_layout()
plt.savefig(SHOTS / "confusion_matrix.png", dpi=130)
plt.close()

# Misclassified tickets - useful for the error analysis in the report
wrong = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": y_pred})
wrong = wrong[wrong.actual != wrong.predicted]
print(f"\nMisclassified test tickets: {len(wrong)}")
if len(wrong):
    print(wrong.to_string(index=False))

# Top words per class - explains HOW the model decides
tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
vocab = np.array(tfidf.get_feature_names_out())
print("\nTop 6 features per category (highest logistic-regression weights):")
for i, cls in enumerate(clf.classes_):
    print(f"  {cls:<18}", ", ".join(vocab[np.argsort(clf.coef_[i])[-6:][::-1]]))

# Retrain on ALL data for the deployed model (standard practice after evaluation)
final_model = make_pipeline(LogisticRegression(max_iter=1000, C=10, random_state=RANDOM_STATE))
final_model.fit(X, y)
joblib.dump(final_model, MODEL_DIR / "ticket_classifier.joblib")

metrics = {
    "n_train": len(X_train), "n_test": len(X_test),
    "accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1,
    "cv_macro_f1": {k: {"mean": v[0], "std": v[1]} for k, v in cv_results.items()},
    "n_misclassified": int(len(wrong)),
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nSaved model to {MODEL_DIR / 'ticket_classifier.joblib'}")
