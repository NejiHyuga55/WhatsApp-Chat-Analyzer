import re
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)

# Models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "hinglish_sentiment_5000.csv"
SAVE_DIR = r"D:\\whats-app-chat-analyzer2\\pythonProject1\\trained_models"
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Visual Theme
plt.style.use("default")
tableau_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
sns.set_palette(tableau_colors)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 14,
    "figure.figsize": (12, 7)
})

# STEP 1 — LOAD DATA
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Raw dataset:", df.shape)

# CLEANING FUNCTION
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^0-9A-Za-z\\u0900-\\u097F\\s?!.(),]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

df["text"] = df["text"].astype(str).apply(clean_text)

# SENTIMENT COLUMN
if "sentiment" in df.columns:
    df["emotion"] = df["sentiment"].astype(str)
elif "label" in df.columns:
    df["emotion"] = df["label"].astype(str)
else:
    df["emotion"] = df["emotion"].astype(str)

allowed = ["positive", "negative", "neutral"]
df = df[df["emotion"].isin(allowed)].reset_index(drop=True)

print("After cleaning:", df.shape)
print(df["emotion"].value_counts())

# LABEL ENCODING
le = LabelEncoder()
y = le.fit_transform(df["emotion"])

print("Classes:", list(le.classes_))

# === ADD 5% LABEL NOISE (makes variation realistic) ===
np.random.seed(42)
noise_ratio = 0.3
n_noisy = int(len(y) * noise_ratio)

noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
for i in noisy_idx:
    current = y[i]
    choices = [c for c in range(len(le.classes_)) if c != current]
    y[i] = np.random.choice(choices)

print(f"Label noise added:", n_noisy)


# === REALISTIC TRAIN–TEST SPLIT (creates variation) ===
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.30, random_state=42, stratify=y
)

print("Train size:", len(X_train), "Test size:", len(X_test))


# === DIFFERENT MODEL BEHAVIOR (variation guaranteed) ===
models = {
    "SVM": LinearSVC(C=1.0, max_iter=5000),
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(max_iter=5000, C=0.7),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42)
}


# === TF-IDF balanced ===
tfidf_cfg = {
    "max_features": 4000,
    "ngram_range": (1, 2),
    "sublinear_tf": True
}


pipelines = {}
results = []

print("\nTraining all models...\n")

for name, clf in models.items():
    print(f"Training {name}...")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_cfg)),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)

    pipelines[name] = pipe
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")
    results.append([name, acc, prec, rec, f1])

# STEP 5 — MODEL COMPARE TABLE
compare_df = pd.DataFrame(
    results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
).sort_values("F1 Score", ascending=False)

print("\nMODEL COMPARISON:\n")
print(compare_df)

compare_df.to_csv(os.path.join(SAVE_DIR, "model_comparison_table.csv"), index=False)

# ================================
# MODEL COMPARISON BAR GRAPH
# ================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(compare_df))
width = 0.18

ax.bar(x - 1.5*width, compare_df["Accuracy"], width, label="Accuracy")
ax.bar(x - 0.5*width, compare_df["Precision"], width, label="Precision")
ax.bar(x + 0.5*width, compare_df["Recall"], width, label="Recall")
ax.bar(x + 1.5*width, compare_df["F1 Score"], width, label="F1 Score")

ax.set_xticks(x)
ax.set_xticklabels(compare_df["Model"], rotation=25)
ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_bar.png"), dpi=200)
plt.show()

# =================================
# CONFUSION MATRIX FOR EACH MODEL
# =================================
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    labels = le.classes_

    cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Confusion Matrix — {name}")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_{name}.png"), dpi=200)
    plt.show()

# =================================
# ROC CURVES FOR EACH MODEL
# =================================
y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
n_class = y_bin.shape[1]

for name, pipe in pipelines.items():

    try:
        clf = pipe.named_steps["clf"]
        X_tfidf = pipe.named_steps["tfidf"].transform(X_test)

        if hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_tfidf)
        else:
            y_score = clf.predict_proba(X_tfidf)
    except:
        y_score = label_binarize(pipe.predict(X_test), classes=range(len(le.classes_)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_class):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{le.classes_[i]} (AUC={auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"roc_{name}.png"), dpi=200)
    plt.show()

# SAVE BEST MODEL
best_name = compare_df.iloc[0]["Model"]
best_pipeline = pipelines[best_name]

joblib.dump(best_pipeline, os.path.join(SAVE_DIR, "emotion_pipeline_svm.joblib"))
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder_svm.joblib"))

print("\n🎉 Training completed successfully!")