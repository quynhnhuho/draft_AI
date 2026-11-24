# LƯU Ý: script thử nghiệm cũ, KHÔNG dùng để tạo mô hình cho chatbot.
# Mô hình chatbot (models/intent_model.keras, tokenizer.json, config.json…) do
# notebooks/N31_De03_Train.ipynb sinh ra; script này ghi kết quả vào models/thu_nghiem/.
"""
train_nb.py - Baseline Naive Bayes (TF-IDF + MultinomialNB) cho bai toan phan loai y dinh.

Chay:
    cd d:/N31_Nhap
    python src/nlu/train_nb.py
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ── Thêm thư mục gốc dự án vào sys.path để import được src.nlu.preprocess ──
import sys

ROOT = Path(__file__).resolve().parents[2]  # d:\N31_Nhap
sys.path.insert(0, str(ROOT))
from src.nlu.preprocess import preprocess

# ── Đường dẫn ──────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "intents.csv"
MODEL_DIR = ROOT / "models" / "thu_nghiem"  # KHÔNG ghi đè mô hình chatbot trong models/
IMAGE_DIR = ROOT / "images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load & preprocess data
# ═══════════════════════════════════════════════════════════════════════════
def load_data(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"])
    df["text_clean"] = df["text"].astype(str).map(preprocess)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Build pipeline
# ═══════════════════════════════════════════════════════════════════════════
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Plot helpers
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, label_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Naive Bayes – Confusion Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_f1_per_class(report_dict, label_names, save_path: Path, model_name="Naive Bayes"):
    f1_scores = [report_dict[lbl]["f1-score"] for lbl in label_names if lbl in report_dict]
    present_labels = [lbl for lbl in label_names if lbl in report_dict]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(present_labels, f1_scores, color="steelblue")
    ax.set_xlabel("Intent", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title(f"{model_name} – F1 per Class", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(present_labels, rotation=30, ha="right")
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Naive Bayes Intent Classifier – Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    print(f"\nTotal samples: {len(df)}")
    print("Label distribution:\n", df["label"].value_counts())

    # ── Labels ───────────────────────────────────────────────────
    label_names = sorted(df["label"].unique().tolist())
    with open(MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)
    print(f"\nLabels ({len(label_names)}): {label_names}")

    # ── Train/test split ─────────────────────────────────────────
    X = df["text_clean"].tolist()
    y = df["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── Train ────────────────────────────────────────────────────
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    report_str = classification_report(y_test, y_pred, target_names=label_names)
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=label_names)
    plot_confusion_matrix(cm, label_names, IMAGE_DIR / "nb_confusion_matrix.png")

    # ── F1 bar chart ─────────────────────────────────────────────
    plot_f1_per_class(report_dict, label_names, IMAGE_DIR / "nb_f1_per_class.png")

    # ── Save model ───────────────────────────────────────────────
    model_path = MODEL_DIR / "nb_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[saved] {model_path}")

    # ── Save NB F1 scores for later comparison ───────────────────
    nb_f1 = {lbl: report_dict[lbl]["f1-score"] for lbl in label_names if lbl in report_dict}
    with open(MODEL_DIR / "nb_f1_scores.json", "w", encoding="utf-8") as f:
        json.dump(nb_f1, f, ensure_ascii=False, indent=2)
    print(f"[saved] {MODEL_DIR / 'nb_f1_scores.json'}")

    print("\nDone.")
    return report_dict, label_names


if __name__ == "__main__":
    main()
