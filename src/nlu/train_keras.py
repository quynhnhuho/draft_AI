import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Thêm gốc dự án vào sys.path ─────────────────────────────────────────
import sys

ROOT = Path(__file__).resolve().parents[2]  # d:\N31_Nhap
sys.path.insert(0, str(ROOT))
from src.nlu.preprocess import preprocess

# ── TensorFlow / Keras ───────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ── Đường dẫn ────────────────────────────────────────────────────────────
DATA_PATH   = ROOT / "data" / "intents.csv"
MODEL_DIR   = ROOT / "models"
IMAGE_DIR   = ROOT / "images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hằng số ───────────────────────────────────────────────────────────────
MAX_LEN            = 30
EMBED_DIM          = 64
EPOCHS             = 30
BATCH_SIZE         = 32
VALIDATION_SPLIT   = 0.1
RANDOM_STATE       = 42


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN A – Chuẩn bị dữ liệu
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Load, preprocess, encode, tokenize, pad, split."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"])
    df["text_clean"] = df["text"].astype(str).map(preprocess)

    # ── Label encoding ───────────────────────────────────────────
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    label_names = le.classes_.tolist()   # alphabetically sorted

    # Lưu labels.json (chung cho cả train_nb.py nếu chạy sau)
    with open(MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)
    print(f"Labels ({len(label_names)}): {label_names}")

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text_clean"].tolist())

    VOCAB_SIZE = min(len(tokenizer.word_index) + 1, 8000)

    # Lưu tokenizer
    with open(MODEL_DIR / "tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    print(f"Vocabulary size (capped): {VOCAB_SIZE}")

    # ── Sequences & padding ──────────────────────────────────────
    seqs   = tokenizer.texts_to_sequences(df["text_clean"].tolist())
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X = padded
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, tokenizer, label_names, VOCAB_SIZE


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN B – Model LSTM
# ═══════════════════════════════════════════════════════════════════════════

def build_lstm(vocab_size: int, embed_dim: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN C – Model Conv1D
# ═══════════════════════════════════════════════════════════════════════════

def build_conv1d(vocab_size: int, embed_dim: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN D – Training
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model: Sequential, X_train, y_train, checkpoint_path: str):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN E – Đánh giá & biểu đồ
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(history, model_name: str, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title(f"{model_name} – Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title(f"{model_name} – Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f"{model_name} Training History", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_confusion_matrix(cm, label_names, title: str, save_path: Path):
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
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def evaluate_model(model: Sequential, X_test, y_test, label_names, model_name: str,
                   cm_save_path: Path):
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"\n{'='*60}")
    print(f"{model_name} – Classification Report")
    print("=" * 60)
    report_str = classification_report(y_test, y_pred, target_names=label_names)
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, label_names,
        title=f"{model_name} – Confusion Matrix",
        save_path=cm_save_path,
    )
    return report_dict


def plot_model_comparison_f1(label_names, nb_f1: dict, lstm_report: dict,
                              conv1d_report: dict, save_path: Path):
    """Bar chart so sánh F1 Naive Bayes vs LSTM vs Conv1D cho từng class."""
    nb_scores     = [nb_f1.get(lbl, 0.0) for lbl in label_names]
    lstm_scores   = [lstm_report.get(lbl, {}).get("f1-score", 0.0) for lbl in label_names]
    conv1d_scores = [conv1d_report.get(lbl, {}).get("f1-score", 0.0) for lbl in label_names]

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, nb_scores,     width, label="Naive Bayes", color="steelblue")
    ax.bar(x,          lstm_scores,  width, label="LSTM",        color="darkorange")
    ax.bar(x + width, conv1d_scores, width, label="Conv1D",      color="green")

    ax.set_xlabel("Intent", fontsize=12)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("F1-score per Class – Model Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN F – Ngưỡng tin cậy
# ═══════════════════════════════════════════════════════════════════════════

def predict_with_threshold(text: str, model: Sequential, tokenizer: Tokenizer,
                           label_names: list, threshold: float = 0.6) -> dict:
    """
    Phân loại ý định có ngưỡng tin cậy.

    Returns:
        dict với 'intent', 'confidence', 'transferred'
        transferred=True nghĩa là câu ngoài phạm vi (confidence < threshold)
    """
    clean = preprocess(text)
    seq   = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba  = model.predict(padded, verbose=0)[0]

    max_idx    = int(np.argmax(proba))
    confidence = float(proba[max_idx])

    if confidence < threshold:
        return {
            "intent": "ngoai_pham_vi",
            "confidence": confidence,
            "transferred": True,
        }

    return {
        "intent": label_names[max_idx],
        "confidence": confidence,
        "transferred": False,
        "all_proba": {label_names[i]: float(proba[i]) for i in range(len(label_names))},
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Keras Intent Classifier – Training (LSTM + Conv1D)")
    print("=" * 60)

    # ── A: Data ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, tokenizer, label_names, VOCAB_SIZE = (
        load_and_prepare_data()
    )
    num_classes = len(label_names)

    # Load NB F1 scores nếu có (để vẽ so sánh)
    nb_f1_path = MODEL_DIR / "nb_f1_scores.json"
    if nb_f1_path.exists():
        with open(nb_f1_path, encoding="utf-8") as f:
            nb_f1 = json.load(f)
    else:
        print("[warn] nb_f1_scores.json not found – run train_nb.py first for comparison chart.")
        nb_f1 = {lbl: 0.0 for lbl in label_names}

    # ── B+D: LSTM training ───────────────────────────────────────
    print("\n--- Building & Training LSTM ---")
    lstm_model = build_lstm(VOCAB_SIZE, EMBED_DIM, MAX_LEN, num_classes)
    lstm_model.summary()

    lstm_history = train_model(
        lstm_model, X_train, y_train,
        checkpoint_path=MODEL_DIR / "lstm_best.keras",
    )
    plot_training_history(lstm_history, "LSTM", IMAGE_DIR / "lstm_training_history.png")

    # ── C+D: Conv1D training ─────────────────────────────────────
    print("\n--- Building & Training Conv1D ---")
    conv1d_model = build_conv1d(VOCAB_SIZE, EMBED_DIM, MAX_LEN, num_classes)
    conv1d_model.summary()

    conv1d_history = train_model(
        conv1d_model, X_train, y_train,
        checkpoint_path=MODEL_DIR / "conv1d_best.keras",
    )
    plot_training_history(conv1d_history, "Conv1D", IMAGE_DIR / "conv1d_training_history.png")

    # ── E: Đánh giá ──────────────────────────────────────────────
    # Load saved best checkpoints
    lstm_best   = tf.keras.models.load_model(MODEL_DIR / "lstm_best.keras")
    conv1d_best = tf.keras.models.load_model(MODEL_DIR / "conv1d_best.keras")

    lstm_report = evaluate_model(
        lstm_best, X_test, y_test, label_names,
        model_name="LSTM",
        cm_save_path=IMAGE_DIR / "lstm_confusion_matrix.png",
    )
    conv1d_report = evaluate_model(
        conv1d_best, X_test, y_test, label_names,
        model_name="Conv1D",
        cm_save_path=IMAGE_DIR / "conv1d_confusion_matrix.png",
    )

    plot_model_comparison_f1(
        label_names, nb_f1, lstm_report, conv1d_report,
        save_path=IMAGE_DIR / "model_comparison_f1.png",
    )

    # ── F: Predict with threshold demo ───────────────────────────
    print("\n--- Predict with Threshold Demo (threshold=0.6) ---")
    test_sentences = [
        "Điểm chuẩn ngành CNTT năm nay là bao nhiêu?",
        "Trường có ký túc xá không?",
        "Học phí mỗi kỳ là bao nhiêu tiền?",
        "Thời tiết hôm nay thế nào?",   # → ngoai_pham_vi
        "Tỉ giá đô la hôm nay?",         # → ngoai_pham_vi
    ]
    for sent in test_sentences:
        result = predict_with_threshold(sent, lstm_best, tokenizer, label_names, threshold=0.6)
        print(f"  Input : {sent}")
        print(f"  Output: intent={result['intent']}, confidence={result['confidence']:.4f}, "
              f"transferred={result['transferred']}\n")

    print("Training complete. Models & charts saved.")


if __name__ == "__main__":
    main()
