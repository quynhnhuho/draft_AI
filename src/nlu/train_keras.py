"""Mô hình chính: TextVectorization → Embedding → BiLSTM (hoặc GRU / Conv1D) → softmax.

    python -m src.nlu.train_keras --data data/intents.csv --arch lstm

Đầu ra (HỢP ĐỒNG với src/nlu/classifier.py):
    models/intent_model.keras   – mô hình nhận chuỗi ĐÃ preprocess(), trả xác suất
    models/labels.json          – danh sách nhãn theo đúng thứ tự đầu ra
    models/config.json          – {"threshold": τ, ...}  (τ chọn trên val ∪ out_of_scope)

TODO (C3): so sánh 2–3 kiến trúc, vẽ learning curve, ma trận nhầm lẫn, quét ngưỡng τ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from .train_nb import SEED, load_split


def build_model(train_texts, n_classes: int, arch: str = "lstm", max_tokens: int = 5000, seq_len: int = 30):
    vec = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens, output_sequence_length=seq_len, standardize=None, split="whitespace")
    vec.adapt(tf.constant(list(train_texts)))

    inp = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = vec(inp)
    x = tf.keras.layers.Embedding(vec.vocabulary_size(), 128, mask_zero=(arch != "cnn"))(x)
    if arch == "lstm":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    elif arch == "gru":
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x)
    else:  # cnn
        x = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents.csv")
    ap.add_argument("--arch", choices=["lstm", "gru", "cnn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.6, help="τ tạm; thay bằng giá trị chọn từ thí nghiệm")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    train, val, test = load_split(args.data)
    labels = sorted(set(train["intent"]) | set(val["intent"]) | set(test["intent"]))
    idx = {l: i for i, l in enumerate(labels)}
    to_xy = lambda d: (tf.constant(d["clean"].tolist()), np.array([idx[l] for l in d["intent"]]))
    (xtr, ytr), (xva, yva), (xte, yte) = to_xy(train), to_xy(val), to_xy(test)

    model = build_model(train["clean"], len(labels), args.arch)
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=args.epochs, batch_size=32, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    pred = model.predict(xte, verbose=0).argmax(1)
    print(classification_report(yte, pred, labels=range(len(labels)), target_names=labels, digits=3, zero_division=0))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "intent_model.keras")
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"threshold": args.threshold, "arch": args.arch}, indent=2), encoding="utf-8")
    print("Đã lưu mô hình vào", out)


if __name__ == "__main__":
    main()
