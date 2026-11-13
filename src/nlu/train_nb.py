"""Baseline: TF-IDF + MultinomialNB.

    python -m src.nlu.train_nb --data data/intents.csv

Đầu vào: CSV có cột text, intent. Đầu ra: models/nb_pipeline.pkl + in classification_report.
TODO (C3): thử ngram_range, alpha, thêm char n-gram; ghi bảng so sánh vào báo cáo.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocess import preprocess

SEED = 42


def load_split(path: str, seed: int = SEED):
    """Đọc, tiền xử lý, XÓA TRÙNG rồi mới chia 70/15/15 stratified (dùng chung cho NB và Keras)."""
    df = pd.read_csv(path)
    df["clean"] = df["text"].astype(str).map(preprocess)
    df = df.drop_duplicates("clean").reset_index(drop=True)
    strat = df["intent"] if df["intent"].value_counts().min() >= 4 else None
    train, rest = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)
    strat_rest = rest["intent"] if strat is not None else None
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=strat_rest)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents.csv")
    ap.add_argument("--out", default="models/nb_pipeline.pkl")
    args = ap.parse_args()

    train, val, test = load_split(args.data)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")),
        ("nb", MultinomialNB(alpha=0.3)),
    ])
    pipe.fit(train["clean"], train["intent"])
    print(classification_report(test["intent"], pipe.predict(test["clean"]), digits=3, zero_division=0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(pipe, f)
    print("Đã lưu", args.out)


if __name__ == "__main__":
    main()
