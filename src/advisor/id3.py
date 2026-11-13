"""C2 – ID3 (Phần 4 – 4.1): kiểm tra lại số liệu tính tay + đối chiếu sklearn + rút luật.

    python -m src.advisor.id3 --data data/survey.csv --target nganh_chon

TODO (C2):
  1. Tạo data/survey.csv (≥14 mẫu, ≥4 thuộc tính rời rạc, cột nhãn nganh_chon).
  2. Cài entropy() và information_gain() – kết quả phải khớp bảng tính tay trong báo cáo.
  3. sklearn_tree(): DecisionTreeClassifier(criterion="entropy") + one-hot, export_text → luật.
"""
from __future__ import annotations

import argparse

import pandas as pd


def entropy(labels: pd.Series) -> float:
    """H(S) = -Σ p_i log2 p_i.  TODO (C2)."""
    raise NotImplementedError("entropy chưa được cài đặt (C2)")


def information_gain(df: pd.DataFrame, attr: str, target: str) -> float:
    """IG(S, A) = H(S) - Σ |S_v|/|S| · H(S_v).  TODO (C2)."""
    raise NotImplementedError("information_gain chưa được cài đặt (C2)")


def sklearn_tree(df: pd.DataFrame, target: str) -> str:
    """TODO (C2): huấn luyện cây sklearn, trả về chuỗi export_text để rút luật."""
    raise NotImplementedError("Đối chiếu sklearn chưa được cài đặt (C2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/survey.csv")
    ap.add_argument("--target", default="nganh_chon")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    print(f"H(S) = {entropy(df[args.target]):.4f}")
    for a in (c for c in df.columns if c != args.target):
        print(f"IG({a}) = {information_gain(df, a, args.target):.4f}")
    print(sklearn_tree(df, args.target))


if __name__ == "__main__":
    main()
