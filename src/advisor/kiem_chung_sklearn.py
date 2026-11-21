# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

GOC = Path(__file__).resolve().parents[2]
DU_LIEU = GOC / "data" / "survey.csv"
HINH = GOC / "docs" / "hinh" / "cay_quyet_dinh_sklearn.png"

TEN = {"A1": "Tổ hợp", "A2": "Điểm", "A3": "Lĩnh vực", "A4": "Định hướng"}
NHAN = "Y"


def entropy(nhan: pd.Series) -> float:
    return -sum(p * math.log2(p) for p in nhan.value_counts(normalize=True) if p > 0)


def do_loi(df: pd.DataFrame, a: str) -> float:
    return entropy(df[NHAN]) - sum(len(g) / len(df) * entropy(g[NHAN]) for _, g in df.groupby(a))


def in_hai_tang(df: pd.DataFrame) -> None:
    print(f"===== ID3 TẦNG 1 (đối chiếu tính tay) =====\nH(S) = {entropy(df[NHAN]):.4f}")
    ig = {a: do_loi(df, a) for a in TEN}
    for a, v in ig.items():
        print(f"  IG(S, {a} - {TEN[a]}) = {v:.4f}")
    goc = max(ig, key=ig.get)
    print(f"Nút gốc: {goc} - {TEN[goc]}\n\n===== ID3 TẦNG 2 =====")
    for gia_tri, g in df.groupby(goc):
        h = entropy(g[NHAN])
        if h == 0:
            print(f"  {goc} = {gia_tri}: H = 0, lá {g[NHAN].iloc[0]} ({len(g)} mẫu)")
            continue
        ig2 = {a: do_loi(g, a) for a in TEN if a != goc}
        chon = max(ig2, key=ig2.get)
        chi_tiet = ", ".join(f"IG({a}) = {v:.4f}" for a, v in ig2.items())
        print(f"  {goc} = {gia_tri}: H = {h:.4f}; {chi_tiet} -> chọn {chon} - {TEN[chon]}")


def rut_luat(model: DecisionTreeClassifier, cot: list[str], gia_tri: dict) -> list[tuple[list[str], str, int]]:
    t = model.tree_

    def gop(dieu_kien):
        ket_qua = []
        for a in dict.fromkeys(x for x, _, _ in dieu_kien):
            con_lai = list(gia_tri[a])
            for x, phep, v in dieu_kien:
                if x == a:
                    con_lai = [v] if phep == "=" else [c for c in con_lai if c != v]
            if len(con_lai) == 1:
                ket_qua.append(f"{TEN[a]} = {con_lai[0]}")
            elif len(con_lai) < len(gia_tri[a]):
                ket_qua.append(f"{TEN[a]} ∈ {{{', '.join(con_lai)}}}")
        return ket_qua

    def duyet(nut, dieu_kien, luat):
        if t.children_left[nut] == t.children_right[nut]:
            luat.append((gop(dieu_kien), model.classes_[t.value[nut][0].argmax()], int(t.n_node_samples[nut])))
            return
        a, v = cot[t.feature[nut]].split("=", 1)
        duyet(t.children_left[nut], dieu_kien + [(a, "≠", v)], luat)
        duyet(t.children_right[nut], dieu_kien + [(a, "=", v)], luat)

    luat: list = []
    duyet(0, [], luat)
    return luat


def main() -> None:
    df = pd.read_csv(DU_LIEU)
    in_hai_tang(df)

    X = pd.get_dummies(df[list(TEN)], prefix_sep="=")
    model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, df[NHAN])
    t = model.tree_
    print("\n===== CÂY QUYẾT ĐỊNH (scikit-learn, entropy) =====")
    print(export_text(model, feature_names=list(X.columns)))
    print(f"Entropy nút gốc: {t.impurity[0]:.4f}; tách đầu tiên: {X.columns[t.feature[0]]}")
    print(f"Độ chính xác trên {len(df)} mẫu: {int(round(model.score(X, df[NHAN]) * len(df)))}/{len(df)}")

    print("\n===== TẬP LUẬT RÚT TRÍCH =====")
    gia_tri = {a: list(df[a].unique()) for a in TEN}
    for i, (dk, nganh, n) in enumerate(rut_luat(model, list(X.columns), gia_tri), 1):
        print(f"L{i}: IF {' AND '.join(dk)} THEN Ngành = {nganh}   ({n} mẫu)")

    HINH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 6))
    plot_tree(model, feature_names=list(X.columns), class_names=list(model.classes_),
              filled=True, rounded=True, fontsize=10)
    plt.savefig(HINH, dpi=150, bbox_inches="tight")
    print(f"\nĐã lưu hình cây: {HINH.relative_to(GOC)}")
    plt.show()


if __name__ == "__main__":
    main()
