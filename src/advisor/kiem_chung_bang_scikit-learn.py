import pandas as pd
import matplotlib.pyplot as plt
import os
THU_MUC = os.path.dirname(os.path.abspath(__file__))
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, _tree

# ----------------------------------------------------------------------
# 1. DU LIEU KHAO SAT
# ----------------------------------------------------------------------
TEN = {"A1": "Tổ hợp", "A2": "Điểm", "A3": "Lĩnh vực", "A4": "Định hướng"}

data = [
    ("A01", "Cao",        "Dạy học",              "Giáo dục",     "SP Tin"),
    ("A01", "Cao",        "Thiết kế học liệu số", "Doanh nghiệp", "CNGD"),
    ("A01", "Cao",        "Dạy học",              "Giáo dục",     "SP Tin"),
    ("A01", "Cao",        "Dạy học",              "Giáo dục",     "SP Tin"),
    ("D07", "Trung bình", "Dạy học",              "Giáo dục",     "CNTT"),
    ("A01", "Trung bình", "Dạy học",              "Giáo dục",     "CNTT"),
    ("D07", "Cao",        "Lập trình",            "Nghiên cứu",   "CNTT"),
    ("A01", "Trung bình", "Dạy học",              "Giáo dục",     "CNTT"),
    ("D07", "Cao",        "Dạy học",              "Giáo dục",     "SP Tin"),
    ("A01", "Cao",        "Dạy học",              "Nghiên cứu",   "SP Tin"),
    ("A01", "Trung bình", "Thiết kế học liệu số", "Nghiên cứu",   "CNGD"),
    ("D07", "Cao",        "Thiết kế học liệu số", "Doanh nghiệp", "CNGD"),
    ("A01", "Trung bình", "Thiết kế học liệu số", "Giáo dục",     "CNGD"),
    ("A01", "Cao",        "Dạy học",              "Giáo dục",     "SP Tin"),
    ("A01", "Trung bình", "Lập trình",            "Doanh nghiệp", "CNTT"),
]
df = pd.DataFrame(data, columns=["A1", "A2", "A3", "A4", "Y"])

# ----------------------------------------------------------------------
# 2. MA HOA ONE-HOT (scikit-learn chi lam viec voi so)
#    Vi du: A3 -> 3 cot "A3=Dạy học", "A3=Lập trình", "A3=Thiết kế học liệu số"
# ----------------------------------------------------------------------
X = pd.get_dummies(df[["A1", "A2", "A3", "A4"]], prefix_sep="=")
y = df["Y"]

# ----------------------------------------------------------------------
# 3. HUAN LUYEN CAY VOI TIEU CHI ENTROPY
# ----------------------------------------------------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X, y)

# ----------------------------------------------------------------------
# 4. IN KET QUA DE DOI CHIEU VOI PHAN TINH TAY
# ----------------------------------------------------------------------
print("===== CÂY QUYẾT ĐỊNH (scikit-learn) =====")
print(export_text(model, feature_names=list(X.columns)))

t = model.tree_
print("Entropy tại nút gốc      :", round(t.impurity[0], 4), "   (tính tay: 1,5656)")
print("Thuộc tính tại nút gốc   :", X.columns[t.feature[0]], "   (tính tay: A3 - Lĩnh vực)")
print("Độ chính xác trên 15 mẫu :", f"{int(model.score(X, y) * len(y))}/{len(y)}")

plt.figure(figsize=(13, 6))
plot_tree(model, feature_names=list(X.columns), class_names=list(model.classes_),
          filled=True, rounded=True, fontsize=10)
plt.savefig(os.path.join(THU_MUC, "cay_sklearn.png"), dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 5. RUT TRICH TAP LUAT
#    Moi duong di tu goc den la = 1 luat. Duyet cay bang de quy.
# ----------------------------------------------------------------------
GIA_TRI = {A: list(df[A].unique()) for A in TEN}   

def duyet(nut, dieu_kien, ds_luat):
    if t.feature[nut] == _tree.TREE_UNDEFINED:        
        nganh = model.classes_[t.value[nut][0].argmax()]
        ds_luat.append((dieu_kien, nganh, int(t.n_node_samples[nut])))
        return
    A, v = X.columns[t.feature[nut]].split("=", 1)      
    duyet(t.children_left[nut],  dieu_kien + [(A, "≠", v)], ds_luat) 
    duyet(t.children_right[nut], dieu_kien + [(A, "=", v)], ds_luat)   

def rut_gon(dieu_kien):
    """Gop cac dieu kien cung thuoc tinh cho de doc.
       Vd: A2 ≠ Trung bình  ->  A2 = Cao (vi A2 chi co 2 gia tri)"""
    ket_qua = []
    thu_tu = list(dict.fromkeys(a for a, _, _ in dieu_kien))  
    for A in thu_tu:
        con_lai = list(GIA_TRI[A])
        for (a, phep, v) in dieu_kien:
            if a != A:
                continue
            con_lai = [v] if phep == "=" else [x for x in con_lai if x != v]
        if len(con_lai) == len(GIA_TRI[A]):
            continue                                    
        if len(con_lai) == 1:
            ket_qua.append(f"{TEN[A]} = {con_lai[0]}")
        else:
            ket_qua.append(f"{TEN[A]} ∈ {{{', '.join(con_lai)}}}")
    return ket_qua

ds_luat = []
duyet(0, [], ds_luat)

print("\n===== TẬP LUẬT RÚT TRÍCH TỪ CÂY SCIKIT-LEARN =====")
for i, (dk, nganh, n) in enumerate(ds_luat, 1):
    print(f"R{i}: IF {' AND '.join(rut_gon(dk))} THEN Ngành = {nganh}   ({n} mẫu)")