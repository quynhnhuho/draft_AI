"""Demo không giao diện – chạy Chức năng 1, 2, 3 trên dữ liệu mẫu trong data/sample/ và in kết quả.

    python demo.py            # cả 3 chức năng
    python demo.py c1         # chỉ Chức năng 1 (tương tự: c2, c3; có thể ghép: python demo.py c1 c3)

  C1 – Lộ trình học: 3 tình huống × 3 thuật toán (BFS, A*, quay lui – nhánh cận): lộ trình, chi phí
       (số học kỳ), số nút mở rộng / sinh, thời gian.
  C2 – Tư vấn ngành: suy diễn tiến (gợi ý ngành + luật đã kích hoạt), suy diễn lùi (kết luận + chuỗi luật),
       ID3 tầng 1 trên bảng khảo sát.
  C3 – Phân loại ý định: mô hình Keras đã huấn luyện (models/intent_model.keras) trên 99 câu test và
       41 câu ngoài phạm vi: accuracy, macro F1, P/R/F1 từng ý định, tác dụng của ngưỡng τ.

Chỉ dùng CPU, không cần khóa API, chạy xong trong khoảng 1 phút (lần đầu nạp TensorFlow lâu hơn).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # bớt log của TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
for stream in (sys.stdout, sys.stderr):                  # in được tiếng Việt trên cửa sổ lệnh Windows
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "sample"
sys.path.insert(0, str(ROOT))

GIOI_HAN_MS = 60_000   # mỗi thuật toán tối đa 60 giây/tình huống (thực tế < 1 giây)


def tieu_de(s: str) -> None:
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ============================================================================== C1
def demo_c1() -> None:
    from src.planner.a_sao import a_sao
    from src.planner.bfs import bfs
    from src.planner.model import NAM_AP_DUNG, NGUON_CTDT, doc_du_lieu, tao_bai_toan, ten_hoc_ky
    from src.planner.nhanh_can import nhanh_can

    tieu_de("CHỨC NĂNG 1 – LẬP LỘ TRÌNH HỌC (BFS, A*, quay lui – nhánh cận)")
    ds = doc_du_lieu(SAMPLE / "hoc_phan.csv")
    print(f"Dữ liệu: {len(ds)} học phần, {sum(h.tin_chi for h in ds.values())} tín chỉ "
          f"(data/sample/hoc_phan.csv). Chi phí = số học kỳ; h(n) = ⌈TC còn lại / MAX_TC⌉.")
    print(f"Nguồn: {NGUON_CTDT} – năm áp dụng {NAM_AP_DUNG}")

    for th in json.loads((SAMPLE / "tinh_huong.json").read_text(encoding="utf-8"))["lo_trinh"]:
        bt = tao_bai_toan(ds, set(th["da_dat"]), set(th["rot"]), th["hoc_ky"], th["hoc_vuot"])
        print(f"\n--- {th['ten']}")
        print(f"    bắt đầu {ten_hoc_ky(th['hoc_ky'])}, tối đa {bt.max_tc} "
              f"TC/kỳ, còn {bt.tin_chi_con_lai(bt.trang_thai_dau())} TC")
        print(f"    {'Thuật toán':<12}{'Chi phí (số kỳ)':>16}{'Nút mở rộng':>14}{'Nút sinh':>11}{'Thời gian':>12}  Tối ưu")
        ket_qua = {}
        for f in (bfs, a_sao, nhanh_can):
            kq = f(bt, gioi_han_ms=GIOI_HAN_MS)
            ket_qua[kq.thuat_toan] = kq
            print(f"    {kq.thuat_toan:<12}{str(kq.so_ky):>16}{kq.so_nut_mo_rong:>14}{kq.so_nut_sinh:>11}"
                  f"{kq.thoi_gian_ms:>9.1f} ms  {'có' if kq.toi_uu else 'chưa chắc'}")
        so_ky = {k.so_ky for k in ket_qua.values()}
        print(f"    → {'Ba thuật toán cùng ra' if len(so_ky) == 1 else 'CẢNH BÁO: khác nhau'} "
              f"{min(s for s in so_ky if s)} học kỳ.")
        kq = next(k for k in ket_qua.values() if k.thuat_toan.startswith("A"))
        print("    Lộ trình (A*):")
        for hk, tap in kq.lo_trinh:
            mon = ", ".join(sorted(tap, key=lambda m: ds[m].stt)) or "(chờ kỳ sau)"
            print(f"      {ten_hoc_ky(hk):<22} {sum(ds[m].tin_chi for m in tap):>2} TC  {mon}")


# ============================================================================== C2
def demo_c2() -> None:
    import pandas as pd

    from src.advisor.kiem_chung_sklearn import NHAN, TEN, do_loi, entropy
    from src.advisor.tham_so import NAM, NGUON, ten_nganh
    from src.advisor.tu_van import kiem_tra_dieu_kien, tu_van

    tieu_de("CHỨC NĂNG 2 – TƯ VẤN CHỌN NGÀNH (hệ luật, suy diễn tiến / lùi, ID3)")
    print(f"Nguồn tham số tuyển sinh: {NGUON['nguon']} – năm áp dụng {NAM}")
    th = json.loads((SAMPLE / "tinh_huong.json").read_text(encoding="utf-8"))

    for ca in th["tu_van_nganh"]:
        kq = tu_van(ca["user"])
        print(f"\n--- {ca['ten']}")
        if kq.loi:
            print("    Lỗi dữ liệu:", "; ".join(kq.loi))
            continue
        print(f"    {len(kq.forward.fired)} lần kích hoạt luật: "
              + " → ".join(dict.fromkeys(l.rule.id for l in kq.forward.fired)))
        print("    Kết luận (gợi ý ngành):")
        for g in kq.goi_y:
            print(f"      {ten_nganh(g.nganh):<22} mức: {g.ten_muc:<14} khả năng trúng tuyển: {g.kha_nang}")
        for tb in kq.thong_bao:
            print("    " + tb.replace("**", ""))

    for ca in th["kiem_tra_dieu_kien"]:
        kq = kiem_tra_dieu_kien(ca["user"], ca["nganh"])
        print(f"\n--- {ca['ten']}")
        print(f"    Mục tiêu: du_dk({ca['nganh']}) → Kết luận: {kq.ket_luan}")
        if kq.lui and kq.lui.proven:
            for buoc in kq.lui.proof:
                print(f"      {buoc}")
        elif kq.lui:
            for s in kq.lui.that_bai[:4]:
                print(f"      thất bại: {s}")
            if kq.lui.thieu_diem is not None:
                print(f"      còn thiếu {kq.lui.thieu_diem} điểm")
            if kq.can_hoi:
                print(f"      cần hỏi thêm: {', '.join(kq.can_hoi)}")

    df = pd.read_csv(SAMPLE / "survey.csv")
    print(f"\n--- ID3 tầng 1 trên bảng khảo sát ({len(df)} mẫu, data/sample/survey.csv)")
    print(f"    H(S) = {entropy(df[NHAN]):.4f}")
    ig = {a: do_loi(df, a) for a in TEN}
    for a, v in ig.items():
        print(f"    IG(S, {a} – {TEN[a]}) = {v:.4f}")
    goc = max(ig, key=ig.get)
    print(f"    → Nút gốc: {goc} – {TEN[goc]}   (cây đầy đủ: python -m src.advisor.kiem_chung_sklearn)")


# ============================================================================== C3
def demo_c3() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from src.nlu.classifier import MODELS_DIR, load_classifier, load_threshold
    from src.nlu.preprocess import preprocess

    tieu_de("CHỨC NĂNG 3 – PHÂN LOẠI Ý ĐỊNH (mô hình Keras đã huấn luyện)")
    t0 = time.perf_counter()
    clf = load_classifier()
    tau = load_threshold(classifier=clf)
    print(f"Mô hình: {clf.name} – models/{json.loads((MODELS_DIR / 'config.json').read_text())['model_file']} "
          f"(nạp trong {time.perf_counter() - t0:.1f} s) · ngưỡng τ = {tau}")
    for w in getattr(clf, "warnings", []):
        print("CẢNH BÁO:", w)

    test = pd.read_csv(SAMPLE / "intents_test.csv")
    oos = pd.read_csv(SAMPLE / "out_of_scope_test.csv")
    p_in = clf.predict_proba([preprocess(t) for t in test["text"]])
    p_oos = clf.predict_proba([preprocess(t) for t in oos["text"]])
    y_pred = [clf.labels[i] for i in p_in.argmax(1)]
    y = test["label"].tolist()

    print(f"\nTập test: {len(test)} câu, {test['label'].nunique()} ý định (data/sample/intents_test.csv – "
          "không dùng khi huấn luyện)")
    print(f"  Accuracy = {accuracy_score(y, y_pred):.4f}   Macro F1 = {f1_score(y, y_pred, average='macro'):.4f}")
    print(classification_report(y, y_pred, digits=3, zero_division=0))

    conf_in, conf_oos = p_in.max(1), p_oos.max(1)
    dung = np.mean((conf_in >= tau) & (np.array(y_pred) == np.array(y)))
    print(f"Ngưỡng τ = {tau}: trả lời đúng {dung:.1%} câu trong phạm vi · có trả lời {np.mean(conf_in >= tau):.1%} · "
          f"chặn {np.mean(conf_oos < tau):.1%} trong {len(oos)} câu ngoài phạm vi (chuyển cán bộ)")

    ref = json.loads((MODELS_DIR / "metrics_c3.json").read_text(encoding="utf-8"))
    best = ref.get("best_model")
    if best in ref.get("models", {}) and clf.name.startswith("keras"):
        ok = abs(ref["models"][best]["test_acc"] - accuracy_score(y, y_pred)) < 1e-6
        print(f"Đối chiếu models/metrics_c3.json (lúc huấn luyện): test accuracy {ref['models'][best]['test_acc']:.4f} "
              f"→ {'KHỚP' if ok else 'KHÁC – kiểm tra lại tiền xử lý / phiên bản thư viện'}")

    print("\nVí dụ dự đoán:")
    for s in ["Điểm chuẩn ngành CNTT năm 2025 là bao nhiêu?", "hp cntt 1 năm bn tiền",
              "Em nên chọn ngành nào nếu thích lập trình?", "Thời tiết hôm nay thế nào?"]:
        p = clf.predict(s)
        ket = p.intent if p.confidence >= tau else "chuyển cán bộ (dưới ngưỡng)"
        print(f"  {s:<48} → {ket:<30} tin cậy {p.confidence:.2f}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Demo Chức năng 1–3 trên dữ liệu mẫu (không cần giao diện)")
    # không dùng choices=: argparse của Python 3.10/3.11 báo lỗi với nargs="*" khi không truyền tham số
    ap.add_argument("chuc_nang", nargs="*", metavar="{c1,c2,c3}", help="mặc định: cả ba")
    chon = [c.lower() for c in ap.parse_args(argv).chuc_nang] or ["c1", "c2", "c3"]
    if sai := [c for c in chon if c not in ("c1", "c2", "c3")]:
        ap.error(f"không có chức năng {', '.join(sai)} (chọn c1, c2, c3)")
    t0 = time.perf_counter()
    for f in [{"c1": demo_c1, "c2": demo_c2, "c3": demo_c3}[c] for c in chon]:
        t = time.perf_counter()
        f()
        print(f"\n[{f.__name__}: {time.perf_counter() - t:.1f} s]")
    print(f"\nHoàn tất sau {time.perf_counter() - t0:.1f} s. Giao diện chatbot (Chức năng 4): python app.py")


if __name__ == "__main__":
    main()
