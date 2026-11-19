"""C2 – Kịch bản minh họa (dùng cho báo cáo và kiểm thử). Chạy:

    python -m src.advisor.kich_ban            # tất cả kịch bản, suy diễn tiến + lùi
    python -m src.advisor.kich_ban --vet      # kèm nhật ký suy diễn lùi đầy đủ
    python -m src.advisor.kich_ban --luat     # in 33 luật và tầng của từng vị từ
"""
from __future__ import annotations

import argparse

from .rules import RULES, phan_tang
from .tu_van import kiem_tra_dieu_kien, markdown_kiem_tra, markdown_tu_van, tu_van

KHONG_THI = {"diem_Toan": None, "diem_Anh": None, "diem_Ly": None, "diem_Hoa": None, "diem_Sinh": None,
             "diem_Tin": None}
SO_THICH_TRONG = {"thich_lap_trinh": False, "thich_du_lieu": False, "thich_thiet_ke_hoc_lieu": False,
                  "thich_day_hoc": False, "lap_trinh_vien": False, "phan_tich_du_lieu": False,
                  "giao_vien": False, "chuyen_vien_edtech": False}


def _thpt(kv: str, **diem) -> dict:
    return {"phuong_thuc": "THPT", "khu_vuc": kv, **KHONG_THI, **{f"diem_{k}": v for k, v in diem.items()}}


KICH_BAN: list[tuple[str, dict, str]] = [
    ("KB1 – THPT, KV1, tổng 19.2: ưu tiên cộng đủ 0.75 (tổng < 22.5); thiếu 0.05 điểm vào SPTH",
     {**_thpt("KV1", Toan=7, Anh=6.2, Ly=6), **SO_THICH_TRONG,
      "thich_lap_trinh": True, "thich_day_hoc": True, "giao_vien": True}, "SPTH"),
    ("KB2 – THPT, KV3, điểm cao 26.3: SPTH Rất phù hợp, khả năng Cao",
     {**_thpt("KV3", Toan=9, Anh=8.5, Tin=8.8, Ly=7), **SO_THICH_TRONG,
      "thich_lap_trinh": True, "thich_day_hoc": True, "giao_vien": True}, "SPTH"),
    ("KB3 – ĐGNL, KV2, tổ hợp X26: CNGD Rất phù hợp; tổng ≥ 22.5 nên ưu tiên giảm dần",
     {"phuong_thuc": "DGNL", "khu_vuc": "KV2", "diem_dgnl": 7.5, "HB_mon1": 8, "HB_mon2": 8.5, "mon_hb2": "Tin",
      **SO_THICH_TRONG, "thich_thiet_ke_hoc_lieu": True, "thich_lap_trinh": True, "chuyen_vien_edtech": True},
     "CNGD"),
    ("KB4 – Điểm liệt: Tiếng Anh = 1.0 làm mọi tổ hợp bị loại → R33",
     {**_thpt("KV2", Toan=8, Anh=1.0, Ly=7, Hoa=7.5), **SO_THICH_TRONG, "thich_lap_trinh": True}, "CNTT"),
    ("KB5 – Đủ điều kiện nhưng không chọn sở thích/nghề nào → R32",
     {**_thpt("KV2-NT", Toan=7, Anh=7, Hoa=6.5), **SO_THICH_TRONG}, "CNGD"),
    ("KB6 – Thiếu tổ hợp: chỉ thi Toán, Vật lý, Hóa học → R33 (không luật R1–R4 nào chạy)",
     {**_thpt("KV3", Toan=8, Ly=8, Hoa=8), **SO_THICH_TRONG, "thich_du_lieu": True}, "CNTT"),
    ("KB7 – Sở thích dạy học nhưng định hướng lập trình viên: CNTT Tham khảo (R31)",
     {**_thpt("KV2", Toan=7.5, Anh=7, Sinh=7), **SO_THICH_TRONG, "thich_day_hoc": True, "lap_trinh_vien": True},
     "CNTT"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vet", action="store_true", help="in nhật ký suy diễn lùi")
    ap.add_argument("--luat", action="store_true", help="in 33 luật và phân tầng")
    args = ap.parse_args()
    if args.luat:
        tang = phan_tang(RULES)
        for r in RULES:
            print(f"[tầng {tang[r.ket_luan[0]]}] {r}")
        return
    for ten, user, nganh in KICH_BAN:
        print("=" * 100, f"\n{ten}\n", "=" * 100, sep="")
        print(markdown_tu_van(tu_van(user)))
        print("-" * 100)
        print(markdown_kiem_tra(kiem_tra_dieu_kien(user, nganh), vet=args.vet))
        print()
    print("=" * 100, "\nKB8 – Suy diễn lùi khi chưa biết gì: hệ thống trả về dữ kiện cần hỏi\n", "=" * 100, sep="")
    for user in ({}, {"phuong_thuc": "THPT"}, {"phuong_thuc": "THPT", **KHONG_THI, "diem_Toan": 8, "diem_Anh": 7,
                                                "diem_Ly": 7}):
        kq = kiem_tra_dieu_kien(user, "CNTT")
        print(f"user = {user}\n  → {kq.ket_luan}; cần hỏi: {kq.can_hoi}")


if __name__ == "__main__":
    main()
