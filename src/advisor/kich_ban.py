"""Chức năng 2 - Kịch bản minh họa suy diễn tiến và suy diễn lùi.

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
from __future__ import annotations

import argparse

from .rules import RULES, phan_tang
from .tham_so import DINH_HUONG, MON, SO_THICH
from .tu_van import kiem_tra_dieu_kien, markdown_kiem_tra, markdown_tu_van, tu_van

KHONG_THI = {f"diem_{m}": None for m in MON}
SO_THICH_TRONG = dict.fromkeys(SO_THICH + DINH_HUONG, False)


def _thpt(kv: str, **diem) -> dict:
    return {"phuong_thuc": "THPT", "khu_vuc": kv, **KHONG_THI, **{f"diem_{k}": v for k, v in diem.items()}}


def _dgnl(kv: str, nl: float, hb_anh: float, mon2: str, hb2: float) -> dict:
    return {"phuong_thuc": "DGNL", "khu_vuc": kv, "diem_dgnl": nl, "HB_mon1": hb_anh, "HB_mon2": hb2,
            "mon_hb2": mon2}


def _thich(*khoa: str) -> dict:
    return {**SO_THICH_TRONG, **dict.fromkeys(khoa, True)}


KICH_BAN: list[tuple[str, str, dict, str]] = [
    ("KB1", "THPT, KV1, tổng 19.2: cộng đủ điểm ưu tiên 0.75, thiếu 0.05 điểm vào SPTH",
     {**_thpt("KV1", Toan=7, Anh=6.2, Ly=6), **_thich("thich_lap_trinh", "thich_day_hoc", "giao_vien")}, "SPTH"),
    ("KB2", "THPT, KV3, điểm cao 26.3: SPTH Rất phù hợp, khả năng Cao",
     {**_thpt("KV3", Toan=9, Anh=8.5, Tin=8.8, Ly=7), **_thich("thich_lap_trinh", "thich_day_hoc", "giao_vien")},
     "SPTH"),
    ("KB3", "ĐGNL, KV2, tổ hợp X26: CNGD Rất phù hợp, tổng từ 22.5 nên điểm ưu tiên giảm dần",
     {**_dgnl("KV2", 7.5, 8, "Tin", 8.5),
      **_thich("thich_thiet_ke_hoc_lieu", "thich_lap_trinh", "chuyen_vien_edtech")}, "CNGD"),
    ("KB4", "Điểm liệt: Tiếng Anh 1.0 làm mọi tổ hợp bị loại, kết luận R33",
     {**_thpt("KV2", Toan=8, Anh=1.0, Ly=7, Hoa=7.5), **_thich("thich_lap_trinh")}, "CNTT"),
    ("KB5", "Đủ điều kiện nhưng không chọn sở thích hay nghề nào, kết luận R32",
     {**_thpt("KV2-NT", Toan=7, Anh=7, Hoa=6.5), **_thich()}, "CNGD"),
    ("KB6", "Thiếu tổ hợp: chỉ thi Toán, Vật lý, Hóa học, kết luận R33",
     {**_thpt("KV3", Toan=8, Ly=8, Hoa=8), **_thich("thich_du_lieu")}, "CNTT"),
    ("KB7", "Thích dạy học nhưng muốn làm lập trình viên: CNTT ở mức Tham khảo (R31)",
     {**_thpt("KV2", Toan=7.5, Anh=7, Sinh=7), **_thich("thich_day_hoc", "lap_trinh_vien")}, "CNTT"),
]

HOI_LAN_LUOT = [{}, {"phuong_thuc": "THPT"},
                {"phuong_thuc": "THPT", **KHONG_THI, "diem_Toan": 8, "diem_Anh": 7, "diem_Ly": 7}]


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
    vach = "=" * 100
    for ma, mo_ta, user, nganh in KICH_BAN:
        print(f"{vach}\n{ma} - {mo_ta}\n{vach}")
        print(markdown_tu_van(tu_van(user)))
        print("-" * 100)
        print(markdown_kiem_tra(kiem_tra_dieu_kien(user, nganh), vet=args.vet))
        print()
    print(f"{vach}\nKB8 - Suy diễn lùi khi chưa đủ thông tin: trả về dữ kiện cần hỏi\n{vach}")
    for user in HOI_LAN_LUOT:
        kq = kiem_tra_dieu_kien(user, "CNTT")
        print(f"user = {user}\n  {kq.ket_luan}; cần hỏi: {kq.can_hoi}")


if __name__ == "__main__":
    main()
