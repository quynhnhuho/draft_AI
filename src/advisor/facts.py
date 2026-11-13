"""C2 – Chuyển thông tin người dùng (có số) thành sự kiện ký hiệu cho hệ luật.

Ví dụ: {"to_hop": "A00", "diem": 24.0, "so_thich": ["lap_trinh"], "dinh_huong": "doanh_nghiep"}
  → {"to_hop=A00", "diem_dat_cntt", "so_thich=lap_trinh", "dinh_huong=doanh_nghiep", ...}

Điểm được so với điểm chuẩn NĂM GẦN NHẤT trong cơ sở tri thức (phương thức THPT).
"""
from __future__ import annotations

from typing import Any


def latest_cutoffs(kb: dict) -> dict[str, tuple[float, int]]:
    """{ma_nganh: (điểm chuẩn, năm)} lấy năm mới nhất có số liệu."""
    out: dict[str, tuple[float, int]] = {}
    for f in kb.get("facts", []):
        if f.get("loai") == "diem_chuan" and f.get("gia_tri") is not None and f.get("phuong_thuc", "thpt") == "thpt":
            if f["nganh"] not in out or f["nam"] > out[f["nganh"]][1]:
                out[f["nganh"]] = (float(f["gia_tri"]), int(f["nam"]))
    return out


def build_facts(user: dict[str, Any], kb: dict) -> set[str]:
    facts: set[str] = set()
    if user.get("to_hop"):
        facts.add(f"to_hop={user['to_hop'].upper()}")
    for s in user.get("so_thich", []) or []:
        facts.add(f"so_thich={s}")
    if user.get("dinh_huong"):
        facts.add(f"dinh_huong={user['dinh_huong']}")
    diem = user.get("diem")
    if diem is not None:
        for ma, (cutoff, _nam) in latest_cutoffs(kb).items():
            if diem >= cutoff:
                facts.add(f"diem_dat_{ma}")
            elif diem >= cutoff - 1:
                facts.add(f"diem_sat_nguong_{ma}")
    return facts
