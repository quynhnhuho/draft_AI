"""Chức năng 2 - Dữ kiện đầu vào: kiểm tra hợp lệ, chuyển thành sự kiện, ngữ cảnh số cho các hàm tính.

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from . import diem as D
from .tham_so import DIEM_LIET, DINH_HUONG, MON, MON_HB2, MUC_UU_TIEN, PHUONG_THUC, SO_THICH, TO_HOP

Atom = tuple[str, ...]

KHOA_DIEM = tuple(f"diem_{m}" for m in MON) + ("diem_dgnl", "HB_mon1", "HB_mon2")
KHOA_BOOL = SO_THICH + DINH_HUONG
CO_DIEM = {"NL": "diem_dgnl", "HB1": "HB_mon1", "HB2": "HB_mon2"}


def kiem_tra_dau_vao(user: dict) -> list[str]:
    loi = []
    if "phuong_thuc" in user and user["phuong_thuc"] not in PHUONG_THUC:
        loi.append(f"Phương thức '{user['phuong_thuc']}' không hợp lệ (chỉ THPT hoặc DGNL).")
    if "khu_vuc" in user and user["khu_vuc"] not in MUC_UU_TIEN:
        loi.append(f"Khu vực '{user['khu_vuc']}' không hợp lệ (KV1, KV2-NT, KV2, KV3).")
    for k in KHOA_DIEM:
        v = user.get(k)
        if v is None:
            continue
        try:
            x = D.so(v)
        except Exception:
            loi.append(f"{k} = {v!r} không phải là số.")
            continue
        if not Decimal(0) <= x <= Decimal(10):
            loi.append(f"{k} = {v} nằm ngoài khoảng [0; 10].")
    if user.get("mon_hb2") is not None and user["mon_hb2"] not in MON_HB2:
        loi.append(f"Môn học bạ thứ hai '{user['mon_hb2']}' không hợp lệ (Ly, Hoa, Sinh, Tin).")
    return loi


def build_facts(user: dict) -> set[Atom]:
    f: set[Atom] = set()
    for k in ("phuong_thuc", "khu_vuc", "mon_hb2"):
        if user.get(k):
            f.add((k, user[k]))
    f |= {("co_mon", m) for m in MON if user.get(f"diem_{m}") is not None}
    f |= {("co_diem", ten) for ten, khoa in CO_DIEM.items() if user.get(khoa) is not None}
    f |= {(k,) for k in KHOA_BOOL if user.get(k)}
    return f


def khoa_can_hoi(atom: Atom) -> str | None:
    p = atom[0]
    if p in ("phuong_thuc", "khu_vuc", "mon_hb2") or p in KHOA_BOOL:
        return p
    if p == "co_mon" and len(atom) == 2:
        return f"diem_{atom[1]}"
    if p == "co_diem" and len(atom) == 2:
        return CO_DIEM.get(atom[1])
    return None


@dataclass
class DuLieu:
    user: dict = field(default_factory=dict)

    def biet(self, khoa: str) -> bool:
        return khoa in self.user

    @property
    def pt(self) -> str | None:
        return self.user.get("phuong_thuc")

    @property
    def khu_vuc(self) -> str | None:
        return self.user.get("khu_vuc")

    def diem_mon(self, mon: str) -> Decimal | None:
        v = self.user.get(f"diem_{mon}")
        return None if v is None else D.so(v)

    def diem_thap_nhat(self, to_hop: str) -> Decimal:
        return min(self.diem_mon(m) for m in TO_HOP[to_hop])

    def co_diem_liet(self, to_hop: str) -> bool:
        return self.diem_thap_nhat(to_hop) <= DIEM_LIET

    def tong_diem(self, to_hop: str) -> Decimal:
        if self.pt == "DGNL":
            u = self.user
            return D.tong_dgnl(u["diem_dgnl"], u["HB_mon1"], u["HB_mon2"])
        return D.tong_thpt(*(self.diem_mon(m) for m in TO_HOP[to_hop]))

    def dut(self, to_hop: str) -> Decimal:
        return D.diem_uu_tien(self.tong_diem(to_hop), self.khu_vuc)

    def dxt(self, to_hop: str) -> Decimal:
        return D.diem_xet_tuyen(self.tong_diem(to_hop), self.khu_vuc)


__all__ = ["Atom", "DuLieu", "build_facts", "kiem_tra_dau_vao", "khoa_can_hoi", "KHOA_DIEM", "KHOA_BOOL"]
