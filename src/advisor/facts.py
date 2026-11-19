"""C2 – Dữ kiện đầu vào: kiểm tra hợp lệ, tiền xử lý thành sự kiện ký hiệu, và "ngữ cảnh số" cho các hàm tính.

Người dùng nhập (mục 1 của tập luật) – dict `user`, khóa:
    phuong_thuc   "THPT" | "DGNL"
    khu_vuc       "KV1" | "KV2-NT" | "KV2" | "KV3"
    diem_Toan, diem_Anh, diem_Ly, diem_Hoa, diem_Sinh, diem_Tin   (THPT; None = không thi)
    diem_dgnl, HB_mon1 (Tiếng Anh), HB_mon2, mon_hb2 ∈ {Ly, Hoa, Sinh, Tin}   (ĐGNL)
    thich_lap_trinh, thich_du_lieu, thich_thiet_ke_hoc_lieu, thich_day_hoc         (bool)
    lap_trinh_vien, phan_tich_du_lieu, giao_vien, chuyen_vien_edtech               (bool)

Phân biệt "chưa biết" và "không có":
    - khóa KHÔNG có trong dict   → chưa hỏi (suy diễn lùi sẽ đưa vào danh sách cần hỏi)
    - khóa có, giá trị None      → đã hỏi, môn đó không thi / không có (quy ước: rỗng, không dùng 0)

Sự kiện là tuple: ("phuong_thuc", "THPT"), ("co_mon", "Toan"), ("thich_lap_trinh",) …
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from . import diem as D
from .tham_so import DINH_HUONG, DIEM_LIET, MON, MON_HB2, MUC_UU_TIEN, PHUONG_THUC, SO_THICH, TO_HOP

Atom = tuple[str, ...]

KHOA_DIEM = tuple(f"diem_{m}" for m in MON) + ("diem_dgnl", "HB_mon1", "HB_mon2")
KHOA_BOOL = SO_THICH + DINH_HUONG
CO_DIEM = {"NL": "diem_dgnl", "HB1": "HB_mon1", "HB2": "HB_mon2"}   # co_diem(NL) ↔ diem_dgnl


# --------------------------------------------------------------------------- kiểm tra đầu vào
def kiem_tra_dau_vao(user: dict) -> list[str]:
    """Trả về danh sách lỗi (rỗng = hợp lệ). Điểm hợp lệ trong [0; 10]; ngoài khoảng → báo lỗi, hỏi lại."""
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
        if not (Decimal(0) <= x <= Decimal(10)):
            loi.append(f"{k} = {v} nằm ngoài khoảng [0; 10].")
    if user.get("mon_hb2") is not None and user["mon_hb2"] not in MON_HB2:
        loi.append(f"Môn học bạ thứ hai '{user['mon_hb2']}' không hợp lệ (Ly, Hoa, Sinh, Tin).")
    return loi


# --------------------------------------------------------------------------- tiền xử lý → sự kiện
def build_facts(user: dict) -> set[Atom]:
    """Dữ kiện đầu vào → tập sự kiện ban đầu của bộ nhớ làm việc (working memory)."""
    f: set[Atom] = set()
    if user.get("phuong_thuc"):
        f.add(("phuong_thuc", user["phuong_thuc"]))
    if user.get("khu_vuc"):
        f.add(("khu_vuc", user["khu_vuc"]))
    for m in MON:                                       # co_mon(X): môn X có điểm (khác rỗng)
        if user.get(f"diem_{m}") is not None:
            f.add(("co_mon", m))
    for ten, khoa in CO_DIEM.items():                   # có NL, có HB1, có HB2
        if user.get(khoa) is not None:
            f.add(("co_diem", ten))
    if user.get("mon_hb2"):
        f.add(("mon_hb2", user["mon_hb2"]))
    for k in KHOA_BOOL:
        if user.get(k):
            f.add((k,))
    return f


def khoa_can_hoi(atom: Atom) -> str | None:
    """Sự kiện lá (không luật nào sinh ra) ↔ dữ kiện đầu vào cần hỏi người dùng. None = không hỏi được."""
    p = atom[0]
    if p in ("phuong_thuc", "khu_vuc", "mon_hb2"):
        return p
    if p == "co_mon" and len(atom) == 2:
        return f"diem_{atom[1]}"
    if p == "co_diem" and len(atom) == 2:
        return CO_DIEM.get(atom[1])
    if p in KHOA_BOOL:
        return p
    return None


# --------------------------------------------------------------------------- ngữ cảnh số cho hàm tính
@dataclass
class DuLieu:
    """Giá trị số của người dùng + các hàm tính tong_diem / DUT / DXT (mục 3). Luật gọi qua điều kiện Test."""
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
        """DXT_THPT(T) hoặc DXT_DGNL (ĐGNL chỉ có một tổ hợp nên không phụ thuộc T)."""
        return D.diem_xet_tuyen(self.tong_diem(to_hop), self.khu_vuc)


__all__ = ["Atom", "DuLieu", "build_facts", "kiem_tra_dau_vao", "khoa_can_hoi", "KHOA_DIEM", "KHOA_BOOL"]
