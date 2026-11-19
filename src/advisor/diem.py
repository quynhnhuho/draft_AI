"""C2 – Công thức tính điểm (mục 3 của tập luật). Các "hàm tính" được luật gọi tới (procedural attachment).

    tong_diem  : THPT = M1 + M2 + M3 ;  ĐGNL = (0.5·NL + 0.35·HB1 + 0.15·HB2) · 3     (thang 30)
    DUT        : tong ≥ 22.5 → (30 − tong) / 7.5 × mức ưu tiên ;  tong < 22.5 → mức ưu tiên
    DXT        : tong_diem + DUT, làm tròn 2 chữ số thập phân (làm tròn nửa lên) TRƯỚC khi so sánh

Dùng Decimal để kết quả khớp với tính tay (float cho 0.35*8.4 = 2.9399999…).
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .tham_so import HE_SO_DGNL, MUC_UU_TIEN, NGUONG_GIAM_UU_TIEN, NHAN_DGNL

D30 = Decimal(30)
D7_5 = Decimal("7.5")


def so(x) -> Decimal:
    """float/int/str → Decimal theo đúng chữ số người dùng nhập (8.2 → Decimal('8.2'))."""
    return x if isinstance(x, Decimal) else Decimal(str(x))


def lam_tron(x: Decimal, chu_so: int = 2) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-chu_so), rounding=ROUND_HALF_UP)


def tong_thpt(m1, m2, m3) -> Decimal:
    return so(m1) + so(m2) + so(m3)


def tong_dgnl(nl, hb1, hb2) -> Decimal:
    return (HE_SO_DGNL["NL"] * so(nl) + HE_SO_DGNL["HB1"] * so(hb1) + HE_SO_DGNL["HB2"] * so(hb2)) * NHAN_DGNL


def diem_uu_tien(tong: Decimal, khu_vuc: str) -> Decimal:
    muc = MUC_UU_TIEN[khu_vuc]
    if tong >= NGUONG_GIAM_UU_TIEN:
        return (D30 - tong) / D7_5 * muc
    return muc


def diem_xet_tuyen(tong: Decimal, khu_vuc: str) -> Decimal:
    return lam_tron(tong + diem_uu_tien(tong, khu_vuc))
