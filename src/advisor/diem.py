# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .tham_so import HE_SO_DGNL, MUC_UU_TIEN, NGUONG_GIAM_UU_TIEN, NHAN_DGNL

D30 = Decimal(30)
D7_5 = Decimal("7.5")


def so(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def lam_tron(x: Decimal, chu_so: int = 2) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-chu_so), rounding=ROUND_HALF_UP)


def tong_thpt(m1, m2, m3) -> Decimal:
    return so(m1) + so(m2) + so(m3)


def tong_dgnl(nl, hb1, hb2) -> Decimal:
    return (HE_SO_DGNL["NL"] * so(nl) + HE_SO_DGNL["HB1"] * so(hb1) + HE_SO_DGNL["HB2"] * so(hb2)) * NHAN_DGNL


def diem_uu_tien(tong, khu_vuc: str) -> Decimal:
    tong, muc = so(tong), MUC_UU_TIEN[khu_vuc]
    if tong >= NGUONG_GIAM_UU_TIEN:
        return (D30 - tong) / D7_5 * muc
    return muc


def diem_xet_tuyen(tong, khu_vuc: str) -> Decimal:
    return lam_tron(so(tong) + diem_uu_tien(tong, khu_vuc))
