"""Chức năng 2 - Bảng tham số tuyển sinh năm 2026.

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "tuyen_sinh_2026.json"

_RAW: dict = json.loads(DATA_PATH.read_text(encoding="utf-8"))

NAM: int = _RAW["nam"]
NGUON: dict = {k: _RAW.get(k) for k in ("nguon", "url", "ngay_truy_cap", "da_kiem_chung", "ghi_chu")}

PHUONG_THUC = ("THPT", "DGNL")
MON = ("Toan", "Anh", "Ly", "Hoa", "Sinh", "Tin")
MON_HB2 = ("Ly", "Hoa", "Sinh", "Tin")
TEN_MON = {"Toan": "Toán", "Anh": "Tiếng Anh", "Ly": "Vật lý", "Hoa": "Hóa học", "Sinh": "Sinh học",
           "Tin": "Tin học"}

TO_HOP: dict[str, tuple[str, ...]] = {t: tuple(m) for t, m in _RAW["to_hop"].items()}
MUC_UU_TIEN: dict[str, Decimal] = {kv: Decimal(v) for kv, v in _RAW["muc_uu_tien"].items()}

HE_SO_DGNL: dict[str, Decimal] = {k: Decimal(v) for k, v in _RAW["cong_thuc"]["dgnl_he_so"].items()}
NHAN_DGNL = Decimal(_RAW["cong_thuc"]["dgnl_nhan"])
NGUONG_GIAM_UU_TIEN = Decimal(_RAW["cong_thuc"]["nguong_giam_uu_tien"])
DIEM_LIET = Decimal(_RAW["cong_thuc"]["diem_liet"])

NGANH: dict[str, dict] = _RAW["nganh"]
MA_NGANH = tuple(NGANH)

SO_THICH = ("thich_lap_trinh", "thich_du_lieu", "thich_thiet_ke_hoc_lieu", "thich_day_hoc")
DINH_HUONG = ("lap_trinh_vien", "phan_tich_du_lieu", "giao_vien", "chuyen_vien_edtech")


def nguong(nganh: str, pt: str) -> Decimal:
    return Decimal(NGANH[nganh]["nguong"][pt])


def diem_chuan(nganh: str, pt: str) -> Decimal:
    return Decimal(NGANH[nganh]["diem_chuan"][pt])


def ten_nganh(nganh: str) -> str:
    return NGANH[nganh]["ten"]


def tu_ma_kb(ma_kb: str) -> str | None:
    return next((k for k, v in NGANH.items() if v["ma_kb"] == ma_kb), None)


def nguon_trich_dan() -> dict:
    url = NGUON.get("url") or []
    return {"nguon": NGUON["nguon"], "nam": NAM, "url": url[0] if url else "",
            "da_kiem_chung": bool(NGUON.get("da_kiem_chung"))}
