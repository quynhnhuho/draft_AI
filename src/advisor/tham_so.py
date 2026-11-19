"""C2 – Bảng tham số tuyển sinh năm 2026 (mục 4 của tập luật), đọc từ data/tuyen_sinh_2026.json.

Mọi con số dùng trong luật (ngưỡng, điểm chuẩn, mức ưu tiên, hệ số ĐGNL) đều lấy từ file này,
không viết cứng trong code → năm sau chỉ cần thay file JSON.
Số được giữ ở dạng Decimal để cộng/làm tròn chính xác (tránh lỗi 0.1 + 0.2 của float).
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
MON = ("Toan", "Anh", "Ly", "Hoa", "Sinh", "Tin")             # tên môn không dấu dùng trong luật
MON_HB2 = ("Ly", "Hoa", "Sinh", "Tin")                       # môn học bạ thứ hai hợp lệ
TEN_MON = {"Toan": "Toán", "Anh": "Tiếng Anh", "Ly": "Vật lý", "Hoa": "Hóa học", "Sinh": "Sinh học",
           "Tin": "Tin học"}

TO_HOP: dict[str, tuple[str, ...]] = {t: tuple(m) for t, m in _RAW["to_hop"].items()}
MUC_UU_TIEN: dict[str, Decimal] = {kv: Decimal(v) for kv, v in _RAW["muc_uu_tien"].items()}

_CT = _RAW["cong_thuc"]
HE_SO_DGNL: dict[str, Decimal] = {k: Decimal(v) for k, v in _CT["dgnl_he_so"].items()}
NHAN_DGNL = Decimal(_CT["dgnl_nhan"])
NGUONG_GIAM_UU_TIEN = Decimal(_CT["nguong_giam_uu_tien"])    # 22.5
DIEM_LIET = Decimal(_CT["diem_liet"])                        # 1.0

NGANH: dict[str, dict] = _RAW["nganh"]                       # khóa: CNTT, SPTH, CNGD
MA_NGANH = tuple(NGANH)

SO_THICH = ("thich_lap_trinh", "thich_du_lieu", "thich_thiet_ke_hoc_lieu", "thich_day_hoc")
DINH_HUONG = ("lap_trinh_vien", "phan_tich_du_lieu", "giao_vien", "chuyen_vien_edtech")


def nguong(nganh: str, pt: str) -> Decimal:
    """Ngưỡng đảm bảo chất lượng đầu vào – dùng cho du_dk (R11–R16)."""
    return Decimal(NGANH[nganh]["nguong"][pt])


def diem_chuan(nganh: str, pt: str) -> Decimal:
    """Điểm chuẩn năm 2026 – chỉ dùng cho kha_nang (R17–R18)."""
    return Decimal(NGANH[nganh]["diem_chuan"][pt])


def ten_nganh(nganh: str) -> str:
    return NGANH[nganh]["ten"]


def tu_ma_kb(ma_kb: str) -> str | None:
    """Mã ngành trong knowledge_base.json ("cntt", "sp_tin", "cngd") → mã trong luật ("CNTT", ...)."""
    return next((k for k, v in NGANH.items() if v["ma_kb"] == ma_kb), None)


def nguon_trich_dan() -> dict:
    """Dạng 'fact' có nguon + nam để chatbot hiển thị kèm câu trả lời (hàm cite() của C4)."""
    url = NGUON.get("url") or []
    return {"nguon": NGUON["nguon"], "nam": NAM, "url": url[0] if url else "",
            "da_kiem_chung": bool(NGUON.get("da_kiem_chung"))}
