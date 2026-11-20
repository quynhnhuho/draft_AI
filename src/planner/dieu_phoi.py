# AI-assisted: Claude (Anthropic) – hỗ trợ thiết kế và viết bộ điều phối (A* chính, nhánh cận dự phòng); nhóm đã kiểm tra lại.
"""Chức năng 1 – BỘ ĐIỀU PHỐI: kết hợp A* và nhánh cận, là điểm vào DUY NHẤT mà chatbot gọi.

BFS (bfs.py) được cài đặt theo yêu cầu đề nhưng không dùng trong chatbot: BFS mở nhiều nút nhất,
trong khi A* đã cho lộ trình ít học kỳ nhất và nhánh cận luôn có lời giải khi A* chậm.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from .model import BaiToan, KetQua

# tên khóa -> (module, hàm, tên hiển thị)
DANH_SACH_THUAT_TOAN = {
    "a_sao": ("a_sao", "a_sao", "A*"),
    "nhanh_can": ("nhanh_can", "nhanh_can", "Nhánh cận"),
}

GIOI_HAN_MS = 3000          # tổng thời gian tối đa cho một lần trả lời chatbot
TI_LE_CHO_A_SAO = 0.6       # A* được dùng 60% thời gian, phần còn lại cho nhánh cận


def nap_thuat_toan(khoa: str) -> Callable[..., KetQua] | None:
    """Nạp hàm thuật toán nếu file đã có trong repo, ngược lại trả None."""
    module, ham, _ = DANH_SACH_THUAT_TOAN[khoa]
    try:
        return getattr(importlib.import_module(f".{module}", __package__), ham)
    except (ImportError, AttributeError):
        return None


def thuat_toan_san_co() -> dict[str, bool]:
    return {ten: nap_thuat_toan(k) is not None for k, (_, _, ten) in DANH_SACH_THUAT_TOAN.items()}


@dataclass
class KetQuaDieuPhoi:
    ket_qua: KetQua | None                                   # kết quả dùng để trả lời
    cac_lan_chay: list[KetQua] = field(default_factory=list)  # mọi lần chạy (để debug/báo cáo)
    ghi_chu: list[str] = field(default_factory=list)


def lap_lo_trinh(bt: BaiToan, gioi_han_ms: float = GIOI_HAN_MS) -> KetQuaDieuPhoi:
    """Điểm vào cho chatbot. Raise NotImplementedError nếu chưa có A* và nhánh cận trong repo."""
    a_sao, nhanh_can = nap_thuat_toan("a_sao"), nap_thuat_toan("nhanh_can")
    if not (a_sao or nhanh_can):
        raise NotImplementedError("Chưa có thuật toán lập lộ trình (a_sao.py, nhanh_can.py)")

    kq_dp = KetQuaDieuPhoi(None)
    con_lai = gioi_han_ms

    if a_sao:
        kq = a_sao(bt, gioi_han_ms=gioi_han_ms * TI_LE_CHO_A_SAO if nhanh_can else gioi_han_ms)
        kq_dp.cac_lan_chay.append(kq)
        con_lai -= kq.thoi_gian_ms
        if not kq.het_gioi_han:
            kq_dp.ket_qua = kq
        else:
            kq_dp.ghi_chu.append(f"A* dừng sau {kq.thoi_gian_ms:.0f} ms (hết giới hạn), chuyển sang nhánh cận.")

    if kq_dp.ket_qua is None and nhanh_can:
        kq = nhanh_can(bt, gioi_han_ms=max(con_lai, 200))
        kq_dp.cac_lan_chay.append(kq)
        kq_dp.ket_qua = kq
        if not kq.toi_uu:
            kq_dp.ghi_chu.append("Nhánh cận hết giới hạn: trả về lộ trình tốt nhất đã tìm, có thể chưa tối ưu.")
    return kq_dp
