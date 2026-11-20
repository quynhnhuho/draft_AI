# AI-assisted: Claude (Anthropic) – hỗ trợ thiết kế và viết bộ điều phối (A* chính, nhánh cận dự phòng); nhóm đã kiểm tra lại.
"""Chức năng 1 – BỘ ĐIỀU PHỐI: kết hợp A* và nhánh cận, là điểm vào DUY NHẤT mà chatbot gọi.

BFS (bfs.py) được cài đặt theo yêu cầu đề nhưng không dùng trong chatbot: BFS mở nhiều nút nhất,
trong khi A* đã cho lộ trình ít học kỳ nhất và nhánh cận luôn có lời giải khi A* chậm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .a_sao import a_sao
from .model import BaiToan, KetQua
from .nhanh_can import nhanh_can

GIOI_HAN_MS = 3000          # tổng thời gian tối đa cho một lần trả lời chatbot
TI_LE_CHO_A_SAO = 0.6       # A* được dùng 60% thời gian, phần còn lại cho nhánh cận


@dataclass
class KetQuaDieuPhoi:
    ket_qua: KetQua                                          # kết quả dùng để trả lời
    cac_lan_chay: list[KetQua] = field(default_factory=list)  # mọi lần chạy (để debug/báo cáo)
    ghi_chu: list[str] = field(default_factory=list)


def lap_lo_trinh(bt: BaiToan, gioi_han_ms: float = GIOI_HAN_MS) -> KetQuaDieuPhoi:
    """Điểm vào cho chatbot: chạy A* trước; A* hết giờ thì chuyển sang nhánh cận."""
    kq = a_sao(bt, gioi_han_ms=gioi_han_ms * TI_LE_CHO_A_SAO)
    if not kq.het_gioi_han:
        return KetQuaDieuPhoi(kq, [kq])

    kq_dp = KetQuaDieuPhoi(kq, [kq], [f"A* dừng sau {kq.thoi_gian_ms:.0f} ms (hết giới hạn), chuyển sang nhánh cận."])
    kq_dp.ket_qua = nhanh_can(bt, gioi_han_ms=max(gioi_han_ms - kq.thoi_gian_ms, 200))
    kq_dp.cac_lan_chay.append(kq_dp.ket_qua)
    if not kq_dp.ket_qua.toi_uu:
        kq_dp.ghi_chu.append("Nhánh cận hết giới hạn: trả về lộ trình tốt nhất đã tìm, có thể chưa tối ưu.")
    return kq_dp
