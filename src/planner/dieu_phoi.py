"""Chức năng 1 – BỘ ĐIỀU PHỐI: kết hợp 3 thuật toán, là điểm vào DUY NHẤT mà chatbot gọi.
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
    "bfs": ("bfs", "bfs", "BFS"),
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
    doi_chieu: KetQua | None = None                          # kết quả BFS đối chiếu (nếu có)
    ghi_chu: list[str] = field(default_factory=list)

    @property
    def khop_doi_chieu(self) -> bool | None:
        if self.doi_chieu is None or self.doi_chieu.so_ky is None or self.ket_qua is None:
            return None
        return self.doi_chieu.so_ky == self.ket_qua.so_ky


def lap_lo_trinh(bt: BaiToan, gioi_han_ms: float = GIOI_HAN_MS, doi_chieu: bool = False) -> KetQuaDieuPhoi:
    """Điểm vào cho chatbot. Raise NotImplementedError nếu chưa có thuật toán nào trong repo."""
    a_sao, nhanh_can, bfs = (nap_thuat_toan(k) for k in ("a_sao", "nhanh_can", "bfs"))
    if not (a_sao or nhanh_can or bfs):
        raise NotImplementedError("Chưa có thuật toán lập lộ trình nào (bfs.py, a_sao.py, nhanh_can.py)")

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

    if kq_dp.ket_qua is None and bfs:                         # chỉ còn BFS
        kq = bfs(bt, gioi_han_ms=gioi_han_ms)
        kq_dp.cac_lan_chay.append(kq)
        kq_dp.ket_qua = kq

    if doi_chieu and bfs and kq_dp.ket_qua is not None and kq_dp.ket_qua.thuat_toan != "BFS":
        kq_dp.doi_chieu = bfs(bt, gioi_han_ms=gioi_han_ms)
    return kq_dp


def so_sanh(bt: BaiToan, gioi_han_ms: float = 30000, them_h2: bool = True) -> list[KetQua]:
    """Chạy MỌI thuật toán sẵn có trên cùng bài toán."""
    ket_qua = []
    for khoa in ("bfs", "a_sao", "nhanh_can"):
        ham = nap_thuat_toan(khoa)
        if ham is None:
            continue
        ket_qua.append(ham(bt, gioi_han_ms=gioi_han_ms))
        if them_h2 and khoa != "bfs":
            ket_qua.append(ham(bt, heuristic="h2", gioi_han_ms=gioi_han_ms))
    return ket_qua
