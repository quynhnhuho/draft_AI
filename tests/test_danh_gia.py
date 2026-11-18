"""Kiểm thử MINH CHỨNG vai trò 3 thuật toán trong chatbot (Chức năng 1).

Chỉ kiểm tra các kết luận KHÔNG phụ thuộc tốc độ máy (số kỳ, số nút, có/không có lời giải).
Số liệu thời gian/bộ nhớ xem bằng: python -m src.planner.danh_gia
"""
import random

import pytest

from src.planner import dieu_phoi, kich_ban
from src.planner.danh_gia import cac_thuat_toan, sinh_vien_ngau_nhien
from src.planner.model import doc_du_lieu, kiem_tra_lo_trinh, tao_bai_toan

DS = doc_du_lieu()
ALGS = cac_thuat_toan()
DU_3 = pytest.mark.skipif(len(ALGS) < 3, reason="cần đủ bfs.py, a_sao.py, nhanh_can.py")
KICH_BAN = list(kich_ban.cac_kich_ban(DS))
ID_KB = [f"tinh_huong_{ten.split('.')[0]}" for ten in KICH_BAN]   # tên test dễ đọc


def _bai_toan_ngau_nhien(n=100, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        dat, rot, hk, vuot = sinh_vien_ngau_nhien(DS, rng)
        yield tao_bai_toan(DS, dat, rot, hk, vuot)


@DU_3
def test_bfs_la_chuan_a_sao_va_nhanh_can_luon_khop_tren_100_sinh_vien():
    """Tính đúng: A* và nhánh cận cho cùng số kỳ tối ưu với BFS; mọi lộ trình hợp lệ."""
    for bt in _bai_toan_ngau_nhien():
        kqs = {k: f(bt) for k, f in ALGS.items()}
        assert kqs["a_sao"].so_ky == kqs["bfs"].so_ky == kqs["nhanh_can"].so_ky
        for kq in kqs.values():
            assert not kiem_tra_lo_trinh(bt, kq.lo_trinh)


@DU_3
def test_a_sao_mo_it_nut_hon_hoac_bang_bfs_tren_100_sinh_vien():
    """Hiệu quả: lý do chọn A* làm bộ giải chính thay vì BFS."""
    tong_a = tong_bfs = 0
    for bt in _bai_toan_ngau_nhien():
        a, b = ALGS["a_sao"](bt), ALGS["bfs"](bt)
        assert a.so_nut_mo_rong <= b.so_nut_mo_rong
        tong_a, tong_bfs = tong_a + a.so_nut_mo_rong, tong_bfs + b.so_nut_mo_rong
    assert tong_a * 3 < tong_bfs            # tổng số nút của A* ít hơn 3 lần BFS


@DU_3
@pytest.mark.parametrize("ten", KICH_BAN, ids=ID_KB)
def test_het_gio_chi_nhanh_can_con_tra_loi(ten):
    """Lý do chọn nhánh cận làm dự phòng: giới hạn thời gian = 0 thì BFS và A* chưa có lời giải,
    nhánh cận vẫn có lộ trình hợp lệ (lời giải tham lam)."""
    bt = kich_ban.tao(ten, DS)
    assert ALGS["bfs"](bt, gioi_han_ms=0).lo_trinh is None
    assert ALGS["a_sao"](bt, gioi_han_ms=0).lo_trinh is None
    kq = ALGS["nhanh_can"](bt, gioi_han_ms=0)
    assert kq.lo_trinh is not None and kq.het_gioi_han
    assert not kiem_tra_lo_trinh(bt, kq.lo_trinh)
    assert kq.so_ky >= ALGS["bfs"](bt).so_ky            # có thể chưa tối ưu, không bao giờ "tốt hơn tối ưu"


@DU_3
@pytest.mark.parametrize("ten", KICH_BAN, ids=ID_KB)
def test_bo_dieu_phoi_chon_dung_vai_tro(ten, monkeypatch):
    """Bình thường -> A*; A* hết giờ -> nhánh cận; debug -> đối chiếu BFS khớp."""
    bt = kich_ban.tao(ten, DS)
    toi_uu = ALGS["bfs"](bt).so_ky

    kq = dieu_phoi.lap_lo_trinh(bt, doi_chieu=True)
    assert kq.ket_qua.thuat_toan == "A*" and kq.ket_qua.so_ky == toi_uu
    assert kq.khop_doi_chieu

    monkeypatch.setattr(dieu_phoi, "TI_LE_CHO_A_SAO", 0.0)
    kq = dieu_phoi.lap_lo_trinh(bt)
    assert [k.thuat_toan for k in kq.cac_lan_chay] == ["A*", "Nhánh cận"]
    assert kq.ket_qua.thuat_toan == "Nhánh cận" and kq.ket_qua.so_ky == toi_uu
