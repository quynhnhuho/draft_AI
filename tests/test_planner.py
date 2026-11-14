"""Kiểm thử Chức năng 1 – lập lộ trình học.  Chạy: pytest -q

Test thuật toán tự bỏ qua (skip) nếu file thuật toán đó chưa được commit vào repo.
"""
import pytest

from src.planner import dieu_phoi, kich_ban
from src.planner.model import (MAX_TC_BINH_THUONG, MAX_TC_HOC_VUOT, BaiToan, doc_du_lieu, hk1_co_dinh,
                               kiem_tra_dau_vao, kiem_tra_lo_trinh, tao_bai_toan)

DS = doc_du_lieu()
CO_THUAT_TOAN = {k: dieu_phoi.nap_thuat_toan(k) for k in dieu_phoi.DANH_SACH_THUAT_TOAN}


def can(khoa):
    return pytest.mark.skipif(CO_THUAT_TOAN[khoa] is None, reason=f"chưa có src/planner/{khoa}.py")


# ----------------------------------------------------------------------------- dữ liệu & mô hình
def test_du_lieu_va_hk1_co_dinh():
    assert len(DS) >= 25
    hk1 = hk1_co_dinh(DS)
    assert len(hk1) == 7 and sum(DS[m].tin_chi for m in hk1) == 18


def test_hoc_truoc_chi_can_da_hoc_ke_ca_rot():
    # Rớt COMP1010 vẫn được học COMP1017 (học trước COMP1010) ở HK2
    assert not kiem_tra_dau_vao(DS, {"COMP1017"}, {"COMP1010"}, 3)
    assert kiem_tra_dau_vao(DS, {"COMP1017"}, set(), 3)          # chưa học COMP1010 -> lỗi
    bt = tao_bai_toan(DS, set(hk1_co_dinh(DS)) - {"COMP1010"}, {"COMP1010"}, 2)
    assert "COMP1017" in bt.du_dieu_kien(bt.trang_thai_dau())


def test_kiem_tra_dau_vao():
    assert kiem_tra_dau_vao(DS, {"XYZ123"}, set(), 3)              # mã không tồn tại
    assert kiem_tra_dau_vao(DS, {"COMP1010"}, {"COMP1010"}, 3)     # vừa đạt vừa rớt
    assert kiem_tra_dau_vao(DS, {"COMP1010"}, set(), 1)            # HK1 chưa học gì


def test_hk1_chi_co_mot_hanh_dong_va_max_tc():
    bt = tao_bai_toan(DS, set(), set(), 1)
    assert bt.hanh_dong(bt.trang_thai_dau()) == [hk1_co_dinh(DS)]
    assert bt.max_tc == MAX_TC_BINH_THUONG
    assert tao_bai_toan(DS, set(), set(), 1, hoc_vuot=True).max_tc == MAX_TC_HOC_VUOT


def test_hanh_dong_la_tap_toi_dai():
    bt = tao_bai_toan(DS, set(hk1_co_dinh(DS)), set(), 2)
    s = bt.trang_thai_dau()
    ung_vien = bt.du_dieu_kien(s)
    for a in bt.hanh_dong(s):
        tong = sum(DS[m].tin_chi for m in a)
        assert tong <= bt.max_tc
        assert all(DS[m].tin_chi > bt.max_tc - tong for m in ung_vien if m not in a)


# ----------------------------------------------------------------------------- thuật toán
KICH_BAN = list(kich_ban.cac_kich_ban(DS))


@pytest.mark.parametrize("ten", KICH_BAN)
def test_cac_thuat_toan_cung_so_ky_va_hop_le(ten):
    bt = kich_ban.tao(ten, DS)
    kqs = dieu_phoi.so_sanh(bt)
    if not kqs:
        pytest.skip("chưa có thuật toán nào")
    for kq in kqs:
        assert kq.lo_trinh is not None and kq.toi_uu, kq.thuat_toan
        assert not kiem_tra_lo_trinh(bt, kq.lo_trinh), (kq.thuat_toan, kiem_tra_lo_trinh(bt, kq.lo_trinh))
    assert len({kq.so_ky for kq in kqs}) == 1, [(kq.thuat_toan, kq.so_ky) for kq in kqs]


@can("a_sao")
@pytest.mark.parametrize("ten", KICH_BAN)
def test_heuristic_chap_nhan_duoc(ten):
    bt = kich_ban.tao(ten, DS)
    toi_uu = CO_THUAT_TOAN["a_sao"](bt).so_ky
    s0 = bt.trang_thai_dau()
    assert bt.h(s0) <= toi_uu and bt.h2(s0) <= toi_uu


@can("nhanh_can")
def test_nhanh_can_tham_lam_va_anytime():
    from src.planner.nhanh_can import loi_giai_tham_lam
    bt = kich_ban.tao(KICH_BAN[0], DS)
    tham_lam = loi_giai_tham_lam(bt)
    assert tham_lam is not None and not kiem_tra_lo_trinh(bt, tham_lam)
    kq = CO_THUAT_TOAN["nhanh_can"](bt, gioi_han_ms=0)      # hết giờ ngay: vẫn phải có lời giải
    assert kq.lo_trinh is not None and kq.het_gioi_han
    assert kq.chi_tiet["can_tren_ban_dau"] >= CO_THUAT_TOAN["nhanh_can"](bt).so_ky


@can("a_sao")
@can("nhanh_can")
def test_dieu_phoi_chuyen_sang_nhanh_can_khi_a_sao_het_gio(monkeypatch):
    monkeypatch.setattr(dieu_phoi, "TI_LE_CHO_A_SAO", 0.0)  # A* hết giờ ngay
    kq = dieu_phoi.lap_lo_trinh(kich_ban.tao(KICH_BAN[0], DS), gioi_han_ms=2000)
    assert kq.ket_qua.thuat_toan == "Nhánh cận" and kq.ghi_chu


@can("a_sao")
@can("bfs")
def test_dieu_phoi_dung_a_sao_va_doi_chieu_bfs():
    kq = dieu_phoi.lap_lo_trinh(kich_ban.tao(KICH_BAN[1], DS), doi_chieu=True)
    assert kq.ket_qua.thuat_toan == "A*" and kq.khop_doi_chieu


def test_dieu_phoi_bao_chua_cai_dat(monkeypatch):
    monkeypatch.setattr(dieu_phoi, "nap_thuat_toan", lambda khoa: None)
    with pytest.raises(NotImplementedError):
        dieu_phoi.lap_lo_trinh(kich_ban.tao(KICH_BAN[0], DS))


def test_muc_tieu_la_nhom_hoc_phan():
    # Chỉ cần đạt COMP1314 (Trí tuệ nhân tạo): phải học thêm COMP1701 trước đó
    bt = BaiToan(DS, frozenset({"COMP1314"}), frozenset(hk1_co_dinh(DS)), 2, truong_xep={1: hk1_co_dinh(DS)})
    assert bt.can_hoc == {"COMP1314", "COMP1701"}
