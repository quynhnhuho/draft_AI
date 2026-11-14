"""Chức năng 1 – Thuật toán QUAY LUI – NHÁNH CẬN (Branch and Bound).

Thành viên phụ trách : <Họ tên – MSSV>
Vai trò trong chatbot: BỘ GIẢI DỰ PHÒNG. Khi A* vượt giới hạn thời gian, chatbot chạy nhánh cận.
                       Nhánh cận LUÔN có sẵn một lời giải (khởi tạo bằng tham lam) nên dừng lúc nào
                       cũng trả lời được ("anytime"); hết giờ thì trả lời giải tốt nhất đã tìm.

Ý tưởng:
    1. Cận trên ban đầu = lời giải THAM LAM: mỗi kỳ chọn tập học phần nhiều tín chỉ nhất
       (hòa thì ưu tiên học phần đứng trước trong CTĐT).
    2. DFS quay lui; tại mỗi nút, cận dưới = g + h(n). Nếu g + h(n) >= tốt nhất -> CẮT NHÁNH.
    3. Gặp lại trạng thái đã xét với g nhỏ hơn hoặc bằng -> CẮT (trùng lặp).
    4. Duyệt con có h nhỏ trước để sớm tìm lời giải tốt, cận trên giảm nhanh.

Chạy riêng:  python -m src.planner.nhanh_can
Ghi chú hỗ trợ AI: Claude (Anthropic) hỗ trợ rà soát và tái cấu trúc; nhóm đã kiểm tra lại.
"""
from __future__ import annotations

import math

from .model import BaiToan, DongHo, KetQua, LoTrinh, TrangThai


def loi_giai_tham_lam(bt: BaiToan) -> LoTrinh | None:
    """Mỗi kỳ chọn tập học phần có tổng tín chỉ lớn nhất. None nếu không thể hoàn thành."""
    s, lo_trinh, ky_trong_lien_tiep = bt.trang_thai_dau(), [], 0
    while not bt.la_dich(s):
        a = max(bt.hanh_dong(s), key=lambda tap: (sum(bt.hoc_phan[m].tin_chi for m in tap),
                                                  [-bt.hoc_phan[m].stt for m in sorted(
                                                      tap, key=lambda x: bt.hoc_phan[x].stt)]))
        ky_trong_lien_tiep = ky_trong_lien_tiep + 1 if not a else 0
        if ky_trong_lien_tiep >= 2:                 # 2 kỳ liên tiếp không học được gì -> bế tắc
            return None
        lo_trinh.append((s[1], a))
        s = bt.ke_tiep(s, a)
    return lo_trinh


def nhanh_can(bt: BaiToan, heuristic: str = "h", gioi_han_ms: float | None = None) -> KetQua:
    ten = "Nhánh cận" if heuristic == "h" else f"Nhánh cận ({heuristic})"
    ham_h = bt.ham_heuristic(heuristic)
    dong_ho = DongHo(gioi_han_ms)

    tham_lam = loi_giai_tham_lam(bt)
    lo_trinh_tot = tham_lam                         # lời giải tốt nhất đã có
    tot_nhat = math.inf if tham_lam is None else len(tham_lam)   # CẬN TRÊN
    can_tren_ban_dau = tot_nhat
    g_da_gap: dict = {}                             # khóa trạng thái -> g nhỏ nhất đã gặp
    duong: LoTrinh = []                             # lộ trình đang xây trên nhánh hiện tại
    mo_rong = sinh = cat_theo_can = cat_trung_lap = 0
    het_gio = False

    def tim_kiem(s: TrangThai, g: int) -> None:
        nonlocal tot_nhat, lo_trinh_tot, mo_rong, sinh, cat_theo_can, cat_trung_lap, het_gio
        if het_gio or dong_ho.het():
            het_gio = True
            return
        if bt.la_dich(s):                           # tới đích: cập nhật lời giải tốt nhất
            if g < tot_nhat:
                tot_nhat, lo_trinh_tot = g, list(duong)
            return
        if g + ham_h(s) >= tot_nhat:                # CẬN: nhánh này không thể tốt hơn -> cắt
            cat_theo_can += 1
            return
        k = bt.khoa(s)
        if g_da_gap.get(k, math.inf) <= g:          # đã xét trạng thái này với g nhỏ hơn/bằng
            cat_trung_lap += 1
            return
        g_da_gap[k] = g
        mo_rong += 1
        con = [(a, bt.ke_tiep(s, a)) for a in bt.hanh_dong(s)]
        sinh += len(con)
        con.sort(key=lambda x: ham_h(x[1]))         # nhánh hứa hẹn (h nhỏ) xét trước
        for a, s_moi in con:
            duong.append((s[1], a))                 # thử đăng ký tập môn a
            tim_kiem(s_moi, g + 1)
            duong.pop()                             # quay lui: bỏ lựa chọn a

    tim_kiem(bt.trang_thai_dau(), 0)
    # Hết giờ nhưng lời giải đã bằng cận dưới h(s0) thì vẫn chắc chắn tối ưu.
    toi_uu = not het_gio or (lo_trinh_tot is not None and tot_nhat == ham_h(bt.trang_thai_dau()))
    return KetQua(ten, lo_trinh_tot, mo_rong, sinh, dong_ho.ms(), toi_uu=toi_uu, het_gioi_han=het_gio,
                  chi_tiet={"heuristic": heuristic, "can_tren_ban_dau": can_tren_ban_dau,
                            "cat_theo_can": cat_theo_can, "cat_trung_lap": cat_trung_lap})


if __name__ == "__main__":
    from .cli import main
    main(["--thuat-toan", "nhanh_can"])
