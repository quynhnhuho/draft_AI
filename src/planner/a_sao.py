"""Chức năng 1 – Thuật toán A*.

Thành viên phụ trách : <Họ tên – MSSV>
Vai trò trong chatbot: BỘ GIẢI CHÍNH. Chatbot luôn chạy A* trước (có giới hạn thời gian); lời giải
                       A* là lộ trình ít học kỳ nhất vì h chấp nhận được.

Ý tưởng: hàng đợi ưu tiên theo f = g + h, g = số học kỳ đã dùng.
    h(n)  = ceil(tín chỉ còn phải học / MAX_TC)  – heuristic theo đề (mặc định)
    h2(n) = max(h theo kỳ lẻ/chẵn, chuỗi học trước) – heuristic mở rộng để so sánh
Cả hai đều chấp nhận được và nhất quán (mỗi bước giảm tối đa 1) => kiểm tra đích khi lấy nút ra
khỏi hàng đợi cho lời giải tối ưu. Hòa f thì ưu tiên g lớn (gần đích hơn).

Chạy riêng:  python -m src.planner.a_sao
Ghi chú hỗ trợ AI: Claude (Anthropic) hỗ trợ rà soát và tái cấu trúc; nhóm đã kiểm tra lại.
"""
from __future__ import annotations

import heapq
import math
from itertools import count

from .model import BaiToan, DongHo, KetQua, TrangThai, dung_lai_lo_trinh


def a_sao(bt: BaiToan, heuristic: str = "h", gioi_han_ms: float | None = None) -> KetQua:
    ten = "A*" if heuristic == "h" else f"A* ({heuristic})"
    ham_h = bt.ham_heuristic(heuristic)
    dong_ho = DongHo(gioi_han_ms)
    s0 = bt.trang_thai_dau()
    thu_tu = count()                                       # phá hòa ổn định
    hang_doi_uu_tien = [(ham_h(s0), 0, next(thu_tu), 0, s0)]  # (f, -g, thứ tự, g, trạng thái)
    g_tot = {bt.khoa(s0): 0}
    cha: dict[TrangThai, tuple[TrangThai, frozenset[str]] | None] = {s0: None}
    mo_rong = sinh = 0

    while hang_doi_uu_tien:
        if dong_ho.het():
            return KetQua(ten, None, mo_rong, sinh, dong_ho.ms(), toi_uu=False, het_gioi_han=True)
        f, _, _, g, s = heapq.heappop(hang_doi_uu_tien)
        if g > g_tot[bt.khoa(s)]:                          # bản cũ trong heap, bỏ qua
            continue
        if bt.la_dich(s):                                  # kiểm tra đích khi lấy ra
            return KetQua(ten, dung_lai_lo_trinh(cha, s), mo_rong, sinh, dong_ho.ms(),
                          chi_tiet={"heuristic": heuristic, "h_ban_dau": ham_h(s0)})
        mo_rong += 1
        for a in bt.hanh_dong(s):
            s_moi = bt.ke_tiep(s, a)
            sinh += 1
            g_moi = g + 1
            if g_moi < g_tot.get(bt.khoa(s_moi), math.inf):
                g_tot[bt.khoa(s_moi)] = g_moi
                cha[s_moi] = (s, a)
                heapq.heappush(hang_doi_uu_tien,
                               (g_moi + ham_h(s_moi), -g_moi, next(thu_tu), g_moi, s_moi))

    return KetQua(ten, None, mo_rong, sinh, dong_ho.ms())


if __name__ == "__main__":
    from .cli import main
    main(["--thuat-toan", "a_sao"])
