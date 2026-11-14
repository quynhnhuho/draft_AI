"""Chức năng 1 – Thuật toán TÌM KIẾM THEO CHIỀU RỘNG (BFS).

Thành viên phụ trách : <Họ tên – MSSV>
Vai trò trong chatbot: CHUẨN ĐỐI CHIẾU. BFS không dùng để trả lời người dùng (tốn bộ nhớ khi bài
                       toán lớn) mà dùng để kiểm chứng A*/nhánh cận cho đúng số học kỳ tối ưu
                       (chế độ debug, tab "So sánh thuật toán", test tự động).

Ý tưởng: hàng đợi FIFO, mỗi tầng của cây tìm kiếm = 1 học kỳ, chi phí mỗi bước bằng nhau
=> lời giải đầu tiên tìm thấy là lộ trình ít học kỳ nhất. Kiểm tra đích ngay khi sinh nút.

Chạy riêng:  python -m src.planner.bfs
Ghi chú hỗ trợ AI: Claude (Anthropic) hỗ trợ rà soát và tái cấu trúc; nhóm đã kiểm tra lại.
"""
from __future__ import annotations

from collections import deque

from .model import BaiToan, DongHo, KetQua, TrangThai, dung_lai_lo_trinh

TEN = "BFS"


def bfs(bt: BaiToan, gioi_han_ms: float | None = None) -> KetQua:
    dong_ho = DongHo(gioi_han_ms)
    s0 = bt.trang_thai_dau()
    if bt.la_dich(s0):
        return KetQua(TEN, [], 0, 0, dong_ho.ms())

    hang_doi: deque[TrangThai] = deque([s0])        # FIFO
    da_duyet = {bt.khoa(s0)}                        # tập trạng thái đã gặp (theo khóa)
    cha: dict[TrangThai, tuple[TrangThai, frozenset[str]] | None] = {s0: None}
    mo_rong = sinh = 0

    while hang_doi:
        if dong_ho.het():                           # hết giới hạn thời gian
            return KetQua(TEN, None, mo_rong, sinh, dong_ho.ms(), toi_uu=False, het_gioi_han=True)
        s = hang_doi.popleft()
        mo_rong += 1
        for a in bt.hanh_dong(s):
            s_moi = bt.ke_tiep(s, a)
            sinh += 1
            if bt.khoa(s_moi) in da_duyet:
                continue
            da_duyet.add(bt.khoa(s_moi))
            cha[s_moi] = (s, a)
            if bt.la_dich(s_moi):                   # kiểm tra đích khi sinh nút (đúng với BFS)
                return KetQua(TEN, dung_lai_lo_trinh(cha, s_moi), mo_rong, sinh, dong_ho.ms())
            hang_doi.append(s_moi)

    return KetQua(TEN, None, mo_rong, sinh, dong_ho.ms())


if __name__ == "__main__":
    from .cli import main
    main(["--thuat-toan", "bfs"])
