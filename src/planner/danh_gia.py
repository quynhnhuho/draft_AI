"""Chức năng 1 – ĐÁNH GIÁ & MINH CHỨNG lựa chọn thuật toán (báo cáo Phần 2 – mục 2.2).

Chứng minh bằng số liệu vì sao bộ điều phối dùng:
    A*        làm bộ giải CHÍNH      (đúng như BFS, mở ít nút hơn, nhanh hơn, ít bộ nhớ hơn)
    Nhánh cận làm bộ giải DỰ PHÒNG   (khi bị giới hạn thời gian, chỉ nhánh cận vẫn có lời giải)
    BFS       làm CHUẨN ĐỐI CHIẾU    (chắc chắn tối ưu nhưng tốn nút/bộ nhớ nhất)

Với mỗi tình huống in 3 bảng, sau đó 1 bảng tổng hợp trên N sinh viên ngẫu nhiên:
    Bảng 1 – Kết quả và hiệu quả     : số kỳ, khớp BFS, hợp lệ, nút mở rộng/sinh, thời gian TB, bộ nhớ đỉnh
    Bảng 2 – Khi giới hạn thời gian   : thuật toán nào còn trả lời được
    Bảng 3 – Bộ điều phối chọn gì     : bình thường / A* chậm (hết giờ) / chế độ debug
    Bảng 4 – Ngẫu nhiên N sinh viên   : tỉ lệ khớp, hợp lệ, A* ít nút hơn BFS, A* nhanh hơn nhánh cận

    python -m src.planner.danh_gia                         # in ra màn hình (Markdown)
    python -m src.planner.danh_gia --out bao_cao_danh_gia.md
    python -m src.planner.danh_gia --so-sv 500 --lap 30     # tăng số mẫu

Thời gian phụ thuộc máy chạy; số kỳ, số nút, khớp/hợp lệ thì không đổi.
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import random
import statistics
import tracemalloc

from . import dieu_phoi, kich_ban
from .model import (MAX_TC_BINH_THUONG, BaiToan, HocPhan, doc_du_lieu, hk1_co_dinh, kiem_tra_lo_trinh,
                    tao_bai_toan)

TEN_HIEN_THI = {"bfs": "BFS", "a_sao": "A*", "nhanh_can": "Nhánh cận"}
GIOI_HAN_THU_MS = (0, 0.5, 1, 2, 5, 20)


# ----------------------------------------------------------------------------- tiện ích
def cac_thuat_toan() -> dict:
    """{khóa: hàm} các thuật toán đã có trong repo, theo thứ tự BFS, A*, nhánh cận."""
    out = {}
    for k in ("bfs", "a_sao", "nhanh_can"):
        ham = dieu_phoi.nap_thuat_toan(k)
        if ham is not None:
            out[k] = ham
    return out


def bo_nho_dinh_kb(ham, bt: BaiToan) -> float:
    tracemalloc.start()
    try:
        ham(bt)
        return tracemalloc.get_traced_memory()[1] / 1024
    finally:
        tracemalloc.stop()


def sinh_vien_ngau_nhien(ds: dict[str, HocPhan], rng: random.Random) -> tuple[set, set, int, bool]:
    """Mô phỏng một sinh viên hợp lệ: học từ HK1 tới trước học kỳ hk, mỗi môn rớt với xác suất 15%.
    Trả về (đã đạt, rớt, học kỳ sắp đăng ký, có học vượt)."""
    hk = rng.randint(1, 6)
    dat, rot = set(), set()
    for ky in range(1, hk):
        if ky == 1:
            hoc = set(hk1_co_dinh(ds))
        else:
            ung_vien = [m for m, h in ds.items()
                        if m not in dat and h.mo_trong_ky(ky) and h.hoc_truoc <= dat | rot]
            rng.shuffle(ung_vien)
            hoc, tc = set(), 0
            for m in ung_vien:
                if tc + ds[m].tin_chi <= MAX_TC_BINH_THUONG and rng.random() < 0.8:
                    hoc.add(m)
                    tc += ds[m].tin_chi
        for m in hoc:
            if rng.random() < 0.15:
                rot.add(m)
            else:
                dat.add(m)
                rot.discard(m)
    return dat, rot, hk, rng.random() < 0.4


def _o(kq) -> str:
    """Ô của bảng 2: '—' = không có lời giải; '6 ⚠' = có nhưng chưa chắc tối ưu."""
    if kq.lo_trinh is None:
        return "— (chưa có lời giải)"
    return f"{kq.so_ky}" if kq.toi_uu else f"{kq.so_ky} ⚠ chưa chắc tối ưu"


# ----------------------------------------------------------------------------- các bảng
def bang_hieu_qua(bt: BaiToan, algs: dict, lap: int) -> list[str]:
    kqs = {k: f(bt) for k, f in algs.items()}
    chuan = kqs["bfs"].so_ky if "bfs" in kqs else None
    dong = ["| Thuật toán | Số kỳ | Khớp BFS | Hợp lệ | Nút mở rộng | Nút sinh | Thời gian TB (ms) | Bộ nhớ đỉnh (KB) |",
            "|---|---|---|---|---|---|---|---|"]
    for k, f in algs.items():
        kq = kqs[k]
        tg = statistics.mean(f(bt).thoi_gian_ms for _ in range(lap))
        khop = "–" if chuan is None else ("✔" if kq.so_ky == chuan else "✘")
        hop_le = "✔" if kq.lo_trinh is not None and not kiem_tra_lo_trinh(bt, kq.lo_trinh) else "✘"
        dong.append(f"| {TEN_HIEN_THI[k]} | {kq.so_ky} | {khop} | {hop_le} | {kq.so_nut_mo_rong} | "
                    f"{kq.so_nut_sinh} | {tg:.2f} | {bo_nho_dinh_kb(f, bt):.0f} |")
    return dong


def bang_gioi_han(bt: BaiToan, algs: dict) -> list[str]:
    dong = ["| Giới hạn thời gian | " + " | ".join(TEN_HIEN_THI[k] for k in algs) + " |",
            "|---|" + "---|" * len(algs)]
    for lim in GIOI_HAN_THU_MS:
        dong.append(f"| {lim} ms | " + " | ".join(_o(f(bt, gioi_han_ms=lim)) for f in algs.values()) + " |")
    dong.append("| không giới hạn | " + " | ".join(_o(f(bt)) for f in algs.values()) + " |")
    return dong


def bang_dieu_phoi(bt: BaiToan) -> list[str]:
    dong = ["| Điều kiện | Thuật toán trả lời | Số kỳ | Ghi chú |", "|---|---|---|---|"]
    try:
        kq = dieu_phoi.lap_lo_trinh(bt)
        dong.append(f"| Bình thường | {kq.ket_qua.thuat_toan} | {kq.ket_qua.so_ky} | "
                    f"{kq.ket_qua.so_nut_mo_rong} nút, {kq.ket_qua.thoi_gian_ms:.1f} ms |")
        cu = dieu_phoi.TI_LE_CHO_A_SAO
        dieu_phoi.TI_LE_CHO_A_SAO = 0.0               # mô phỏng A* chậm: hết giờ ngay
        try:
            kq = dieu_phoi.lap_lo_trinh(bt)
        finally:
            dieu_phoi.TI_LE_CHO_A_SAO = cu
        dong.append(f"| A* chậm (hết giới hạn) | {kq.ket_qua.thuat_toan} | {kq.ket_qua.so_ky} | "
                    f"{'tối ưu' if kq.ket_qua.toi_uu else 'chưa chắc tối ưu'}; {' '.join(kq.ghi_chu)} |")
        kq = dieu_phoi.lap_lo_trinh(bt, doi_chieu=True)
        if kq.doi_chieu is not None:
            dong.append(f"| Chế độ debug | {kq.ket_qua.thuat_toan} + đối chiếu BFS | {kq.ket_qua.so_ky} | "
                        f"BFS {kq.doi_chieu.so_ky} kỳ – {'✔ khớp' if kq.khop_doi_chieu else '✘ KHÔNG khớp'} |")
    except NotImplementedError as e:
        dong.append(f"| – | {e} | – | – |")
    return dong


def bang_ngau_nhien(ds, algs: dict, so_sv: int, seed: int) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    tg = {k: [] for k in algs}
    nut = {k: [] for k in algs}
    dem = {"khop": 0, "hop_le": 0, "a_it_nut": 0, "a_nhanh_nc": 0}
    for _ in range(so_sv):
        dat, rot, hk, vuot = sinh_vien_ngau_nhien(ds, rng)
        bt = tao_bai_toan(ds, dat, rot, hk, vuot)
        kqs = {k: f(bt) for k, f in algs.items()}
        dem["khop"] += len({kq.so_ky for kq in kqs.values()}) == 1
        dem["hop_le"] += all(kq.lo_trinh is not None and not kiem_tra_lo_trinh(bt, kq.lo_trinh)
                             for kq in kqs.values())
        if "a_sao" in kqs and "bfs" in kqs:
            dem["a_it_nut"] += kqs["a_sao"].so_nut_mo_rong <= kqs["bfs"].so_nut_mo_rong
        if "a_sao" in kqs and "nhanh_can" in kqs:
            dem["a_nhanh_nc"] += kqs["a_sao"].thoi_gian_ms <= kqs["nhanh_can"].thoi_gian_ms
        for k, kq in kqs.items():
            tg[k].append(kq.thoi_gian_ms)
            nut[k].append(kq.so_nut_mo_rong)

    pct = lambda x: f"{x}/{so_sv} ({100 * x / so_sv:.0f}%)"
    dong = [f"- Cả {len(algs)} thuật toán cho **cùng số học kỳ**: {pct(dem['khop'])}",
            f"- Mọi lộ trình **hợp lệ** (kiểm tra độc lập): {pct(dem['hop_le'])}"]
    if "a_sao" in algs and "bfs" in algs:
        dong.append(f"- A* mở **ít nút hơn hoặc bằng** BFS: {pct(dem['a_it_nut'])}")
    if "a_sao" in algs and "nhanh_can" in algs:
        dong.append(f"- A* **nhanh hơn hoặc bằng** nhánh cận: {pct(dem['a_nhanh_nc'])}")
    dong += ["", "| Thuật toán | Thời gian TB (ms) | Trung vị | P95 | Lớn nhất | Nút mở rộng TB | Nút lớn nhất |",
             "|---|---|---|---|---|---|---|"]
    for k in algs:
        t = sorted(tg[k])
        dong.append(f"| {TEN_HIEN_THI[k]} | {statistics.mean(t):.2f} | {statistics.median(t):.2f} | "
                    f"{t[int(0.95 * (len(t) - 1))]:.2f} | {t[-1]:.2f} | {statistics.mean(nut[k]):.1f} | {max(nut[k])} |")
    return dong, dem


# ----------------------------------------------------------------------------- chương trình
def bao_cao(so_sv: int = 300, lap: int = 20, seed: int = 42) -> str:
    ds = doc_du_lieu()
    algs = cac_thuat_toan()
    out = ["# Đánh giá thuật toán lập lộ trình (Chức năng 1)", "",
           f"Thời điểm chạy: {dt.datetime.now():%d/%m/%Y %H:%M} · Python {platform.python_version()} · "
           f"{platform.system()} {platform.machine()} · thời gian trung bình qua {lap} lần chạy", ""]
    if not algs:
        return "\n".join(out + ["Chưa có thuật toán nào trong `src/planner/` (bfs.py, a_sao.py, nhanh_can.py)."])
    for ten, kb in kich_ban.cac_kich_ban(ds).items():
        bt = tao_bai_toan(ds, kb["da_dat"], kb["rot"], kb["hoc_ky"], kb["hoc_vuot"])
        out += [f"## {ten}", "", kb["mo_ta"], "",
                "**Bảng 1 – Kết quả và hiệu quả**", "", *bang_hieu_qua(bt, algs, lap), "",
                "**Bảng 2 – Khi bị giới hạn thời gian**", "", *bang_gioi_han(bt, algs), "",
                "**Bảng 3 – Bộ điều phối của chatbot chọn thuật toán nào**", "", *bang_dieu_phoi(bt), ""]
    dong, _ = bang_ngau_nhien(ds, algs, so_sv, seed)
    out += [f"## Tổng hợp trên {so_sv} sinh viên ngẫu nhiên (seed = {seed})", "", *dong, ""]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Đánh giá và minh chứng lựa chọn thuật toán C1")
    ap.add_argument("--so-sv", type=int, default=300, help="số sinh viên ngẫu nhiên")
    ap.add_argument("--lap", type=int, default=20, help="số lần chạy để lấy thời gian trung bình")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="ghi ra file Markdown")
    args = ap.parse_args(argv)
    noi_dung = bao_cao(args.so_sv, args.lap, args.seed)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(noi_dung)
        print(f"Đã ghi {args.out}")
    else:
        print(noi_dung)


if __name__ == "__main__":
    main()
