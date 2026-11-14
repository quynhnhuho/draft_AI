"""Chức năng 1 – Ba tình huống bắt buộc + bảng so sánh thuật toán (báo cáo Phần 2 – mục 2.2).

    python -m src.planner.kich_ban            # in bảng + lộ trình
    python -m src.planner.kich_ban --md       # in bảng Markdown để dán vào báo cáo
"""
from __future__ import annotations

import argparse

from .dieu_phoi import lap_lo_trinh, so_sanh
from .model import BaiToan, doc_du_lieu, hk1_co_dinh, kiem_tra_lo_trinh, tao_bai_toan, ten_hoc_ky


def cac_kich_ban(ds=None) -> dict[str, dict]:
    ds = ds or doc_du_lieu()
    hk1 = set(hk1_co_dinh(ds))
    return {
        "1. Sinh viên năm nhất": dict(
            mo_ta="Bắt đầu HK1 (Trường xếp), tối đa 18 TC/kỳ.",
            da_dat=set(), rot=set(), hoc_ky=1, hoc_vuot=False),
        "2. Sinh viên rớt 2 học phần": dict(
            mo_ta="Đầu HK3. HK1 rớt COMP1010 (Lập trình cơ bản); HK2 rớt COMP1016 (Cấu trúc dữ liệu). "
                  "Học phần học trước chỉ cần đã học nên HK2 vẫn học được các môn cần COMP1010.",
            da_dat=(hk1 - {"COMP1010"}) | {"POLI2002", "POLI2003", "COMP1017", "COMP1013", "COMP1011"},
            rot={"COMP1010", "COMP1016"}, hoc_ky=3, hoc_vuot=False),
        "3a. Sinh viên muốn học vượt từ năm nhất": dict(
            mo_ta="Bắt đầu HK1 (Trường xếp), từ HK2 đăng ký tối đa 22 TC/kỳ. Học vượt KHÔNG rút ngắn được: "
                  "nút thắt là chuỗi học trước + kỳ mở (POLI2001→2003→2005→2004; COMP1701→COMP1314 "
                  "đều mở kỳ lẻ), không phải số tín chỉ.",
            da_dat=set(), rot=set(), hoc_ky=1, hoc_vuot=True, doi_chung=True),
        "3b. Sinh viên năm 2 rớt 1 học phần, học vượt để bù": dict(
            mo_ta="Đầu HK3, đã học HK1–HK2 theo lộ trình chuẩn nhưng rớt COMP1016 (Cấu trúc dữ liệu); "
                  "đăng ký tối đa 22 TC/kỳ. Ở đây nút thắt là số tín chỉ nên học vượt rút ngắn được 1 kỳ.",
            da_dat=hk1 | {"POLI2002", "POLI2003", "COMP1017", "COMP1013", "PSYC2804", "COMP1011"},
            rot={"COMP1016"}, hoc_ky=3, hoc_vuot=True, doi_chung=True),
    }


def tao(ten: str, ds=None) -> BaiToan:
    ds = ds or doc_du_lieu()
    kb = cac_kich_ban(ds)[ten]
    return tao_bai_toan(ds, kb["da_dat"], kb["rot"], kb["hoc_ky"], kb["hoc_vuot"])


def in_lo_trinh(bt: BaiToan, lo_trinh) -> None:
    for ky, tap in lo_trinh:
        tc = sum(bt.hoc_phan[m].tin_chi for m in tap)
        nhan = " [Trường xếp]" if ky in bt.truong_xep else ""
        mon = ", ".join(m + (" (học lại)" if m in bt.hoc_phan_rot else "")
                        for m in sorted(tap, key=lambda x: bt.hoc_phan[x].stt))
        print(f"    {ten_hoc_ky(ky)}{nhan} – {tc} TC: {mon or '(chờ kỳ sau)'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="in bảng Markdown")
    args = ap.parse_args(argv)
    ds = doc_du_lieu()
    for ten, kb in cac_kich_ban(ds).items():
        bt = tao(ten, ds)
        kqs = so_sanh(bt)
        print(f"\n### {ten}\n{kb['mo_ta']}\n")
        if not kqs:
            print("  (chưa có thuật toán nào trong src/planner/)")
            continue
        if kb.get("doi_chung"):  # so với cùng sinh viên nhưng không học vượt (18 TC)
            goc = lap_lo_trinh(tao_bai_toan(ds, kb["da_dat"], kb["rot"], kb["hoc_ky"], False)).ket_qua
            print(f"Đối chứng không học vượt (18 TC/kỳ): {goc.so_ky} kỳ\n")
        if args.md:
            print("| Thuật toán | Số kỳ | Nút mở rộng | Nút sinh | Thời gian (ms) | Ghi chú |")
            print("|---|---|---|---|---|---|")
        for kq in kqs:
            ct = kq.chi_tiet
            ghi_chu = (f"cận trên tham lam {ct['can_tren_ban_dau']}, cắt theo cận {ct['cat_theo_can']}, "
                       f"cắt trùng lặp {ct['cat_trung_lap']}") if "can_tren_ban_dau" in ct else ""
            hop_le = "hợp lệ" if kq.lo_trinh is not None and not kiem_tra_lo_trinh(bt, kq.lo_trinh) else "LỖI"
            if args.md:
                print(f"| {kq.thuat_toan} | {kq.so_ky} | {kq.so_nut_mo_rong} | {kq.so_nut_sinh} | "
                      f"{kq.thoi_gian_ms:.1f} | {ghi_chu} |")
            else:
                print(f"  {kq.thuat_toan:16} {kq.so_ky} kỳ | mở rộng {kq.so_nut_mo_rong:>6} | sinh {kq.so_nut_sinh:>7} | "
                      f"{kq.thoi_gian_ms:8.1f} ms | {hop_le} {ghi_chu}")
        if kqs and kqs[0].lo_trinh is not None:
            print(f"\n  Lộ trình ({kqs[0].thuat_toan}):")
            in_lo_trinh(bt, kqs[0].lo_trinh)


if __name__ == "__main__":
    main()
