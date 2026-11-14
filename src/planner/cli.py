"""Chức năng 1 – Chạy lập lộ trình ở dòng lệnh (nhập thông tin sinh viên từ bàn phím).

    python -m src.planner.cli                       # tự chọn thuật toán như chatbot
    python -m src.planner.cli --thuat-toan a_sao    # chạy riêng 1 thuật toán (bfs | a_sao | nhanh_can)
    python -m src.planner.cli --thuat-toan tat_ca   # chạy cả 3 để so sánh
"""
from __future__ import annotations

import argparse

from .dieu_phoi import DANH_SACH_THUAT_TOAN, lap_lo_trinh, nap_thuat_toan, so_sanh
from .model import (GIA_DINH, MAX_TC_BINH_THUONG, MAX_TC_HOC_VUOT, NGUON_CTDT, BaiToan, HocPhan, KetQua,
                    doc_du_lieu, hk1_co_dinh, kiem_tra_dau_vao, tao_bai_toan, ten_hoc_ky)


# ----------------------------------------------------------------------------- nhập liệu
def in_danh_sach(ds: dict[str, HocPhan]) -> None:
    hk1 = hk1_co_dinh(ds)
    print(f"\n{'Mã HP':<10}{'Tên học phần':<42}{'TC':>3}  {'Học trước':<10}{'Kỳ mở':>6}")
    print("-" * 75)
    for hp in ds.values():
        ht = ", ".join(sorted(hp.hoc_truoc)) or "-"
        print(f"{hp.ma:<10}{hp.ten:<42}{hp.tin_chi:>3}  {ht:<10}{hp.ky_mo:>6}{'  [HK1]' if hp.ma in hk1 else ''}")
    print()


def hoi_co_khong(thong_bao: str) -> bool:
    while True:
        chuoi = input(f"{thong_bao} (c/k): ").strip().lower()
        if chuoi in ("c", "có", "co"):
            return True
        if chuoi in ("k", "không", "khong"):
            return False
        print("  ! Vui lòng nhập c (có) hoặc k (không).")


def nhap_hoc_ky() -> int:
    while True:
        chuoi = input("Học kỳ sắp đăng ký (1 = sinh viên năm nhất): ").strip()
        if chuoi.isdigit() and 1 <= int(chuoi) <= 16:
            return int(chuoi)
        print("  ! Học kỳ phải là số nguyên từ 1 đến 16.")


def nhap_ma(thong_bao: str, ds: dict[str, HocPhan]) -> set[str]:
    while True:
        ma_nhap = {m.upper() for m in input(f"{thong_bao} (cách nhau bằng dấu phẩy, Enter nếu không có): ")
                   .replace(",", " ").split()}
        sai = sorted(ma_nhap - ds.keys())
        if not sai:
            return ma_nhap
        print(f"  ! Không có học phần: {', '.join(sai)}. Vui lòng nhập lại.")


def nhap_thong_tin(ds: dict[str, HocPhan]) -> BaiToan:
    print("\n" + "-" * 75 + "\nNHẬP THÔNG TIN SINH VIÊN")
    while True:
        hk = nhap_hoc_ky()
        da_dat, rot = set(), set()
        if hk > 1:
            da_dat = nhap_ma("Mã các học phần ĐÃ ĐẠT", ds)
            rot = nhap_ma("Mã các học phần BỊ RỚT (chưa học lại đạt)", ds)
        hoc_vuot = hoi_co_khong("Sinh viên có muốn học vượt không?")
        loi = kiem_tra_dau_vao(ds, da_dat, rot, hk)
        if not loi:
            break
        for dong in loi:
            print(f"  ! {dong}")
        print("  Vui lòng nhập lại.")
    print(f"  Số tín chỉ tối đa mỗi kỳ: {MAX_TC_HOC_VUOT if hoc_vuot else MAX_TC_BINH_THUONG} TC")
    return tao_bai_toan(ds, da_dat, rot, hk, hoc_vuot)


# ----------------------------------------------------------------------------- in kết quả
def in_ket_qua(bt: BaiToan, kq: KetQua) -> None:
    s0 = bt.trang_thai_dau()
    print("\n" + "=" * 75 + f"\nKẾT QUẢ {kq.thuat_toan.upper()}")
    print(f"  Đã đạt: {len(bt.da_hoan_thanh)}/{len(bt.muc_tieu)} HP | Tín chỉ còn phải học: "
          f"{bt.tin_chi_con_lai(s0)} | Bắt đầu HK{bt.hoc_ky_bat_dau} | MAX_TC = {bt.max_tc}")
    print("-" * 75)
    if kq.lo_trinh is None:
        print("  Không tìm được lộ trình" + (" (hết giới hạn thời gian)." if kq.het_gioi_han else "."))
    elif not kq.lo_trinh:
        print("  Sinh viên đã hoàn thành toàn bộ học phần mục tiêu.")
    else:
        for ky, tap in kq.lo_trinh:
            tc = sum(bt.hoc_phan[m].tin_chi for m in tap)
            print(f"  {ten_hoc_ky(ky)} – {tc} TC{'  [Trường xếp]' if ky in bt.truong_xep else ''}")
            if not tap:
                print("      (không có học phần nào đủ điều kiện, chờ sang kỳ sau)")
            for m in sorted(tap, key=lambda x: bt.hoc_phan[x].stt):
                hp = bt.hoc_phan[m]
                print(f"      {hp.ma:<9}{hp.ten} ({hp.tin_chi} TC){'  (học lại)' if m in bt.hoc_phan_rot else ''}")
        print("-" * 75)
        print(f"  Số học kỳ cần học thêm : {kq.so_ky}{'' if kq.toi_uu else '  (có thể chưa tối ưu)'}")
    print(f"  Số nút mở rộng         : {kq.so_nut_mo_rong}")
    print(f"  Số nút sinh ra         : {kq.so_nut_sinh}")
    for k, v in kq.chi_tiet.items():
        print(f"  {k:<23}: {v}")
    print(f"  Thời gian chạy         : {kq.thoi_gian_ms:.2f} ms")
    print("=" * 75)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Lập lộ trình học tối ưu")
    ap.add_argument("--thuat-toan", choices=["tu_dong", "tat_ca", *DANH_SACH_THUAT_TOAN], default="tu_dong")
    args = ap.parse_args(argv)

    ds = doc_du_lieu()
    print("LẬP LỘ TRÌNH HỌC TỐI ƯU")
    print(f"Nguồn: {NGUON_CTDT}\nDữ liệu: {len(ds)} học phần, {sum(h.tin_chi for h in ds.values())} tín chỉ")
    print("Giả định:\n" + "\n".join(f"  - {g}" for g in GIA_DINH))
    in_danh_sach(ds)

    try:
        while True:
            bt = nhap_thong_tin(ds)
            if args.thuat_toan == "tu_dong":
                kq_dp = lap_lo_trinh(bt, doi_chieu=True)
                for dong in kq_dp.ghi_chu:
                    print(f"  * {dong}")
                in_ket_qua(bt, kq_dp.ket_qua)
                if kq_dp.doi_chieu:
                    print(f"  Đối chiếu BFS: {kq_dp.doi_chieu.so_ky} kỳ – "
                          f"{'KHỚP' if kq_dp.khop_doi_chieu else 'KHÔNG KHỚP'}")
            elif args.thuat_toan == "tat_ca":
                for kq in so_sanh(bt, them_h2=False):
                    in_ket_qua(bt, kq)
            else:
                ham = nap_thuat_toan(args.thuat_toan)
                if ham is None:
                    print(f"Chưa có file src/planner/{args.thuat_toan}.py")
                    return
                in_ket_qua(bt, ham(bt))
            if input("\nNhập sinh viên khác? (c/k): ").strip().lower() not in ("c", "co", "có", "y"):
                break
    except (KeyboardInterrupt, EOFError):
        pass
    print("Kết thúc chương trình.")


if __name__ == "__main__":
    main()
