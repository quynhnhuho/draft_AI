"""Chức năng 1 – MÔ HÌNH CHUNG của bài toán lập lộ trình học (Phần 2 – mục 2.1).

File này dùng chung cho cả 3 thuật toán (bfs.py, a_sao.py, nhanh_can.py), cho bộ điều phối
(dieu_phoi.py) và cho chatbot. Mỗi thuật toán chỉ cần import từ đây.

Mô hình hóa:
    Trạng thái  = (tập học phần ĐÃ ĐẠT, học kỳ hiện tại)
    Toán tử     = đăng ký một tập học phần trong học kỳ hiện tại:
                    - học phần mở trong kỳ (lẻ/chẵn), chưa đạt
                    - đã HỌC các học phần học trước (đạt hoặc rớt đều được)
                    - tổng tín chỉ <= MAX_TC
                    - học kỳ 1: chỉ được học đúng các học phần do Trường xếp
    Chi phí     = 1 cho mỗi học kỳ
    Đích        = đạt toàn bộ học phần mục tiêu
    Heuristic   = h(n)  = ceil(số tín chỉ còn phải học / MAX_TC)      (theo đề, chấp nhận được)
                  h2(n) = max(h theo kỳ lẻ/chẵn, chuỗi học trước dài nhất) (mở rộng, chấp nhận được)

Ghi chú hỗ trợ AI: mô hình được nhóm viết, Claude (Anthropic) hỗ trợ rà soát, sửa lỗi và tái cấu trúc;
nhóm đã kiểm tra lại toàn bộ.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

DUONG_DAN_DU_LIEU = Path(__file__).resolve().parents[2] / "data" / "hoc_phan.csv"

# =====================================================================================
# QUY ƯỚC VÀ GIẢ ĐỊNH NHÓM ĐÃ THỐNG NHẤT
# =====================================================================================
MAX_TC_BINH_THUONG = 18          # số tín chỉ tối đa mỗi kỳ
MAX_TC_HOC_VUOT = 22             # số tín chỉ tối đa mỗi kỳ khi học vượt
SO_HK_MOI_NAM = 2                # không có học kỳ hè
SO_HP_CO_DINH_HK1 = 7            # HK1 = 7 học phần đầu tiên trong hoc_phan.csv (18 TC)
HOC_KY_TOI_DA = 16

NGUON_CTDT = ("Danh mục học phần trích từ chương trình đào tạo ban hành kèm theo Quyết định số "
              "2619/QĐ-ĐHSP ngày 04/09/2024 của Hiệu trưởng Trường Đại học Sư phạm Thành phố Hồ Chí Minh")
NAM_AP_DUNG = 2024

GIA_DINH = [
    "Không có học kỳ hè: mỗi năm 2 học kỳ; học kỳ lẻ là HK1, học kỳ chẵn là HK2 của năm học.",
    f"Học kỳ 1 do Trường xếp cố định: {SO_HP_CO_DINH_HK1} học phần đầu tiên trong danh sách (18 TC); "
    "sinh viên không tự chọn thêm hoặc bớt.",
    "Điều kiện học trước: chỉ cần ĐÃ HỌC học phần trước (đạt hoặc rớt đều được); học phần rớt phải học lại.",
    f"Tối đa {MAX_TC_BINH_THUONG} TC/kỳ; {MAX_TC_HOC_VUOT} TC/kỳ khi học vượt.",
    "Mọi học phần đăng ký trong lộ trình được giả định là đạt.",
    "Học kỳ mở (Lẻ/Chẵn) của từng học phần lấy theo cột hoc_ky_mo trong data/hoc_phan.csv.",
]


# =====================================================================================
# 1. DỮ LIỆU HỌC PHẦN
# =====================================================================================
@dataclass(frozen=True)
class HocPhan:
    stt: int                        # thứ tự trong chương trình đào tạo (dùng để sắp xếp)
    ma: str
    ten: str
    tin_chi: int
    hoc_truoc: frozenset[str]
    ky_mo: str                      # "Lẻ" hoặc "Chẵn"

    def mo_trong_ky(self, hoc_ky: int) -> bool:
        return (hoc_ky % 2 == 1) == (self.ky_mo == "Lẻ")


def doc_du_lieu(duong_dan: Path = DUONG_DAN_DU_LIEU) -> dict[str, HocPhan]:
    """Đọc hoc_phan.csv (cột: ma, ten, tin_chi, hoc_truoc, hoc_ky_mo) và kiểm tra dữ liệu."""
    ds: dict[str, HocPhan] = {}
    with open(duong_dan, encoding="utf-8") as f:
        for stt, dong in enumerate(csv.DictReader(f), start=1):
            ky_mo = dong["hoc_ky_mo"].strip()
            if ky_mo not in ("Lẻ", "Chẵn"):
                raise ValueError(f"{dong['ma']}: hoc_ky_mo phải là 'Lẻ' hoặc 'Chẵn', nhận được {ky_mo!r}")
            hoc_truoc = frozenset(m.strip() for m in dong["hoc_truoc"].split(";") if m.strip())
            ds[dong["ma"].strip()] = HocPhan(stt, dong["ma"].strip(), dong["ten"].strip(),
                                             int(dong["tin_chi"]), hoc_truoc, ky_mo)
    thieu = [(hp.ma, p) for hp in ds.values() for p in hp.hoc_truoc if p not in ds]
    if thieu:
        raise ValueError(f"Học phần học trước không có trong danh sách: {thieu}")
    return ds


def hk1_co_dinh(ds: dict[str, HocPhan]) -> frozenset[str]:
    """Các học phần HK1 do Trường xếp = SO_HP_CO_DINH_HK1 học phần đầu tiên (theo stt)."""
    hk1 = [hp for hp in sorted(ds.values(), key=lambda h: h.stt)][:SO_HP_CO_DINH_HK1]
    loi = [hp.ma for hp in hk1 if hp.ky_mo != "Lẻ" or hp.hoc_truoc]
    if loi or sum(hp.tin_chi for hp in hk1) > MAX_TC_BINH_THUONG:
        raise ValueError(f"HK1 cố định không hợp lệ (phải mở kỳ lẻ, không có học trước, ≤ "
                         f"{MAX_TC_BINH_THUONG} TC): {loi or [hp.ma for hp in hk1]}")
    return frozenset(hp.ma for hp in hk1)


def ten_hoc_ky(ky: int) -> str:
    return f"HK{ky} (Năm {(ky - 1) // SO_HK_MOI_NAM + 1} – HK{1 if ky % 2 else 2})"


# =====================================================================================
# 2. BÀI TOÁN TÌM KIẾM
# =====================================================================================
TrangThai = tuple[frozenset[str], int]          # (đã đạt, học kỳ hiện tại)


@dataclass
class BaiToan:
    hoc_phan: dict[str, HocPhan]
    muc_tieu: frozenset[str]                     # nhóm học phần cần đạt
    da_hoan_thanh: frozenset[str]                # đã đạt trước khi lập lộ trình
    hoc_ky_bat_dau: int                          # học kỳ sắp đăng ký
    max_tc: int = MAX_TC_BINH_THUONG
    hoc_phan_rot: frozenset[str] = frozenset()   # đã học nhưng rớt -> vẫn tính "đã học"
    truong_xep: dict[int, frozenset[str]] = field(default_factory=dict)  # học kỳ -> HP do Trường xếp

    def __post_init__(self) -> None:
        # Học phần còn phải học = mục tiêu chưa đạt + các học phần học trước chưa từng học (bao đóng).
        da_hoc = self.da_hoan_thanh | self.hoc_phan_rot
        can, stack = set(), list(self.muc_tieu - self.da_hoan_thanh)
        while stack:
            m = stack.pop()
            if m not in can:
                can.add(m)
                stack.extend(p for p in self.hoc_phan[m].hoc_truoc if p not in da_hoc)
        self.can_hoc: frozenset[str] = frozenset(can)

    # ---- trạng thái -------------------------------------------------------------------
    def trang_thai_dau(self) -> TrangThai:
        return (self.da_hoan_thanh, self.hoc_ky_bat_dau)

    def la_dich(self, s: TrangThai) -> bool:
        return self.muc_tieu <= s[0]

    def da_hoc(self, da_dat: frozenset[str]) -> frozenset[str]:
        """'Đã học' = đã đạt hoặc đã rớt – dùng để xét điều kiện học trước."""
        return da_dat | self.hoc_phan_rot

    def tin_chi_con_lai(self, s: TrangThai) -> int:
        return sum(self.hoc_phan[m].tin_chi for m in self.can_hoc - s[0])

    def du_dieu_kien(self, s: TrangThai) -> list[str]:
        """Học phần còn phải học, mở trong kỳ, đã HỌC đủ học phần học trước.
        Sắp theo thứ tự trong CTĐT (stt) để lộ trình tự nhiên và lời giải tốt được sinh trước."""
        da_dat, ky = s
        da_hoc = self.da_hoc(da_dat)
        return sorted((m for m in self.can_hoc - da_dat
                       if self.hoc_phan[m].mo_trong_ky(ky) and self.hoc_phan[m].hoc_truoc <= da_hoc),
                      key=lambda m: self.hoc_phan[m].stt)

    # ---- toán tử ----------------------------------------------------------------------
    def hanh_dong(self, s: TrangThai) -> list[frozenset[str]]:
        """Các tập học phần có thể đăng ký trong kỳ.

        - Kỳ do Trường xếp: đúng 1 hành động là tập học phần đã xếp (trừ học phần đã đạt).
        - Kỳ khác: chỉ sinh tập "tối đại" (không thêm được học phần đủ điều kiện nào nữa mà vẫn
          <= MAX_TC). Học thêm môn không bao giờ làm lộ trình dài ra (trạng thái đạt nhiều hơn luôn
          trội) nên cắt này không làm mất lời giải tối ưu. Không có môn nào -> tập rỗng (chờ kỳ sau).
        """
        da_dat, ky = s
        if ky in self.truong_xep:
            da_hoc = self.da_hoc(da_dat)
            return [frozenset(m for m in self.truong_xep[ky] - da_dat
                              if self.hoc_phan[m].mo_trong_ky(ky) and self.hoc_phan[m].hoc_truoc <= da_hoc)]

        ung_vien = self.du_dieu_kien(s)
        tc = [self.hoc_phan[m].tin_chi for m in ung_vien]
        con_lai = [0] * (len(tc) + 1)                       # con_lai[i] = tổng TC từ i đến hết
        for i in range(len(tc) - 1, -1, -1):
            con_lai[i] = con_lai[i + 1] + tc[i]
        ket_qua: list[frozenset[str]] = []

        def quay_lui(i: int, chon: list[str], tong: int, bo_nho_nhat: float) -> None:
            # Cắt: kể cả lấy hết các môn còn lại, môn nhỏ nhất đã bỏ vẫn vừa -> không thể tối đại.
            if bo_nho_nhat <= self.max_tc - min(self.max_tc, tong + con_lai[i]):
                return
            if i == len(ung_vien):
                if bo_nho_nhat > self.max_tc - tong:        # tập tối đại
                    ket_qua.append(frozenset(chon))
                return
            if tong + tc[i] <= self.max_tc:                  # chọn môn i
                chon.append(ung_vien[i])
                quay_lui(i + 1, chon, tong + tc[i], bo_nho_nhat)
                chon.pop()
            quay_lui(i + 1, chon, tong, min(bo_nho_nhat, tc[i]))   # bỏ môn i

        quay_lui(0, [], 0, math.inf)
        return ket_qua

    @staticmethod
    def ke_tiep(s: TrangThai, a: frozenset[str]) -> TrangThai:
        return (s[0] | a, s[1] + 1)

    def khoa(self, s: TrangThai) -> tuple:
        """Cùng tập đã đạt + cùng tính chẵn/lẻ của học kỳ => tương lai giống nhau.
        Kỳ do Trường xếp có tương lai khác kỳ thường nên được tách khóa riêng."""
        return (s[0], s[1] % 2, s[1] if s[1] in self.truong_xep else 0)

    # ---- heuristic --------------------------------------------------------------------
    def h(self, s: TrangThai) -> int:
        """h(n) = ceil(tín chỉ còn phải học / MAX_TC) – heuristic theo đề.
        Chấp nhận được: mỗi học kỳ học tối đa MAX_TC tín chỉ."""
        return math.ceil(self.tin_chi_con_lai(s) / self.max_tc)

    def h2(self, s: TrangThai) -> int:
        """h2(n) = max(h_le_chan, h_chuoi) – heuristic mở rộng, vẫn chấp nhận được.
        - h_le_chan: TC môn kỳ lẻ cần ít nhất a kỳ lẻ, TC môn kỳ chẵn cần ít nhất b kỳ chẵn.
        - h_chuoi : chuỗi học trước dài nhất còn lại, có tính chờ tới kỳ mở môn.
        """
        da_dat, ky = s
        con = self.can_hoc - da_dat
        if not con:
            return 0
        le = sum(self.hoc_phan[m].tin_chi for m in con if self.hoc_phan[m].ky_mo == "Lẻ")
        chan = sum(self.hoc_phan[m].tin_chi for m in con if self.hoc_phan[m].ky_mo == "Chẵn")
        a, b = math.ceil(le / self.max_tc), math.ceil(chan / self.max_tc)
        h_le_chan = max(2 * a - 1, 2 * b, 0) if ky % 2 == 1 else max(2 * b - 1, 2 * a, 0)

        som_nhat: dict[str, int] = {}               # độ lệch học kỳ sớm nhất có thể học môn m

        def tinh(m: str) -> int:
            if m not in som_nhat:
                t = max((tinh(p) + 1 for p in self.hoc_phan[m].hoc_truoc
                         if p in con and p not in self.hoc_phan_rot), default=0)
                if not self.hoc_phan[m].mo_trong_ky(ky + t):
                    t += 1
                som_nhat[m] = t
            return som_nhat[m]

        h_chuoi = max(tinh(m) + 1 for m in con)
        return max(h_le_chan, h_chuoi)

    def ham_heuristic(self, ten: str):
        return {"h": self.h, "h2": self.h2}[ten]


# =====================================================================================
# 3. KẾT QUẢ VÀ TIỆN ÍCH DÙNG CHUNG CHO CÁC THUẬT TOÁN
# =====================================================================================
LoTrinh = list[tuple[int, frozenset[str]]]       # [(học kỳ, tập học phần), ...]


@dataclass
class KetQua:
    thuat_toan: str
    lo_trinh: LoTrinh | None                     # None = không tìm được
    so_nut_mo_rong: int = 0
    so_nut_sinh: int = 0
    thoi_gian_ms: float = 0.0
    toi_uu: bool = True                          # False: dừng vì hết giới hạn, chưa chắc tối ưu
    het_gioi_han: bool = False
    chi_tiet: dict = field(default_factory=dict)  # số liệu riêng của từng thuật toán

    @property
    def so_ky(self) -> int | None:
        return None if self.lo_trinh is None else len(self.lo_trinh)


class DongHo:
    """Đo thời gian và kiểm tra giới hạn (None = không giới hạn)."""

    def __init__(self, gioi_han_ms: float | None = None):
        self.t0 = time.perf_counter()
        self.gioi_han_ms = gioi_han_ms

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def het(self) -> bool:
        return self.gioi_han_ms is not None and self.ms() > self.gioi_han_ms


def dung_lai_lo_trinh(cha: dict, s: TrangThai) -> LoTrinh:
    lo_trinh = []
    while cha[s] is not None:
        s_truoc, a = cha[s]
        lo_trinh.append((s_truoc[1], a))
        s = s_truoc
    return lo_trinh[::-1]


# =====================================================================================
# 4. TẠO BÀI TOÁN TỪ THÔNG TIN SINH VIÊN + KIỂM TRA
# =====================================================================================
def kiem_tra_dau_vao(ds: dict[str, HocPhan], da_dat: set[str], rot: set[str], hoc_ky: int) -> list[str]:
    """Trả về danh sách lỗi (rỗng = hợp lệ)."""
    loi = []
    sai = sorted((set(da_dat) | set(rot)) - ds.keys())
    if sai:
        loi.append(f"Không có học phần: {', '.join(sai)}")
    if not 1 <= hoc_ky <= HOC_KY_TOI_DA:
        loi.append(f"Học kỳ phải từ 1 đến {HOC_KY_TOI_DA}")
    trung = sorted(set(da_dat) & set(rot))
    if trung:
        loi.append(f"{', '.join(trung)} không thể vừa đạt vừa rớt")
    if hoc_ky == 1 and (da_dat or rot):
        loi.append("Sinh viên bắt đầu từ HK1 chưa học học phần nào")
    if not loi:
        da_hoc = set(da_dat) | set(rot)
        for m in sorted(da_hoc, key=lambda x: ds[x].stt):
            thieu = ds[m].hoc_truoc - da_hoc
            if thieu:
                loi.append(f"{m} cần học trước {', '.join(sorted(thieu))}")
    return loi


def tao_bai_toan(ds: dict[str, HocPhan], da_dat: set[str], rot: set[str], hoc_ky: int,
                 hoc_vuot: bool = False, muc_tieu: set[str] | None = None) -> BaiToan:
    loi = kiem_tra_dau_vao(ds, da_dat, rot, hoc_ky)
    if loi:
        raise ValueError("; ".join(loi))
    return BaiToan(ds, frozenset(muc_tieu or ds), frozenset(da_dat), hoc_ky,
                   MAX_TC_HOC_VUOT if hoc_vuot else MAX_TC_BINH_THUONG, frozenset(rot),
                   truong_xep={1: hk1_co_dinh(ds)})


def kiem_tra_lo_trinh(bt: BaiToan, lo_trinh: LoTrinh) -> list[str]:
    """Kiểm tra độc lập một lộ trình có hợp lệ không (dùng trong test và để đối chiếu)."""
    loi, da_dat, ky_mong_doi = [], set(bt.da_hoan_thanh), bt.hoc_ky_bat_dau
    for ky, tap in lo_trinh:
        if ky != ky_mong_doi:
            loi.append(f"Học kỳ không liên tiếp: {ky}")
        ky_mong_doi = ky + 1
        tong = sum(bt.hoc_phan[m].tin_chi for m in tap)
        if tong > bt.max_tc:
            loi.append(f"HK{ky}: {tong} TC > {bt.max_tc}")
        if ky in bt.truong_xep and set(tap) != set(bt.truong_xep[ky]) - set(bt.da_hoan_thanh):
            loi.append(f"HK{ky}: không đúng các học phần Trường xếp")
        da_hoc = da_dat | set(bt.hoc_phan_rot)
        for m in tap:
            hp = bt.hoc_phan[m]
            if not hp.mo_trong_ky(ky):
                loi.append(f"HK{ky}: {m} không mở")
            if not hp.hoc_truoc <= da_hoc:
                loi.append(f"HK{ky}: {m} chưa học {', '.join(sorted(hp.hoc_truoc - da_hoc))}")
            if m in da_dat:
                loi.append(f"HK{ky}: {m} đã đạt rồi")
        da_dat |= tap
    if not bt.muc_tieu <= da_dat:
        loi.append(f"Chưa đạt: {', '.join(sorted(bt.muc_tieu - da_dat))}")
    return loi
