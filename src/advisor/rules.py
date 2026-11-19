"""C2 – Cơ sở luật tư vấn chọn ngành (Phần 3 – 3.3). 33 luật theo "Tập luật C2 (v2)", năm áp dụng 2026.

Biểu diễn
---------
- Sự kiện (atom) là tuple:  ("du_dk", "CNTT"), ("tohop_hople", "A01"), ("thich_lap_trinh",).
  Viết trong luật bằng chuỗi "du_dk(CNTT)"; tham số bắt đầu bằng "?" là BIẾN: "tohop_xet(?T)".
- Điều kiện của luật là một trong 3 loại:
    * atom dương        "tohop_hople(?T)"      – khớp với sự kiện (gán giá trị cho biến)
    * Not(atom)         ¬diem_liet(?T)         – phủ định theo kiểu "không chứng minh được"
                                                  (biến chưa gán trong Not nghĩa là ¬∃, dùng cho R32, R33)
    * Test(...)         DXT(?T) ≥ ngưỡng       – điều kiện số, gọi hàm tính ở diem.py (procedural attachment)
- Luật: IF tất cả điều kiện THEN một atom kết luận.

Chuyển từ tập luật trên giấy sang code – 2 chỗ viết lại tương đương (ghi trong tài liệu):
  1. R11–R16 "∃T: tohop_xet(T) ∧ DXT_best ≥ ngưỡng"  ⇔  "tohop_xet(?T) ∧ DXT(?T) ≥ ngưỡng"
     (DXT_best = max DXT(T) nên max ≥ ngưỡng ⇔ tồn tại T có DXT(T) ≥ ngưỡng).
  2. kha_nang(X) = Cao / Trung bình tách thành 2 vị từ kha_nang_cao(X), kha_nang_tb(X);
     R18 viết thành  du_dk(X) ∧ ¬kha_nang_cao(X)  (tương đương DXT_best < điểm chuẩn khi đã du_dk)
     để vị từ không phủ định chính nó → hệ luật PHÂN TẦNG được (xem phan_tang()).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from .tham_so import DIEM_LIET, NAM, diem_chuan, nguong

if TYPE_CHECKING:
    from .facts import DuLieu

Atom = tuple[str, ...]

_ATOM_RE = re.compile(r"^\s*([a-z_0-9]+)\s*(?:\((.*)\))?\s*$")


def A(s: str) -> Atom:
    """"du_dk(CNTT)" → ("du_dk", "CNTT");  "thich_lap_trinh" → ("thich_lap_trinh",)."""
    m = _ATOM_RE.match(s)
    if not m:
        raise ValueError(f"Atom không hợp lệ: {s!r}")
    args = tuple(a.strip() for a in m.group(2).split(",")) if m.group(2) else ()
    return (m.group(1),) + args


def fmt(a: Atom) -> str:
    return a[0] + (f"({', '.join(a[1:])})" if len(a) > 1 else "")


def la_bien(x: str) -> bool:
    return x.startswith("?")


def gia_tri(x: str, theta: dict) -> str:
    """Lần theo chuỗi gán biến: ?A → ?B → "A01"."""
    while la_bien(x) and x in theta:
        x = theta[x]
    return x


def thay_the(a: Atom, theta: dict) -> Atom:
    return (a[0],) + tuple(gia_tri(x, theta) for x in a[1:])


def bien_cua(a: Atom) -> set[str]:
    return {x for x in a[1:] if la_bien(x)}


# =========================================================================== điều kiện
@dataclass(frozen=True)
class Not:
    atom: Atom

    def __str__(self) -> str:
        return "¬" + fmt(self.atom)


@dataclass(frozen=True)
class KetQuaTest:
    dung: bool
    mo_ta: str                     # vd "DXT(A01) = 19.95 ≥ 18.00"
    thieu: Decimal | None = None   # số điểm còn thiếu khi điều kiện ngưỡng sai


class Test:
    """Điều kiện số. `can` = dữ kiện đầu vào phải biết trước khi tính (để suy diễn lùi biết cần hỏi gì)."""
    can: tuple[str, ...] = ()

    def bien(self) -> set[str]:
        return {x for x in self.__dict__.values() if isinstance(x, str) and la_bien(x)}

    def kiem_tra(self, ctx: "DuLieu", theta: dict) -> KetQuaTest:
        raise NotImplementedError


@dataclass(frozen=True)
class DiemLiet(Test):
    """R9: min(3 môn của tổ hợp T) ≤ 1.0 (chỉ phương thức THPT)."""
    T: str

    def __str__(self) -> str:
        return f"min(3 môn của {self.T}) ≤ {DIEM_LIET}"

    def kiem_tra(self, ctx, theta) -> KetQuaTest:
        t = gia_tri(self.T, theta)
        m = ctx.diem_thap_nhat(t)
        return KetQuaTest(m <= DIEM_LIET, f"min(3 môn của {t}) = {m} {'≤' if m <= DIEM_LIET else '>'} {DIEM_LIET}")


@dataclass(frozen=True)
class DatMuc(Test):
    """DXT(T) ≥ mức của ngành X. loai = "nguong" (R11–R16) hoặc "chuan" (R17). pt=None → phương thức đang xét."""
    T: str
    nganh: str
    loai: str = "nguong"
    pt: str | None = None
    can: tuple[str, ...] = field(default=("khu_vuc",), compare=False)

    def _muc(self, ctx, theta) -> tuple[str, str, Decimal]:
        x, pt = gia_tri(self.nganh, theta), self.pt or ctx.pt
        return x, pt, (nguong(x, pt) if self.loai == "nguong" else diem_chuan(x, pt))

    def __str__(self) -> str:
        ten = "ngưỡng" if self.loai == "nguong" else f"điểm chuẩn {NAM}"
        if self.pt and not la_bien(self.nganh):
            muc = nguong(self.nganh, self.pt) if self.loai == "nguong" else diem_chuan(self.nganh, self.pt)
            return f"DXT({self.T}) ≥ {ten} {self.nganh}-{self.pt} ({muc})"
        return f"DXT({self.T}) ≥ {ten} của {self.nganh}"

    def kiem_tra(self, ctx, theta) -> KetQuaTest:
        x, pt, muc = self._muc(ctx, theta)
        t = gia_tri(self.T, theta)
        dxt = ctx.dxt(t)
        ten = "ngưỡng" if self.loai == "nguong" else f"điểm chuẩn {NAM}"
        nhan = f"DXT({t})" if pt == "THPT" else "DXT_DGNL"
        if dxt >= muc:
            return KetQuaTest(True, f"{nhan} = {dxt} ≥ {ten} {x} ({muc})")
        return KetQuaTest(False, f"{nhan} = {dxt} < {ten} {x} ({muc}), thiếu {muc - dxt}", muc - dxt)



# =========================================================================== luật
@dataclass(frozen=True)
class Rule:
    id: str
    nhom: str                      # "a" điều kiện xét tuyển, "b" sở thích/định hướng, "c" gợi ý
    dieu_kien: tuple               # tuple[Atom | Not | Test, ...]
    ket_luan: Atom
    giai_thich: str = ""
    nguon: str = ""

    def __str__(self) -> str:
        dk = " ∧ ".join(fmt(c) if isinstance(c, tuple) else str(c) for c in self.dieu_kien)
        return f"{self.id}: {dk} → {fmt(self.ket_luan)}"


def R(id_: str, nhom: str, dieu_kien: list, ket_luan: str, giai_thich: str, nguon: str = "") -> Rule:
    dk = tuple(A(c) if isinstance(c, str) else c for c in dieu_kien)
    return Rule(id_, nhom, dk, A(ket_luan), giai_thich, nguon)


NGUON_DE_AN = f"Đề án tuyển sinh {NAM} – tổ hợp xét tuyển"
NGUON_NGUONG = f"Đề án tuyển sinh {NAM} – ngưỡng đảm bảo chất lượng đầu vào"
NGUON_CHUAN = f"Thông báo điểm trúng tuyển {NAM}"
NGUON_QUY_CHE = "Quy chế tuyển sinh – điểm liệt (≤ 1.0)"
NGUON_NHOM = "Tập luật nhóm (sở thích, định hướng nghề nghiệp)"

_TOHOP_MON3 = {"A01": "Ly", "B08": "Sinh", "D07": "Hoa", "X26": "Tin"}
_NGANH = {"CNTT": "Công nghệ thông tin", "SPTH": "Sư phạm Tin học", "CNGD": "Công nghệ giáo dục"}

RULES: list[Rule] = []

# ---- Nhóm (a) – điều kiện xét tuyển: R1–R18 -------------------------------------------------
for i, (t, m3) in enumerate(_TOHOP_MON3.items(), start=1):            # R1–R4: tổ hợp hợp lệ (THPT)
    RULES.append(R(f"R{i}", "a", ["phuong_thuc(THPT)", "co_mon(Toan)", "co_mon(Anh)", f"co_mon({m3})"],
                   f"tohop_hople({t})", f"Có điểm thi Toán, Anh, {m3} → tổ hợp {t} đủ môn (THPT)", NGUON_DE_AN))
for i, (t, m3) in enumerate(_TOHOP_MON3.items(), start=5):            # R5–R8: tổ hợp hợp lệ (ĐGNL)
    RULES.append(R(f"R{i}", "a", ["phuong_thuc(DGNL)", "co_diem(NL)", "co_diem(HB1)", "co_diem(HB2)",
                                  f"mon_hb2({m3})"],
                   f"tohop_hople({t})", f"Có NL, HB Anh, HB {m3} → tổ hợp {t} đủ môn (ĐGNL)", NGUON_DE_AN))

RULES += [
    R("R9", "a", ["phuong_thuc(THPT)", "tohop_hople(?T)", DiemLiet("?T")], "diem_liet(?T)",
      "Tổ hợp có môn ≤ 1.0 bị loại (điểm liệt, chỉ xét THPT)", NGUON_QUY_CHE),
    R("R10", "a", ["tohop_hople(?T)", Not(A("diem_liet(?T)"))], "tohop_xet(?T)",
      "Tổ hợp đủ môn và không có điểm liệt thì dùng được để xét tuyển", NGUON_QUY_CHE),
]
_id = 11
for pt in ("THPT", "DGNL"):                                           # R11–R16: đủ điều kiện (so với NGƯỠNG)
    for x in ("CNTT", "SPTH", "CNGD"):
        RULES.append(R(f"R{_id}", "a", [f"phuong_thuc({pt})", "tohop_xet(?T)", DatMuc("?T", x, "nguong", pt)],
                       f"du_dk({x})", f"Có tổ hợp dùng được và DXT đạt ngưỡng {pt} ngành {_NGANH[x]}",
                       NGUON_NGUONG))
        _id += 1
RULES += [
    R("R17", "a", ["du_dk(?X)", "tohop_xet(?T)", DatMuc("?T", "?X", "chuan")], "kha_nang_cao(?X)",
      "Đủ điều kiện và DXT_best ≥ điểm chuẩn năm 2026 → khả năng trúng tuyển Cao", NGUON_CHUAN),
    R("R18", "a", ["du_dk(?X)", Not(A("kha_nang_cao(?X)"))], "kha_nang_tb(?X)",
      "Đủ điều kiện nhưng DXT_best < điểm chuẩn → khả năng Trung bình", NGUON_CHUAN),
]

# ---- Nhóm (b) – sở thích, định hướng: R19–R28 -----------------------------------------------
RULES += [
    R("R19", "b", ["thich_lap_trinh"], "nhom_nganh(CNTT)", "Thích lập trình → CNTT", NGUON_NHOM),
    R("R20", "b", ["thich_du_lieu"], "nhom_nganh(CNTT)", "Thích dữ liệu → CNTT", NGUON_NHOM),
    R("R21", "b", ["thich_day_hoc"], "nhom_nganh(SPTH)", "Thích dạy học → Sư phạm Tin học", NGUON_NHOM),
    R("R22", "b", ["thich_thiet_ke_hoc_lieu"], "nhom_nganh(CNGD)", "Thích thiết kế học liệu → CN giáo dục",
      NGUON_NHOM),
    R("R23", "b", ["thich_day_hoc", "thich_lap_trinh"], "khop_manh(SPTH)",
      "Vừa thích dạy học vừa thích lập trình → khớp mạnh Sư phạm Tin học", NGUON_NHOM),
    R("R24", "b", ["thich_thiet_ke_hoc_lieu", "thich_lap_trinh"], "khop_manh(CNGD)",
      "Vừa thích thiết kế học liệu vừa thích lập trình → khớp mạnh CN giáo dục", NGUON_NHOM),
    R("R25", "b", ["lap_trinh_vien"], "nghe_phu_hop(CNTT)", "Muốn làm lập trình viên → CNTT", NGUON_NHOM),
    R("R26", "b", ["phan_tich_du_lieu"], "nghe_phu_hop(CNTT)", "Muốn phân tích dữ liệu → CNTT", NGUON_NHOM),
    R("R27", "b", ["giao_vien"], "nghe_phu_hop(SPTH)", "Muốn làm giáo viên → Sư phạm Tin học", NGUON_NHOM),
    R("R28", "b", ["chuyen_vien_edtech"], "nghe_phu_hop(CNGD)", "Muốn làm chuyên viên EdTech → CN giáo dục",
      NGUON_NHOM),
]

# ---- Nhóm (c) – kết hợp → gợi ý: R29–R33 -----------------------------------------------------
RULES += [
    R("R29", "c", ["nhom_nganh(?X)", "du_dk(?X)", "nghe_phu_hop(?X)"], "goi_y(?X, rat_phu_hop)",
      "Hợp sở thích + đủ điều kiện + đúng định hướng nghề → Rất phù hợp", NGUON_NHOM),
    R("R30", "c", ["nhom_nganh(?X)", "du_dk(?X)", Not(A("nghe_phu_hop(?X)"))], "goi_y(?X, phu_hop)",
      "Hợp sở thích + đủ điều kiện, định hướng nghề chưa khớp → Phù hợp", NGUON_NHOM),
    R("R31", "c", ["nghe_phu_hop(?X)", "du_dk(?X)", Not(A("nhom_nganh(?X)"))], "goi_y(?X, tham_khao)",
      "Đúng định hướng nghề + đủ điều kiện, sở thích chưa khớp → Tham khảo", NGUON_NHOM),
    R("R32", "c", ["du_dk(?X)", Not(A("goi_y(?Y, ?M)"))], "du_dk_chua_khop(?X)",
      "Không có gợi ý nào nhưng đủ điều kiện ngành X → thông báo chưa khớp sở thích/nghề", NGUON_NHOM),
    R("R33", "c", [Not(A("du_dk(?X)"))], "chua_du_dk",
      "Không đủ điều kiện ngành nào → thông báo chưa đủ điều kiện kèm lý do", NGUON_NHOM),
]

MUC_GOI_Y = {"rat_phu_hop": "Rất phù hợp", "phu_hop": "Phù hợp", "tham_khao": "Tham khảo"}


# =========================================================================== kiểm tra & phân tầng
def kiem_tra_luat(rules: list[Rule]) -> None:
    """Luật "an toàn": biến ở kết luận và trong Test phải được gán bởi một atom dương đứng TRƯỚC.
    (Biến chỉ xuất hiện trong Not được hiểu là ¬∃ – ví dụ R32, R33.)"""
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Trùng id luật")
    for r in rules:
        da_gan: set[str] = set()
        for c in r.dieu_kien:
            if isinstance(c, tuple):
                da_gan |= bien_cua(c)
            elif isinstance(c, Test) and not c.bien() <= da_gan:
                raise ValueError(f"{r.id}: biến {c.bien() - da_gan} trong điều kiện số chưa được gán")
        if not bien_cua(r.ket_luan) <= da_gan:
            raise ValueError(f"{r.id}: biến {bien_cua(r.ket_luan) - da_gan} ở kết luận chưa được gán")


def phan_tang(rules: list[Rule]) -> dict[str, int]:
    """Phân tầng (stratification) để phủ định "không chứng minh được" có nghĩa đúng khi suy diễn tiến.

    tầng(kết luận) ≥ tầng(vị từ dương ở điều kiện);  tầng(kết luận) ≥ tầng(vị từ trong Not) + 1.
    Lặp đến khi ổn định; nếu tầng vượt số vị từ ⇒ có vòng phủ định ⇒ hệ luật không phân tầng được.
    """
    preds = {r.ket_luan[0] for r in rules}
    for r in rules:
        for c in r.dieu_kien:
            if isinstance(c, tuple):
                preds.add(c[0])
            elif isinstance(c, Not):
                preds.add(c.atom[0])
    tang = dict.fromkeys(preds, 0)
    doi = True
    while doi:
        doi = False
        for r in rules:
            h = r.ket_luan[0]
            for c in r.dieu_kien:
                can = tang[c[0]] if isinstance(c, tuple) else tang[c.atom[0]] + 1 if isinstance(c, Not) else 0
                if tang[h] < can:
                    tang[h] = can
                    doi = True
                    if can > len(preds):
                        raise ValueError(f"Hệ luật không phân tầng được (vòng phủ định qua '{h}')")
    return tang


kiem_tra_luat(RULES)
