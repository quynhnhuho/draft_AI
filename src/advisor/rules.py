# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from .tham_so import DIEM_LIET, NAM, diem_chuan, nguong, ten_nganh

if TYPE_CHECKING:
    from .facts import DuLieu

Atom = tuple[str, ...]

_ATOM_RE = re.compile(r"^\s*([a-z_0-9]+)\s*(?:\((.*)\))?\s*$")


def A(s: str) -> Atom:
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
    while la_bien(x) and x in theta:
        x = theta[x]
    return x


def thay_the(a: Atom, theta: dict) -> Atom:
    return (a[0],) + tuple(gia_tri(x, theta) for x in a[1:])


@dataclass(frozen=True)
class Not:
    atom: Atom

    def __str__(self) -> str:
        return "¬" + fmt(self.atom)


@dataclass(frozen=True)
class KetQuaTest:
    dung: bool
    mo_ta: str
    thieu: Decimal | None = None


class Test:
    can: tuple[str, ...] = ()

    def kiem_tra(self, ctx: "DuLieu", theta: dict) -> KetQuaTest:
        raise NotImplementedError


@dataclass(frozen=True)
class DiemLiet(Test):
    T: str

    def __str__(self) -> str:
        return f"min(3 môn của {self.T}) ≤ {DIEM_LIET}"

    def kiem_tra(self, ctx, theta) -> KetQuaTest:
        t = gia_tri(self.T, theta)
        m = ctx.diem_thap_nhat(t)
        dau = "≤" if m <= DIEM_LIET else ">"
        return KetQuaTest(m <= DIEM_LIET, f"min(3 môn của {t}) = {m} {dau} {DIEM_LIET}")


@dataclass(frozen=True)
class DatMuc(Test):
    T: str
    nganh: str
    loai: str = "nguong"
    pt: str | None = None
    can: tuple[str, ...] = field(default=("khu_vuc",), compare=False)

    @property
    def ten_muc(self) -> str:
        return "ngưỡng" if self.loai == "nguong" else f"điểm chuẩn {NAM}"

    def muc(self, nganh: str, pt: str) -> Decimal:
        return nguong(nganh, pt) if self.loai == "nguong" else diem_chuan(nganh, pt)

    def __str__(self) -> str:
        if self.pt and not la_bien(self.nganh):
            return f"DXT({self.T}) ≥ {self.ten_muc} {self.nganh}-{self.pt} ({self.muc(self.nganh, self.pt)})"
        return f"DXT({self.T}) ≥ {self.ten_muc} của {self.nganh}"

    def kiem_tra(self, ctx, theta) -> KetQuaTest:
        x, t, pt = gia_tri(self.nganh, theta), gia_tri(self.T, theta), self.pt or ctx.pt
        muc, dxt = self.muc(x, pt), ctx.dxt(t)
        nhan = f"DXT({t})" if pt == "THPT" else "DXT_DGNL"
        if dxt >= muc:
            return KetQuaTest(True, f"{nhan} = {dxt} ≥ {self.ten_muc} {x} ({muc})")
        return KetQuaTest(False, f"{nhan} = {dxt} < {self.ten_muc} {x} ({muc}), thiếu {muc - dxt}", muc - dxt)


@dataclass(frozen=True)
class Rule:
    id: str
    dieu_kien: tuple
    ket_luan: Atom
    giai_thich: str = ""

    def __str__(self) -> str:
        dk = " ∧ ".join(fmt(c) if isinstance(c, tuple) else str(c) for c in self.dieu_kien)
        return f"{self.id}: {dk} → {fmt(self.ket_luan)}"


def R(id_: str, dieu_kien: list, ket_luan: str, giai_thich: str) -> Rule:
    dk = tuple(A(c) if isinstance(c, str) else c for c in dieu_kien)
    return Rule(id_, dk, A(ket_luan), giai_thich)


MON_THU_BA = {"A01": "Ly", "B08": "Sinh", "D07": "Hoa", "X26": "Tin"}
MUC_GOI_Y = {"rat_phu_hop": "Rất phù hợp", "phu_hop": "Phù hợp", "tham_khao": "Tham khảo"}


def _nhom_a() -> list[Rule]:
    luat = []
    for i, (t, m3) in enumerate(MON_THU_BA.items()):
        luat.append(R(f"R{i + 1}", ["phuong_thuc(THPT)", "co_mon(Toan)", "co_mon(Anh)", f"co_mon({m3})"],
                      f"tohop_hople({t})", f"Có điểm thi Toán, Anh, {m3} thì tổ hợp {t} đủ môn (THPT)"))
    for i, (t, m3) in enumerate(MON_THU_BA.items()):
        luat.append(R(f"R{i + 5}", ["phuong_thuc(DGNL)", "co_diem(NL)", "co_diem(HB1)", "co_diem(HB2)",
                                    f"mon_hb2({m3})"],
                      f"tohop_hople({t})", f"Có NL, học bạ Anh và {m3} thì tổ hợp {t} đủ môn (ĐGNL)"))
    luat += [
        R("R9", ["phuong_thuc(THPT)", "tohop_hople(?T)", DiemLiet("?T")], "diem_liet(?T)",
          "Tổ hợp có môn không quá 1.0 bị loại (điểm liệt, chỉ xét THPT)"),
        R("R10", ["tohop_hople(?T)", Not(A("diem_liet(?T)"))], "tohop_xet(?T)",
          "Tổ hợp đủ môn và không có điểm liệt thì dùng được để xét tuyển"),
    ]
    so = 11
    for pt in ("THPT", "DGNL"):
        for x in ("CNTT", "SPTH", "CNGD"):
            luat.append(R(f"R{so}", [f"phuong_thuc({pt})", "tohop_xet(?T)", DatMuc("?T", x, "nguong", pt)],
                          f"du_dk({x})", f"Có tổ hợp dùng được và DXT đạt ngưỡng {pt} ngành {ten_nganh(x)}"))
            so += 1
    luat += [
        R("R17", ["du_dk(?X)", "tohop_xet(?T)", DatMuc("?T", "?X", "chuan")], "kha_nang_cao(?X)",
          "Đủ điều kiện và DXT_best đạt điểm chuẩn năm 2026 thì khả năng trúng tuyển Cao"),
        R("R18", ["du_dk(?X)", Not(A("kha_nang_cao(?X)"))], "kha_nang_tb(?X)",
          "Đủ điều kiện nhưng DXT_best dưới điểm chuẩn thì khả năng Trung bình"),
    ]
    return luat


def _nhom_b() -> list[Rule]:
    return [
        R("R19", ["thich_lap_trinh"], "nhom_nganh(CNTT)", "Thích lập trình hợp ngành CNTT"),
        R("R20", ["thich_du_lieu"], "nhom_nganh(CNTT)", "Thích dữ liệu hợp ngành CNTT"),
        R("R21", ["thich_day_hoc"], "nhom_nganh(SPTH)", "Thích dạy học hợp ngành Sư phạm Tin học"),
        R("R22", ["thich_thiet_ke_hoc_lieu"], "nhom_nganh(CNGD)",
          "Thích thiết kế học liệu hợp ngành Công nghệ giáo dục"),
        R("R23", ["thich_day_hoc", "thich_lap_trinh"], "khop_manh(SPTH)",
          "Vừa thích dạy học vừa thích lập trình thì khớp mạnh Sư phạm Tin học"),
        R("R24", ["thich_thiet_ke_hoc_lieu", "thich_lap_trinh"], "khop_manh(CNGD)",
          "Vừa thích thiết kế học liệu vừa thích lập trình thì khớp mạnh Công nghệ giáo dục"),
        R("R25", ["lap_trinh_vien"], "nghe_phu_hop(CNTT)", "Muốn làm lập trình viên hợp ngành CNTT"),
        R("R26", ["phan_tich_du_lieu"], "nghe_phu_hop(CNTT)", "Muốn phân tích dữ liệu hợp ngành CNTT"),
        R("R27", ["giao_vien"], "nghe_phu_hop(SPTH)", "Muốn làm giáo viên hợp ngành Sư phạm Tin học"),
        R("R28", ["chuyen_vien_edtech"], "nghe_phu_hop(CNGD)",
          "Muốn làm chuyên viên EdTech hợp ngành Công nghệ giáo dục"),
    ]


def _nhom_c() -> list[Rule]:
    return [
        R("R29", ["nhom_nganh(?X)", "du_dk(?X)", "nghe_phu_hop(?X)"], "goi_y(?X, rat_phu_hop)",
          "Hợp sở thích, đủ điều kiện và đúng định hướng nghề nên Rất phù hợp"),
        R("R30", ["nhom_nganh(?X)", "du_dk(?X)", Not(A("nghe_phu_hop(?X)"))], "goi_y(?X, phu_hop)",
          "Hợp sở thích, đủ điều kiện nhưng định hướng nghề chưa khớp nên Phù hợp"),
        R("R31", ["nghe_phu_hop(?X)", "du_dk(?X)", Not(A("nhom_nganh(?X)"))], "goi_y(?X, tham_khao)",
          "Đúng định hướng nghề, đủ điều kiện nhưng sở thích chưa khớp nên Tham khảo"),
        R("R32", ["du_dk(?X)", Not(A("goi_y(?Y, ?M)"))], "du_dk_chua_khop(?X)",
          "Không có gợi ý nào nhưng đủ điều kiện ngành X"),
        R("R33", [Not(A("du_dk(?X)"))], "chua_du_dk",
          "Không đủ điều kiện ngành nào"),
    ]


RULES: list[Rule] = _nhom_a() + _nhom_b() + _nhom_c()


def _vi_tu(c) -> str | None:
    if isinstance(c, tuple):
        return c[0]
    if isinstance(c, Not):
        return c.atom[0]
    return None


def phan_tang(rules: list[Rule]) -> dict[str, int]:
    preds = {r.ket_luan[0] for r in rules} | {p for r in rules for c in r.dieu_kien if (p := _vi_tu(c))}
    tang = dict.fromkeys(preds, 0)
    doi = True
    while doi:
        doi = False
        for r in rules:
            h = r.ket_luan[0]
            for c in r.dieu_kien:
                if isinstance(c, Test):
                    continue
                can = tang[_vi_tu(c)] + (1 if isinstance(c, Not) else 0)
                if tang[h] < can:
                    if can > len(preds):
                        raise ValueError(f"Hệ luật không phân tầng được (vòng phủ định qua '{h}')")
                    tang[h] = can
                    doi = True
    return tang

