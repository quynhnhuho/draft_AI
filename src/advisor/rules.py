"""Chức năng 2 - Biểu diễn luật và cơ sở luật tư vấn chọn ngành (33 luật, năm áp dụng 2026).

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
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


def bien_cua(a: Atom) -> set[str]:
    return {x for x in a[1:] if la_bien(x)}


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

    def bien(self) -> set[str]:
        return {x for x in self.__dict__.values() if isinstance(x, str) and la_bien(x)}

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
    nhom: str
    dieu_kien: tuple
    ket_luan: Atom
    giai_thich: str = ""
    nguon: str = ""

    def __str__(self) -> str:
        dk = " ∧ ".join(fmt(c) if isinstance(c, tuple) else str(c) for c in self.dieu_kien)
        return f"{self.id}: {dk} → {fmt(self.ket_luan)}"


def R(id_: str, nhom: str, dieu_kien: list, ket_luan: str, giai_thich: str, nguon: str = "") -> Rule:
    dk = tuple(A(c) if isinstance(c, str) else c for c in dieu_kien)
    return Rule(id_, nhom, dk, A(ket_luan), giai_thich, nguon)


NGUON_DE_AN = f"Đề án tuyển sinh {NAM} - tổ hợp xét tuyển"
NGUON_NGUONG = f"Đề án tuyển sinh {NAM} - ngưỡng đảm bảo chất lượng đầu vào"
NGUON_CHUAN = f"Thông báo điểm trúng tuyển {NAM}"
NGUON_QUY_CHE = "Quy chế tuyển sinh - điểm liệt"
NGUON_NHOM = "Tập luật nhóm (sở thích, định hướng nghề nghiệp)"

MON_THU_BA = {"A01": "Ly", "B08": "Sinh", "D07": "Hoa", "X26": "Tin"}
MUC_GOI_Y = {"rat_phu_hop": "Rất phù hợp", "phu_hop": "Phù hợp", "tham_khao": "Tham khảo"}


def _nhom_a() -> list[Rule]:
    luat = []
    for i, (t, m3) in enumerate(MON_THU_BA.items()):
        luat.append(R(f"R{i + 1}", "a", ["phuong_thuc(THPT)", "co_mon(Toan)", "co_mon(Anh)", f"co_mon({m3})"],
                      f"tohop_hople({t})", f"Có điểm thi Toán, Anh, {m3} thì tổ hợp {t} đủ môn (THPT)",
                      NGUON_DE_AN))
    for i, (t, m3) in enumerate(MON_THU_BA.items()):
        luat.append(R(f"R{i + 5}", "a", ["phuong_thuc(DGNL)", "co_diem(NL)", "co_diem(HB1)", "co_diem(HB2)",
                                         f"mon_hb2({m3})"],
                      f"tohop_hople({t})", f"Có NL, học bạ Anh và {m3} thì tổ hợp {t} đủ môn (ĐGNL)", NGUON_DE_AN))
    luat += [
        R("R9", "a", ["phuong_thuc(THPT)", "tohop_hople(?T)", DiemLiet("?T")], "diem_liet(?T)",
          "Tổ hợp có môn không quá 1.0 bị loại (điểm liệt, chỉ xét THPT)", NGUON_QUY_CHE),
        R("R10", "a", ["tohop_hople(?T)", Not(A("diem_liet(?T)"))], "tohop_xet(?T)",
          "Tổ hợp đủ môn và không có điểm liệt thì dùng được để xét tuyển", NGUON_QUY_CHE),
    ]
    so = 11
    for pt in ("THPT", "DGNL"):
        for x in ("CNTT", "SPTH", "CNGD"):
            luat.append(R(f"R{so}", "a", [f"phuong_thuc({pt})", "tohop_xet(?T)", DatMuc("?T", x, "nguong", pt)],
                          f"du_dk({x})", f"Có tổ hợp dùng được và DXT đạt ngưỡng {pt} ngành {ten_nganh(x)}",
                          NGUON_NGUONG))
            so += 1
    luat += [
        R("R17", "a", ["du_dk(?X)", "tohop_xet(?T)", DatMuc("?T", "?X", "chuan")], "kha_nang_cao(?X)",
          "Đủ điều kiện và DXT_best đạt điểm chuẩn năm 2026 thì khả năng trúng tuyển Cao", NGUON_CHUAN),
        R("R18", "a", ["du_dk(?X)", Not(A("kha_nang_cao(?X)"))], "kha_nang_tb(?X)",
          "Đủ điều kiện nhưng DXT_best dưới điểm chuẩn thì khả năng Trung bình", NGUON_CHUAN),
    ]
    return luat


def _nhom_b() -> list[Rule]:
    return [
        R("R19", "b", ["thich_lap_trinh"], "nhom_nganh(CNTT)", "Thích lập trình hợp ngành CNTT", NGUON_NHOM),
        R("R20", "b", ["thich_du_lieu"], "nhom_nganh(CNTT)", "Thích dữ liệu hợp ngành CNTT", NGUON_NHOM),
        R("R21", "b", ["thich_day_hoc"], "nhom_nganh(SPTH)", "Thích dạy học hợp ngành Sư phạm Tin học",
          NGUON_NHOM),
        R("R22", "b", ["thich_thiet_ke_hoc_lieu"], "nhom_nganh(CNGD)",
          "Thích thiết kế học liệu hợp ngành Công nghệ giáo dục", NGUON_NHOM),
        R("R23", "b", ["thich_day_hoc", "thich_lap_trinh"], "khop_manh(SPTH)",
          "Vừa thích dạy học vừa thích lập trình thì khớp mạnh Sư phạm Tin học", NGUON_NHOM),
        R("R24", "b", ["thich_thiet_ke_hoc_lieu", "thich_lap_trinh"], "khop_manh(CNGD)",
          "Vừa thích thiết kế học liệu vừa thích lập trình thì khớp mạnh Công nghệ giáo dục", NGUON_NHOM),
        R("R25", "b", ["lap_trinh_vien"], "nghe_phu_hop(CNTT)", "Muốn làm lập trình viên hợp ngành CNTT",
          NGUON_NHOM),
        R("R26", "b", ["phan_tich_du_lieu"], "nghe_phu_hop(CNTT)", "Muốn phân tích dữ liệu hợp ngành CNTT",
          NGUON_NHOM),
        R("R27", "b", ["giao_vien"], "nghe_phu_hop(SPTH)", "Muốn làm giáo viên hợp ngành Sư phạm Tin học",
          NGUON_NHOM),
        R("R28", "b", ["chuyen_vien_edtech"], "nghe_phu_hop(CNGD)",
          "Muốn làm chuyên viên EdTech hợp ngành Công nghệ giáo dục", NGUON_NHOM),
    ]


def _nhom_c() -> list[Rule]:
    return [
        R("R29", "c", ["nhom_nganh(?X)", "du_dk(?X)", "nghe_phu_hop(?X)"], "goi_y(?X, rat_phu_hop)",
          "Hợp sở thích, đủ điều kiện và đúng định hướng nghề nên Rất phù hợp", NGUON_NHOM),
        R("R30", "c", ["nhom_nganh(?X)", "du_dk(?X)", Not(A("nghe_phu_hop(?X)"))], "goi_y(?X, phu_hop)",
          "Hợp sở thích, đủ điều kiện nhưng định hướng nghề chưa khớp nên Phù hợp", NGUON_NHOM),
        R("R31", "c", ["nghe_phu_hop(?X)", "du_dk(?X)", Not(A("nhom_nganh(?X)"))], "goi_y(?X, tham_khao)",
          "Đúng định hướng nghề, đủ điều kiện nhưng sở thích chưa khớp nên Tham khảo", NGUON_NHOM),
        R("R32", "c", ["du_dk(?X)", Not(A("goi_y(?Y, ?M)"))], "du_dk_chua_khop(?X)",
          "Không có gợi ý nào nhưng đủ điều kiện ngành X", NGUON_NHOM),
        R("R33", "c", [Not(A("du_dk(?X)"))], "chua_du_dk",
          "Không đủ điều kiện ngành nào", NGUON_NHOM),
    ]


RULES: list[Rule] = _nhom_a() + _nhom_b() + _nhom_c()


def kiem_tra_luat(rules: list[Rule]) -> None:
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


kiem_tra_luat(RULES)
