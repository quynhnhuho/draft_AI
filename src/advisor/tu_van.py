# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .diem import lam_tron
from .facts import Atom, DuLieu, build_facts, kiem_tra_dau_vao
from .inference import BackwardResult, ForwardResult, backward_chaining, forward_chaining
from .rules import MUC_GOI_Y
from .tham_so import DIEM_LIET, MA_NGANH, NAM, NGUON, TEN_MON, TO_HOP, diem_chuan, nguong, ten_nganh

CAU_HOI = {
    "phuong_thuc": "phương thức xét tuyển (THPT hay ĐGNL)",
    "khu_vuc": "khu vực ưu tiên (KV1, KV2-NT, KV2, KV3)",
    **{f"diem_{m}": f"điểm thi THPT môn {ten}" for m, ten in TEN_MON.items()},
    "diem_dgnl": "điểm ĐGNL môn Toán (thang 10)",
    "HB_mon1": "điểm học bạ lớp 12 môn Tiếng Anh",
    "HB_mon2": "điểm học bạ lớp 12 môn thứ hai của tổ hợp",
    "mon_hb2": "môn học bạ thứ hai (Lý, Hóa, Sinh hoặc Tin)",
}


@dataclass
class DongDiem:
    to_hop: str
    tong: Decimal
    dut: Decimal
    dxt: Decimal
    diem_liet: bool


@dataclass
class GoiY:
    nganh: str
    muc: str
    kha_nang: str
    diem_phu_hop: int
    chenh: Decimal
    thanh_phan: dict = field(default_factory=dict)

    @property
    def ten_muc(self) -> str:
        return MUC_GOI_Y[self.muc]


@dataclass
class KetQuaTuVan:
    user: dict
    loi: list[str] = field(default_factory=list)
    forward: ForwardResult | None = None
    goi_y: list[GoiY] = field(default_factory=list)
    du_dk: list[str] = field(default_factory=list)
    thong_bao: list[str] = field(default_factory=list)
    diem: list[DongDiem] = field(default_factory=list)
    best: tuple[str, Decimal] | None = None
    chua_khop: list[str] = field(default_factory=list)


@dataclass
class KetQuaKiemTra:
    user: dict
    nganh: str
    loi: list[str] = field(default_factory=list)
    lui: BackwardResult | None = None
    kha_nang: BackwardResult | None = None

    @property
    def ket_luan(self) -> str:
        return "Lỗi dữ liệu" if self.loi else self.lui.ket_luan

    @property
    def can_hoi(self) -> list[str]:
        if not self.lui:
            return []
        thu_tu = list(CAU_HOI)
        return sorted(self.lui.missing, key=lambda k: thu_tu.index(k) if k in thu_tu else len(thu_tu))


def bang_diem(facts: set[Atom], ctx: DuLieu) -> list[DongDiem]:
    if not ctx.khu_vuc:
        return []
    return [DongDiem(t, ctx.tong_diem(t), ctx.dut(t), ctx.dxt(t), ("diem_liet", t) in facts)
            for _, t in sorted(f for f in facts if f[0] == "tohop_hople")]


def dxt_best(facts: set[Atom], ctx: DuLieu) -> tuple[str, Decimal] | None:
    to_hop = [f[1] for f in facts if f[0] == "tohop_xet"]
    if not to_hop or not ctx.khu_vuc:
        return None
    return max(((t, ctx.dxt(t)) for t in to_hop), key=lambda p: (p[1], p[0]))


def diem_phu_hop(facts: set[Atom], x: str) -> tuple[int, dict]:
    tp = {"nhom_nganh": int(("nhom_nganh", x) in facts), "khop_manh": int(("khop_manh", x) in facts),
          "nghe_phu_hop": 2 * int(("nghe_phu_hop", x) in facts), "kha_nang_cao": int(("kha_nang_cao", x) in facts)}
    return sum(tp.values()), tp


def xep_hang(fwd: ForwardResult, ctx: DuLieu) -> list[GoiY]:
    best = dxt_best(fwd.facts, ctx)
    ds = []
    for x, muc in fwd.recommendations:
        d, tp = diem_phu_hop(fwd.facts, x)
        cao = ("kha_nang_cao", x) in fwd.facts
        chenh = best[1] - diem_chuan(x, ctx.pt) if best else Decimal(0)
        ds.append(GoiY(x, muc, "Cao" if cao else "Trung bình", d, chenh, tp))
    return sorted(ds, key=lambda g: (-g.diem_phu_hop, g.kha_nang != "Cao", -g.chenh, g.nganh))


def ly_do_chua_du_dk(facts: set[Atom], ctx: DuLieu) -> list[str]:
    if not any(f[0] == "tohop_hople" for f in facts):
        if ctx.pt == "DGNL":
            return ["Thiếu tổ hợp: chưa đủ điểm ĐGNL môn Toán, học bạ Tiếng Anh và học bạ môn thứ hai "
                    "(không luật R5-R8 nào được kích hoạt)."]
        return ["Thiếu tổ hợp: chưa có đủ điểm thi Toán, Tiếng Anh và một môn Vật lý, Sinh học, Hóa học hoặc "
                "Tin học (không luật R1-R4 nào được kích hoạt)."]
    ly_do = []
    liet = sorted(f[1] for f in facts if f[0] == "diem_liet")
    if liet:
        chi_tiet = [f"{t} có môn {TEN_MON[min(TO_HOP[t], key=ctx.diem_mon)]} = {ctx.diem_thap_nhat(t)} "
                    f"(không quá {DIEM_LIET})" for t in liet]
        ly_do.append("Điểm liệt (R9): tổ hợp " + ", ".join(chi_tiet) + " bị loại.")
    best = dxt_best(facts, ctx)
    if best:
        t, d = best
        x = min(MA_NGANH, key=lambda m: nguong(m, ctx.pt))
        n = nguong(x, ctx.pt)
        ly_do.append(f"DXT_best = {d} (tổ hợp {t}) chưa đạt ngưỡng thấp nhất {n} ({ten_nganh(x)}), "
                     f"còn thiếu {n - d} điểm.")
    return ly_do


def tu_van(user: dict) -> KetQuaTuVan:
    kq = KetQuaTuVan(user, kiem_tra_dau_vao(user))
    kq.loi += [f"Thiếu {CAU_HOI[k]}." for k in ("phuong_thuc", "khu_vuc") if not user.get(k)]
    if kq.loi:
        return kq
    ctx = DuLieu(user)
    fwd = kq.forward = forward_chaining(build_facts(user), ctx)
    kq.diem = bang_diem(fwd.facts, ctx)
    kq.best = dxt_best(fwd.facts, ctx)
    kq.du_dk = [f[1] for f in fwd.tat_ca("du_dk")]
    kq.goi_y = xep_hang(fwd, ctx)
    chua_khop = kq.chua_khop = [f[1] for f in fwd.tat_ca("du_dk_chua_khop")]
    if chua_khop:
        kq.thong_bao.append("Bạn **đủ điều kiện** xét tuyển ngành " + ", ".join(map(ten_nganh, chua_khop))
                            + " nhưng các ngành này **chưa khớp sở thích hoặc định hướng nghề nghiệp** "
                            "bạn đã chọn (R32).")
    if fwd.co("chua_du_dk"):
        kq.thong_bao.append("Bạn **chưa đủ điều kiện xét tuyển** ngành nào (R33). Lý do:\n"
                            + "\n".join(f"  - {s}" for s in ly_do_chua_du_dk(fwd.facts, ctx)))
    return kq


def kiem_tra_dieu_kien(user: dict, nganh: str) -> KetQuaKiemTra:
    kq = KetQuaKiemTra(user, nganh, kiem_tra_dau_vao(user))
    if kq.loi:
        return kq
    ctx, facts = DuLieu(user), build_facts(user)
    kq.lui = backward_chaining(("du_dk", nganh), facts, ctx)
    if kq.lui.proven:
        kq.kha_nang = backward_chaining(("kha_nang_cao", nganh), facts, ctx)
    return kq


# ------------------------------------------------------------------ trình bày cho người dùng (web, Chức năng 4)
# Ngôn ngữ thường, xưng "mình – bạn", KHÔNG hiện mã luật (R1…R33), tên vị từ (du_dk, tohop_xet…) hay vết suy diễn.
# Phần kỹ thuật (luật kích hoạt, chuỗi chứng minh) vẫn có trong KetQuaTuVan/KetQuaKiemTra và được in bởi demo.py.

def _nguon() -> str:
    ten = str(NGUON["nguon"]).split(";")[0].strip()   # bỏ phần ghi chú nội bộ về tập luật
    return f"Nguồn: {ten} – năm áp dụng {NAM}"


def _loi(ds: list[str]) -> str:
    return "Thông tin chưa hợp lệ:\n" + "\n".join(f"- {s}" for s in ds)


def _so_voi(chenh: Decimal) -> str:
    if chenh > 0:
        return f"cao hơn {chenh} điểm"
    if chenh < 0:
        return f"thấp hơn {-chenh} điểm"
    return "bằng điểm chuẩn"


def _bang_diem_md(rows: list[DongDiem], pt: str) -> str:
    nhan = "Tổ hợp" if pt == "THPT" else "Tổ hợp (ĐGNL)"
    out = [f"| {nhan} | Tổng điểm (chưa cộng ưu tiên) | Điểm ưu tiên | Điểm xét tuyển | Ghi chú |",
           "|---|---|---|---|---|"]
    out += [f"| {r.to_hop} | {lam_tron(r.tong)} | {lam_tron(r.dut)} | **{r.dxt}** | "
            f"{'có điểm liệt – không dùng được' if r.diem_liet else 'dùng được'} |" for r in rows]
    return "\n".join(out)


def _ly_do_than_thien(user: dict) -> list[str]:
    """Lý do chưa đủ điều kiện, nói bằng lời thường (tính lại từ dữ liệu, không dùng mã luật)."""
    ctx = DuLieu(user)
    facts = forward_chaining(build_facts(user), ctx).facts
    if not any(f[0] == "tohop_hople" for f in facts):
        if ctx.pt == "DGNL":
            return ["Chưa đủ điểm ĐGNL môn Toán, học bạ Tiếng Anh và học bạ môn thứ hai để tính điểm xét tuyển."]
        return ["Chưa đủ điểm thi để ghép thành tổ hợp xét tuyển: cần Toán, Tiếng Anh và một môn Vật lý, "
                "Hóa học, Sinh học hoặc Tin học."]
    ly_do = []
    for t in sorted(f[1] for f in facts if f[0] == "diem_liet"):
        mon = min(TO_HOP[t], key=ctx.diem_mon)
        ly_do.append(f"Tổ hợp {t} có môn {TEN_MON[mon]} {ctx.diem_thap_nhat(t)} điểm (từ {DIEM_LIET} điểm trở "
                     "xuống là điểm liệt) nên không dùng được.")
    best = dxt_best(facts, ctx)
    if best:
        t, d = best
        x = min(MA_NGANH, key=lambda m: nguong(m, ctx.pt))
        n = nguong(x, ctx.pt)
        ly_do.append(f"Điểm xét tuyển cao nhất của bạn là {d} (tổ hợp {t}), chưa đạt mức điểm tối thiểu {n} "
                     f"của ngành {ten_nganh(x)}; còn thiếu {n - d} điểm.")
    return ly_do


def markdown_tu_van(kq: KetQuaTuVan) -> str:
    if kq.loi:
        return _loi(kq.loi)
    pt = kq.user["phuong_thuc"]
    parts = []
    if kq.diem:
        parts.append(_bang_diem_md(kq.diem, pt))
    if kq.best:
        parts.append(f"**Điểm xét tuyển cao nhất của bạn: {kq.best[1]}** (tổ hợp {kq.best[0]})")
    if kq.goi_y:
        rows = [f"| # | Ngành | Mức độ phù hợp | Khả năng trúng tuyển | So với điểm chuẩn {NAM} |",
                "|---|---|---|---|---|"]
        rows += [f"| {i} | {ten_nganh(g.nganh)} | **{g.ten_muc}** | {g.kha_nang} | {_so_voi(g.chenh)} |"
                 for i, g in enumerate(kq.goi_y, 1)]
        vi_sao = []
        for g in kq.goi_y:
            lan = next(f for f in kq.forward.fired if f.ket_luan == ("goi_y", g.nganh, g.muc))
            vi_sao.append(f"- **{ten_nganh(g.nganh)}**: {lan.rule.giai_thich}.")
        parts.append("**Ngành gợi ý cho bạn:**\n\n" + "\n".join(rows) + "\n\n**Vì sao?**\n" + "\n".join(vi_sao))
    if kq.chua_khop:
        parts.append("Bạn **đủ điều kiện** xét tuyển ngành " + ", ".join(map(ten_nganh, kq.chua_khop))
                     + " nhưng chưa khớp với sở thích hoặc định hướng nghề nghiệp bạn đã chọn.")
    if kq.forward.co("chua_du_dk"):
        parts.append("Bạn **chưa đủ điều kiện xét tuyển** ngành nào của Khoa. Lý do:\n"
                     + "\n".join(f"- {s}" for s in _ly_do_than_thien(kq.user)))
    if kq.du_dk and kq.goi_y:
        parts.append("Các ngành bạn đủ điều kiện xét tuyển: " + ", ".join(map(ten_nganh, kq.du_dk)) + ".")
    parts.append(_nguon())
    return "\n\n".join(parts)


def markdown_kiem_tra(kq: KetQuaKiemTra) -> str:
    if kq.loi:
        return _loi(kq.loi)
    ten, pt, b = ten_nganh(kq.nganh), kq.user.get("phuong_thuc"), kq.lui
    parts = []
    if b.proven:
        parts.append(f"**Có.** Bạn đủ điều kiện xét tuyển ngành **{ten}**.")
        ctx = DuLieu(kq.user)
        best = dxt_best(forward_chaining(build_facts(kq.user), ctx).facts, ctx)
        if best:
            parts.append(f"Điểm xét tuyển cao nhất của bạn là **{best[1]}** (tổ hợp {best[0]}), đạt mức điểm "
                         f"tối thiểu {nguong(kq.nganh, pt)} của ngành.")
        muc = "**Cao**" if kq.kha_nang.proven else "**Trung bình**"
        parts.append(f"Khả năng trúng tuyển: {muc} (điểm chuẩn năm {NAM}: {diem_chuan(kq.nganh, pt)}).")
    elif b.missing:
        parts.append("Mình cần thêm thông tin để trả lời: " + ", ".join(CAU_HOI.get(k, k) for k in kq.can_hoi) + ".")
    else:
        parts.append(f"**Chưa.** Bạn chưa đủ điều kiện xét tuyển ngành **{ten}**.")
        ly_do = _ly_do_nganh(kq.user, kq.nganh)
        if ly_do:
            parts.append("Lý do:\n" + "\n".join(f"- {s}" for s in ly_do))
        if b.thieu_diem is not None:
            parts.append(f"Bạn còn thiếu **{b.thieu_diem}** điểm so với mức điểm tối thiểu của ngành.")
    parts.append(_nguon())
    return "\n\n".join(parts)


def _ly_do_nganh(user: dict, nganh: str) -> list[str]:
    """Vì sao chưa đủ điều kiện một ngành: thiếu tổ hợp, điểm liệt, hoặc từng tổ hợp thiếu bao nhiêu điểm."""
    ctx = DuLieu(user)
    facts = forward_chaining(build_facts(user), ctx).facts
    rows = bang_diem(facts, ctx)
    if not rows:
        return _ly_do_than_thien(user)[:1]
    n = nguong(nganh, ctx.pt)
    out = []
    for r in rows:
        if r.diem_liet:
            out.append(f"Tổ hợp {r.to_hop} có điểm liệt nên không dùng được.")
        elif r.dxt < n:
            out.append(f"Tổ hợp {r.to_hop}: điểm xét tuyển {r.dxt}, thấp hơn mức tối thiểu {n} "
                       f"(thiếu {n - r.dxt} điểm).")
    return out
