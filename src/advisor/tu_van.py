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
    chua_khop = [f[1] for f in fwd.tat_ca("du_dk_chua_khop")]
    if chua_khop:
        kq.thong_bao.append("Em **đủ điều kiện** xét tuyển ngành " + ", ".join(map(ten_nganh, chua_khop))
                            + " nhưng các ngành này **chưa khớp sở thích hoặc định hướng nghề nghiệp** "
                            "em đã chọn (R32).")
    if fwd.co("chua_du_dk"):
        kq.thong_bao.append("Em **chưa đủ điều kiện xét tuyển** ngành nào (R33). Lý do:\n"
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


def _nguon() -> str:
    s = f"Nguồn: {NGUON['nguon']} - năm áp dụng {NAM}"
    if not NGUON.get("da_kiem_chung"):
        s += "  \nLưu ý: số liệu chưa đối chiếu với Đề án tuyển sinh chính thức."
    return s


def _loi(ds: list[str]) -> str:
    return "Thông tin chưa hợp lệ:\n" + "\n".join(f"- {s}" for s in ds)


def _bang_diem_md(rows: list[DongDiem], pt: str) -> str:
    nhan = "Tổ hợp" if pt == "THPT" else "Tổ hợp (ĐGNL)"
    out = [f"| {nhan} | Tổng (chưa ưu tiên) | Điểm ưu tiên | DXT | Ghi chú |", "|---|---|---|---|---|"]
    out += [f"| {r.to_hop} | {lam_tron(r.tong)} | {lam_tron(r.dut)} | **{r.dxt}** | "
            f"{'điểm liệt - loại' if r.diem_liet else 'dùng được'} |" for r in rows]
    return "\n".join(out)


def _bang_goi_y_md(kq: KetQuaTuVan) -> str:
    rows = ["| # | Ngành | Mức gợi ý | Khả năng trúng tuyển | Điểm phù hợp | DXT_best - điểm chuẩn |",
            "|---|---|---|---|---|---|"]
    rows += [f"| {i} | {ten_nganh(g.nganh)} | **{g.ten_muc}** | {g.kha_nang} | {g.diem_phu_hop} | {g.chenh:+} |"
             for i, g in enumerate(kq.goi_y, 1)]
    vi_sao = []
    for g in kq.goi_y:
        lan = next(f for f in kq.forward.fired if f.ket_luan == ("goi_y", g.nganh, g.muc))
        tp = g.thanh_phan
        vi_sao.append(f"- **{ten_nganh(g.nganh)}** - {lan.rule.id}: {lan.rule.giai_thich}. Điểm phù hợp = "
                      f"sở thích {tp['nhom_nganh']} + khớp mạnh {tp['khop_manh']} + nghề {tp['nghe_phu_hop']} "
                      f"+ khả năng Cao {tp['kha_nang_cao']} = **{g.diem_phu_hop}**")
    return "**Ngành gợi ý (suy diễn tiến):**\n\n" + "\n".join(rows) + "\n\n**Vì sao?**\n" + "\n".join(vi_sao)


def markdown_tu_van(kq: KetQuaTuVan, vet: bool = True) -> str:
    if kq.loi:
        return _loi(kq.loi)
    pt = kq.user["phuong_thuc"]
    parts = [f"**Phương thức:** {pt}, **khu vực:** {kq.user['khu_vuc']}"]
    if kq.diem:
        parts.append(_bang_diem_md(kq.diem, pt))
    if kq.best:
        parts.append(f"**DXT_best = {kq.best[1]}** (tổ hợp {kq.best[0]})")
    if kq.goi_y:
        parts.append(_bang_goi_y_md(kq))
    parts += kq.thong_bao
    if kq.du_dk and kq.goi_y:
        parts.append("Các ngành em đủ điều kiện xét tuyển: " + ", ".join(map(ten_nganh, kq.du_dk)) + ".")
    if vet:
        parts.append("**Vết suy diễn tiến** (luật kích hoạt theo thứ tự):\n"
                     + "\n".join(f"{i}. {f}" for i, f in enumerate(kq.forward.fired, 1)))
    parts.append(_nguon())
    return "\n\n".join(parts)


def markdown_kiem_tra(kq: KetQuaKiemTra, vet: bool = False) -> str:
    if kq.loi:
        return _loi(kq.loi)
    ten, pt, b = ten_nganh(kq.nganh), kq.user.get("phuong_thuc"), kq.lui
    parts = [f"**Mục tiêu suy diễn lùi:** `{b.goal}` - em có đủ điều kiện xét tuyển ngành **{ten}** không?"]
    if b.proven:
        parts.append(f"**Có.** Em đủ điều kiện xét tuyển ngành {ten} (đạt ngưỡng đảm bảo chất lượng đầu vào "
                     f"{nguong(kq.nganh, pt)}).")
        muc = "**Cao**" if kq.kha_nang.proven else "**Trung bình**"
        parts.append(f"Khả năng trúng tuyển: {muc}, so với điểm chuẩn {NAM} là {diem_chuan(kq.nganh, pt)}.")
        parts.append("**Chuỗi luật đã dùng:**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(b.proof, 1)))
    elif b.missing:
        parts.append("**Chưa đủ thông tin** để kết luận. Mình cần biết thêm: "
                     + ", ".join(CAU_HOI.get(k, k) for k in kq.can_hoi) + ".")
    else:
        parts.append(f"**Không.** Em chưa đủ điều kiện xét tuyển ngành {ten}.")
        if b.thieu_diem is not None:
            parts.append(f"Còn thiếu **{b.thieu_diem}** điểm so với ngưỡng.")
        if b.that_bai:
            parts.append("**Điều kiện thất bại:**\n" + "\n".join(f"- {s}" for s in b.that_bai))
    if vet:
        parts.append("**Nhật ký suy diễn lùi:**\n```\n" + "\n".join(b.trace) + "\n```")
    parts.append(_nguon())
    return "\n\n".join(parts)
