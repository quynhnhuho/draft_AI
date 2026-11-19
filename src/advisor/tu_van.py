"""C2 – Lớp dịch vụ: điểm vào DUY NHẤT cho chatbot (C4) và kịch bản demo.

    tu_van(user)                     → suy diễn TIẾN: danh sách ngành gợi ý đã xếp hạng + vết + thông báo R32/R33
    kiem_tra_dieu_kien(user, "CNTT") → suy diễn LÙI: Có / Không / Chưa đủ thông tin (+ câu cần hỏi)
    markdown_tu_van(kq), markdown_kiem_tra(kq) → văn bản trả lời (Markdown) có giải thích và nguồn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .diem import lam_tron
from .facts import Atom, DuLieu, build_facts, kiem_tra_dau_vao
from .inference import BackwardResult, ForwardResult, backward_chaining, forward_chaining
from .rules import MUC_GOI_Y
from .tham_so import MA_NGANH, NAM, NGUON, TEN_MON, TO_HOP, diem_chuan, nguong, ten_nganh

CAU_HOI = {
    "phuong_thuc": "phương thức xét tuyển (THPT hay ĐGNL)",
    "khu_vuc": "khu vực ưu tiên (KV1, KV2-NT, KV2, KV3)",
    **{f"diem_{m}": f"điểm thi THPT môn {ten}" for m, ten in TEN_MON.items()},
    "diem_dgnl": "điểm ĐGNL môn Toán (thang 10)",
    "HB_mon1": "điểm học bạ lớp 12 môn Tiếng Anh",
    "HB_mon2": "điểm học bạ lớp 12 môn thứ hai của tổ hợp",
    "mon_hb2": "môn học bạ thứ hai (Lý, Hóa, Sinh hoặc Tin)",
}


# =========================================================================== điểm số hiển thị
@dataclass
class DongDiem:
    to_hop: str
    tong: Decimal
    dut: Decimal
    dxt: Decimal
    diem_liet: bool


def bang_diem(facts: set[Atom], ctx: DuLieu) -> list[DongDiem]:
    """Điểm của từng tổ hợp hợp lệ (để hiển thị/giải thích; luật vẫn tự tính qua Test)."""
    if not ctx.khu_vuc:
        return []
    rows = []
    for f in sorted(x for x in facts if x[0] == "tohop_hople"):
        t = f[1]
        rows.append(DongDiem(t, ctx.tong_diem(t), ctx.dut(t), ctx.dxt(t), ("diem_liet", t) in facts))
    return rows


def dxt_best(facts: set[Atom], ctx: DuLieu) -> tuple[str, Decimal] | None:
    """(best_tohop, DXT_best) trên các tổ hợp tohop_xet."""
    ung_vien = [(ctx.dxt(f[1]), f[1]) for f in facts if f[0] == "tohop_xet"]
    if not ung_vien or not ctx.khu_vuc:
        return None
    d, t = max(ung_vien)
    return t, d


# =========================================================================== SUY DIỄN TIẾN → gợi ý
@dataclass
class GoiY:
    nganh: str
    muc: str                 # rat_phu_hop | phu_hop | tham_khao
    kha_nang: str            # "Cao" | "Trung bình"
    diem_phu_hop: int
    chenh: Decimal           # DXT_best − điểm chuẩn
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


def diem_phu_hop(facts: set[Atom], x: str) -> tuple[int, dict]:
    """diem_phu_hop(X) = 1·[X ∈ nhom_nganh] + 1·[khop_manh(X)] + 2·[nghe_phu_hop(X)] + 1·[kha_nang(X) = Cao]."""
    tp = {"nhom_nganh": int(("nhom_nganh", x) in facts), "khop_manh": int(("khop_manh", x) in facts),
          "nghe_phu_hop": 2 * int(("nghe_phu_hop", x) in facts), "kha_nang_cao": int(("kha_nang_cao", x) in facts)}
    return sum(tp.values()), tp


def xep_hang(fwd: ForwardResult, ctx: DuLieu) -> list[GoiY]:
    """Sắp giảm dần theo diem_phu_hop; bằng nhau → ưu tiên kha_nang = Cao, rồi DXT_best − điểm chuẩn lớn hơn."""
    best = dxt_best(fwd.facts, ctx)
    ds = []
    for x, muc in fwd.recommendations:
        d, tp = diem_phu_hop(fwd.facts, x)
        cao = ("kha_nang_cao", x) in fwd.facts
        chenh = (best[1] - diem_chuan(x, ctx.pt)) if best else Decimal(0)
        ds.append(GoiY(x, muc, "Cao" if cao else "Trung bình", d, chenh, tp))
    return sorted(ds, key=lambda g: (-g.diem_phu_hop, g.kha_nang != "Cao", -g.chenh, g.nganh))


def ly_do_chua_du_dk(facts: set[Atom], ctx: DuLieu) -> list[str]:
    """Lý do cho R33: thiếu tổ hợp / điểm liệt / DXT_best chưa đạt ngưỡng (nêu số điểm còn thiếu)."""
    hople = sorted(f[1] for f in facts if f[0] == "tohop_hople")
    liet = sorted(f[1] for f in facts if f[0] == "diem_liet")
    if not hople:
        if ctx.pt == "DGNL":
            return ["Thiếu tổ hợp: chưa đủ điểm ĐGNL môn Toán, học bạ Tiếng Anh và học bạ môn thứ hai "
                    "(không luật R5–R8 nào được kích hoạt)."]
        return ["Thiếu tổ hợp: chưa có đủ điểm thi Toán, Tiếng Anh và một môn Vật lý / Sinh học / Hóa học / "
                "Tin học (không luật R1–R4 nào được kích hoạt)."]
    out = []
    if liet:
        out.append("Điểm liệt (R9): tổ hợp " + ", ".join(
            f"{t} có môn {min(TO_HOP[t], key=lambda m: ctx.diem_mon(m))} = {ctx.diem_thap_nhat(t)} ≤ 1.0"
            for t in liet) + " bị loại.")
    best = dxt_best(facts, ctx)
    if best:
        t, d = best
        x_min = min(MA_NGANH, key=lambda x: nguong(x, ctx.pt))
        n = nguong(x_min, ctx.pt)
        out.append(f"DXT_best = {d} (tổ hợp {t}) chưa đạt ngưỡng thấp nhất {n} ({ten_nganh(x_min)}) "
                   f"– còn thiếu {n - d} điểm.")
    return out


def tu_van(user: dict) -> KetQuaTuVan:
    kq = KetQuaTuVan(user)
    kq.loi = kiem_tra_dau_vao(user)
    for k in ("phuong_thuc", "khu_vuc"):
        if not user.get(k):
            kq.loi.append(f"Thiếu {CAU_HOI[k]}.")
    if kq.loi:
        return kq
    ctx = DuLieu(user)
    fwd = forward_chaining(build_facts(user), ctx)
    kq.forward = fwd
    kq.diem = bang_diem(fwd.facts, ctx)
    kq.best = dxt_best(fwd.facts, ctx)
    kq.du_dk = [f[1] for f in fwd.tat_ca("du_dk")]
    kq.goi_y = xep_hang(fwd, ctx)
    chua_khop = [f[1] for f in fwd.tat_ca("du_dk_chua_khop")]
    if chua_khop:                                                           # R32
        kq.thong_bao.append("Em **đủ điều kiện** xét tuyển ngành " + ", ".join(map(ten_nganh, chua_khop)) +
                            " nhưng các ngành này **chưa khớp sở thích hoặc định hướng nghề nghiệp** em đã chọn (R32).")
    if fwd.co("chua_du_dk"):                                                # R33
        kq.thong_bao.append("Em **chưa đủ điều kiện xét tuyển** ngành nào (R33). Lý do:\n" +
                            "\n".join(f"  - {s}" for s in ly_do_chua_du_dk(fwd.facts, ctx)))
    return kq


# =========================================================================== SUY DIỄN LÙI → đủ điều kiện?
@dataclass
class KetQuaKiemTra:
    user: dict
    nganh: str
    loi: list[str] = field(default_factory=list)
    lui: BackwardResult | None = None
    kha_nang: BackwardResult | None = None      # suy diễn lùi thứ hai: kha_nang_cao(X)?
    best: tuple[str, Decimal] | None = None

    @property
    def ket_luan(self) -> str:
        return "Lỗi dữ liệu" if self.loi else self.lui.ket_luan

    @property
    def can_hoi(self) -> list[str]:
        """Dữ kiện còn thiếu theo thứ tự nên hỏi."""
        if not self.lui:
            return []
        uu_tien = list(CAU_HOI)
        return sorted(self.lui.missing, key=lambda k: uu_tien.index(k) if k in uu_tien else 99)


def kiem_tra_dieu_kien(user: dict, nganh: str) -> KetQuaKiemTra:
    """Trả lời "Em có đủ điều kiện xét tuyển ngành X không?" bằng suy diễn lùi với mục tiêu du_dk(X)."""
    kq = KetQuaKiemTra(user, nganh, kiem_tra_dau_vao(user))
    if kq.loi:
        return kq
    ctx = DuLieu(user)
    facts = build_facts(user)
    kq.lui = backward_chaining(("du_dk", nganh), facts, ctx)
    if kq.lui.proven:
        kq.kha_nang = backward_chaining(("kha_nang_cao", nganh), facts, ctx)
        # DXT_best để hiển thị: các tổ hợp chứng minh được tohop_xet(T) (cũng bằng suy diễn lùi)
        dung_duoc = [t for t in TO_HOP if backward_chaining(("tohop_xet", t), facts, ctx).proven]
        kq.best = max(((t, ctx.dxt(t)) for t in dung_duoc), key=lambda p: p[1]) if dung_duoc else None
    return kq


# =========================================================================== định dạng câu trả lời
def _nguon() -> str:
    s = f"📌 Nguồn: {NGUON['nguon']} – năm áp dụng {NAM}"
    if not NGUON.get("da_kiem_chung"):
        s += "  \n⚠️ Số liệu chưa đối chiếu với Đề án tuyển sinh chính thức."
    return s


def _bang_diem_md(rows: list[DongDiem], pt: str) -> str:
    if not rows:
        return ""
    nhan = "Tổ hợp" if pt == "THPT" else "Tổ hợp (ĐGNL)"
    out = [f"| {nhan} | Tổng (chưa ưu tiên) | Điểm ưu tiên | DXT | Ghi chú |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.to_hop} | {lam_tron(r.tong)} | {lam_tron(r.dut)} | **{r.dxt}** | "
                   f"{'❌ điểm liệt – loại' if r.diem_liet else '✔ dùng được'} |")
    return "\n".join(out)


def markdown_tu_van(kq: KetQuaTuVan, vet: bool = True) -> str:
    if kq.loi:
        return "Thông tin chưa hợp lệ:\n" + "\n".join(f"- {s}" for s in kq.loi)
    pt = kq.user["phuong_thuc"]
    parts = [f"**Phương thức:** {pt} · **Khu vực:** {kq.user['khu_vuc']}"]
    if kq.diem:
        parts.append(_bang_diem_md(kq.diem, pt))
    if kq.best:
        parts.append(f"→ **DXT_best = {kq.best[1]}** (tổ hợp {kq.best[0]})")
    if kq.goi_y:
        rows = ["| # | Ngành | Mức gợi ý | Khả năng trúng tuyển | Điểm phù hợp | DXT_best − điểm chuẩn |",
                "|---|---|---|---|---|---|"]
        for i, g in enumerate(kq.goi_y, 1):
            rows.append(f"| {i} | {ten_nganh(g.nganh)} | **{g.ten_muc}** | {g.kha_nang} | {g.diem_phu_hop} | "
                        f"{g.chenh:+} |")
        parts.append("**Ngành gợi ý (suy diễn tiến):**\n\n" + "\n".join(rows))
        vi_sao = []
        for g in kq.goi_y:                  # luật nào đã sinh ra goi_y(X, mức)
            lan = next(f for f in kq.forward.fired if f.ket_luan == ("goi_y", g.nganh, g.muc))
            tp = g.thanh_phan
            vi_sao.append(f"- **{ten_nganh(g.nganh)}** – {lan.rule.id}: {lan.rule.giai_thich}. Điểm phù hợp = "
                          f"sở thích {tp['nhom_nganh']} + khớp mạnh {tp['khop_manh']} + nghề {tp['nghe_phu_hop']} "
                          f"+ khả năng Cao {tp['kha_nang_cao']} = **{g.diem_phu_hop}**")
        parts.append("**Vì sao?**\n" + "\n".join(vi_sao))
    parts += kq.thong_bao
    if kq.du_dk and kq.goi_y:
        parts.append("Các ngành em đủ điều kiện xét tuyển: " + ", ".join(map(ten_nganh, kq.du_dk)) + ".")
    if vet and kq.forward:
        parts.append("**Vết suy diễn tiến** (luật kích hoạt theo thứ tự):\n" +
                     "\n".join(f"{i}. {f}" for i, f in enumerate(kq.forward.fired, 1)))
    parts.append(_nguon())
    return "\n\n".join(parts)


def markdown_kiem_tra(kq: KetQuaKiemTra, vet: bool = False) -> str:
    ten = ten_nganh(kq.nganh)
    if kq.loi:
        return "Thông tin chưa hợp lệ:\n" + "\n".join(f"- {s}" for s in kq.loi)
    b = kq.lui
    parts = [f"**Mục tiêu suy diễn lùi:** `{b.goal}` – em có đủ điều kiện xét tuyển ngành **{ten}** không?"]
    if b.proven:
        parts.append(f"✅ **Có.** Em đủ điều kiện xét tuyển ngành {ten} (đạt ngưỡng đảm bảo chất lượng đầu vào "
                     f"{nguong(kq.nganh, kq.user['phuong_thuc'])}).")
        if kq.kha_nang is not None:
            dc = diem_chuan(kq.nganh, kq.user["phuong_thuc"])
            muc = "**Cao**" if kq.kha_nang.proven else "**Trung bình**"
            them = f" (DXT_best = {kq.best[1]}, tổ hợp {kq.best[0]})" if kq.best else ""
            parts.append(f"Khả năng trúng tuyển: {muc} – so với điểm chuẩn {NAM} là {dc}{them}.")
        parts.append("**Chuỗi luật đã dùng:**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(b.proof, 1)))
    elif b.missing:
        parts.append("❔ **Chưa đủ thông tin** để kết luận. Mình cần biết thêm: " +
                     ", ".join(CAU_HOI.get(k, k) for k in kq.can_hoi) + ".")
    else:
        parts.append(f"❌ **Không.** Em chưa đủ điều kiện xét tuyển ngành {ten}.")
        if b.thieu_diem is not None:
            parts.append(f"Còn thiếu **{b.thieu_diem}** điểm so với ngưỡng.")
        if b.that_bai:
            parts.append("**Điều kiện thất bại:**\n" + "\n".join(f"- {s}" for s in b.that_bai))
    if vet:
        parts.append("**Nhật ký suy diễn lùi:**\n```\n" + "\n".join(b.trace) + "\n```")
    parts.append(_nguon())
    return "\n\n".join(parts)
