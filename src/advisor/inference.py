"""C2 – Suy diễn tiến và suy diễn lùi (Phần 3 – 3.4). TỰ CÀI ĐẶT, không dùng thư viện suy diễn.

    forward_chaining(facts, ctx)          → danh sách ngành gợi ý (goi_y) + vết các luật đã kích hoạt
    backward_chaining("du_dk(CNTT)", ...) → "Em có đủ điều kiện xét tuyển ngành X không?"
                                            Có / Không / Chưa đủ thông tin (+ dữ kiện còn thiếu cần hỏi)

facts : tập sự kiện ban đầu (facts.build_facts)
ctx   : DuLieu – giá trị số của người dùng, dùng cho điều kiện số (Test) và để biết dữ kiện nào chưa hỏi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import count
from typing import Iterator

from .facts import DuLieu, khoa_can_hoi
from .rules import (RULES, A, Atom, Not, Rule, Test, fmt, gia_tri, la_bien, phan_tang, thay_the)


# =========================================================================== khớp mẫu / hợp nhất
def hop_nhat(a: Atom, b: Atom, theta: dict) -> dict | None:
    """Hợp nhất (unification) hai atom có thể chứa biến. Trả về phép gán mở rộng hoặc None."""
    if a[0] != b[0] or len(a) != len(b):
        return None
    theta = dict(theta)
    for x, y in zip(a[1:], b[1:]):
        x, y = gia_tri(x, theta), gia_tri(y, theta)
        if x == y:
            continue
        if la_bien(x):
            theta[x] = y
        elif la_bien(y):
            theta[y] = x
        else:
            return None
    return theta


def _cung_vi_tu(facts: set[Atom], a: Atom) -> list[Atom]:
    return [f for f in facts if f[0] == a[0] and len(f) == len(a)]


def _mo_ta_dk(c, ctx: DuLieu, theta: dict) -> str:
    """Điều kiện sau khi thay biến – dùng để ghi vết/giải thích."""
    if isinstance(c, tuple):
        return fmt(thay_the(c, theta))
    if isinstance(c, Not):
        return "¬" + fmt(thay_the(c.atom, theta))
    return c.kiem_tra(ctx, theta).mo_ta


# =========================================================================== SUY DIỄN TIẾN
@dataclass(frozen=True)
class LanKichHoat:
    """Một lần luật được kích hoạt: luật nào, với phép gán nào, sinh ra sự kiện gì, ở tầng/vòng nào."""
    rule: Rule
    theta: dict
    ket_luan: Atom
    dieu_kien: tuple[str, ...]     # điều kiện đã thay biến (điều kiện số kèm giá trị)
    tang: int
    vong: int

    def __str__(self) -> str:
        return f"{self.rule.id}: {' ∧ '.join(self.dieu_kien) or '(không điều kiện)'} ⇒ {fmt(self.ket_luan)}"


@dataclass
class ForwardResult:
    facts: set[Atom]
    fired: list[LanKichHoat] = field(default_factory=list)

    def co(self, s: str) -> bool:
        return A(s) in self.facts

    def tat_ca(self, vi_tu: str) -> list[Atom]:
        return sorted(f for f in self.facts if f[0] == vi_tu)

    @property
    def recommendations(self) -> list[tuple[str, str]]:
        """[(ngành, mức)] từ các sự kiện goi_y(X, mức)."""
        return [(f[1], f[2]) for f in self.tat_ca("goi_y")]


def _khop(dieu_kien: tuple, wm: set[Atom], ctx: DuLieu, theta: dict) -> Iterator[dict]:
    """Liệt kê mọi phép gán làm TẤT CẢ điều kiện đúng trên bộ nhớ làm việc wm (duyệt trái → phải)."""
    if not dieu_kien:
        yield theta
        return
    c, con_lai = dieu_kien[0], dieu_kien[1:]
    if isinstance(c, tuple):                                        # atom dương
        for f in _cung_vi_tu(wm, c):
            th = hop_nhat(c, f, theta)
            if th is not None:
                yield from _khop(con_lai, wm, ctx, th)
    elif isinstance(c, Not):                                        # ¬atom: không có sự kiện nào khớp
        if not any(hop_nhat(c.atom, f, theta) is not None for f in _cung_vi_tu(wm, c.atom)):
            yield from _khop(con_lai, wm, ctx, theta)
    elif isinstance(c, Test):                                       # điều kiện số
        if all(ctx.biet(k) for k in c.can) and c.kiem_tra(ctx, theta).dung:
            yield from _khop(con_lai, wm, ctx, theta)


def forward_chaining(facts: set[Atom], ctx: DuLieu | None = None, rules: list[Rule] = RULES) -> ForwardResult:
    """Suy diễn tiến theo tầng.

    Với từng tầng (tăng dần): lặp – duyệt các luật của tầng theo thứ tự, tìm mọi phép gán thỏa điều kiện,
    kết luận chưa có trong bộ nhớ → thêm vào và ghi vết. Dừng tầng khi một vòng không sinh sự kiện mới.
    Chạy theo tầng bảo đảm khi luật dùng ¬p thì p đã được suy ra đầy đủ ở tầng thấp hơn.
    """
    ctx = ctx or DuLieu()
    wm = set(facts)
    ket_qua = ForwardResult(wm)
    tang_vi_tu = phan_tang(rules)
    for tang in sorted({tang_vi_tu[r.ket_luan[0]] for r in rules}):
        luat_tang = [r for r in rules if tang_vi_tu[r.ket_luan[0]] == tang]
        for vong in count(1):
            moi = False
            for r in luat_tang:
                for theta in list(_khop(r.dieu_kien, wm, ctx, {})):   # list(): không sửa wm khi đang duyệt
                    kl = thay_the(r.ket_luan, theta)
                    if kl not in wm:
                        wm.add(kl)
                        moi = True
                        ket_qua.fired.append(LanKichHoat(r, theta, kl, tuple(_mo_ta_dk(c, ctx, theta)
                                                                             for c in r.dieu_kien), tang, vong))
            if not moi:
                break
    return ket_qua


# =========================================================================== SUY DIỄN LÙI
@dataclass(frozen=True)
class BuocChungMinh:
    rule: Rule
    ket_luan: Atom
    dieu_kien: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.rule.id}: {' ∧ '.join(self.dieu_kien)} ⇒ {fmt(self.ket_luan)}"


@dataclass
class BackwardResult:
    goal: str
    proven: bool
    trace: list[str] = field(default_factory=list)          # nhật ký tìm kiếm (thụt lề theo độ sâu)
    missing: set[str] = field(default_factory=set)          # dữ kiện đầu vào chưa biết → chatbot hỏi tiếp
    proof: list[BuocChungMinh] = field(default_factory=list)  # chuỗi luật đã dùng (nếu chứng minh được)
    that_bai: list[str] = field(default_factory=list)       # điều kiện thất bại (để giải thích khi "Không")
    thieu_diem: Decimal | None = None                       # số điểm còn thiếu ít nhất (nếu có)

    @property
    def ket_luan(self) -> str:
        if self.proven:
            return "Có"
        return "Chưa đủ thông tin" if self.missing else "Không"


class _SuyDienLui:
    """Tìm kiếm theo chiều sâu kiểu Prolog: mỗi hàm trả về generator các phép gán làm mục tiêu đúng."""

    def __init__(self, facts: set[Atom], ctx: DuLieu, rules: list[Rule]):
        self.facts, self.ctx, self.rules = facts, ctx, rules
        self.co_luat = {r.ket_luan[0] for r in rules}       # vị từ suy ra được (không phải lá)
        self.trace: list[str] = []
        self.missing: set[str] = set()
        self.that_bai: list[str] = []
        self.thieu: list[Decimal] = []
        self.trong_phu_dinh = 0                             # >0: đang chứng minh atom bên trong một ¬
        self._dem = count(1)

    def log(self, sau: int, s: str) -> None:
        self.trace.append("  " * sau + s)

    def _ghi_that_bai(self, s: str) -> None:
        # Thất bại BÊN TRONG phép phủ định là điều mong muốn (vd. tổ hợp không bị điểm liệt) → không ghi.
        s = re.sub(r"(\?[A-Za-z]+)_\d+", r"\1", s)          # ?T_3 → ?T cho dễ đọc
        if not self.trong_phu_dinh and s not in self.that_bai:
            self.that_bai.append(s)

    def _doi_ten(self, r: Rule) -> tuple[tuple, Atom]:
        """Đổi tên biến của luật (?T → ?T_7) để không trùng biến của mục tiêu đang chứng minh."""
        n = next(self._dem)
        ten = lambda x: f"{x}_{n}" if la_bien(x) else x  # noqa: E731
        doi = lambda a: (a[0],) + tuple(ten(x) for x in a[1:])  # noqa: E731
        dk = []
        for c in r.dieu_kien:
            if isinstance(c, tuple):
                dk.append(doi(c))
            elif isinstance(c, Not):
                dk.append(Not(doi(c.atom)))
            else:
                dk.append(type(c)(**{k: ten(v) if isinstance(v, str) else v for k, v in c.__dict__.items()}))
        return tuple(dk), doi(r.ket_luan)

    # ------------------------------------------------------------------ một mục tiêu
    def chung_minh(self, muc_tieu: Atom, theta: dict, sau: int,
                   to_tien: frozenset = frozenset()) -> Iterator[tuple[dict, list[BuocChungMinh]]]:
        """to_tien = các mục tiêu tổ tiên trên nhánh chứng minh hiện tại (chống lặp vô hạn).
        Truyền theo tham số chứ không giữ trong đối tượng: generator đang tạm dừng (chờ quay lui) không được
        coi là "đang xét", nếu không nhánh anh em sẽ bị bỏ qua sai."""
        g = thay_the(muc_tieu, theta)
        self.log(sau, f"? {fmt(g)}")
        tim_thay = False

        for f in _cung_vi_tu(self.facts, g):                    # 1) có sẵn trong dữ kiện
            th = hop_nhat(g, f, theta)
            if th is not None:
                tim_thay = True
                self.log(sau + 1, f"✔ {fmt(f)} – dữ kiện")
                yield th, []

        if g in to_tien:                                        # 2) chống lặp: g là tổ tiên của chính nó
            self.log(sau + 1, "↺ mục tiêu lặp lại trên cùng nhánh – bỏ qua")
            return
        to_tien = to_tien | {g}
        for r in self.rules:                                    # 3) các luật có kết luận khớp mục tiêu
            if r.ket_luan[0] != g[0]:
                continue
            dk, kl = self._doi_ten(r)
            th = hop_nhat(kl, g, theta)
            if th is None:
                continue
            self.log(sau + 1, f"thử {r}")
            dung = False
            for th2, buoc in self.chung_minh_ds(dk, th, sau + 2, r, to_tien):
                dung = tim_thay = True
                kl_cu_the = thay_the(kl, th2)
                self.log(sau + 1, f"✔ {r.id} ⇒ {fmt(kl_cu_the)}")
                yield th2, buoc + [BuocChungMinh(r, kl_cu_the, tuple(_mo_ta_dk(c, self.ctx, th2) for c in dk))]
            if not dung:
                self.log(sau + 1, f"✘ {r.id} không thỏa")

        if not tim_thay and g[0] not in self.co_luat:           # 4) lá: hỏi người dùng nếu chưa biết
            khoa = khoa_can_hoi(g)
            if khoa and not self.ctx.biet(khoa):
                self.missing.add(khoa)
                self.log(sau + 1, f"? chưa biết – cần hỏi '{khoa}'")
            else:
                self.log(sau + 1, "✘ không có trong dữ kiện")

    # ------------------------------------------------------------------ danh sách điều kiện (phép AND)
    def chung_minh_ds(self, dieu_kien: tuple, theta: dict, sau: int, r: Rule,
                      to_tien: frozenset = frozenset()) -> Iterator[tuple[dict, list[BuocChungMinh]]]:
        if not dieu_kien:
            yield theta, []
            return
        c, con_lai = dieu_kien[0], dieu_kien[1:]

        if isinstance(c, tuple):
            co_nghiem = False
            for th, b1 in self.chung_minh(c, theta, sau, to_tien):
                co_nghiem = True
                for th2, b2 in self.chung_minh_ds(con_lai, th, sau, r, to_tien):
                    yield th2, b1 + b2
                if con_lai:                                     # các điều kiện sau thất bại → thử nghiệm khác
                    self.log(sau, f"↩ quay lui: tìm cách khác thỏa {fmt(thay_the(c, theta))}")
            if not co_nghiem and c[0] in self.co_luat:
                self._ghi_that_bai(f"{r.id}: không chứng minh được {fmt(thay_the(c, theta))}")

        elif isinstance(c, Not):                                # phủ định: thử chứng minh atom bên trong
            p = thay_the(c.atom, theta)
            truoc = len(self.missing)
            self.log(sau, f"? ¬{fmt(p)}")
            self.trong_phu_dinh += 1
            try:
                nghiem = next(self.chung_minh(p, theta, sau + 1, to_tien), None)
            finally:
                self.trong_phu_dinh -= 1
            if nghiem is not None:
                dung = fmt(thay_the(p, nghiem[0]))
                vi = f" vì {nghiem[1][-1].dieu_kien[-1]}" if nghiem[1] else ""
                self.log(sau, f"✘ ¬{fmt(p)} sai vì {dung} đúng")
                self._ghi_that_bai(f"{r.id}: {dung} đúng{vi}")
            elif len(self.missing) > truoc:
                self.log(sau, f"? ¬{fmt(p)} chưa xác định được (thiếu dữ kiện)")
            else:
                self.log(sau, f"✔ ¬{fmt(p)}")
                yield from self.chung_minh_ds(con_lai, theta, sau, r, to_tien)

        else:                                                   # điều kiện số
            thieu = [k for k in c.can if not self.ctx.biet(k)]
            if thieu:
                self.missing.update(thieu)
                self.log(sau, f"? {c} – cần hỏi {', '.join(thieu)}")
                return
            kq = c.kiem_tra(self.ctx, theta)
            if kq.dung:
                self.log(sau, f"✔ {kq.mo_ta}")
                yield from self.chung_minh_ds(con_lai, theta, sau, r, to_tien)
            else:
                self.log(sau, f"✘ {kq.mo_ta}")
                self._ghi_that_bai(f"{r.id}: {kq.mo_ta}")
                if kq.thieu is not None and not self.trong_phu_dinh:
                    self.thieu.append(kq.thieu)


def backward_chaining(goal: str | Atom, facts: set[Atom], ctx: DuLieu | None = None,
                      rules: list[Rule] = RULES) -> BackwardResult:
    """Chứng minh `goal` (vd "du_dk(CNTT)") bằng suy diễn lùi.

    - goal có trong dữ kiện → đúng.
    - Thử lần lượt các luật có kết luận khớp goal; chứng minh đệ quy từng điều kiện từ trái sang phải
      (atom → đệ quy; ¬atom → atom không chứng minh được; điều kiện số → gọi hàm tính).
    - Dừng ở chứng minh ĐẦU TIÊN tìm được. Sự kiện lá chưa biết → đưa vào `missing` để chatbot hỏi tiếp.
    """
    ctx = ctx or DuLieu()
    g = A(goal) if isinstance(goal, str) else goal
    sd = _SuyDienLui(set(facts), ctx, rules)
    nghiem = next(sd.chung_minh(g, {}, 0), None)
    kq = BackwardResult(fmt(g), nghiem is not None, sd.trace, sd.missing)
    if nghiem is not None:
        kq.proof = nghiem[1]
        kq.missing = set()          # đã chứng minh được thì không cần hỏi thêm
    else:
        kq.that_bai = sd.that_bai
        kq.thieu_diem = min(sd.thieu) if sd.thieu else None
    return kq
