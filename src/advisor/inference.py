"""Chức năng 2 - Suy diễn tiến và suy diễn lùi (tự cài đặt).

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import count
from typing import Iterator

from .facts import DuLieu, khoa_can_hoi
from .rules import RULES, A, Atom, Not, Rule, Test, fmt, gia_tri, la_bien, phan_tang, thay_the

Loi_giai = Iterator[tuple[dict, list["BuocChungMinh"]]]


def hop_nhat(a: Atom, b: Atom, theta: dict) -> dict | None:
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
    if isinstance(c, tuple):
        return fmt(thay_the(c, theta))
    if isinstance(c, Not):
        return "¬" + fmt(thay_the(c.atom, theta))
    return c.kiem_tra(ctx, theta).mo_ta


@dataclass(frozen=True)
class LanKichHoat:
    rule: Rule
    theta: dict
    ket_luan: Atom
    dieu_kien: tuple[str, ...]
    tang: int
    vong: int

    def __str__(self) -> str:
        return f"{self.rule.id}: {' ∧ '.join(self.dieu_kien)} ⇒ {fmt(self.ket_luan)}"


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
        return [(f[1], f[2]) for f in self.tat_ca("goi_y")]


def _khop(dieu_kien: tuple, wm: set[Atom], ctx: DuLieu, theta: dict) -> Iterator[dict]:
    if not dieu_kien:
        yield theta
        return
    c, con_lai = dieu_kien[0], dieu_kien[1:]
    if isinstance(c, tuple):
        for f in _cung_vi_tu(wm, c):
            th = hop_nhat(c, f, theta)
            if th is not None:
                yield from _khop(con_lai, wm, ctx, th)
    elif isinstance(c, Not):
        if not any(hop_nhat(c.atom, f, theta) is not None for f in _cung_vi_tu(wm, c.atom)):
            yield from _khop(con_lai, wm, ctx, theta)
    elif all(ctx.biet(k) for k in c.can) and c.kiem_tra(ctx, theta).dung:
        yield from _khop(con_lai, wm, ctx, theta)


def forward_chaining(facts: set[Atom], ctx: DuLieu | None = None, rules: list[Rule] = RULES) -> ForwardResult:
    ctx = ctx or DuLieu()
    ket_qua = ForwardResult(set(facts))
    wm = ket_qua.facts
    tang_vi_tu = phan_tang(rules)
    for tang in sorted({tang_vi_tu[r.ket_luan[0]] for r in rules}):
        luat_tang = [r for r in rules if tang_vi_tu[r.ket_luan[0]] == tang]
        for vong in count(1):
            moi = False
            for r in luat_tang:
                for theta in list(_khop(r.dieu_kien, wm, ctx, {})):
                    kl = thay_the(r.ket_luan, theta)
                    if kl in wm:
                        continue
                    wm.add(kl)
                    moi = True
                    dk = tuple(_mo_ta_dk(c, ctx, theta) for c in r.dieu_kien)
                    ket_qua.fired.append(LanKichHoat(r, theta, kl, dk, tang, vong))
            if not moi:
                break
    return ket_qua


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
    trace: list[str] = field(default_factory=list)
    missing: set[str] = field(default_factory=set)
    proof: list[BuocChungMinh] = field(default_factory=list)
    that_bai: list[str] = field(default_factory=list)
    thieu_diem: Decimal | None = None

    @property
    def ket_luan(self) -> str:
        if self.proven:
            return "Có"
        return "Chưa đủ thông tin" if self.missing else "Không"


class _SuyDienLui:
    def __init__(self, facts: set[Atom], ctx: DuLieu, rules: list[Rule]):
        self.facts, self.ctx, self.rules = facts, ctx, rules
        self.co_luat = {r.ket_luan[0] for r in rules}
        self.trace: list[str] = []
        self.missing: set[str] = set()
        self.that_bai: list[str] = []
        self.thieu: list[Decimal] = []
        self.trong_phu_dinh = 0
        self._dem = count(1)

    def log(self, sau: int, s: str) -> None:
        self.trace.append("  " * sau + s)

    def _ghi_that_bai(self, s: str) -> None:
        s = re.sub(r"(\?[A-Za-z]+)_\d+", r"\1", s)
        if not self.trong_phu_dinh and s not in self.that_bai:
            self.that_bai.append(s)

    def _doi_ten(self, r: Rule) -> tuple[tuple, Atom]:
        n = next(self._dem)

        def ten(x):
            return f"{x}_{n}" if isinstance(x, str) and la_bien(x) else x

        def doi(a: Atom) -> Atom:
            return (a[0],) + tuple(ten(x) for x in a[1:])

        dk = []
        for c in r.dieu_kien:
            if isinstance(c, tuple):
                dk.append(doi(c))
            elif isinstance(c, Not):
                dk.append(Not(doi(c.atom)))
            else:
                dk.append(type(c)(**{k: ten(v) for k, v in c.__dict__.items()}))
        return tuple(dk), doi(r.ket_luan)

    def chung_minh(self, muc_tieu: Atom, theta: dict, sau: int, to_tien: frozenset = frozenset()) -> Loi_giai:
        g = thay_the(muc_tieu, theta)
        self.log(sau, f"? {fmt(g)}")
        tim_thay = False

        for f in _cung_vi_tu(self.facts, g):
            th = hop_nhat(g, f, theta)
            if th is not None:
                tim_thay = True
                self.log(sau + 1, f"[đúng] {fmt(f)} - dữ kiện")
                yield th, []

        if g in to_tien:
            self.log(sau + 1, "mục tiêu lặp lại trên cùng nhánh - bỏ qua")
            return
        to_tien = to_tien | {g}
        for r in self.rules:
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
                self.log(sau + 1, f"[đúng] {r.id} ⇒ {fmt(kl_cu_the)}")
                yield th2, buoc + [BuocChungMinh(r, kl_cu_the, tuple(_mo_ta_dk(c, self.ctx, th2) for c in dk))]
            if not dung:
                self.log(sau + 1, f"[sai] {r.id} không thỏa")

        if not tim_thay and g[0] not in self.co_luat:
            khoa = khoa_can_hoi(g)
            if khoa and not self.ctx.biet(khoa):
                self.missing.add(khoa)
                self.log(sau + 1, f"chưa biết - cần hỏi '{khoa}'")
            else:
                self.log(sau + 1, "[sai] không có trong dữ kiện")

    def chung_minh_ds(self, dieu_kien: tuple, theta: dict, sau: int, r: Rule,
                      to_tien: frozenset = frozenset()) -> Loi_giai:
        if not dieu_kien:
            yield theta, []
            return
        c, con_lai = dieu_kien[0], dieu_kien[1:]
        if isinstance(c, tuple):
            yield from self._atom(c, con_lai, theta, sau, r, to_tien)
        elif isinstance(c, Not):
            yield from self._phu_dinh(c, con_lai, theta, sau, r, to_tien)
        else:
            yield from self._dieu_kien_so(c, con_lai, theta, sau, r, to_tien)

    def _atom(self, c: Atom, con_lai: tuple, theta: dict, sau: int, r: Rule, to_tien: frozenset) -> Loi_giai:
        co_nghiem = False
        for th, b1 in self.chung_minh(c, theta, sau, to_tien):
            co_nghiem = True
            for th2, b2 in self.chung_minh_ds(con_lai, th, sau, r, to_tien):
                yield th2, b1 + b2
            if con_lai:
                self.log(sau, f"quay lui: tìm cách khác thỏa {fmt(thay_the(c, theta))}")
        if not co_nghiem and c[0] in self.co_luat:
            self._ghi_that_bai(f"{r.id}: không chứng minh được {fmt(thay_the(c, theta))}")

    def _phu_dinh(self, c: Not, con_lai: tuple, theta: dict, sau: int, r: Rule, to_tien: frozenset) -> Loi_giai:
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
            self.log(sau, f"[sai] ¬{fmt(p)} vì {dung} đúng")
            self._ghi_that_bai(f"{r.id}: {dung} đúng{vi}")
        elif len(self.missing) > truoc:
            self.log(sau, f"¬{fmt(p)} chưa xác định được (thiếu dữ kiện)")
        else:
            self.log(sau, f"[đúng] ¬{fmt(p)}")
            yield from self.chung_minh_ds(con_lai, theta, sau, r, to_tien)

    def _dieu_kien_so(self, c: Test, con_lai: tuple, theta: dict, sau: int, r: Rule,
                      to_tien: frozenset) -> Loi_giai:
        thieu = [k for k in c.can if not self.ctx.biet(k)]
        if thieu:
            self.missing.update(thieu)
            self.log(sau, f"{c} - cần hỏi {', '.join(thieu)}")
            return
        kq = c.kiem_tra(self.ctx, theta)
        if kq.dung:
            self.log(sau, f"[đúng] {kq.mo_ta}")
            yield from self.chung_minh_ds(con_lai, theta, sau, r, to_tien)
            return
        self.log(sau, f"[sai] {kq.mo_ta}")
        self._ghi_that_bai(f"{r.id}: {kq.mo_ta}")
        if kq.thieu is not None and not self.trong_phu_dinh:
            self.thieu.append(kq.thieu)


def backward_chaining(goal: str | Atom, facts: set[Atom], ctx: DuLieu | None = None,
                      rules: list[Rule] = RULES) -> BackwardResult:
    ctx = ctx or DuLieu()
    g = A(goal) if isinstance(goal, str) else goal
    sd = _SuyDienLui(set(facts), ctx, rules)
    nghiem = next(sd.chung_minh(g, {}, 0), None)
    if nghiem is not None:
        return BackwardResult(fmt(g), True, sd.trace, set(), nghiem[1])
    return BackwardResult(fmt(g), False, sd.trace, sd.missing, [], sd.that_bai,
                          min(sd.thieu) if sd.thieu else None)
