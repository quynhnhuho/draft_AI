"""Chức năng 2 - Kiểm thử hệ luật, suy diễn tiến và suy diễn lùi.

Ghi chú hỗ trợ AI: mã nguồn được viết với sự hỗ trợ của trợ lý AI Claude (Anthropic); nhóm đã kiểm tra lại.
"""
import random
from decimal import Decimal

import pytest

from src.advisor.diem import diem_uu_tien, diem_xet_tuyen, lam_tron, tong_dgnl
from src.advisor.facts import DuLieu, build_facts, kiem_tra_dau_vao
from src.advisor.inference import backward_chaining, forward_chaining
from src.advisor.kich_ban import KICH_BAN, KHONG_THI, SO_THICH_TRONG
from src.advisor.rules import RULES, A, Not, R, kiem_tra_luat, phan_tang
from src.advisor.tham_so import DINH_HUONG, MA_NGANH, MON, MON_HB2, SO_THICH, TO_HOP
from src.advisor.tu_van import kiem_tra_dieu_kien, tu_van

KB = {ma: (user, nganh) for ma, _, user, nganh in KICH_BAN}


def chay_tien(user):
    return forward_chaining(build_facts(user), DuLieu(user))


def test_33_luat_va_phan_tang():
    assert [r.id for r in RULES] == [f"R{i}" for i in range(1, 34)]
    t = phan_tang(RULES)
    assert t["tohop_hople"] == t["diem_liet"] == 0
    assert t["tohop_xet"] > t["diem_liet"]
    assert t["kha_nang_tb"] > t["kha_nang_cao"]
    assert t["du_dk_chua_khop"] > t["goi_y"] and t["chua_du_dk"] > t["du_dk"]


def test_phat_hien_vong_phu_dinh_va_luat_khong_an_toan():
    with pytest.raises(ValueError):
        phan_tang([R("X1", "a", ["q", Not(A("p"))], "p", "")])
    with pytest.raises(ValueError):
        kiem_tra_luat([R("X2", "a", ["q"], "p(?X)", "")])


def test_diem_uu_tien_hai_nhanh():
    assert diem_uu_tien(Decimal("19.2"), "KV1") == Decimal("0.75")
    assert diem_uu_tien(Decimal("24"), "KV1") == Decimal("0.6")
    assert diem_uu_tien(Decimal("22.5"), "KV2") == Decimal("0.25")
    assert diem_xet_tuyen(Decimal("19.2"), "KV1") == Decimal("19.95")


def test_dgnl_va_lam_tron():
    tong = tong_dgnl(7.5, 8, 8.5)
    assert tong == Decimal("23.475")
    assert diem_xet_tuyen(tong, "KV2") == Decimal("23.69")
    assert lam_tron(Decimal("20.125")) == Decimal("20.13")


def test_kiem_tra_dau_vao():
    assert kiem_tra_dau_vao({"diem_Toan": 10.5})
    assert kiem_tra_dau_vao({"mon_hb2": "Van"})
    assert kiem_tra_dau_vao({"phuong_thuc": "THPT", "khu_vuc": "KV2", "diem_Toan": 0, "diem_Ly": None}) == []


def test_kb1_suy_dien_tien():
    r = chay_tien(KB["KB1"][0])
    assert r.recommendations == [("CNTT", "phu_hop")]
    assert r.co("du_dk(CNGD)") and not r.co("du_dk(SPTH)")
    assert [f.rule.id for f in r.fired][:1] == ["R1"]


def test_kb2_xep_hang():
    kq = tu_van(KB["KB2"][0])
    assert [(g.nganh, g.muc, g.kha_nang) for g in kq.goi_y] == [("SPTH", "rat_phu_hop", "Cao"),
                                                               ("CNTT", "phu_hop", "Cao")]
    assert kq.goi_y[0].diem_phu_hop == 5 and kq.best == ("X26", Decimal("26.30"))


def test_kb3_dgnl():
    kq = tu_van(KB["KB3"][0])
    assert kq.goi_y[0].nganh == "CNGD" and kq.goi_y[0].muc == "rat_phu_hop"
    assert kq.forward.co("kha_nang_tb(SPTH)")


def test_kb4_diem_liet_r33():
    kq = tu_van(KB["KB4"][0])
    assert kq.forward.co("chua_du_dk") and not kq.du_dk
    assert any("Điểm liệt" in t for t in kq.thong_bao)


def test_diem_liet_chi_ap_dung_thpt():
    u = {"phuong_thuc": "DGNL", "khu_vuc": "KV3", "diem_dgnl": 9, "HB_mon1": 1.0, "HB_mon2": 9, "mon_hb2": "Ly"}
    assert chay_tien(u).co("tohop_xet(A01)")


def test_kb5_r32_va_kb6_r33_thieu_to_hop():
    r5 = chay_tien(KB["KB5"][0])
    assert {f[1] for f in r5.tat_ca("du_dk_chua_khop")} == set(MA_NGANH) and not r5.recommendations
    kq6 = tu_van(KB["KB6"][0])
    assert kq6.forward.co("chua_du_dk") and "Thiếu tổ hợp" in kq6.thong_bao[0]


def test_kb7_tham_khao():
    kq = tu_van(KB["KB7"][0])
    assert ("CNTT", "tham_khao") in kq.forward.recommendations
    assert kq.goi_y[0].nganh == "CNTT"


def test_thieu_khu_vuc_bao_loi():
    assert tu_van({"phuong_thuc": "THPT"}).loi


def test_lui_khong_va_so_diem_thieu():
    kq = kiem_tra_dieu_kien(KB["KB1"][0], "SPTH")
    assert kq.ket_luan == "Không" and kq.lui.thieu_diem == Decimal("0.05")


def test_lui_co_va_chuoi_luat():
    kq = kiem_tra_dieu_kien(KB["KB1"][0], "CNTT")
    assert kq.ket_luan == "Có"
    assert [b.rule.id for b in kq.lui.proof] == ["R1", "R10", "R11"]
    assert kq.kha_nang.proven


def test_lui_diem_liet_giai_thich():
    kq = kiem_tra_dieu_kien(KB["KB4"][0], "CNTT")
    assert kq.ket_luan == "Không" and any("diem_liet(A01) đúng" in s for s in kq.lui.that_bai)


def test_lui_hoi_lan_luot_du_kien_thieu():
    assert kiem_tra_dieu_kien({}, "CNTT").can_hoi == ["phuong_thuc"]
    assert kiem_tra_dieu_kien({"phuong_thuc": "THPT"}, "CNTT").can_hoi == ["diem_Toan"]
    u = {"phuong_thuc": "THPT", **KHONG_THI, "diem_Toan": 8, "diem_Anh": 7, "diem_Ly": 7}
    assert kiem_tra_dieu_kien(u, "CNTT").can_hoi == ["khu_vuc"]
    assert kiem_tra_dieu_kien({"phuong_thuc": "DGNL"}, "CNTT").can_hoi == ["diem_dgnl"]


def _nguoi_dung_ngau_nhien(rng: random.Random) -> dict:
    pt = rng.choice(["THPT", "DGNL"])
    u = {"phuong_thuc": pt, "khu_vuc": rng.choice(["KV1", "KV2-NT", "KV2", "KV3"])}

    def diem():
        return round(rng.choice([rng.uniform(0, 1.2), rng.uniform(4, 10)]), 2)

    if pt == "THPT":
        u |= {f"diem_{m}": (diem() if rng.random() < 0.7 else None) for m in MON}
    else:
        u |= {"diem_dgnl": diem(), "HB_mon1": diem(), "HB_mon2": diem(), "mon_hb2": rng.choice(MON_HB2)}
    u |= {k: rng.random() < 0.4 for k in SO_THICH + DINH_HUONG}
    return u


def test_tien_va_lui_cho_cung_ket_qua():
    rng = random.Random(2026)
    muc_tieu = ([("du_dk", x) for x in MA_NGANH] + [("kha_nang_cao", x) for x in MA_NGANH] +
                [("kha_nang_tb", x) for x in MA_NGANH] + [("tohop_xet", t) for t in TO_HOP] +
                [("goi_y", x, m) for x in MA_NGANH for m in ("rat_phu_hop", "phu_hop", "tham_khao")] +
                [("du_dk_chua_khop", x) for x in MA_NGANH] + [("chua_du_dk",)])
    for _ in range(300):
        u = _nguoi_dung_ngau_nhien(rng)
        f, ctx = build_facts(u), DuLieu(u)
        tien = forward_chaining(f, ctx).facts
        for g in muc_tieu:
            lui = backward_chaining(g, f, ctx)
            assert lui.proven == (g in tien), (u, g)
            assert not lui.missing, (u, g)


def test_lui_nhanh_anh_em_khong_bi_bo_qua():
    u = {"phuong_thuc": "DGNL", "khu_vuc": "KV3", "diem_dgnl": 9, "HB_mon1": 6, "HB_mon2": 6, "mon_hb2": "Tin"}
    f, ctx = build_facts(u), DuLieu(u)
    assert backward_chaining("kha_nang_cao(CNTT)", f, ctx).proven
    assert not backward_chaining("kha_nang_tb(CNTT)", f, ctx).proven


def test_lui_dung_khi_luat_de_quy():
    luat = [R("X1", "a", ["q"], "p", ""), R("X2", "a", ["p"], "q", "")]
    kq = backward_chaining("p", set(), DuLieu(), luat)
    assert not kq.proven and any("lặp lại" in s for s in kq.trace)
