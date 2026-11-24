"""Đánh giá end-to-end Chức năng 4 trên tests/scenarios.yaml (mục 7.9 HUONG_DAN_DO_AN).

    python -m src.chatbot.danh_gia                         # mô hình chính (Keras) trong models/
    python -m src.chatbot.danh_gia --backend nb            # baseline Naive Bayes
    python -m src.chatbot.danh_gia --out docs/C4_DANH_GIA.md

Chỉ số báo cáo:
  * tỷ lệ đúng ý định (các lượt có ghi "y_dinh")
  * tỷ lệ câu trả lời số liệu CÓ NGUỒN + NĂM (phải 100%)
  * chuyển cán bộ: đúng / chuyển thừa / bỏ sót
  * thời gian phản hồi (trung bình, lớn nhất)
  * số lượt ý định do luật quyết định (luật ưu tiên / luật bổ trợ / chặn phạm vi)
"""
from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..nlu.classifier import load_classifier, load_threshold
from .engine import ChatEngine
from .kb import KnowledgeBase
from .schema import DialogueState

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "tests" / "scenarios.yaml"
_CHUA_DAT = object()


@dataclass
class KetQuaLuot:
    hoi_thoai: str
    noi: str
    y_dinh: str | None
    tin_cay: float
    chuyen_can_bo: bool
    nguon_y_dinh: str | None
    ms: float
    loi: list[str] = field(default_factory=list)
    co_so_lieu: bool = False   # câu trả lời có đưa ra fact từ cơ sở tri thức
    co_nguon: bool = False     # … và có dòng "Nguồn: … – năm áp dụng …"


def doc_kich_ban(path: Path = SCENARIOS) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _co_nguon(text: str) -> bool:
    return "Nguồn:" in text and "năm áp dụng" in text


def can_chuyen(luot: dict) -> bool | None:
    """Kỳ vọng chuyển cán bộ: ghi rõ "chuyen_can_bo", hoặc suy ra False khi lượt có ý định mong đợi."""
    if "chuyen_can_bo" in luot:
        return bool(luot["chuyen_can_bo"])
    return False if "y_dinh" in luot else None


def kiem_luot(luot: dict, reply, state: DialogueState) -> list[str]:
    loi = []
    if "y_dinh" in luot:
        ok = luot["y_dinh"] if isinstance(luot["y_dinh"], list) else [luot["y_dinh"]]
        if reply.intent not in ok:
            loi.append(f"ý định {reply.intent!r} ≠ {ok}")
    if (can := can_chuyen(luot)) is not None and bool(reply.handoff) != can:
        loi.append("phải chuyển cán bộ" if can else "chuyển cán bộ thừa")
    if reply.sources and not _co_nguon(reply.text):
        loi.append("câu trả lời số liệu thiếu nguồn/năm")
    if luot.get("so_lieu") and not reply.handoff and not reply.sources:
        loi.append("không đưa ra số liệu từ cơ sở tri thức")
    for s in luot.get("chua", []):
        if s not in reply.text:
            loi.append(f"thiếu {s!r}")
    if luot.get("luong", _CHUA_DAT) is not _CHUA_DAT and state.active_flow != luot["luong"]:
        loi.append(f"luồng {state.active_flow!r} ≠ {luot['luong']!r}")
    return loi


def chay(engine: ChatEngine, kich_ban: list[dict]) -> list[KetQuaLuot]:
    out = []
    for hk in kich_ban:
        state = DialogueState()
        for luot in hk["luot"]:
            t0 = time.perf_counter()
            reply = engine.respond(luot["noi"], state)
            ms = (time.perf_counter() - t0) * 1000
            out.append(KetQuaLuot(hk["ten"], luot["noi"], reply.intent, reply.confidence, reply.handoff,
                                  reply.debug.get("nguon_y_dinh"), ms, kiem_luot(luot, reply, state),
                                  bool(reply.sources), _co_nguon(reply.text)))
            out[-1].luot = luot  # giữ lại kỳ vọng để tính chỉ số
    return out


def tong_hop(kq: list[KetQuaLuot]) -> dict:
    co_y_dinh = [r for r in kq if "y_dinh" in r.luot]
    so_lieu = [r for r in kq if r.co_so_lieu]
    hb = [(r, can_chuyen(r.luot)) for r in kq if can_chuyen(r.luot) is not None]
    mo_hinh = [r for r in kq if r.nguon_y_dinh]    # lượt đi qua bộ phân loại (không phải đang trong luồng)
    return {
        "so_hoi_thoai": len({r.hoi_thoai for r in kq}),
        "so_luot": len(kq),
        "luot_dat": sum(not r.loi for r in kq),
        "dung_y_dinh": (sum(not any(x.startswith("ý định") for x in r.loi) for r in co_y_dinh), len(co_y_dinh)),
        "so_lieu_co_nguon": (sum(r.co_nguon for r in so_lieu), len(so_lieu)),
        "chuyen_dung": sum(can and r.chuyen_can_bo for r, can in hb),
        "chuyen_can": sum(can for _, can in hb),
        "chuyen_thua": sum((not can) and r.chuyen_can_bo for r, can in hb),
        "khong_chuyen_can": sum(not can for _, can in hb),
        "ms_tb": sum(r.ms for r in kq) / len(kq),
        "ms_max": max(r.ms for r in kq),
        "nguon_y_dinh": {k: sum(r.nguon_y_dinh == k for r in mo_hinh)
                         for k in ("mo_hinh", "luat_uu_tien", "luat_bo_tro", "chan_pham_vi", "duoi_nguong")},
    }


def bao_cao(kq: list[KetQuaLuot], th: dict, ten_mo_hinh: str, tau: float) -> str:
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "–"  # noqa: E731
    dong = [
        f"# Đánh giá end-to-end Chức năng 4\n",
        f"Bộ phân loại: **{ten_mo_hinh}** · τ = {tau:.2f} · kịch bản: `tests/scenarios.yaml` "
        f"({th['so_hoi_thoai']} hội thoại, {th['so_luot']} lượt)\n",
        "| Chỉ số | Kết quả |", "|---|---|",
        f"| Lượt đạt mọi kỳ vọng | {pct(th['luot_dat'], th['so_luot'])} |",
        f"| Đúng ý định | {pct(*th['dung_y_dinh'])} |",
        f"| Câu trả lời có số liệu kèm nguồn + năm (mọi lượt đưa ra fact) | {pct(*th['so_lieu_co_nguon'])} |",
        f"| Chuyển cán bộ đúng (trên các câu cần chuyển) | {pct(th['chuyen_dung'], th['chuyen_can'])} |",
        f"| Chuyển cán bộ thừa (trên các câu không cần chuyển) | {pct(th['chuyen_thua'], th['khong_chuyen_can'])} |",
        f"| Thời gian phản hồi trung bình / lớn nhất | {th['ms_tb']:.0f} ms / {th['ms_max']:.0f} ms |",
        "",
        "Ý định được quyết định bởi (các lượt không nằm giữa luồng nhiều lượt): "
        + ", ".join(f"`{k}` {v}" for k, v in th["nguon_y_dinh"].items()),
        "",
        "## Chi tiết từng lượt\n",
        "| # | Hội thoại | Câu người dùng | Ý định | Tin cậy | Quyết định bởi | Chuyển CB | Kết quả |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(kq, 1):
        dong.append(f"| {i} | {r.hoi_thoai} | {r.noi} | `{r.y_dinh}` | {r.tin_cay:.2f} | "
                    f"{r.nguon_y_dinh or '(luồng)'} | {'có' if r.chuyen_can_bo else ''} | "
                    f"{'đạt' if not r.loi else 'LỖI: ' + '; '.join(r.loi)} |")
    return "\n".join(dong) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "keras", "nb", "keyword"])
    ap.add_argument("--tau", type=float, default=None, help="ghi đè ngưỡng (mặc định: theo models/config.json)")
    ap.add_argument("--out", type=Path, default=None, help="ghi báo cáo markdown ra file")
    args = ap.parse_args(argv)

    clf = load_classifier(backend=args.backend)
    tau = args.tau if args.tau is not None else load_threshold(classifier=clf)
    engine = ChatEngine(KnowledgeBase(), clf, tau, log_dir=Path(tempfile.mkdtemp()))  # không ghi đè logs/
    kq = chay(engine, doc_kich_ban())
    th = tong_hop(kq)
    md = bao_cao(kq, th, clf.name, tau)
    print(md)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"Đã ghi {args.out}")
    return th


if __name__ == "__main__":
    main()
