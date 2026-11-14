"""Hội thoại nhiều lượt (slot filling) để gọi mô-đun C1 (lộ trình) và C2 (tư vấn ngành) ngay trong khung chat.

Mỗi luồng = danh sách Slot. Slot được điền sẵn từ thực thể của câu kích hoạt; slot còn thiếu
thì hỏi lần lượt. Gõ "hủy" / "thoát" để dừng luồng.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..nlu.preprocess import normalize
from .entities import Entities, extract, parse_score
from .schema import BotReply, DialogueState

if TYPE_CHECKING:
    from .engine import ChatEngine

CANCEL_WORDS = {"hủy", "huỷ", "thoát", "dừng", "thôi", "cancel", "huy", "thoat"}
SKIP_WORDS = ("không biết", "bỏ qua", "chưa biết", "không có", "tùy", "tuỳ")

Parser = Callable[[str, Entities], Any]   # trả None nếu không hiểu → hỏi lại


@dataclass
class Slot:
    name: str
    question: str
    parse: Parser
    when: Callable[[dict], bool] = lambda slots: True   # điều kiện cần hỏi slot này


def _skippable(parser: Parser, empty: Any) -> Parser:
    def wrapped(text: str, e: Entities):
        if any(w in normalize(text) for w in SKIP_WORDS):
            return empty
        return parser(text, e)
    return wrapped


def _p_to_hop(text, e):
    return e.to_hop[0] if e.to_hop else None


def _p_diem(text, e):
    return e.diem if e.diem is not None else parse_score(text)


def _p_so_thich(text, e):
    return e.so_thich or None


def _p_dinh_huong(text, e):
    return e.dinh_huong


_NAM_HK = re.compile(r"năm\s?(\d)\s*(?:học kỳ|hk|kỳ)\s?(\d)")
_HK = re.compile(r"(?:học kỳ|hk|kỳ)\s?(\d{1,2})")


def _p_ky(text, e):
    """Học kỳ sắp đăng ký: "3", "học kỳ 3", "năm 2 học kỳ 1" (= 3), "năm nhất" (= 1)."""
    t = normalize(text)
    if "năm nhất" in t or "tân sinh viên" in t or "mới nhập học" in t:
        return 1
    if (m := _NAM_HK.search(t)):
        return (int(m.group(1)) - 1) * 2 + int(m.group(2))
    if (m := _HK.search(t)):
        return int(m.group(1))
    digits = [int(w) for w in t.split() if w.isdigit()]
    return digits[0] if len(digits) == 1 and 1 <= digits[0] <= 16 else None


def _p_list_hp(text, e):
    """Danh sách học phần (mã hoặc tên). "hk1"/"học kỳ 1" = toàn bộ HK1 do Trường xếp."""
    hp = list(e.ma_hp) + (["@HK1"] if e.nhac_hk1 else [])
    if hp:
        return hp
    t = f" {normalize(text)} "
    if any(f" {w} " in t for w in ("chưa", "không", "không có", "chưa có", "không rớt", "không môn nào")):
        return []
    return None


def _p_co_khong(text, e):
    t = f" {normalize(text)} "
    if any(f" {w} " in t for w in ("không", "thôi", "bình thường", "không muốn")):
        return False
    if any(f" {w} " in t for w in ("có", "muốn", "ừ", "ok", "đồng ý", "học vượt", "yes", "vâng", "dạ có")):
        return True
    return None


SLOT_TO_HOP = Slot("to_hop", "Em xét tuyển bằng **tổ hợp** nào? (ví dụ: A00, A01, D01)", _p_to_hop)
SLOT_DIEM = Slot("diem", "Tổng **điểm** 3 môn của tổ hợp đó (thang 30) là bao nhiêu?", _p_diem)

FLOWS: dict[str, list[Slot]] = {
    # C2 – suy diễn tiến: gợi ý danh sách ngành
    "tu_van_nganh": [
        SLOT_TO_HOP,
        SLOT_DIEM,
        Slot("so_thich", "Em **thích** gì nhất: lập trình, dạy học, dữ liệu/AI, mạng – bảo mật, thiết kế/game? "
                         "(gõ \"bỏ qua\" nếu chưa rõ)", _skippable(_p_so_thich, [])),
        Slot("dinh_huong", "Sau khi ra trường em muốn: làm **giáo viên**, đi làm **doanh nghiệp**, "
                           "**nghiên cứu** hay **khởi nghiệp**? (gõ \"bỏ qua\" nếu chưa rõ)", _skippable(_p_dinh_huong, "")),
    ],
    # C2 – suy diễn lùi: "Em có đủ điều kiện xét tuyển ngành X không?" (slot "nganh" điền từ câu hỏi)
    "kiem_tra_dieu_kien": [SLOT_TO_HOP, SLOT_DIEM],
    # C1 – lập lộ trình học
    "lo_trinh": [
        Slot("ky", "Học kỳ **sắp đăng ký** là học kỳ thứ mấy của em? (ví dụ: \"3\" hoặc \"năm 2 học kỳ 1\"; "
                   "tân sinh viên gõ \"năm nhất\")", _p_ky),
        Slot("da_dat", "Em liệt kê các học phần **đã đạt** (mã hoặc tên, ví dụ: COMP1010, Toán rời rạc và ứng dụng). "
                       "Gõ \"hk1\" nếu đã đạt toàn bộ học kỳ 1, có thể kết hợp: \"hk1, COMP1017\".", _p_list_hp,
             when=lambda s: s.get("ky", 1) > 1),
        Slot("bi_rot", "Em có học phần nào **bị rớt** (chưa học lại đạt) không? Liệt kê mã/tên, hoặc gõ \"không\".",
             _p_list_hp, when=lambda s: s.get("ky", 1) > 1),
        Slot("hoc_vuot", "Em có muốn **học vượt** (đăng ký tối đa 22 TC/kỳ thay vì 18 TC) không? (có/không)",
             _p_co_khong),
    ],
}


def _next_slot(flow: str, slots: dict) -> Slot | None:
    return next((s for s in FLOWS[flow] if s.name not in slots and s.when(slots)), None)


def _prefill(flow: str, slots: dict, e: Entities, text: str) -> None:
    """Điền sẵn slot từ câu kích hoạt (chỉ các slot có thực thể rõ ràng)."""
    if flow == "lo_trinh":
        t = normalize(text)
        if "năm nhất" in t or "tân sinh viên" in t:
            slots.setdefault("ky", 1)
        if "học vượt" in t:
            slots.setdefault("hoc_vuot", True)
        return
    if e.to_hop:
        slots.setdefault("to_hop", e.to_hop[0])
    if e.diem is not None:
        slots.setdefault("diem", e.diem)
    if e.so_thich:
        slots.setdefault("so_thich", e.so_thich)
    if e.dinh_huong:
        slots.setdefault("dinh_huong", e.dinh_huong)


def start_flow(flow: str, text: str, e: Entities, state: DialogueState, engine: "ChatEngine",
               extra: dict | None = None) -> BotReply:
    state.reset_flow()
    state.active_flow = flow
    state.slots.update(extra or {})
    _prefill(flow, state.slots, e, text)
    return _ask_or_finish(state, engine)


def continue_flow(text: str, state: DialogueState, engine: "ChatEngine") -> BotReply:
    if normalize(text).strip() in CANCEL_WORDS:
        state.reset_flow()
        return BotReply("Đã dừng. Em cứ hỏi tiếp câu khác nhé!", intent="huy_luong", confidence=1.0)

    slot = _next_slot(state.active_flow, state.slots)
    e = extract(text, engine.kb.majors, engine.courses)
    value = slot.parse(text, e) if slot else None
    if slot and value is None:
        return BotReply(f"Mình chưa hiểu câu trả lời 😅. {slot.question}\n\n_(gõ \"hủy\" để dừng)_",
                        intent=state.active_flow, confidence=1.0)
    if slot:
        state.slots[slot.name] = value
    return _ask_or_finish(state, engine)


def _ask_or_finish(state: DialogueState, engine: "ChatEngine") -> BotReply:
    flow = state.active_flow
    slot = _next_slot(flow, state.slots)
    if slot:
        state.pending_slot = slot.name
        return BotReply(f"{slot.question}\n\n_(gõ \"hủy\" để dừng)_", intent=flow, confidence=1.0,
                        debug={"slots": dict(state.slots)})

    slots = dict(state.slots)
    state.reset_flow()
    if flow == "tu_van_nganh":
        return engine.advise(slots)
    if flow == "kiem_tra_dieu_kien":
        return engine.check_eligibility(slots["nganh"], slots)
    if flow == "lo_trinh":
        return engine.make_plan_from_slots(slots)
    raise ValueError(f"Luồng không xác định: {flow}")
