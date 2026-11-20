"""Hội thoại nhiều lượt (slot filling) để gọi mô-đun C1 (lộ trình) và C2 (tư vấn ngành) ngay trong khung chat.

Mỗi luồng = danh sách Slot. Slot được điền sẵn từ thực thể của câu kích hoạt; slot còn thiếu
thì hỏi lần lượt. Gõ "hủy" / "thoát" để dừng luồng.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..advisor.tham_so import DINH_HUONG as KHOA_DINH_HUONG
from ..advisor.tham_so import MON, MON_HB2
from ..advisor.tham_so import SO_THICH as KHOA_SO_THICH
from ..nlu.preprocess import normalize
from .entities import Entities, extract, parse_diem_mon
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


def _p_phuong_thuc(text, e):
    return e.phuong_thuc


def _p_khu_vuc(text, e):
    if e.khu_vuc:
        return e.khu_vuc
    t = normalize(text).replace(" ", "")
    return {"1": "KV1", "2nt": "KV2-NT", "2": "KV2", "3": "KV3"}.get(t)


def _hop_le(d: dict) -> bool:
    return bool(d) and all(0 <= v <= 10 for v in d.values())


def _p_diem_thpt(text, e):
    d = e.diem_mon or parse_diem_mon(text)
    if not _hop_le(d):
        return None
    return {f"diem_{m}": d.get(m) for m in MON}


def _p_so_10(text, e):
    t = text.replace(",", ".")
    nums = [float(x) for x in re.findall(r"(?<![\d.])\d{1,2}(?:\.\d{1,3})?(?![\d.])", t)]
    return nums[0] if len(nums) == 1 and 0 <= nums[0] <= 10 else None


def _p_hoc_ba(text, e):
    d = e.diem_mon or parse_diem_mon(text)
    mon2 = [m for m in d if m in MON_HB2]
    if not _hop_le(d) or "Anh" not in d or len(mon2) != 1:
        return None
    return {"HB_mon1": d["Anh"], "mon_hb2": mon2[0], "HB_mon2": d[mon2[0]]}


def _p_so_thich(text, e):
    return e.so_thich or None


_DINH_HUONG_LONG = {
    "lap_trinh_vien": ("lập trình",), "phan_tich_du_lieu": ("dữ liệu", "phân tích"),
    "giao_vien": ("dạy", "giáo viên"), "chuyen_vien_edtech": ("edtech", "công nghệ giáo dục", "học liệu"),
}


def _p_dinh_huong(text, e):
    t = normalize(text)
    ds = list(e.dinh_huong) or [k for k, kws in _DINH_HUONG_LONG.items() if any(w in t for w in kws)]
    return ds or None


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


def _la_thpt(slots: dict) -> bool:
    return slots.get("phuong_thuc") == "THPT"


def _la_dgnl(slots: dict) -> bool:
    return slots.get("phuong_thuc") == "DGNL"


SLOT_PT = Slot("phuong_thuc", "Em xét tuyển theo **phương thức** nào: **THPT** (điểm thi tốt nghiệp) hay "
                              "**ĐGNL** (học bạ + thi đánh giá năng lực chuyên biệt)?", _p_phuong_thuc)
SLOT_KV = Slot("khu_vuc", "Em thuộc **khu vực ưu tiên** nào: KV1, KV2-NT, KV2 hay KV3?", _p_khu_vuc)
SLOT_DIEM_THPT = Slot("diem_thpt", "Em nhập **điểm thi THPT** các môn đã thi (thang 10), ví dụ: "
                                   "`Toán 8.2, Anh 7.5, Lý 6.75, Tin 8`. Môn không thi thì không cần ghi.",
                      _p_diem_thpt, when=_la_thpt)
SLOT_DGNL = Slot("diem_dgnl", "Điểm **ĐGNL môn Toán** của em (đã quy về thang 10) là bao nhiêu?", _p_so_10,
                 when=_la_dgnl)
SLOT_HB = Slot("hoc_ba", "Điểm **học bạ lớp 12** môn Tiếng Anh và môn thứ hai của tổ hợp (Lý, Hóa, Sinh hoặc Tin), "
                         "ví dụ: `Anh 8.5, Tin 9`.", _p_hoc_ba, when=_la_dgnl)
SLOT_SO_THICH = Slot("so_thich", "Em **thích** những gì? (chọn nhiều được): lập trình, dữ liệu, thiết kế học liệu, "
                                 "dạy học. Gõ \"bỏ qua\" nếu chưa rõ.", _skippable(_p_so_thich, []))
SLOT_DINH_HUONG = Slot("dinh_huong", "Sau khi ra trường em muốn làm gì? (chọn nhiều được): **lập trình viên**, "
                                     "**phân tích dữ liệu**, **giáo viên**, **chuyên viên công nghệ giáo dục (EdTech)**. "
                                     "Gõ \"bỏ qua\" nếu chưa rõ.", _skippable(_p_dinh_huong, []))

KHOA_SLOT = {"phuong_thuc": SLOT_PT, "khu_vuc": SLOT_KV, "diem_dgnl": SLOT_DGNL,
             "HB_mon1": SLOT_HB, "HB_mon2": SLOT_HB, "mon_hb2": SLOT_HB,
             **{f"diem_{m}": SLOT_DIEM_THPT for m in MON}}

FLOWS: dict[str, list[Slot]] = {
    "tu_van_nganh": [SLOT_PT, SLOT_DIEM_THPT, SLOT_DGNL, SLOT_HB, SLOT_KV, SLOT_SO_THICH, SLOT_DINH_HUONG],
    "kiem_tra_dieu_kien": [SLOT_PT, SLOT_DIEM_THPT, SLOT_DGNL, SLOT_HB, SLOT_KV],
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


def slots_to_user(slots: dict) -> dict:
    user = {k: slots[k] for k in ("phuong_thuc", "khu_vuc", "diem_dgnl") if k in slots}
    user.update(slots.get("diem_thpt") or {})
    user.update(slots.get("hoc_ba") or {})
    if "so_thich" in slots:
        user.update({k: k in slots["so_thich"] for k in KHOA_SO_THICH})
    if "dinh_huong" in slots:
        user.update({k: k in slots["dinh_huong"] for k in KHOA_DINH_HUONG})
    return user


def _next_slot(flow: str, slots: dict, engine: "ChatEngine | None" = None) -> Slot | None:
    if flow == "kiem_tra_dieu_kien" and engine is not None:
        for khoa in engine.du_kien_con_thieu(slots["nganh"], slots_to_user(slots)):
            slot = KHOA_SLOT.get(khoa)
            if slot and slot.name not in slots:
                return slot
        return None
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
    if e.phuong_thuc:
        slots.setdefault("phuong_thuc", e.phuong_thuc)
    if e.khu_vuc:
        slots.setdefault("khu_vuc", e.khu_vuc)
    if e.diem_mon and slots.get("phuong_thuc") == "THPT" and (d := _p_diem_thpt(text, e)):
        slots.setdefault("diem_thpt", d)
    if flow == "tu_van_nganh":
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

    slot = next((s for s in FLOWS[state.active_flow] if s.name == state.pending_slot), None) \
        or _next_slot(state.active_flow, state.slots, engine)
    e = extract(text, engine.kb.majors, engine.courses)
    value = slot.parse(text, e) if slot else None
    if slot and value is None:
        return BotReply(f"Mình chưa hiểu câu trả lời. {slot.question}\n\n_(gõ \"hủy\" để dừng)_",
                        intent=state.active_flow, confidence=1.0)
    if slot:
        state.slots[slot.name] = value
    return _ask_or_finish(state, engine)


def _ask_or_finish(state: DialogueState, engine: "ChatEngine") -> BotReply:
    flow = state.active_flow
    slot = _next_slot(flow, state.slots, engine)
    if slot:
        state.pending_slot = slot.name
        return BotReply(f"{slot.question}\n\n_(gõ \"hủy\" để dừng)_", intent=flow, confidence=1.0,
                        debug={"slots": dict(state.slots)})

    slots = dict(state.slots)
    state.reset_flow()
    if flow == "tu_van_nganh":
        return engine.advise(slots_to_user(slots))
    if flow == "kiem_tra_dieu_kien":
        return engine.check_eligibility(slots["nganh"], slots_to_user(slots))
    if flow == "lo_trinh":
        return engine.make_plan_from_slots(slots)
    raise ValueError(f"Luồng không xác định: {flow}")
