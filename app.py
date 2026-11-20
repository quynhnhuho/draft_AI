"""Chức năng 4 – Giao diện chatbot tư vấn tuyển sinh (Streamlit).

    streamlit run app.py

Người dùng không chọn thuật toán lập lộ trình: chatbot gọi bộ điều phối của Chức năng 1.
"""
from __future__ import annotations

import datetime as dt
import json

import streamlit as st

from src.chatbot.engine import LOG_DIR, ChatEngine, module_status
from src.advisor.tham_so import MON_HB2, TEN_MON, tu_ma_kb
from src.chatbot.entities import DINH_HUONG, SO_THICH, TEN_DINH_HUONG, TEN_SO_THICH
from src.chatbot.kb import KnowledgeBase
from src.chatbot.schema import BotReply, DialogueState
from src.nlu.classifier import KeywordClassifier, load_classifier, load_threshold

st.set_page_config(page_title="Tư vấn tuyển sinh – Khoa CNTT", layout="wide")


@st.cache_resource
def get_resources():
    return KnowledgeBase(), load_classifier(), load_threshold()


kb, classifier, default_tau = get_resources()
engine = ChatEngine(kb, classifier, default_tau)

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role", "content", "meta"}]
    st.session_state.dialog = DialogueState()


def log_turn(user_text: str, reply: BotReply) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "conversations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"), "user": user_text,
                            "intent": reply.intent, "conf": round(reply.confidence, 3),
                            "handoff": reply.handoff}, ensure_ascii=False) + "\n")


def push_bot(reply: BotReply) -> None:
    st.session_state.messages.append({"role": "assistant", "content": reply.text, "meta": {
        "intent": reply.intent, "confidence": reply.confidence, "top_k": reply.top_k,
        "handoff": reply.handoff, "debug": reply.debug}})


def hp_label(m: str) -> str:
    return f"{m} – {engine.courses[m].ten}"


# ================================================================== sidebar
with st.sidebar:
    st.header("Cấu hình")
    st.caption(f"Bộ phân loại ý định: **{classifier.name}**")
    if isinstance(classifier, KeywordClassifier):
        st.warning("Đang dùng **phân loại từ khóa tạm thời**. Huấn luyện mô hình Keras (Chức năng 3) "
                   "→ `models/intent_model.keras` để thay thế.")
    engine.threshold = st.slider("Ngưỡng tin cậy τ", 0.0, 1.0, float(default_tau), 0.05,
                                 help="Độ tin cậy < τ → chuyển cán bộ tư vấn thay vì trả lời bừa")
    debug = st.toggle("Chế độ debug (ý định, thực thể, thuật toán)", value=False,
                      help="Hiện ý định, độ tin cậy, thực thể; với lộ trình học: thuật toán đã dùng "
                           "và số nút")
    engine.debug = debug

    with st.expander("Trạng thái mô-đun"):
        st.write({"C3 – Mô hình Keras": classifier.name == "keras", **module_status()})

    with st.expander("Tư vấn chọn ngành (Chức năng 2)"):
        with st.form("advisor"):
            pt = st.radio("Phương thức xét tuyển", ["THPT", "DGNL"], horizontal=True)
            kv = st.selectbox("Khu vực ưu tiên", ["KV1", "KV2-NT", "KV2", "KV3"], index=3)
            st.caption("**THPT** – điểm thi các môn (để trống nếu không thi)")
            diem = {m: st.number_input(ten, 0.0, 10.0, value=None, step=0.05, key=f"d_{m}")
                    for m, ten in TEN_MON.items()}
            st.caption("**ĐGNL** – điểm ĐGNL môn Toán và học bạ lớp 12")
            dgnl = st.number_input("Điểm ĐGNL môn Toán (thang 10)", 0.0, 10.0, value=None, step=0.05)
            hb1 = st.number_input("Học bạ Tiếng Anh", 0.0, 10.0, value=None, step=0.05)
            mon_hb2 = st.selectbox("Môn học bạ thứ hai", list(MON_HB2), format_func=TEN_MON.get)
            hb2 = st.number_input("Học bạ môn thứ hai", 0.0, 10.0, value=None, step=0.05)
            so_thich = st.multiselect("Sở thích", list(SO_THICH), format_func=TEN_SO_THICH.get)
            dinh_huong = st.multiselect("Định hướng nghề nghiệp", list(DINH_HUONG), format_func=TEN_DINH_HUONG.get)
            check = st.selectbox("Kiểm tra đủ điều kiện ngành (suy diễn lùi)", ["(không)"] + list(kb.majors),
                                 format_func=lambda c: kb.major_name(c) if c in kb.majors else c)
            if st.form_submit_button("Tư vấn"):
                user = {"phuong_thuc": pt, "khu_vuc": kv,
                        **{k: k in so_thich for k in SO_THICH}, **{k: k in dinh_huong for k in DINH_HUONG}}
                if pt == "THPT":
                    user |= {f"diem_{m}": v for m, v in diem.items()}
                else:
                    user |= {"diem_dgnl": dgnl, "HB_mon1": hb1, "HB_mon2": hb2, "mon_hb2": mon_hb2}
                st.session_state.pending_form = ("advisor", user, check)

    with st.expander("Lộ trình học (Chức năng 1)"):
        with st.form("planner"):
            ids = sorted(engine.course_ids, key=lambda m: engine.courses[m].stt)
            ky = st.number_input("Học kỳ sắp đăng ký (1 = năm nhất)", 1, 16, 1)
            done = st.multiselect("Học phần đã đạt", ids, format_func=hp_label)
            rot = st.multiselect("Học phần bị rớt (chưa học lại đạt)", ids, format_func=hp_label)
            hoc_vuot = st.checkbox("Em muốn học vượt (tối đa 22 TC/kỳ)")
            if st.form_submit_button("Lập lộ trình"):
                st.session_state.pending_form = ("planner", {"da_dat": set(done), "rot": set(rot),
                                                             "ky": int(ky), "hoc_vuot": hoc_vuot}, None)

    if st.button("Xóa hội thoại"):
        st.session_state.messages = []
        st.session_state.dialog = DialogueState()
        st.rerun()


# ================================================================== trang chatbot
def trang_chatbot():
    st.title("Chatbot tư vấn tuyển sinh – Khoa CNTT")
    st.caption(f"{kb.meta.get('truong', '')} · Mọi số liệu đều kèm nguồn và năm áp dụng · "
               "Không dùng API mô hình ngôn ngữ lớn")

    suggestions = ["Điểm chuẩn ngành CNTT năm ngoái bao nhiêu?", "CNTT khác Sư phạm Tin học chỗ nào?",
                   "Em thi THPT Toán 8, Anh 7, Lý 7, KV2 có đủ điều kiện vào CNTT không?",
                   "Em là tân sinh viên, lộ trình học thế nào?"]
    if not st.session_state.messages:
        push_bot(engine.greet("chào"))
    if len(st.session_state.messages) == 1:          # chỉ hiện gợi ý khi chưa hỏi gì
        for c, s in zip(st.columns(len(suggestions)), suggestions):
            if c.button(s, use_container_width=True):
                st.session_state.quick = s

    if form := st.session_state.pop("pending_form", None):
        kind, data, extra = form
        if kind == "advisor":
            st.session_state.messages.append({"role": "user", "content": "[Form tư vấn chọn ngành]"})
            push_bot(engine.advise(data))
            if extra in kb.majors and tu_ma_kb(extra):
                push_bot(engine.check_eligibility(extra, data))
        else:
            st.session_state.messages.append({"role": "user", "content": "[Form lập lộ trình học]"})
            push_bot(engine.make_plan(**data))

    prompt = st.chat_input("Nhập câu hỏi của em…") or st.session_state.pop("quick", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        reply = engine.respond(prompt, st.session_state.dialog)
        push_bot(reply)
        log_turn(prompt, reply)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if debug and meta and meta.get("intent"):
                top = ", ".join(f"{i}: {p:.2f}" for i, p in meta.get("top_k", []))
                st.caption(f"Ý định: `{meta['intent']}` · tin cậy {meta['confidence']:.2f} · top-k: {top}"
                           + (" · đã chuyển cán bộ" if meta.get("handoff") else ""))
                if meta.get("debug", {}).get("entities"):
                    st.caption(f"thực thể: `{meta['debug']['entities']}`")


trang_chatbot()
