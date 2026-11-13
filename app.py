"""Chức năng 4 – Giao diện chatbot tư vấn tuyển sinh (Streamlit).

    streamlit run app.py
"""
from __future__ import annotations

import json
import datetime as dt

import streamlit as st

from src.chatbot.engine import LOG_DIR, ChatEngine, module_status
from src.chatbot.entities import DINH_HUONG, SO_THICH
from src.chatbot.kb import KnowledgeBase
from src.chatbot.schema import BotReply, DialogueState
from src.nlu.classifier import KeywordClassifier, load_classifier, load_threshold
from src.planner.search import ALGORITHMS

st.set_page_config(page_title="Tư vấn tuyển sinh – Khoa CNTT", page_icon="🎓", layout="wide")


@st.cache_resource
def get_resources():
    return KnowledgeBase(), load_classifier(), load_threshold()


kb, classifier, default_tau = get_resources()

# ------------------------------------------------------------------ trạng thái phiên
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


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.caption(f"Bộ phân loại ý định: **{classifier.name}**")
    if isinstance(classifier, KeywordClassifier):
        st.warning("Đang dùng **phân loại từ khóa tạm thời**. Huấn luyện mô hình Keras (Chức năng 3) "
                   "→ `models/intent_model.keras` để thay thế.")
    tau = st.slider("Ngưỡng tin cậy τ", 0.0, 1.0, float(default_tau), 0.05,
                    help="Độ tin cậy < τ → chuyển cán bộ tư vấn thay vì trả lời bừa")
    debug = st.toggle("Chế độ debug (ý định, độ tin cậy, thực thể)", value=True)

    with st.expander("📦 Trạng thái mô-đun"):
        st.write({**{"C3 – Mô hình Keras": classifier.name == "keras"},
                  **module_status()})

    with st.expander("🎯 Tư vấn chọn ngành (Chức năng 2)"):
        with st.form("advisor"):
            to_hop = st.text_input("Tổ hợp xét tuyển", "A00")
            diem = st.number_input("Tổng điểm 3 môn", 0.0, 30.0, 24.0, 0.25)
            so_thich = st.multiselect("Sở thích", list(SO_THICH))
            dinh_huong = st.selectbox("Định hướng nghề nghiệp", [""] + list(DINH_HUONG))
            check = st.selectbox("Kiểm tra đủ điều kiện ngành (suy diễn lùi)", ["(không)"] + list(kb.majors),
                                 format_func=lambda c: kb.major_name(c) if c in kb.majors else c)
            if st.form_submit_button("Tư vấn"):
                user = {"to_hop": to_hop.strip().upper(), "diem": diem, "so_thich": so_thich, "dinh_huong": dinh_huong}
                st.session_state.pending_form = ("advisor", user, check)

    with st.expander("🗓️ Lộ trình học (Chức năng 1)"):
        engine_tmp = ChatEngine(kb, classifier, tau)
        with st.form("planner"):
            ids = sorted(engine_tmp.course_ids)
            label = lambda m: f"{m} – {engine_tmp.courses[m].ten}"
            done = st.multiselect("Học phần đã đạt", ids, format_func=label)
            ky = st.number_input("Học kỳ bắt đầu", 1, 15, 1)
            max_tc = st.number_input("Tín chỉ tối đa mỗi kỳ", 10, 30,
                                     int(engine_tmp.course_meta.get("tin_chi_toi_da_moi_ky", 20)))
            algo = st.selectbox("Thuật toán", list(ALGORITHMS))
            if st.form_submit_button("Lập lộ trình"):
                st.session_state.pending_form = ("planner", {"done": set(done), "ky": int(ky),
                                                             "max_credits": int(max_tc), "algorithm": algo}, None)

    if st.button("🗑️ Xóa hội thoại"):
        st.session_state.messages = []
        st.session_state.dialog = DialogueState()
        st.rerun()

engine = ChatEngine(kb, classifier, tau)

# ------------------------------------------------------------------ khung chat
st.title("🎓 Chatbot tư vấn tuyển sinh – Khoa CNTT")
st.caption(f"{kb.meta.get('truong', '')} · Mọi số liệu đều kèm nguồn và năm áp dụng · "
           "Không dùng API mô hình ngôn ngữ lớn")

SUGGESTIONS = ["Điểm chuẩn ngành CNTT năm ngoái bao nhiêu?", "CNTT khác Sư phạm Tin học chỗ nào?",
               "Em được 24 điểm khối A00 có đủ điều kiện vào CNTT không?", "Em bị rớt 2 môn thì nên học gì?"]
if not st.session_state.messages:
    push_bot(engine.greet("chào"))
if len(st.session_state.messages) == 1:  # chỉ hiện gợi ý khi chưa hỏi gì
    for c, s in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        if c.button(s, use_container_width=True):
            st.session_state.quick = s

# Xử lý form ở sidebar
if form := st.session_state.pop("pending_form", None):
    kind, data, extra = form
    if kind == "advisor":
        st.session_state.messages.append({"role": "user", "content": "[Form tư vấn chọn ngành]"})
        push_bot(engine.advise(data))
        if extra in kb.majors:
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
            st.caption(f"🔎 ý định: `{meta['intent']}` · tin cậy {meta['confidence']:.2f} · top-k: {top}"
                       + (" · ↪️ đã chuyển cán bộ" if meta.get("handoff") else ""))
            if meta.get("debug", {}).get("entities"):
                st.caption(f"thực thể: `{meta['debug']['entities']}`")
