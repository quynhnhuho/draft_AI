"""Chức năng 4 – Giao diện chatbot tư vấn tuyển sinh (Streamlit).

    streamlit run app.py

Hai trang (chọn ở sidebar):
  - 💬 Chatbot             : dành cho thí sinh / sinh viên. Không có lựa chọn thuật toán.
  - 📊 So sánh thuật toán  : dành cho giảng viên / hội đồng – chạy BFS, A*, nhánh cận trên cùng bài toán.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

from src.chatbot.engine import LOG_DIR, ChatEngine, module_status
from src.chatbot.entities import DINH_HUONG, SO_THICH
from src.chatbot.kb import KnowledgeBase
from src.chatbot.schema import BotReply, DialogueState
from src.nlu.classifier import KeywordClassifier, load_classifier, load_threshold
from src.planner import dieu_phoi, kich_ban
from src.planner.model import (GIA_DINH, NAM_AP_DUNG, NGUON_CTDT, kiem_tra_lo_trinh, tao_bai_toan,
                               ten_hoc_ky)

st.set_page_config(page_title="Tư vấn tuyển sinh – Khoa CNTT", page_icon="🎓", layout="wide")


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
    trang = st.radio("Trang", ["💬 Chatbot", "📊 So sánh thuật toán (C1)"], label_visibility="collapsed")
    st.header("⚙️ Cấu hình")
    st.caption(f"Bộ phân loại ý định: **{classifier.name}**")
    if isinstance(classifier, KeywordClassifier):
        st.warning("Đang dùng **phân loại từ khóa tạm thời**. Huấn luyện mô hình Keras (Chức năng 3) "
                   "→ `models/intent_model.keras` để thay thế.")
    engine.threshold = st.slider("Ngưỡng tin cậy τ", 0.0, 1.0, float(default_tau), 0.05,
                                 help="Độ tin cậy < τ → chuyển cán bộ tư vấn thay vì trả lời bừa")
    debug = st.toggle("Chế độ debug (ý định, thực thể, thuật toán)", value=False,
                      help="Hiện ý định, độ tin cậy, thực thể; với lộ trình học: thuật toán đã dùng, "
                           "số nút và đối chiếu BFS")
    engine.debug = debug

    with st.expander("📦 Trạng thái mô-đun"):
        st.write({"C3 – Mô hình Keras": classifier.name == "keras", **module_status()})

    if trang == "💬 Chatbot":
        with st.expander("🎯 Tư vấn chọn ngành (Chức năng 2)"):
            with st.form("advisor"):
                to_hop = st.text_input("Tổ hợp xét tuyển", "A00")
                diem = st.number_input("Tổng điểm 3 môn", 0.0, 30.0, 24.0, 0.25)
                so_thich = st.multiselect("Sở thích", list(SO_THICH))
                dinh_huong = st.selectbox("Định hướng nghề nghiệp", [""] + list(DINH_HUONG))
                check = st.selectbox("Kiểm tra đủ điều kiện ngành (suy diễn lùi)", ["(không)"] + list(kb.majors),
                                     format_func=lambda c: kb.major_name(c) if c in kb.majors else c)
                if st.form_submit_button("Tư vấn"):
                    user = {"to_hop": to_hop.strip().upper(), "diem": diem, "so_thich": so_thich,
                            "dinh_huong": dinh_huong}
                    st.session_state.pending_form = ("advisor", user, check)

        with st.expander("🗓️ Lộ trình học (Chức năng 1)"):
            with st.form("planner"):
                ids = sorted(engine.course_ids, key=lambda m: engine.courses[m].stt)
                ky = st.number_input("Học kỳ sắp đăng ký (1 = năm nhất)", 1, 16, 1)
                done = st.multiselect("Học phần đã đạt", ids, format_func=hp_label)
                rot = st.multiselect("Học phần bị rớt (chưa học lại đạt)", ids, format_func=hp_label)
                hoc_vuot = st.checkbox("Em muốn học vượt (tối đa 22 TC/kỳ)")
                if st.form_submit_button("Lập lộ trình"):
                    st.session_state.pending_form = ("planner", {"da_dat": set(done), "rot": set(rot),
                                                                 "ky": int(ky), "hoc_vuot": hoc_vuot}, None)

        if st.button("🗑️ Xóa hội thoại"):
            st.session_state.messages = []
            st.session_state.dialog = DialogueState()
            st.rerun()


# ================================================================== trang chatbot
def trang_chatbot():
    st.title("🎓 Chatbot tư vấn tuyển sinh – Khoa CNTT")
    st.caption(f"{kb.meta.get('truong', '')} · Mọi số liệu đều kèm nguồn và năm áp dụng · "
               "Không dùng API mô hình ngôn ngữ lớn")

    suggestions = ["Điểm chuẩn ngành CNTT năm ngoái bao nhiêu?", "CNTT khác Sư phạm Tin học chỗ nào?",
                   "Em được 24 điểm khối A00 có đủ điều kiện vào CNTT không?",
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


# ================================================================== trang so sánh thuật toán
def trang_so_sanh():
    st.title("📊 So sánh thuật toán lập lộ trình (Chức năng 1)")
    st.markdown(
        "Trong chatbot, người dùng **không chọn thuật toán**. Bộ điều phối (`src/planner/dieu_phoi.py`) tự quyết:\n"
        "- **A\\*** – bộ giải chính (h = ⌈tín chỉ còn lại / MAX_TC⌉), cho lộ trình ít học kỳ nhất.\n"
        "- **Nhánh cận** – bộ giải dự phòng khi A\\* vượt giới hạn thời gian; khởi tạo bằng lời giải tham lam "
        "nên dừng lúc nào cũng có lộ trình.\n"
        "- **BFS** – chuẩn đối chiếu (chế độ debug, trang này, test tự động).\n\n"
        "Trang này chạy **tất cả** thuật toán trên cùng bài toán để kiểm chứng và lấy số liệu cho báo cáo. "
        "Các dòng *(h2)* dùng heuristic mở rộng h2 = max(h theo kỳ lẻ/chẵn, chuỗi học trước).")
    st.caption(f"📌 Nguồn dữ liệu: {NGUON_CTDT} – năm áp dụng {NAM_AP_DUNG}")
    with st.expander("Giả định của mô hình"):
        st.markdown("\n".join(f"- {g}" for g in GIA_DINH))

    cac_kb = kich_ban.cac_kich_ban(engine.courses)
    chon = st.selectbox("Tình huống", list(cac_kb) + ["Tùy chỉnh"])
    if chon == "Tùy chỉnh":
        ids = sorted(engine.course_ids, key=lambda m: engine.courses[m].stt)
        c1, c2 = st.columns(2)
        hk = c1.number_input("Học kỳ sắp đăng ký", 1, 16, 1)
        hoc_vuot = c2.checkbox("Học vượt (22 TC)")
        da_dat = set(st.multiselect("Đã đạt", ids, format_func=hp_label))
        rot = set(st.multiselect("Bị rớt", ids, format_func=hp_label))
    else:
        kb_ = cac_kb[chon]
        st.info(kb_["mo_ta"])
        hk, hoc_vuot, da_dat, rot = kb_["hoc_ky"], kb_["hoc_vuot"], kb_["da_dat"], kb_["rot"]
    them_h2 = st.checkbox("Thêm biến thể heuristic h2", value=True)

    if st.button("▶️ Chạy so sánh", type="primary"):
        try:
            bt = tao_bai_toan(engine.courses, da_dat, rot, int(hk), hoc_vuot)
        except ValueError as e:
            st.error(str(e))
            return
        kqs = dieu_phoi.so_sanh(bt, them_h2=them_h2)
        if not kqs:
            st.warning("Chưa có thuật toán nào trong `src/planner/` (bfs.py, a_sao.py, nhanh_can.py).")
            return
        rows = []
        for kq in kqs:
            ct = kq.chi_tiet
            rows.append({"Thuật toán": kq.thuat_toan, "Số kỳ": kq.so_ky, "Nút mở rộng": kq.so_nut_mo_rong,
                         "Nút sinh": kq.so_nut_sinh, "Thời gian (ms)": round(kq.thoi_gian_ms, 2),
                         "Tối ưu": "✔" if kq.toi_uu else "chưa chắc",
                         "Hợp lệ": "✔" if kq.lo_trinh is not None and not kiem_tra_lo_trinh(bt, kq.lo_trinh) else "✘",
                         "Cận trên tham lam": ct.get("can_tren_ban_dau"),
                         "Cắt theo cận": ct.get("cat_theo_can"), "Cắt trùng lặp": ct.get("cat_trung_lap")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        so_ky = {kq.so_ky for kq in kqs if kq.lo_trinh is not None}
        if len(so_ky) == 1:
            st.success(f"Mọi thuật toán cho cùng số học kỳ tối ưu: **{so_ky.pop()}** kỳ.")
        else:
            st.error(f"Các thuật toán cho số học kỳ khác nhau: {so_ky}")
        tot = kqs[0]
        if tot.lo_trinh:
            st.subheader(f"Lộ trình ({tot.thuat_toan})")
            st.dataframe(pd.DataFrame([{
                "Học kỳ": ten_hoc_ky(ky) + (" 🏫" if ky in bt.truong_xep else ""),
                "Học phần": "; ".join(m + (" (học lại)" if m in bt.hoc_phan_rot else "")
                                      for m in sorted(tap, key=lambda x: bt.hoc_phan[x].stt)),
                "TC": sum(bt.hoc_phan[m].tin_chi for m in tap)} for ky, tap in tot.lo_trinh]),
                hide_index=True, use_container_width=True)


if trang == "💬 Chatbot":
    trang_chatbot()
else:
    trang_so_sanh()
