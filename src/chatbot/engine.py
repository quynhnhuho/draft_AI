"""ChatEngine – lõi chatbot (Chức năng 4).

Một lượt:  luồng đang chạy? → trích thực thể → nhận ý định → ngưỡng τ (chuyển cán bộ) → handler.
Không dùng API mô hình ngôn ngữ lớn: mọi câu trả lời lấy từ cơ sở tri thức, hệ luật (C2) hoặc planner (C1).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from ..advisor.facts import build_facts, latest_cutoffs
from ..advisor import inference
from ..nlu.preprocess import normalize
from ..planner import dieu_phoi
from ..planner.model import (GIA_DINH, MAX_TC_BINH_THUONG, MAX_TC_HOC_VUOT, NAM_AP_DUNG, NGUON_CTDT,
                             doc_du_lieu, hk1_co_dinh, kiem_tra_dau_vao, tao_bai_toan, ten_hoc_ky)
from . import flows
from .entities import Entities, SO_THICH, DINH_HUONG, extract
from .kb import KnowledgeBase, cite, format_facts
from .schema import BotReply, DialogueState

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# Ý định trả lời trực tiếp từ cơ sở tri thức
KB_INTENTS = {
    "diem_chuan": "điểm chuẩn",
    "to_hop_xet_tuyen": "tổ hợp xét tuyển",
    "hoc_phi": "học phí",
    "hoc_bong": "học bổng",
    "ky_tuc_xa": "ký túc xá",
    "chuong_trinh_dao_tao": "chương trình đào tạo",
    "viec_lam": "việc làm sau tốt nghiệp",
    "thu_tuc_nhap_hoc": "thủ tục nhập học",
}

ELIGIBILITY_CUES = ("đủ điều kiện", "đủ đk", "có đậu", "đậu không", "đỗ không", "có đỗ",
                    "vào được", "trúng tuyển", "có vào", "đủ điểm")

NOT_READY = "🚧 **Mô-đun {name} chưa được cài đặt** – sẽ hoàn thiện ở {where}."


class ChatEngine:
    def __init__(self, kb: KnowledgeBase, classifier, threshold: float = 0.6, log_dir: Path = LOG_DIR):
        self.kb = kb
        self.classifier = classifier
        self.threshold = threshold
        self.log_dir = log_dir
        self.debug = False            # bật từ giao diện: hiện thuật toán, số nút, đối chiếu BFS
        try:
            self.courses = doc_du_lieu()              # {mã: HocPhan} từ data/hoc_phan.csv
            self.hk1 = hk1_co_dinh(self.courses)
        except (FileNotFoundError, ValueError):
            self.courses, self.hk1 = {}, frozenset()
        self.course_ids = set(self.courses)

    # ------------------------------------------------------------------ lượt hội thoại
    def respond(self, text: str, state: DialogueState) -> BotReply:
        state.turns += 1
        text = text.strip()
        if not text:
            return BotReply("Em hãy nhập câu hỏi nhé.")

        if state.active_flow:
            reply = flows.continue_flow(text, state, self)
            reply.confidence = reply.confidence or 1.0   # đang trong luồng: không qua bộ phân loại
            return reply

        e = extract(text, self.kb.majors, self.courses)
        pred = self.classifier.predict(text)
        dbg = {"classifier": getattr(self.classifier, "name", "?"), "entities": e.as_dict(),
               "threshold": self.threshold}

        if pred.confidence < self.threshold:
            reply = self.handoff(text, pred)
        else:
            if e.nganh:
                state.last_major = e.nganh[0]
            reply = self._route(pred.intent, text, e, state)

        reply.intent = reply.intent or pred.intent
        reply.confidence, reply.top_k = pred.confidence, pred.top_k
        reply.debug = {**dbg, **reply.debug}
        return reply

    def _route(self, intent: str, text: str, e: Entities, state: DialogueState) -> BotReply:
        if intent == "chao_hoi":
            return self.greet(text)
        if intent == "tu_van_chon_nganh":
            t = normalize(text)
            if e.nganh and any(c in t for c in ELIGIBILITY_CUES):
                return flows.start_flow("kiem_tra_dieu_kien", text, e, state, self, extra={"nganh": e.nganh[0]})
            return flows.start_flow("tu_van_nganh", text, e, state, self)
        if intent == "lo_trinh_hoc_tap":
            return flows.start_flow("lo_trinh", text, e, state, self)
        if intent in KB_INTENTS:
            return self.answer_from_kb(intent, text, e, state)
        return self.handoff(text, None)

    # ------------------------------------------------------------------ handlers
    def greet(self, text: str) -> BotReply:
        t = normalize(text)
        if any(w in t for w in ("cảm ơn", "cám ơn", "thanks")):
            return BotReply("Không có gì ạ! Chúc em một mùa tuyển sinh thật thành công 🎓")
        if any(w in t for w in ("tạm biệt", "bye")):
            return BotReply("Tạm biệt em, hẹn gặp lại! 👋")
        return BotReply(
            f"Chào em 👋 Mình là trợ lý tư vấn tuyển sinh của {self.kb.meta.get('khoa', 'Khoa CNTT')}. Mình có thể:\n"
            "- Trả lời về **điểm chuẩn, tổ hợp xét tuyển, học phí, học bổng, ký túc xá, chương trình đào tạo, "
            "việc làm, thủ tục nhập học**\n"
            "- **Tư vấn chọn ngành** (CNTT hay Sư phạm Tin học) và kiểm tra em có **đủ điều kiện xét tuyển** không\n"
            "- Gợi ý **lộ trình đăng ký học phần** để ra trường đúng hạn")

    def answer_from_kb(self, intent: str, text: str, e: Entities, state: DialogueState) -> BotReply:
        t = normalize(text)
        # So sánh hai ngành: "CNTT khác Sư phạm Tin chỗ nào?"
        if intent == "chuong_trinh_dao_tao" and ("khác" in t or "so sánh" in t):
            parts = [f"**{m['ten']}** (mã {m['ma_nganh']}): {m['mo_ta']} {m.get('khac_biet', '')}"
                     for m in self.kb.majors.values()]
            return BotReply("\n\n".join(parts) + "\n\n📌 Nguồn: mô tả ngành trong cơ sở tri thức "
                            "(cần dẫn chương trình đào tạo chính thức).")

        nganh = e.nganh[0] if e.nganh else state.last_major
        res = self.kb.query(intent, nganh, e.nam)
        if not res.facts:
            return self.handoff(text, None, reason=f"Hiện mình chưa có dữ liệu đã kiểm chứng về {KB_INTENTS[intent]}"
                                + (f" ngành {self.kb.major_name(nganh)}" if nganh else "") + ".")
        head = ""
        if res.year_fallback:
            years = sorted({f["nam"] for f in res.facts})
            head = (f"Mình chưa có số liệu năm {res.requested_year}; dưới đây là số liệu năm gần nhất "
                    f"({', '.join(map(str, years))}).\n\n")
        elif e.nam is None and intent in ("diem_chuan", "hoc_phi", "to_hop_xet_tuyen"):
            head = f"Số liệu mới nhất mình có (năm {max(f['nam'] for f in res.facts)}):\n\n"
        return BotReply(head + format_facts(res.facts), sources=res.facts)

    def handoff(self, text: str, pred, reason: str | None = None) -> BotReply:
        """Không đủ tin cậy / ngoài phạm vi / thiếu dữ liệu → chuyển cán bộ tư vấn, không trả lời bừa."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "handoff.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["thoi_gian", "cau_hoi", "y_dinh_du_doan", "do_tin_cay", "top_k", "ly_do"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), text,
                        pred.intent if pred else "", f"{pred.confidence:.3f}" if pred else "",
                        json.dumps(pred.top_k if pred else [], ensure_ascii=False),
                        reason or "duoi_nguong"])
        msg = reason or "Câu hỏi này nằm ngoài phạm vi mình chắc chắn trả lời được."
        return BotReply(f"{msg} Mình đã **chuyển câu hỏi đến cán bộ tư vấn** của Khoa. "
                        f"Em có thể liên hệ trực tiếp:\n{self.kb.contact_text()}",
                        intent="chuyen_can_bo", handoff=True)

    # ------------------------------------------------------------------ C2: tư vấn chọn ngành
    def _user_summary(self, user: dict) -> str:
        bits = []
        if user.get("to_hop"):
            bits.append(f"tổ hợp **{user['to_hop']}**")
        if user.get("diem") is not None:
            bits.append(f"điểm **{user['diem']}**")
        if user.get("so_thich"):
            bits.append("sở thích **" + ", ".join(user["so_thich"]) + "**")
        if user.get("dinh_huong"):
            bits.append(f"định hướng **{user['dinh_huong']}**")
        return ", ".join(bits) or "(chưa có thông tin)"

    def advise(self, user: dict) -> BotReply:
        """Suy diễn TIẾN → danh sách ngành gợi ý + vết suy diễn."""
        facts = build_facts(user, {"facts": self.kb.facts})
        info = f"Thông tin của em: {self._user_summary(user)}.\n\nSự kiện đầu vào: `{sorted(facts)}`"
        try:
            result = inference.forward_chaining(facts)
        except NotImplementedError:
            return BotReply(f"{info}\n\n" + NOT_READY.format(name="tư vấn chọn ngành (suy diễn tiến)",
                                                           where="`src/advisor/inference.py` – Chức năng 2"),
                            intent="tu_van_chon_nganh", debug={"facts": sorted(facts)})
        if not result.recommendations:
            body = "Với thông tin hiện tại, hệ luật chưa đưa ra được ngành gợi ý phù hợp."
        else:
            body = "**Ngành gợi ý:** " + ", ".join(self.kb.major_name(m) for m in result.recommendations)
        trace = "\n".join(f"- {r.id}: {' ∧ '.join(r.if_all)} → {r.then}  _{r.giai_thich}_" for r in result.fired)
        return BotReply(f"{info}\n\n{body}\n\n**Vết suy diễn:**\n{trace or '- (không luật nào được kích hoạt)'}"
                        f"\n\n{self._cutoff_note()}", intent="tu_van_chon_nganh", sources=self._cutoff_facts())

    def check_eligibility(self, nganh: str, user: dict) -> BotReply:
        """Suy diễn LÙI cho mục tiêu du_dk_<ngành>."""
        facts = build_facts(user, {"facts": self.kb.facts})
        goal = f"du_dk_{nganh}"
        info = (f"Câu hỏi: em có đủ điều kiện xét tuyển ngành **{self.kb.major_name(nganh)}** không?  \n"
                f"Thông tin: {self._user_summary(user)}. Mục tiêu: `{goal}`")
        try:
            result = inference.backward_chaining(goal, facts)
        except NotImplementedError:
            return BotReply(f"{info}\n\n" + NOT_READY.format(name="kiểm tra điều kiện xét tuyển (suy diễn lùi)",
                                                           where="`src/advisor/inference.py` – Chức năng 2"),
                            intent="tu_van_chon_nganh", debug={"facts": sorted(facts)})
        verdict = "✅ **Có**, em đủ điều kiện xét tuyển" if result.proven else "❌ **Chưa** đủ điều kiện xét tuyển"
        missing = f"\n\nCòn thiếu thông tin: {', '.join(sorted(result.missing))}" if result.missing else ""
        trace = "\n".join(f"- {s}" for s in result.trace)
        return BotReply(f"{info}\n\n{verdict} ngành {self.kb.major_name(nganh)}.{missing}\n\n"
                        f"**Các bước chứng minh:**\n{trace}\n\n{self._cutoff_note(nganh)}",
                        intent="tu_van_chon_nganh", sources=self._cutoff_facts(nganh))

    def _cutoff_facts(self, nganh: str | None = None) -> list[dict]:
        return self.kb.query("diem_chuan", nganh).facts

    def _cutoff_note(self, nganh: str | None = None) -> str:
        facts = self._cutoff_facts(nganh)
        if not facts:
            return ""
        lines = [f"- {self.kb.major_name(f['nganh'])}: {f.get('gia_tri')} điểm\n  {cite(f)}" for f in facts]
        return "_So sánh với điểm chuẩn năm gần nhất:_\n" + "\n".join(lines)

    # ------------------------------------------------------------------ C1: lộ trình học
    def make_plan_from_slots(self, slots: dict) -> BotReply:
        def mo_rong(ds):  # "@HK1" = toàn bộ học phần HK1 do Trường xếp
            out = set()
            for m in ds or []:
                out |= set(self.hk1) if m == "@HK1" else {m}
            return out
        rot = mo_rong(slots.get("bi_rot"))
        return self.make_plan(da_dat=mo_rong(slots.get("da_dat")) - rot, rot=rot,
                              ky=int(slots.get("ky") or 1), hoc_vuot=bool(slots.get("hoc_vuot")))

    def make_plan(self, da_dat: set[str], rot: set[str] = frozenset(), ky: int = 1,
                  hoc_vuot: bool = False) -> BotReply:
        """Lập lộ trình. Người dùng KHÔNG chọn thuật toán – bộ điều phối (C1) tự chọn."""
        intent = "lo_trinh_hoc_tap"
        if not self.courses:
            return BotReply("Chưa có dữ liệu học phần (`data/hoc_phan.csv`).", intent=intent)
        loi = kiem_tra_dau_vao(self.courses, set(da_dat), set(rot), ky)
        if loi:
            return BotReply("Thông tin em cung cấp chưa hợp lệ:\n" + "\n".join(f"- {x}" for x in loi) +
                            "\n\nEm kiểm tra lại rồi gõ **\"lộ trình học\"** để nhập lại nhé.", intent=intent)
        bt = tao_bai_toan(self.courses, set(da_dat), set(rot), ky, hoc_vuot)
        max_tc = MAX_TC_HOC_VUOT if hoc_vuot else MAX_TC_BINH_THUONG
        info = (f"Bắt đầu từ **{ten_hoc_ky(ky)}** · tối đa **{max_tc} TC/kỳ**"
                f"{' (học vượt)' if hoc_vuot else ''} · đã đạt {len(da_dat)} học phần"
                f"{f' · rớt {len(rot)} học phần' if rot else ''} · còn **{bt.tin_chi_con_lai(bt.trang_thai_dau())} TC**.")
        try:
            kq_dp = dieu_phoi.lap_lo_trinh(bt, doi_chieu=self.debug)
        except NotImplementedError:
            return BotReply(f"{info}\n\n" + NOT_READY.format(
                name="lập lộ trình học", where="`src/planner/` (bfs.py, a_sao.py, nhanh_can.py) – Chức năng 1"),
                intent=intent)
        kq = kq_dp.ket_qua
        if kq is None or kq.lo_trinh is None:
            return BotReply(f"{info}\n\nMình chưa tìm được lộ trình phù hợp. Em nên gặp cố vấn học tập để được hỗ trợ.",
                            intent=intent)
        if not kq.lo_trinh:
            return BotReply(f"{info}\n\n🎉 Em đã hoàn thành toàn bộ học phần mục tiêu!", intent=intent)

        rows = ["| Học kỳ | Học phần | TC |", "|---|---|---|"]
        for hk, tap in kq.lo_trinh:
            mon = "; ".join(f"{m} – {self.courses[m].ten}" + (" *(học lại)*" if m in bt.hoc_phan_rot else "")
                              for m in sorted(tap, key=lambda x: self.courses[x].stt)) or "*(chờ kỳ sau)*"
            nhan = " 🏫 Trường xếp" if hk in bt.truong_xep else ""
            rows.append(f"| {ten_hoc_ky(hk)}{nhan} | {mon} | {sum(self.courses[m].tin_chi for m in tap)} |")
        ket_luan = ("✅ Đây là lộ trình **ít học kỳ nhất** để hoàn thành chương trình." if kq.toi_uu else
                    "⚠️ Bài toán lớn nên hệ thống trả về lộ trình tốt nhất tìm được trong thời gian cho phép; "
                    "có thể chưa phải ít học kỳ nhất. Em nên trao đổi thêm với cố vấn học tập.")
        text = (f"{info}\n\n**Lộ trình đề xuất: {kq.so_ky} học kỳ** "
                f"(đến hết {ten_hoc_ky(kq.lo_trinh[-1][0])})\n\n" + "\n".join(rows) + f"\n\n{ket_luan}"
                f"\n\n📌 Nguồn: {NGUON_CTDT} – năm áp dụng {NAM_AP_DUNG}"
                "\n\n**Giả định:**\n" + "\n".join(f"- {g}" for g in GIA_DINH))
        if self.debug:
            dong = [f"Bộ giải: **{kq.thuat_toan}** · {kq.so_nut_mo_rong} nút mở rộng · {kq.so_nut_sinh} nút sinh · "
                    f"{kq.thoi_gian_ms:.1f} ms"]
            dong += kq_dp.ghi_chu
            if kq_dp.doi_chieu is not None:
                dc = kq_dp.doi_chieu
                dong.append(f"Đối chiếu BFS: {dc.so_ky} kỳ, {dc.so_nut_mo_rong} nút, {dc.thoi_gian_ms:.1f} ms – "
                            + ("✔ khớp" if kq_dp.khop_doi_chieu else "✘ KHÔNG khớp"))
            text += "\n\n🔎 " + "  \n".join(dong)
        return BotReply(text, intent=intent, debug={"thuat_toan": kq.thuat_toan, "so_ky": kq.so_ky})


def module_status() -> dict[str, bool]:
    """Kiểm tra mô-đun nào đã được cài đặt (dùng cho sidebar). True = đã cài."""
    def ok(fn) -> bool:
        try:
            fn()
            return True
        except NotImplementedError:
            return False
        except Exception:
            return True  # đã cài nhưng lỗi với đầu vào thử – coi là đã có

    status = {f"C1 – {ten}": co for ten, co in dieu_phoi.thuat_toan_san_co().items()}
    status["C2 – Suy diễn tiến"] = ok(lambda: inference.forward_chaining(set()))
    status["C2 – Suy diễn lùi"] = ok(lambda: inference.backward_chaining("du_dk_cntt", set()))
    return status


__all__ = ["ChatEngine", "module_status", "SO_THICH", "DINH_HUONG", "latest_cutoffs"]
