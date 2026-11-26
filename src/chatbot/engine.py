# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
from __future__ import annotations

import csv
import datetime as dt
import json
import time
from pathlib import Path

from ..advisor import tu_van as c2
from ..advisor.tham_so import TEN_MON, nguon_trich_dan, ten_nganh, tu_ma_kb
from ..nlu.preprocess import normalize
from ..planner import dieu_phoi
from ..planner.model import (GIA_DINH, MAX_TC_BINH_THUONG, MAX_TC_HOC_VUOT, NAM_AP_DUNG, NGUON_CTDT,
                             doc_du_lieu, hk1_co_dinh, kiem_tra_dau_vao, tao_bai_toan, ten_hoc_ky)
from . import flows
from .entities import DINH_HUONG, SO_THICH, TEN_DINH_HUONG, TEN_SO_THICH, Entities, _find_majors, extract
from .kb import KnowledgeBase, format_facts
from .pham_vi import kiem_tra_pham_vi
from .schema import BotReply, DialogueState

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

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

ELIGIBILITY_CUES = ("đủ điều kiện", "có đậu", "đậu không", "đỗ không", "có đỗ",
                    "vào được", "trúng tuyển", "có vào", "đủ điểm")

# LUẬT ƯU TIÊN (mục 7.2 HUONG_DAN_DO_AN): câu có ngành của Khoa + một cụm hỏi "đủ điều kiện/đậu…"
# → luôn đi vào suy diễn lùi của C2, kể cả khi mô hình đoán ý định khác. Cụm từ hẹp hơn
# ELIGIBILITY_CUES để không bắt nhầm câu điểm chuẩn ("điểm trúng tuyển ngành CNTT…"): trên 660 câu
# của data/intents.csv chỉ trùng 2 câu (đều là "em được 27đ, có đủ điểm vào CNTT không").
ELIGIBILITY_RULE = ("đủ điều kiện vào", "đủ điều kiện xét tuyển", "đủ điều kiện học", "đủ điều kiện đậu",
                    "đủ điểm vào", "có đậu", "đậu không", "đỗ không", "có đỗ", "vào được")

# LUẬT BỔ TRỢ: khi độ tin cậy < τ, chỉ XÁC NHẬN một ý định nằm trong top-3 của mô hình nếu câu chứa
# cụm từ đặc trưng. Hai ý định này chỉ mở luồng hỏi thêm (không đưa ra số liệu) nên rủi ro thấp.
RULE_CUES = {
    "tu_van_chon_nganh": ("nên chọn ngành", "chọn ngành nào", "ngành nào phù hợp", "nên học ngành nào",
                          "tư vấn chọn ngành", "tư vấn ngành", "phân vân giữa"),
    "lo_trinh_hoc_tap": ("lộ trình", "đăng ký học phần", "học vượt", "ra trường đúng hạn", "kế hoạch học tập",
                         "học phần nào", "đăng ký môn", "học lại", "rớt môn", "nợ môn"),
}
COMPARE_CUES = ("khác", "so sánh", "giống", "khác biệt")


class ChatEngine:
    def __init__(self, kb: KnowledgeBase, classifier, threshold: float = 0.6, log_dir: Path = LOG_DIR):
        self.kb = kb
        self.classifier = classifier
        self.threshold = threshold
        self.log_dir = log_dir
        self.debug = False
        try:
            self.courses = doc_du_lieu()
            self.hk1 = hk1_co_dinh(self.courses)
        except (FileNotFoundError, ValueError):
            self.courses, self.hk1 = {}, frozenset()
        self.course_ids = set(self.courses)

    def respond(self, text: str, state: DialogueState) -> BotReply:
        state.turns += 1
        text = text.strip()
        if not text:
            return BotReply("Em hãy nhập câu hỏi nhé.")

        if state.active_flow:
            reply = flows.continue_flow(text, state, self)
            reply.confidence = reply.confidence or 1.0
            return reply

        t0 = time.perf_counter()
        e = extract(text, self.kb.majors, self.courses)
        t = normalize(text)
        pred = self.classifier.predict(text)
        dbg = {"classifier": getattr(self.classifier, "name", "?"), "entities": e.as_dict(),
               "threshold": self.threshold}

        intent, nguon = pred.intent, "mo_hinh"
        if (ngoai := kiem_tra_pham_vi(t, self.kb.majors, lambda s: _find_majors(s, self.kb.majors))):
            intent, nguon = None, "chan_pham_vi"
            reply = self.handoff(text, pred, reason=ngoai[1], code=ngoai[0])
        else:
            if e.nganh and tu_ma_kb(e.nganh[0]) and any(c in t for c in ELIGIBILITY_RULE):
                intent, nguon = "tu_van_chon_nganh", "luat_uu_tien"
            elif pred.confidence < self.threshold:
                intent, nguon = self._rule_intent(t, pred), "luat_bo_tro"
            if intent is None:
                nguon = "duoi_nguong"
                reply = self.handoff(text, pred)
            else:
                if e.nganh:
                    state.last_major = e.nganh[0]
                reply = self._route(intent, text, e, state)

        reply.intent = reply.intent or intent
        reply.confidence, reply.top_k = pred.confidence, pred.top_k
        reply.debug = {**dbg, "nguon_y_dinh": nguon, "y_dinh_mo_hinh": pred.intent,
                       "thoi_gian_ms": round((time.perf_counter() - t0) * 1000, 1), **reply.debug}
        return reply

    @staticmethod
    def _rule_intent(t: str, pred) -> str | None:
        """Luật bổ trợ: xác nhận ý định trong top-3 của mô hình nhờ cụm từ đặc trưng."""
        top = {i for i, _ in pred.top_k}
        for intent, cues in RULE_CUES.items():
            if intent in top and any(c in t for c in cues):
                return intent
        return None

    def compare_majors(self, e: Entities) -> BotReply:
        codes = [c for c in e.nganh if c in self.kb.majors]
        codes = codes if len(codes) >= 2 else list(self.kb.majors)
        parts = [f"**{self.kb.majors[c]['ten']}** (mã ngành {self.kb.majors[c]['ma_nganh']}): "
                 f"{self.kb.majors[c]['mo_ta']} {self.kb.majors[c].get('khac_biet', '')}" for c in codes]
        return BotReply("\n\n".join(parts) + f"\n\nNguồn: mô tả ngành trong cơ sở tri thức của "
                        f"{self.kb.meta.get('khoa', 'Khoa')} (phiên bản {self.kb.meta.get('phien_ban', '?')}) "
                        "– cần đối chiếu chương trình đào tạo chính thức."
                        "\n\nNếu em muốn mình **tư vấn chọn ngành theo điểm và sở thích**, gõ "
                        "\"tư vấn chọn ngành\".", intent="chuong_trinh_dao_tao")

    def _route(self, intent: str, text: str, e: Entities, state: DialogueState) -> BotReply:
        if intent == "chao_hoi":
            return self.greet(text)
        if intent == "tu_van_chon_nganh":
            t = normalize(text)
            if e.nganh and tu_ma_kb(e.nganh[0]) and any(c in t for c in ELIGIBILITY_CUES):
                return flows.start_flow("kiem_tra_dieu_kien", text, e, state, self, extra={"nganh": e.nganh[0]})
            if len(e.nganh) >= 2 and any(c in t for c in COMPARE_CUES) and not e.diem_mon:
                return self.compare_majors(e)   # "CNTT khác Sư phạm Tin chỗ nào?" – so sánh, không mở luồng
            return flows.start_flow("tu_van_nganh", text, e, state, self)
        if intent == "lo_trinh_hoc_tap":
            return flows.start_flow("lo_trinh", text, e, state, self)
        if intent in KB_INTENTS:
            return self.answer_from_kb(intent, text, e, state)
        return self.handoff(text, None)

    def greet(self, text: str) -> BotReply:
        t = normalize(text)
        if any(w in t for w in ("cảm ơn", "cám ơn", "thanks")):
            return BotReply("Không có gì ạ! Chúc em một mùa tuyển sinh thật thành công!")
        if any(w in t for w in ("tạm biệt", "bye")):
            return BotReply("Tạm biệt em, hẹn gặp lại!")
        return BotReply(
            f"Chào em! Mình là trợ lý tư vấn tuyển sinh của {self.kb.meta.get('khoa', 'Khoa CNTT')}. Mình có thể:\n"
            "- Trả lời về **điểm chuẩn, tổ hợp xét tuyển, học phí, học bổng, ký túc xá, chương trình đào tạo, "
            "việc làm, thủ tục nhập học**\n"
            "- **Tư vấn chọn ngành** (CNTT, Sư phạm Tin học, Công nghệ giáo dục) và kiểm tra em có "
            "**đủ điều kiện xét tuyển** không\n"
            "- Gợi ý **lộ trình đăng ký học phần** để ra trường đúng hạn")

    def answer_from_kb(self, intent: str, text: str, e: Entities, state: DialogueState) -> BotReply:
        t = normalize(text)
        if intent == "chuong_trinh_dao_tao" and any(c in t for c in COMPARE_CUES) and len(e.nganh) != 1:
            return self.compare_majors(e)

        nganh = e.nganh[0] if e.nganh else state.last_major
        res = self.kb.query(intent, nganh, e.nam)
        if not res.facts:
            return self.handoff(text, None, code="thieu_du_lieu",
                                reason=f"Hiện mình chưa có dữ liệu về {KB_INTENTS[intent]}"
                                + (f" ngành {self.kb.major_name(nganh)}" if nganh else "") + ".")
        head = ""
        if res.year_fallback:
            years = sorted({f["nam"] for f in res.facts if f.get("ap_dung_tu_nam") is None})
            head = (f"Mình chưa có số liệu năm {res.requested_year}; dưới đây là số liệu năm gần nhất "
                    f"({', '.join(map(str, years))}).\n\n")
        elif e.nam is None and intent in ("diem_chuan", "hoc_phi", "to_hop_xet_tuyen"):
            head = f"Số liệu mới nhất mình có (năm {max(f['nam'] for f in res.facts)}):\n\n"
        return BotReply(head + format_facts(res.facts), sources=res.facts)

    def handoff(self, text: str, pred, reason: str | None = None, code: str = "duoi_nguong") -> BotReply:
        """Chuyển cán bộ: trả lời lịch sự + liên hệ, ghi logs/handoff.csv (hàng đợi + dữ liệu bổ sung C3)."""
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
                        code])
        msg = reason or "Câu hỏi này nằm ngoài phạm vi mình chắc chắn trả lời được."
        return BotReply(f"{msg} Mình đã **chuyển câu hỏi đến cán bộ tư vấn** của Khoa. "
                        f"Em có thể liên hệ trực tiếp:\n\n{self.kb.contact_text()}",
                        intent="chuyen_can_bo", handoff=True)

    @staticmethod
    def _user_summary(user: dict) -> str:
        bits = []
        if user.get("phuong_thuc"):
            bits.append(f"phương thức **{user['phuong_thuc']}**")
        if user.get("khu_vuc"):
            bits.append(f"khu vực **{user['khu_vuc']}**")
        mon = [f"{TEN_MON[m]} {user[f'diem_{m}']:g}" for m in TEN_MON if user.get(f"diem_{m}") is not None]
        if mon:
            bits.append("điểm thi " + ", ".join(mon))
        if user.get("diem_dgnl") is not None:
            bits.append(f"ĐGNL {user['diem_dgnl']:g}, học bạ Tiếng Anh {user.get('HB_mon1')}, "
                        f"{TEN_MON.get(user.get('mon_hb2'), '?')} {user.get('HB_mon2')}")
        st = [TEN_SO_THICH[k] for k in SO_THICH if user.get(k)]
        dh = [TEN_DINH_HUONG[k] for k in DINH_HUONG if user.get(k)]
        if st:
            bits.append("thích **" + ", ".join(st) + "**")
        if dh:
            bits.append("muốn làm **" + ", ".join(dh) + "**")
        return "; ".join(bits) or "(chưa có thông tin)"

    def advise(self, user: dict) -> BotReply:
        kq = c2.tu_van(user)
        text = f"Thông tin của em: {self._user_summary(user)}.\n\n" + c2.markdown_tu_van(kq, vet=self.debug)
        return BotReply(text, intent="tu_van_chon_nganh", sources=[nguon_trich_dan()],
                        debug={"goi_y": [(g.nganh, g.muc, g.diem_phu_hop) for g in kq.goi_y],
                               "so_luat_kich_hoat": len(kq.forward.fired) if kq.forward else 0})

    def check_eligibility(self, nganh_kb: str, user: dict) -> BotReply:
        x = tu_ma_kb(nganh_kb) or nganh_kb
        kq = c2.kiem_tra_dieu_kien(user, x)
        text = (f"Câu hỏi: em có đủ điều kiện xét tuyển ngành **{ten_nganh(x)}** không?  \n"
                f"Thông tin: {self._user_summary(user)}.\n\n" + c2.markdown_kiem_tra(kq, vet=self.debug))
        return BotReply(text, intent="tu_van_chon_nganh", sources=[nguon_trich_dan()],
                        debug={"ket_luan": kq.ket_luan, "can_hoi": kq.can_hoi})

    @staticmethod
    def du_kien_con_thieu(nganh_kb: str, user: dict) -> list[str]:
        return c2.kiem_tra_dieu_kien(user, tu_ma_kb(nganh_kb) or nganh_kb).can_hoi

    def make_plan_from_slots(self, slots: dict) -> BotReply:
        def mo_rong(ds):
            out = set()
            for m in ds or []:
                out |= set(self.hk1) if m == "@HK1" else {m}
            return out
        rot = mo_rong(slots.get("bi_rot"))
        return self.make_plan(da_dat=mo_rong(slots.get("da_dat")) - rot, rot=rot,
                              ky=int(slots.get("ky") or 1), hoc_vuot=bool(slots.get("hoc_vuot")))

    def make_plan(self, da_dat: set[str], rot: set[str] = frozenset(), ky: int = 1,
                  hoc_vuot: bool = False) -> BotReply:
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
        kq_dp = dieu_phoi.lap_lo_trinh(bt)
        kq = kq_dp.ket_qua
        if kq.lo_trinh is None:
            return BotReply(f"{info}\n\nMình chưa tìm được lộ trình phù hợp. Em nên gặp cố vấn học tập để được hỗ trợ.",
                            intent=intent)
        if not kq.lo_trinh:
            return BotReply(f"{info}\n\nEm đã hoàn thành toàn bộ học phần mục tiêu!", intent=intent)

        rows = ["| Học kỳ | Học phần | TC |", "|---|---|---|"]
        for hk, tap in kq.lo_trinh:
            mon = "; ".join(f"{m} – {self.courses[m].ten}" + (" *(học lại)*" if m in bt.hoc_phan_rot else "")
                              for m in sorted(tap, key=lambda x: self.courses[x].stt)) or "*(chờ kỳ sau)*"
            nhan = " (Trường xếp)" if hk in bt.truong_xep else ""
            rows.append(f"| {ten_hoc_ky(hk)}{nhan} | {mon} | {sum(self.courses[m].tin_chi for m in tap)} |")
        ket_luan = ("Đây là lộ trình **ít học kỳ nhất** để hoàn thành chương trình." if kq.toi_uu else
                    "Lưu ý: bài toán lớn nên hệ thống trả về lộ trình tốt nhất tìm được trong thời gian cho phép; "
                    "có thể chưa phải ít học kỳ nhất. Em nên trao đổi thêm với cố vấn học tập.")
        text = (f"{info}\n\n**Lộ trình đề xuất: {kq.so_ky} học kỳ** "
                f"(đến hết {ten_hoc_ky(kq.lo_trinh[-1][0])})\n\n" + "\n".join(rows) + f"\n\n{ket_luan}"
                f"\n\nNguồn: {NGUON_CTDT} – năm áp dụng {NAM_AP_DUNG}"
                "\n\n**Giả định:**\n" + "\n".join(f"- {g}" for g in GIA_DINH))
        if self.debug:
            dong = [f"Bộ giải: **{kq.thuat_toan}** · {kq.so_nut_mo_rong} nút mở rộng · {kq.so_nut_sinh} nút sinh · "
                    f"{kq.thoi_gian_ms:.1f} ms"]
            dong += kq_dp.ghi_chu
            text +="\n\nChi tiết: " + "  \n".join(dong)
        return BotReply(text, intent=intent, sources=[{"nguon": NGUON_CTDT, "nam": NAM_AP_DUNG}],
                        debug={"thuat_toan": kq.thuat_toan, "so_ky": kq.so_ky})


def module_status(classifier=None, kb: KnowledgeBase | None = None) -> dict[str, str]:
    """Trạng thái các mô-đun – hiện ở sidebar để kiểm tra trước khi demo."""
    out: dict[str, str] = {}
    try:
        from ..planner import a_sao, nhanh_can  # noqa: F401
        courses = doc_du_lieu()
        out["C1 – Lộ trình (A*, nhánh cận)"] = f"sẵn sàng · {len(courses)} học phần"
    except Exception as exc:  # hiển thị lỗi thay vì làm sập giao diện
        out["C1 – Lộ trình (A*, nhánh cận)"] = f"lỗi: {exc}"
    try:
        from ..advisor import inference
        from ..advisor.rules import RULES
        assert callable(inference.forward_chaining) and callable(inference.backward_chaining)
        out["C2 – Hệ luật (suy diễn tiến/lùi)"] = f"sẵn sàng · {len(RULES)} luật"
    except Exception as exc:
        out["C2 – Hệ luật (suy diễn tiến/lùi)"] = f"lỗi: {exc}"
    if classifier is not None:
        name = getattr(classifier, "name", "?")
        out["C3 – Phân loại ý định"] = (f"{name}" + (" · CHƯA có mô hình đã huấn luyện"
                                                       if name.startswith("keyword") else ""))
    if kb is not None:
        chua = sum(not f.get("da_kiem_chung", False) for f in kb.facts)
        out["C4 – Cơ sở tri thức"] = f"{len(kb.facts)} mục · {chua} mục chưa đối chiếu nguồn chính thức"
    return out


__all__ = ["ChatEngine", "module_status", "SO_THICH", "DINH_HUONG"]
