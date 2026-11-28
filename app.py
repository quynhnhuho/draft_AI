"""Chức năng 4 – Chatbot tư vấn tuyển sinh Khoa CNTT (web Flask + HTML/CSS/JS).

    python app.py                 → mở http://127.0.0.1:5000
    python app.py --port 8000     → đổi cổng
    python app.py --nb            → dùng baseline Naive Bayes (máy không cài được TensorFlow)

Web chỉ là giao diện: mọi xử lý nằm trong mã Python của nhóm
  câu hỏi → src/nlu (C3, mô hình Keras models/intent_model.keras) → src/chatbot/engine.py
          → trả lời từ data/knowledge_base.json (kèm nguồn + năm áp dụng)
          → hoặc gọi src/advisor (C2, suy diễn tiến/lùi) / src/planner (C1, A* + nhánh cận).
Không dùng dịch vụ ngoài, không cần khóa API.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import threading
import uuid
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, abort, jsonify, render_template, request, send_from_directory  # noqa: E402

from src.advisor.tham_so import MON, MON_HB2, NAM, TEN_MON, tu_ma_kb  # noqa: E402
from src.chatbot.engine import LOG_DIR, ChatEngine, module_status  # noqa: E402
from src.chatbot.entities import DINH_HUONG, SO_THICH, TEN_DINH_HUONG, TEN_SO_THICH  # noqa: E402
from src.chatbot.kb import KnowledgeBase  # noqa: E402
from src.chatbot.schema import BotReply, DialogueState  # noqa: E402
from src.nlu.classifier import load_classifier, load_threshold  # noqa: E402
from src.web.markdown_lite import to_html  # noqa: E402

ROOT = Path(__file__).resolve().parent

# Câu hỏi gợi ý: chỉ giữ câu đã kiểm tra chatbot trả lời được (tests/test_web.py).
GOI_Y = [
    "Điểm chuẩn ngành Công nghệ Thông tin năm gần nhất?",
    "Các phương thức xét tuyển năm 2026",
    "Học phí và học bổng Khoa CNTT",
    "Nên chọn ngành nào của Khoa CNTT?",
    "Lập lộ trình học cho tân sinh viên",
]

MAX_HOI_THOAI = 30          # mỗi trình duyệt giữ tối đa 30 cuộc hội thoại (bộ nhớ của máy chủ)
MAX_KHACH = 1000


class KhoHoiThoai:
    """Lưu hội thoại trong bộ nhớ theo client_id (trình duyệt tự sinh, lưu ở localStorage)."""

    def __init__(self):
        self._data: dict[str, dict[str, dict]] = {}
        self.lock = threading.RLock()

    def danh_sach(self, client: str) -> list[dict]:
        with self.lock:
            hs = sorted(self._data.get(client, {}).values(), key=lambda h: h["updated"], reverse=True)
            return [{"id": h["id"], "title": h["title"], "updated": h["updated"]} for h in hs if h["messages"]]

    def lay(self, client: str, cid: str | None, tao_moi: bool = True) -> dict | None:
        with self.lock:
            kho = self._data.get(client)
            if kho is None:
                if len(self._data) >= MAX_KHACH:
                    self._data.pop(next(iter(self._data)))
                kho = self._data[client] = {}
            if cid and cid in kho:
                return kho[cid]
            if not tao_moi:
                return None
            if len(kho) >= MAX_HOI_THOAI:
                kho.pop(min(kho, key=lambda k: kho[k]["updated"]))
            now = dt.datetime.now().isoformat(timespec="seconds")
            h = {"id": uuid.uuid4().hex[:12], "title": "Cuộc hội thoại mới", "created": now, "updated": now,
                 "messages": [], "state": DialogueState()}
            kho[h["id"]] = h
            return h


def _gio() -> str:
    return dt.datetime.now().strftime("%H:%M")


def _json_reply(reply: BotReply, state: DialogueState, index: int) -> dict:
    return {"role": "bot", "index": index, "time": _gio(), "html": to_html(reply.text),
            "intent": reply.intent, "confidence": round(float(reply.confidence or 0), 4), "handoff": reply.handoff,
            "has_source": bool(reply.sources), "flow": state.active_flow}


def _so(v, ten: str, lo: float = 0.0, hi: float = 10.0) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(str(v).replace(",", "."))
    except ValueError:
        raise ValueError(f"{ten} phải là số") from None
    if not lo <= x <= hi:
        raise ValueError(f"{ten} phải từ {lo:g} đến {hi:g}")
    return x


def create_app(classifier=None, log_dir: Path = LOG_DIR) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.json.ensure_ascii = False
    clf = classifier or load_classifier()
    kb = KnowledgeBase()
    tau_mac_dinh = load_threshold(classifier=clf)
    engine = ChatEngine(kb, clf, tau_mac_dinh, log_dir=log_dir)
    kho = KhoHoiThoai()
    engine_lock = threading.Lock()           # một engine dùng chung giữa các request
    app.config.update(ENGINE=engine, KHO=kho, TAU=tau_mac_dinh)

    def ghi_log(file: str, row: dict) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def hoi_thoai_cua(body: dict) -> dict:
        client = str(body.get("client_id") or "")[:64]
        if not client:
            abort(400, "thiếu client_id")
        return kho.lay(client, body.get("conversation_id"))

    def them(h: dict, user_text: str, reply: BotReply) -> dict:
        with kho.lock:
            h["messages"].append({"role": "user", "text": user_text, "time": _gio()})
            out = _json_reply(reply, h["state"], len(h["messages"]))
            h["messages"].append(out)
            if h["title"] == "Cuộc hội thoại mới":
                h["title"] = user_text if len(user_text) <= 48 else user_text[:45].rstrip() + "…"
            h["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        ghi_log("conversations.jsonl", {"t": h["updated"], "hoi_thoai": h["id"], "user": user_text,
                                        "intent": reply.intent, "conf": out["confidence"],
                                        "top_k": [[i, round(float(p), 4)] for i, p in reply.top_k],
                                        "nguon_y_dinh": reply.debug.get("nguon_y_dinh"),
                                        "ms": reply.debug.get("thoi_gian_ms"), "handoff": reply.handoff})
        return {"conversation": {"id": h["id"], "title": h["title"], "updated": h["updated"]},
                "user": h["messages"][-2], "reply": out}

    @app.get("/favicon.ico")
    def favicon():
        # biểu tượng tab = logo Khoa CNTT (static/favicon.ico, tạo từ static/img/logo_khoa_cntt.png)
        return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/x-icon")

    @app.get("/")
    def trang_chu():
        ds_hp = sorted(engine.courses.values(), key=lambda h: h.stt)
        return render_template(
            "index.html", nam=NAM, khoa=kb.meta.get("khoa", "Khoa Công nghệ Thông tin"),
            truong=kb.meta.get("truong", ""), goi_y=GOI_Y, lien_he=kb.lien_he(),
            hom_nay=dt.date.today().strftime("%d/%m/%Y"),
            mon=[(m, TEN_MON[m]) for m in MON], mon_hb2=[(m, TEN_MON[m]) for m in MON_HB2],
            so_thich=[(k, TEN_SO_THICH[k]) for k in SO_THICH], dinh_huong=[(k, TEN_DINH_HUONG[k]) for k in DINH_HUONG],
            nganh=[(c, m["ten"]) for c, m in kb.majors.items() if tu_ma_kb(c)],
            hoc_phan=[(h.ma, h.ten, h.tin_chi) for h in ds_hp],
            hk1=sorted(engine.hk1, key=lambda m: engine.courses[m].stt))

    @app.post("/api/chat")
    def chat():
        body = request.get_json(silent=True) or {}
        text = str(body.get("message") or "").strip()[:1000]
        if not text:
            return jsonify(error="Câu hỏi trống"), 400
        h = hoi_thoai_cua(body)
        with engine_lock:
            reply = engine.respond(text, h["state"])
        return jsonify(them(h, text, reply))

    @app.post("/api/tu-van")
    def tu_van():
        """Biểu mẫu Chức năng 2: suy diễn tiến (gợi ý ngành) + suy diễn lùi (nếu chọn ngành cần kiểm tra)."""
        body = request.get_json(silent=True) or {}
        h = hoi_thoai_cua(body)
        try:
            pt = body.get("phuong_thuc")
            if pt not in ("THPT", "DGNL"):
                raise ValueError("Chọn phương thức THPT hoặc ĐGNL")
            kv = body.get("khu_vuc")
            if kv not in ("KV1", "KV2-NT", "KV2", "KV3"):
                raise ValueError("Chọn khu vực ưu tiên")
            user = {"phuong_thuc": pt, "khu_vuc": kv,
                    **{k: k in (body.get("so_thich") or []) for k in SO_THICH},
                    **{k: k in (body.get("dinh_huong") or []) for k in DINH_HUONG}}
            if pt == "THPT":
                diem = body.get("diem") or {}
                user |= {f"diem_{m}": _so(diem.get(m), f"Điểm {TEN_MON[m]}") for m in MON}
            else:
                if body.get("mon_hb2") not in MON_HB2:
                    raise ValueError("Chọn môn học bạ thứ hai")
                user |= {"diem_dgnl": _so(body.get("diem_dgnl"), "Điểm ĐGNL"),
                         "HB_mon1": _so(body.get("hb1"), "Học bạ Tiếng Anh"),
                         "mon_hb2": body["mon_hb2"], "HB_mon2": _so(body.get("hb2"), "Học bạ môn thứ hai")}
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        nganh = body.get("kiem_tra_nganh") or ""
        with engine_lock:
            replies = [engine.advise(user)]
            if nganh in kb.majors and tu_ma_kb(nganh):
                replies.append(engine.check_eligibility(nganh, user))
        out = [them(h, "Tư vấn chọn ngành cho mình", replies[0])]
        if len(replies) > 1:
            out.append(them(h, f"Mình có đủ điều kiện xét tuyển ngành {kb.major_name(nganh)} không?", replies[1]))
        return jsonify(out)

    @app.post("/api/lo-trinh")
    def lo_trinh():
        """Biểu mẫu Chức năng 1: lập lộ trình (bộ điều phối tự chọn A* / nhánh cận)."""
        body = request.get_json(silent=True) or {}
        h = hoi_thoai_cua(body)
        try:
            ky = int(body.get("hoc_ky") or 1)
        except (TypeError, ValueError):
            return jsonify(error="Học kỳ phải là số nguyên"), 400
        da_dat = {str(m) for m in body.get("da_dat") or []}
        rot = {str(m) for m in body.get("rot") or []}
        with engine_lock:
            reply = engine.make_plan(da_dat=da_dat, rot=rot, ky=ky, hoc_vuot=bool(body.get("hoc_vuot")))
        return jsonify(them(h, f"Lập lộ trình học cho mình từ học kỳ {ky}", reply))

    @app.get("/api/hoi-thoai")
    def ds_hoi_thoai():
        return jsonify(kho.danh_sach(str(request.args.get("client_id") or "")))

    @app.get("/api/hoi-thoai/<cid>")
    def mot_hoi_thoai(cid: str):
        h = kho.lay(str(request.args.get("client_id") or ""), cid, tao_moi=False)
        if h is None:
            return jsonify(error="Không tìm thấy hội thoại"), 404
        return jsonify({"id": h["id"], "title": h["title"], "messages": h["messages"]})

    @app.post("/api/danh-gia")
    def danh_gia():
        """Nút Hữu ích / Chưa đúng dưới mỗi câu trả lời → logs/feedback.csv."""
        body = request.get_json(silent=True) or {}
        h = kho.lay(str(body.get("client_id") or ""), body.get("conversation_id"), tao_moi=False)
        try:
            i = int(body.get("index"))
            msg = h["messages"][i]
            assert msg["role"] == "bot" and body.get("value") in ("tot", "chua_tot")
        except (TypeError, ValueError, IndexError, AssertionError, KeyError):
            return jsonify(error="Đánh giá không hợp lệ"), 400
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "feedback.csv"
        moi = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if moi:
                w.writerow(["thoi_gian", "cau_hoi", "y_dinh", "do_tin_cay", "chuyen_can_bo", "danh_gia"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), h["messages"][i - 1]["text"],
                        msg["intent"], msg["confidence"], msg["handoff"], body["value"]])
        msg["feedback"] = body["value"]
        return jsonify(ok=True)

    @app.get("/api/trang-thai")
    def trang_thai():
        return jsonify(mo_hinh=clf.name, tau=tau_mac_dinh, canh_bao=getattr(clf, "warnings", []),
                       mo_dun=module_status(clf, kb))

    @app.errorhandler(400)
    def loi_400(e):
        return jsonify(error=getattr(e, "description", "Yêu cầu không hợp lệ")), 400

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Chatbot tư vấn tuyển sinh Khoa CNTT – HCMUE")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--nb", action="store_true", help="dùng baseline Naive Bayes thay cho mô hình Keras")
    args = ap.parse_args()
    print("Đang nạp mô hình phân loại ý định (lần đầu có thể mất 10–30 giây)…", flush=True)
    app = create_app(load_classifier(backend="nb") if args.nb else None)
    print(f"Bộ phân loại: {app.config['ENGINE'].classifier.name} · τ = {app.config['TAU']}")
    print(f"Mở trình duyệt tại: http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
