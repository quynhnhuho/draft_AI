"""Kiểm thử chatbot (C4) với bộ phân loại từ khóa tạm thời.  Chạy: pytest -q"""
import pytest

from src.chatbot.engine import ChatEngine
from src.chatbot.entities import extract, parse_score
from src.chatbot.kb import KnowledgeBase
from src.chatbot.schema import DialogueState
from src.nlu.classifier import KeywordClassifier
from src.nlu.preprocess import normalize


@pytest.fixture
def engine(tmp_path):
    return ChatEngine(KnowledgeBase(), KeywordClassifier(), threshold=0.5, log_dir=tmp_path)


def test_normalize_tone_and_teencode():
    assert normalize("Hòa ko đc") == "hoà không được"


@pytest.mark.parametrize("text,ky_vong", [
    ("Kí túc xá bao nhiêu", "ký túc xá bao nhiêu"),       # chính tả i/y → dạng có trong dữ liệu huấn luyện
    ("học kì 2", "học kỳ 2"), ("vật lí", "vật lý"), ("kĩ năng", "kỹ năng"),
    ("ki tuc xa", "ký túc xá"), ("ky tuc xa o dau", "ký túc xá o dau"),   # gõ không dấu
    ("kiến thức lịch học", "kiến thức lịch học"),          # từ khác có chữ i không bị đổi
])
def test_normalize_chinh_ta_i_y(text, ky_vong):
    assert normalize(text) == ky_vong


def test_entities():
    kb = KnowledgeBase()
    e = extract("Em được 24.5 điểm khối A1, muốn vào CNTT năm 2025", kb.majors)
    assert e.nganh == ["cntt"] and e.to_hop == ["A01"] and e.diem == 24.5 and e.nam == 2025
    assert extract("sp tin lấy bao nhiêu", kb.majors).nganh == ["sp_tin"]
    assert parse_score("23,75") == 23.75


def test_kb_answer_has_source_and_year(engine):
    r = engine.respond("Điểm chuẩn ngành CNTT năm 2024", DialogueState())
    assert r.intent == "diem_chuan" and "Nguồn" in r.text and "2024" in r.text and r.sources


def test_context_carry_over(engine):
    s = DialogueState()
    engine.respond("điểm chuẩn sư phạm tin", s)
    r = engine.respond("thế học phí thì sao", s)
    assert all(f["nganh"] in ("sp_tin", None) for f in r.sources)


def test_out_of_scope_handoff(engine, tmp_path):
    r = engine.respond("hôm nay trời có mưa không", DialogueState())
    assert r.handoff and (tmp_path / "handoff.csv").exists()


def test_eligibility_flow_hoi_theo_suy_dien_lui(engine):
    """Câu hỏi tiếp theo do suy diễn lùi quyết định: phương thức → điểm các môn → khu vực → kết luận."""
    s = DialogueState()
    r = engine.respond("Em có đủ điều kiện vào sư phạm tin không?", s)
    assert s.active_flow == "kiem_tra_dieu_kien" and "phương thức" in r.text
    r = engine.respond("thpt", s)
    assert "điểm thi THPT" in r.text
    r = engine.respond("Toán 7, Anh 6.2, Lý 6", s)
    assert "khu vực" in r.text
    r = engine.respond("KV1", s)
    assert s.active_flow is None and "**Chưa.**" in r.text and "0.05" in r.text and "Nguồn" in r.text


def test_eligibility_dien_san_tu_cau_hoi(engine):
    r = engine.respond("Em thi THPT được Toán 8, Anh 7, Lý 7, KV3, có đủ điều kiện vào CNTT không?", DialogueState())
    assert "**Có.**" in r.text and "Điểm xét tuyển cao nhất" in r.text
    assert not any(k in r.text for k in ("R11", "du_dk", "DXT", "suy diễn"))   # không hiện thuật ngữ kỹ thuật


def test_tu_van_nganh_suy_dien_tien(engine):
    s = DialogueState()
    engine.respond("em nên chọn ngành nào", s)
    for q in ["học bạ + đgnl", "8", "Anh 8.5, Tin 9", "kv2", "lập trình, thiết kế học liệu", "edtech"]:
        r = engine.respond(q, s)
    assert s.active_flow is None
    assert "Công nghệ giáo dục" in r.text and "Rất phù hợp" in r.text


def test_dau_vao_ngoai_khoang_hoi_lai(engine):
    s = DialogueState()
    engine.respond("Em có đủ điều kiện vào CNTT không?", s)
    engine.respond("THPT", s)
    r = engine.respond("Toán 12, Anh 7, Lý 6", s)
    assert "chưa hiểu" in r.text and s.pending_slot == "diem_thpt"


def test_nhan_dung_nganh_cong_nghe_giao_duc():
    kb = KnowledgeBase()
    assert extract("Em có đủ điều kiện vào ngành công nghệ giáo dục không?", kb.majors).nganh == ["cngd"]
    assert extract("ngành công nghệ thông tin", kb.majors).nganh == ["cntt"]
    assert extract("CNTT khác Sư phạm Tin chỗ nào?", kb.majors).nganh == ["cntt", "sp_tin"]


def test_kiem_tra_dieu_kien_dung_nganh_cngd(engine):
    s = DialogueState()
    engine.respond("Em có đủ điều kiện vào ngành công nghệ giáo dục không?", s)
    assert s.slots["nganh"] == "cngd"


def test_co_so_tri_thuc_khop_so_lieu_2026():
    from src.advisor.tham_so import NGANH, TO_HOP, diem_chuan
    kb = KnowledgeBase()
    for x, nganh in NGANH.items():
        dc = {f["phuong_thuc"].upper(): f["gia_tri"] for f in kb.query("diem_chuan", nganh["ma_kb"], 2026).facts}
        assert dc == {pt: float(diem_chuan(x, pt)) for pt in ("THPT", "DGNL")}
        th = kb.query("to_hop_xet_tuyen", nganh["ma_kb"], 2026).facts
        assert th and th[0]["to_hop"] == nganh["to_hop"] and set(th[0]["to_hop"]) <= set(TO_HOP)


# ---------------------------------------------------------------- câu hỏi nhiều ý (phương án A)
def test_cau_hoi_nhieu_y_tra_loi_tung_y(engine):
    r = engine.respond("Học phí và học bổng ngành CNTT?", DialogueState())
    assert r.debug["nguon_y_dinh"] == "nhieu_y_dinh" and r.debug["cac_y_dinh"] == ["hoc_phi", "hoc_bong"]
    assert r.text.index("**Học phí**") < r.text.index("**Học bổng**") and r.text.count("Nguồn") >= 2


@pytest.mark.parametrize("text", [
    "Em thi THPT được Toán 8 và Anh 7, Lý 7, KV3, có đủ điều kiện vào CNTT không?",   # luồng điều kiện
    "học phí ngành CNTT bao nhiêu",                                                    # câu một ý
])
def test_khong_tach_cau_mot_y_hoac_luong_dieu_kien(engine, text):
    assert engine.respond(text, DialogueState()).debug.get("nguon_y_dinh") != "nhieu_y_dinh"


def test_xung_ho_minh_ban(engine):
    import re
    for q in ["xin chào", "cảm ơn", "thời tiết hôm nay thế nào", "Em có đủ điều kiện vào CNTT không?"]:
        t = engine.respond(q, DialogueState()).text
        assert not re.search(r"(?<!\w)[Ee]m(?!\w)", t), (q, t)


def test_lo_trinh_khong_hien_gia_dinh(engine):
    t = engine.make_plan(set(), set(), 1).text
    assert "Giả định" not in t and "nút mở rộng" not in t and "Lộ trình đề xuất" in t


def test_game_la_so_thich_lap_trinh():
    kb = KnowledgeBase()
    assert extract("em thích làm game", kb.majors).so_thich == ["thich_lap_trinh"]
    from src.advisor.tu_van import nganh_theo_so_thich
    ds = nganh_theo_so_thich({"thich_day_hoc": True, "thich_lap_trinh": True})
    assert [(x, manh) for x, manh, _ in ds] == [("SPTH", True), ("CNTT", False)]   # R23 khớp mạnh trước R19
