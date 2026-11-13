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


def test_eligibility_flow_reports_missing_module(engine):
    r = engine.respond("Em được 24 điểm khối A00 có đủ điều kiện vào CNTT không?", DialogueState())
    assert "chưa được cài đặt" in r.text or "điều kiện" in r.text


def test_planner_flow_and_cancel(engine):
    s = DialogueState()
    r = engine.respond("em muốn lập lộ trình học", s)
    assert s.active_flow == "lo_trinh"
    engine.respond("hủy", s)
    assert s.active_flow is None
