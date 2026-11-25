"""Kiểm thử giao diện web Flask (Chức năng 4), demo.py và data/sample/."""
import filecmp
from pathlib import Path

import pandas as pd
import pytest

from app import CAU_HOI_THUONG_GAP, GOI_Y, create_app
from src.nlu.classifier import KeywordClassifier, load_classifier
from src.web.markdown_lite import to_html

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path):
    app = create_app(KeywordClassifier(), log_dir=tmp_path)
    app.config["LOG_DIR"] = tmp_path
    return app.test_client()


def _chat(c, text, cid=None, client_id="test"):
    r = c.post("/api/chat", json={"client_id": client_id, "conversation_id": cid, "message": text})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_trang_chu_theo_figma(client):
    html = client.get("/").get_data(as_text=True)
    for s in ["HCMUE FIT Chat", "Trợ lý Tuyển sinh Khoa CNTT", "Cuộc hội thoại mới", "GẦN ĐÂY",
              "CÂU HỎI THƯỜNG GẶP", "LIÊN HỆ KHOA CNTT", "Gợi ý câu hỏi", "Làm mới hội thoại", "Gửi",
              "logo_hcmue.png", "logo_khoa_cntt.png"]:
        assert s in html
    assert "<script>alert" not in html


def test_chat_so_lieu_co_nguon_va_nam(client):
    j = _chat(client, "Điểm chuẩn ngành CNTT năm 2024")
    r = j["reply"]
    assert r["intent"] == "diem_chuan" and r["has_source"]
    assert 'class="source"' in r["html"] and "năm áp dụng 2024" in r["html"]


def test_html_duoc_escape(client):
    j = _chat(client, "<script>alert(1)</script> điểm chuẩn cntt")
    assert "<script>" not in j["reply"]["html"]
    assert "<script>" not in to_html("**a** <img src=x onerror=alert(1)>")


def test_hoi_thoai_nhieu_luot_va_danh_sach(client):
    j = _chat(client, "Em có đủ điều kiện vào sư phạm tin không?")
    cid = j["conversation"]["id"]
    assert j["reply"]["flow"] == "kiem_tra_dieu_kien"
    for t in ["thpt", "Toán 7, Anh 6.2, Lý 6"]:
        _chat(client, t, cid)
    j = _chat(client, "KV1", cid)
    assert j["reply"]["flow"] is None and "Nguồn" in j["reply"]["html"]
    ds = client.get("/api/hoi-thoai?client_id=test").get_json()
    assert ds[0]["id"] == cid and ds[0]["title"].startswith("Em có đủ điều kiện")
    msgs = client.get(f"/api/hoi-thoai/{cid}?client_id=test").get_json()["messages"]
    assert len(msgs) == 8 and msgs[0]["role"] == "user"
    assert client.get(f"/api/hoi-thoai/{cid}?client_id=khac").status_code == 404   # không xem được của người khác


def test_bieu_mau_c2_tien_va_lui(client):
    r = client.post("/api/tu-van", json={"client_id": "t", "phuong_thuc": "THPT", "khu_vuc": "KV3",
                                          "diem": {"Toan": 8, "Anh": 7, "Ly": 7}, "so_thich": ["thich_lap_trinh"],
                                          "dinh_huong": ["lap_trinh_vien"], "kiem_tra_nganh": "cntt"})
    out = r.get_json()
    assert r.status_code == 200 and len(out) == 2
    assert "suy diễn tiến" in out[0]["reply"]["html"] and "<table>" in out[0]["reply"]["html"]
    assert "<strong>Có.</strong>" in out[1]["reply"]["html"]


def test_bieu_mau_c2_bao_loi_du_lieu(client):
    r = client.post("/api/tu-van", json={"client_id": "t", "phuong_thuc": "THPT", "khu_vuc": "KV3",
                                          "diem": {"Toan": 12}})
    assert r.status_code == 400 and "Toán" in r.get_json()["error"]


def test_bieu_mau_c1(client):
    hk1 = ["POLI2001", "POLI1903", "PSYC1001", "COMP1010", "COMP1802", "COMP1801"]
    r = client.post("/api/lo-trinh", json={"client_id": "t", "hoc_ky": 2, "da_dat": hk1, "rot": ["COMP1800"]})
    html = r.get_json()["reply"]["html"]
    assert "Lộ trình đề xuất: 4 học kỳ" in html and "(học lại)" in html and "năm áp dụng 2024" in html


def test_danh_gia_ghi_feedback(client, tmp_path):
    j = _chat(client, "Học phí ngành CNTT bao nhiêu?")
    r = client.post("/api/danh-gia", json={"client_id": "test", "conversation_id": j["conversation"]["id"],
                                            "index": j["reply"]["index"], "value": "tot"})
    assert r.status_code == 200
    assert "Học phí ngành CNTT bao nhiêu?" in (tmp_path / "feedback.csv").read_text(encoding="utf-8")


def test_trang_thai(client):
    st = client.get("/api/trang-thai").get_json()
    assert st["mo_dun"]["C1 – Lộ trình (A*, nhánh cận)"].startswith("sẵn sàng")


@pytest.mark.skipif(not (ROOT / "models" / "intent_model.keras").exists(), reason="chưa có mô hình")
def test_cau_goi_y_tren_giao_dien_deu_tra_loi_duoc(tmp_path):
    """Mọi câu gợi ý / câu hỏi thường gặp hiển thị trên giao diện phải được trả lời bằng mô hình Keras."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("underthesea")
    c = create_app(load_classifier(backend="keras"), log_dir=tmp_path).test_client()
    for cau in GOI_Y + [q for _, q in CAU_HOI_THUONG_GAP]:
        j = _chat(c, cau, client_id=cau)
        assert not j["reply"]["handoff"], cau


# ---------------------------------------------------------------- data/sample + demo.py
def test_du_lieu_mau_khop_du_lieu_day_du():
    s = ROOT / "data" / "sample"
    assert filecmp.cmp(s / "hoc_phan.csv", ROOT / "data" / "hoc_phan.csv", shallow=False)
    assert filecmp.cmp(s / "survey.csv", ROOT / "data" / "survey.csv", shallow=False)
    full = set(pd.read_csv(ROOT / "data" / "intents.csv", encoding="utf-8-sig")["text"])
    test = pd.read_csv(s / "intents_test.csv")
    assert len(test) == 99 and set(test["text"]) <= full and test["label"].nunique() == 11
    oos = set(pd.read_csv(ROOT / "data" / "out_of_scope.csv", encoding="utf-8-sig")["text"])
    assert set(pd.read_csv(s / "out_of_scope_test.csv")["text"]) <= oos


def test_demo_chay_du_ba_chuc_nang(capsys):
    pytest.importorskip("tensorflow")
    pytest.importorskip("underthesea")
    import demo

    demo.main([])
    out = capsys.readouterr().out
    for s in ["CHỨC NĂNG 1", "BFS", "A*", "Nhánh cận", "Ba thuật toán cùng ra", "CHỨC NĂNG 2", "Kết luận: Có",
              "Kết luận: Không", "Nút gốc", "CHỨC NĂNG 3", "Macro F1", "KHỚP"]:
        assert s in out, s
