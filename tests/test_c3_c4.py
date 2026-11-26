"""Kiểm thử phần nối C3 → C4: tiền xử lý, bộ phân loại đã huấn luyện, chặn phạm vi, luật, kịch bản end-to-end.

Các test cần TensorFlow / scikit-learn / PyYAML sẽ tự bỏ qua (skip) nếu máy chưa cài.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.chatbot.engine import ChatEngine, module_status
from src.chatbot.entities import _find_majors
from src.chatbot.kb import KnowledgeBase
from src.chatbot.pham_vi import kiem_tra_pham_vi
from src.chatbot.schema import DialogueState
from src.nlu.classifier import (KeywordClassifier, KerasTokenizerLite, Prediction, load_classifier,
                                load_threshold, pad, read_config)
from src.nlu.preprocess import normalize, preprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
co_mo_hinh = pytest.mark.skipif(not (MODELS / "config.json").exists(), reason="chưa có models/config.json")


# ---------------------------------------------------------------- tiền xử lý (mục C)
def test_teencode_khong_lam_hong_khoi_thi():
    assert normalize("Khối A, khối B có xét không") == "khối a khối b có xét không"
    assert normalize("hp 1 tc bn tiền") == "học phí 1 tín chỉ bao nhiêu tiền"
    assert normalize("đk nhận hb") == "điều kiện nhận học bổng"
    assert normalize("tp.hcm") == "thành phố hồ chí minh"


def test_preprocess_tach_tu_neu_co_underthesea():
    pytest.importorskip("underthesea")
    assert preprocess("Học phí ngành CNTT") == "học_phí ngành công_nghệ_thông_tin"


# ---------------------------------------------------------------- bộ phân loại (C3)
@co_mo_hinh
def test_config_khop_labels_va_tokenizer():
    cfg = read_config(MODELS)
    labels = json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))
    assert cfg["labels"] == labels and len(labels) == 11
    assert {"tu_van_chon_nganh", "lo_trinh_hoc_tap"} <= set(labels)
    assert 0 < cfg["threshold"] < 1


@co_mo_hinh
def test_tokenizer_lite_giong_keras():
    keras_text = pytest.importorskip("tensorflow.keras.preprocessing.text")
    tok_json = (MODELS / "tokenizer.json").read_text(encoding="utf-8")
    ref = keras_text.tokenizer_from_json(tok_json)
    lite = KerasTokenizerLite(MODELS / "tokenizer.json")
    cau = [preprocess(s) for s in ["Học phí ngành CNTT bao nhiêu", "từ_lạ_chưa_gặp xyz", "xin chào"]]
    assert lite.texts_to_sequences(cau) == ref.texts_to_sequences(cau)


def test_pad_giong_keras():
    x = pad([[1, 2, 3], [4, 5, 6, 7, 8, 9]], 5)
    assert x.tolist() == [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]


@co_mo_hinh
@pytest.mark.parametrize("backend", ["keras", "nb"])
def test_mo_hinh_tai_tao_dung_so_lieu_test(backend):
    """Tiền xử lý lúc chạy phải giống lúc huấn luyện → tái tạo đúng accuracy trong metrics_c3.json."""
    pytest.importorskip("sklearn")
    if backend == "keras":
        pytest.importorskip("tensorflow")
    pytest.importorskip("underthesea")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(ROOT / "data" / "intents.csv", encoding="utf-8-sig")
    df["c"] = [preprocess(t) for t in df["text"]]
    df = df.drop_duplicates("c").reset_index(drop=True)
    y = LabelEncoder().fit_transform(df["label"])
    idx = np.arange(len(df))
    _, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)

    clf = load_classifier(MODELS, backend=backend)
    pred = [clf.labels[i] for i in clf.predict_proba(list(df["c"][te])).argmax(1)]
    acc = float(np.mean(np.array(pred) == df["label"].values[te]))
    ten = {"keras": read_config(MODELS)["best_model"], "nb": "NaiveBayes"}[backend]
    ky_vong = json.loads((MODELS / "metrics_c3.json").read_text(encoding="utf-8"))["models"][ten]["test_acc"]
    assert acc == pytest.approx(ky_vong, abs=1e-6)


@co_mo_hinh
def test_hop_dong_predict_va_nguong():
    pytest.importorskip("sklearn")
    clf = load_classifier(MODELS, backend="nb")
    p = clf.predict("học phí ngành công nghệ thông tin bao nhiêu")
    assert isinstance(p, Prediction) and p.intent == "hoc_phi" and len(p.top_k) == 3
    assert load_threshold(MODELS, classifier=clf) == read_config(MODELS)["thresholds_per_model"]["NaiveBayes"]


def test_khong_co_mo_hinh_thi_dung_tu_khoa(tmp_path):
    clf = load_classifier(tmp_path)
    assert isinstance(clf, KeywordClassifier) and load_threshold(tmp_path, classifier=clf) == 0.5


# ---------------------------------------------------------------- chặn phạm vi + luật (C4)
@pytest.fixture
def kb():
    return KnowledgeBase()


@pytest.fixture
def engine(tmp_path, kb):
    return ChatEngine(kb, KeywordClassifier(), threshold=0.5, log_dir=tmp_path)


def _pham_vi(kb, text):
    return kiem_tra_pham_vi(normalize(text), kb.majors, lambda s: _find_majors(s, kb.majors))


@pytest.mark.parametrize("text,code", [
    ("điểm chuẩn ngành sư phạm toán năm 2025", "nganh_khac"),
    ("học phí ngành công nghệ sinh học", "nganh_khac"),
    ("điểm chuẩn CNTT của Bách khoa", "truong_khac"),
    ("Trường Đại học Kinh tế có ngành marketing không", "truong_khac"),
])
def test_chan_nganh_truong_khac(kb, text, code):
    assert _pham_vi(kb, text)[0] == code


@pytest.mark.parametrize("text", ["điểm chuẩn ngành công nghệ thông tin", "ngành công nghệ giáo dục học gì",
                                  "em nên chọn ngành nào", "tư vấn chọn ngành giúp em", "ngành sư phạm tin",
                                  "Trường Đại học Sư phạm TP.HCM có ký túc xá không"])
def test_khong_chan_nham(kb, text):
    assert _pham_vi(kb, text) is None


def test_khong_chan_nham_cau_trong_tap_du_lieu(kb):
    df = pd.read_csv(ROOT / "data" / "intents.csv", encoding="utf-8-sig")
    assert [t for t in df["text"] if _pham_vi(kb, t)] == []


def test_nganh_khac_chuyen_can_bo_khong_tra_so_lieu_cntt(engine, tmp_path):
    r = engine.respond("điểm chuẩn ngành sư phạm toán năm 2025", DialogueState())
    assert r.handoff and not r.sources and "Công nghệ thông tin năm" not in r.text
    assert "nganh_khac" in (tmp_path / "handoff.csv").read_text(encoding="utf-8")


def test_luat_uu_tien_du_dieu_kien_bat_ke_mo_hinh(tmp_path, kb):
    class Sai:  # mô hình đoán sai, tự tin thấp
        name = "gia"

        def predict(self, text, k=3):
            return Prediction("viec_lam", 0.3, [("viec_lam", 0.3)])

    eng = ChatEngine(kb, Sai(), threshold=0.9, log_dir=tmp_path)
    s = DialogueState()
    r = eng.respond("Em có đủ điều kiện vào sư phạm tin không?", s)
    assert s.active_flow == "kiem_tra_dieu_kien" and r.debug["nguon_y_dinh"] == "luat_uu_tien"


def test_luat_bo_tro_chi_xac_nhan_y_dinh_trong_top3(tmp_path, kb):
    class Thap:
        name = "gia"

        def __init__(self, top):
            self.top = top

        def predict(self, text, k=3):
            return Prediction(self.top[0][0], self.top[0][1], self.top)

    s = DialogueState()
    eng = ChatEngine(kb, Thap([("viec_lam", 0.4), ("lo_trinh_hoc_tap", 0.3)]), 0.9, log_dir=tmp_path)
    r = eng.respond("em rớt môn thì học kỳ sau đăng ký gì", s)
    assert s.active_flow == "lo_trinh" and r.debug["nguon_y_dinh"] == "luat_bo_tro"
    eng = ChatEngine(kb, Thap([("viec_lam", 0.4), ("hoc_phi", 0.3)]), 0.9, log_dir=tmp_path)
    assert eng.respond("em rớt môn thì học kỳ sau đăng ký gì", DialogueState()).handoff


def test_so_sanh_hai_nganh_khong_mo_luong(tmp_path, kb):
    class TuVan:
        name = "gia"

        def predict(self, text, k=3):
            return Prediction("tu_van_chon_nganh", 0.99, [("tu_van_chon_nganh", 0.99)])

    s = DialogueState()
    r = ChatEngine(kb, TuVan(), 0.9, log_dir=tmp_path).respond("CNTT khác Sư phạm Tin học chỗ nào?", s)
    assert s.active_flow is None and "Sư phạm Tin học" in r.text and "Công nghệ thông tin" in r.text


def test_lien_he_khoa_va_truong(kb):
    t = kb.contact_text()
    assert "<" not in t and "khoacntt@hcmue.edu.vn" in t and "tuvantuyensinh@hcmue.edu.vn" in t
    assert [ten for ten, _ in kb.lien_he()][0] == "Khoa Công nghệ thông tin"


def test_chuyen_can_bo_co_lien_he(engine):
    r = engine.respond("hôm nay trời có mưa không", DialogueState())
    assert r.handoff and "028 38352020" in r.text and "028 38355077" in r.text


def test_module_status(kb):
    st = module_status(KeywordClassifier(), kb)
    assert st["C1 – Lộ trình (A*, nhánh cận)"].startswith("sẵn sàng")
    assert st["C2 – Hệ luật (suy diễn tiến/lùi)"].startswith("sẵn sàng")


# ---------------------------------------------------------------- kịch bản end-to-end (mục 7.9)
@co_mo_hinh
def test_kich_ban_end_to_end():
    pytest.importorskip("yaml")
    pytest.importorskip("tensorflow")
    pytest.importorskip("underthesea")
    from src.chatbot.danh_gia import main

    th = main(["--backend", "keras"])
    assert th["so_lieu_co_nguon"][0] == th["so_lieu_co_nguon"][1]          # 100% số liệu có nguồn + năm
    assert th["chuyen_dung"] == th["chuyen_can"]                            # không trả lời bừa câu ngoài phạm vi
    dung, tong = th["dung_y_dinh"]
    assert dung / tong >= 0.85


# ---------------------------------------------------------------- học phí (QĐ 2465, QĐ 2849, NĐ 116)
@pytest.mark.parametrize("nganh,nam,gia_tri", [
    ("cntt", 2024, 19680000), ("cntt", 2025, 22200000), ("cntt", 2026, 25080000),
    ("sp_tin", 2024, 14100000), ("sp_tin", 2025, 15900000), ("sp_tin", 2026, 17900000),
    ("cngd", 2025, 15900000), ("cngd", 2026, 17900000),
])
def test_hoc_phi_theo_quyet_dinh(kb, nganh, nam, gia_tri):
    facts = [f for f in kb.query("hoc_phi", nganh, nam).facts if f.get("ap_dung_tu_nam") is None]
    assert [f["gia_tri"] for f in facts] == [gia_tri] and facts[0]["da_kiem_chung"]


def test_nghi_dinh_116_kem_hoc_phi_su_pham_moi_nam(kb):
    for nam in (None, 2024, 2026):
        ids = [f["id"] for f in kb.query("hoc_phi", "sp_tin", nam).facts]
        assert "hotro_sptin_nd116" in ids
    assert "hotro_sptin_nd116" not in [f["id"] for f in kb.query("hoc_phi", "cntt").facts]


def test_nam_khong_co_du_lieu_lay_nam_gan_nhat(kb):
    r = kb.query("hoc_phi", "cngd", 2024)
    assert r.year_fallback and {f["nam"] for f in r.facts} == {2025}
