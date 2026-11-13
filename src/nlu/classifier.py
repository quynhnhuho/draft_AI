"""Bộ phân loại ý định dùng trong chatbot.

HỢP ĐỒNG với C4:  IntentClassifier.predict(text) -> Prediction(intent, confidence, top_k)

Thứ tự nạp:
  1. Mô hình Keras  : models/intent_model.keras + models/labels.json   (mô hình CHÍNH – C3)
  2. Baseline NB    : models/nb_pipeline.pkl                            (dự phòng)
  3. KeywordClassifier – CHỈ để phát triển giao diện khi chưa có mô hình. Không dùng khi demo.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from .preprocess import normalize, preprocess

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class Prediction:
    intent: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)


class KerasIntentModel:
    name = "keras"

    def __init__(self, model_path: Path, labels_path: Path):
        import tensorflow as tf  # nạp muộn: app vẫn chạy khi chưa cài TF

        self._tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))

    def predict(self, text: str, k: int = 3) -> Prediction:
        probs = self.model.predict(self._tf.constant([preprocess(text)]), verbose=0)[0]
        order = probs.argsort()[::-1][:k]
        top = [(self.labels[i], float(probs[i])) for i in order]
        return Prediction(top[0][0], top[0][1], top)


class NBIntentModel:
    name = "naive_bayes"

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            self.pipe = pickle.load(f)

    def predict(self, text: str, k: int = 3) -> Prediction:
        probs = self.pipe.predict_proba([preprocess(text)])[0]
        order = probs.argsort()[::-1][:k]
        top = [(str(self.pipe.classes_[i]), float(probs[i])) for i in order]
        return Prediction(top[0][0], top[0][1], top)


class KeywordClassifier:
    """Phân loại tạm bằng từ khóa – chỉ để dựng/kiểm thử giao diện C4 trước khi có mô hình."""

    name = "keyword (tạm thời)"
    KEYWORDS = {
        "diem_chuan": ["điểm chuẩn", "lấy bao nhiêu điểm", "bao nhiêu điểm", "điểm trúng tuyển"],
        "to_hop_xet_tuyen": ["tổ hợp", "khối", "xét tuyển môn"],
        "hoc_phi": ["học phí", "tiền học", "miễn phí", "đóng bao nhiêu"],
        "hoc_bong": ["học bổng"],
        "ky_tuc_xa": ["ký túc xá", "kí túc xá", "chỗ ở", "phòng ở"],
        "chuong_trinh_dao_tao": ["chương trình đào tạo", "học những môn", "học môn gì", "khác", "khác nhau", "tín chỉ"],
        "viec_lam": ["việc làm", "ra trường làm", "xin việc", "cơ hội nghề", "lương"],
        "thu_tuc_nhap_hoc": ["nhập học", "hồ sơ", "giấy tờ"],
        "chao_hoi": ["xin chào", "chào", "hello", "cảm ơn", "tạm biệt", "hi"],
        "tu_van_chon_nganh": ["nên chọn ngành", "ngành nào", "đủ điều kiện", "có đậu", "vào được", "tư vấn ngành", "phù hợp"],
        "lo_trinh_hoc_tap": ["lộ trình", "đăng ký học phần", "học vượt", "rớt", "ra trường đúng hạn", "học kỳ sau"],
    }

    def predict(self, text: str, k: int = 3) -> Prediction:
        t = f" {normalize(text)} "  # so khớp nguyên từ: tránh "hi" khớp trong "nhiêu"
        scores = {it: sum(len(kw) for kw in kws if f" {kw} " in t) for it, kws in self.KEYWORDS.items()}
        total = sum(scores.values())
        if total == 0:
            return Prediction("chao_hoi", 0.0, [("chao_hoi", 0.0)])
        top = sorted(((it, s / total) for it, s in scores.items()), key=lambda x: -x[1])[:k]
        return Prediction(top[0][0], top[0][1], top)


def load_classifier(models_dir: Path = MODELS_DIR):
    """Nạp mô hình tốt nhất hiện có. Trả về đối tượng có .name và .predict(text)."""
    keras_path, labels_path = models_dir / "intent_model.keras", models_dir / "labels.json"
    if keras_path.exists() and labels_path.exists():
        try:
            return KerasIntentModel(keras_path, labels_path)
        except ImportError:
            pass
    nb_path = models_dir / "nb_pipeline.pkl"
    if nb_path.exists():
        return NBIntentModel(nb_path)
    return KeywordClassifier()


def load_threshold(models_dir: Path = MODELS_DIR, default: float = 0.6) -> float:
    cfg = models_dir / "config.json"
    if cfg.exists():
        return float(json.loads(cfg.read_text(encoding="utf-8")).get("threshold", default))
    return default
