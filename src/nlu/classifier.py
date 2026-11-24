"""Bộ phân loại ý định dùng trong chatbot (C3 → C4).

HỢP ĐỒNG với C4:  classifier.predict(text) -> Prediction(intent, confidence, top_k)
                   classifier.name               – tên hiển thị
                   classifier.default_threshold  – ngưỡng τ đã chọn khi huấn luyện cho chính mô hình đó

Classifier KHÔNG tự chặn câu ngoài phạm vi: ChatEngine so confidence với τ (một chỗ duy nhất).

Thứ tự nạp (load_classifier):
  1. Mô hình Keras chính : models/config.json → model_file (intent_model.keras) + tokenizer.json + labels
  2. Baseline Naive Bayes : models/nb_pipeline.pkl  (khi máy không cài được TensorFlow)
  3. KeywordClassifier    : CHỈ để phát triển giao diện khi chưa có mô hình. Không dùng khi demo.

Tất cả file trong models/ do notebooks/N31_De03_Train.ipynb sinh ra.
"""
from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import normalize, preprocess, word_segmenter

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_THRESHOLD = 0.6


@dataclass
class Prediction:
    intent: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)


def _top_k(proba, labels: list[str], k: int) -> Prediction:
    order = np.argsort(proba)[::-1][:k]
    top = [(labels[i], float(proba[i])) for i in order]
    return Prediction(top[0][0], top[0][1], top)


def read_config(models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def read_labels(models_dir: Path = MODELS_DIR, cfg: dict | None = None) -> list[str]:
    path = Path(models_dir) / "labels.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else (cfg or {}).get("labels", [])
    if isinstance(raw, dict):  # dạng cũ {"0": "chao_hoi", ...}
        return [raw[str(i)] for i in range(len(raw))]
    return list(raw)


class KerasTokenizerLite:
    """Đọc tokenizer.json của keras.preprocessing.text.Tokenizer và tái hiện texts_to_sequences().

    Tự cài để lúc chạy không phụ thuộc mô-đun tiền xử lý cũ của Keras. Notebook tạo Tokenizer với
    filters="", lower=False, split=" " (văn bản đã qua preprocess()), nên chỉ cần tách theo khoảng trắng.
    """

    def __init__(self, path: Path):
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))["config"]
        self.word_index: dict[str, int] = json.loads(cfg["word_index"])
        self.num_words: int | None = cfg.get("num_words")
        self.oov_index: int | None = self.word_index.get(cfg.get("oov_token")) if cfg.get("oov_token") else None
        if cfg.get("filters") or cfg.get("lower") or cfg.get("split", " ") != " ":
            warnings.warn("tokenizer.json dùng filters/lower/split khác notebook – kiểm tra lại tiền xử lý")

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        out = []
        for t in texts:
            seq = []
            for w in t.split(" "):
                if not w:
                    continue
                i = self.word_index.get(w)
                if i is not None and (self.num_words is None or i < self.num_words):
                    seq.append(i)
                elif self.oov_index is not None:
                    seq.append(self.oov_index)
            out.append(seq)
        return out


def pad(seqs: list[list[int]], max_len: int, padding: str = "post", truncating: str = "post") -> np.ndarray:
    """Giống keras pad_sequences(maxlen, padding, truncating, value=0)."""
    x = np.zeros((len(seqs), max_len), dtype="int32")
    for r, s in enumerate(seqs):
        s = s[:max_len] if truncating == "post" else s[-max_len:]
        if s:
            if padding == "post":
                x[r, :len(s)] = s
            else:
                x[r, -len(s):] = s
    return x


class KerasIntentModel:
    def __init__(self, models_dir: Path = MODELS_DIR, cfg: dict | None = None):
        import tensorflow as tf  # nạp muộn: app vẫn chạy (bằng NB) khi chưa cài TensorFlow

        models_dir = Path(models_dir)
        cfg = cfg if cfg is not None else read_config(models_dir)
        self.labels = read_labels(models_dir, cfg)
        self.max_len = int(cfg.get("max_len", 50))
        self.padding = cfg.get("padding", "post")
        self.arch = cfg.get("best_model", "keras")
        self.name = f"keras ({self.arch})"
        self.default_threshold = float(cfg.get("threshold", DEFAULT_THRESHOLD))
        self.tokenizer = KerasTokenizerLite(models_dir / "tokenizer.json")
        self.model = tf.keras.models.load_model(models_dir / cfg.get("model_file", "intent_model.keras"),
                                                compile=False)
        n_out = self.model.output_shape[-1]
        if n_out != len(self.labels):
            raise ValueError(f"Mô hình có {n_out} lớp nhưng labels.json có {len(self.labels)} nhãn")

    def encode(self, texts: list[str]) -> np.ndarray:
        """texts: câu ĐÃ preprocess()."""
        return pad(self.tokenizer.texts_to_sequences(texts), self.max_len, self.padding)

    def predict_proba(self, clean_texts: list[str]) -> np.ndarray:
        return np.asarray(self.model(self.encode(clean_texts), training=False))

    def predict(self, text: str, k: int = 3) -> Prediction:
        return _top_k(self.predict_proba([preprocess(text)])[0], self.labels, k)


class NBIntentModel:
    name = "naive_bayes"

    def __init__(self, models_dir: Path = MODELS_DIR, cfg: dict | None = None):
        models_dir = Path(models_dir)
        cfg = cfg if cfg is not None else read_config(models_dir)
        with open(models_dir / "nb_pipeline.pkl", "rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cảnh báo khác phiên bản scikit-learn giữa Colab và máy chạy
            self.pipe = pickle.load(f)
        labels = read_labels(models_dir, cfg)
        # Notebook huấn luyện NB với nhãn đã mã hóa 0..n-1 → đổi lại tên bằng labels.json
        self.labels = [labels[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                       for c in self.pipe.classes_]
        self.default_threshold = float(cfg.get("thresholds_per_model", {}).get("NaiveBayes", DEFAULT_THRESHOLD))

    def predict_proba(self, clean_texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba(list(clean_texts))

    def predict(self, text: str, k: int = 3) -> Prediction:
        return _top_k(self.predict_proba([preprocess(text)])[0], self.labels, k)


class KeywordClassifier:
    """Phân loại tạm bằng từ khóa – chỉ để dựng/kiểm thử giao diện C4 khi chưa có mô hình."""

    name = "keyword (tạm thời)"
    default_threshold = 0.5
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


def check_preprocess(cfg: dict) -> list[str]:
    """Cảnh báo khi tiền xử lý lúc chạy khác lúc huấn luyện (lỗi 7.10 trong HUONG_DAN_DO_AN)."""
    want = cfg.get("word_segmenter", "none") if cfg.get("use_word_segment", False) else "none"
    have = word_segmenter()
    if want != have:
        return [f"Mô hình được huấn luyện với bộ tách từ '{want}' nhưng máy này đang dùng '{have}'. "
                "Cài đặt: pip install underthesea"]
    return []


def load_classifier(models_dir: Path = MODELS_DIR, backend: str = "auto"):
    """Nạp mô hình tốt nhất hiện có. backend: "auto" | "keras" | "nb" | "keyword".

    Đối tượng trả về có .name, .default_threshold, .predict(text) và .warnings (danh sách cảnh báo).
    """
    models_dir = Path(models_dir)
    cfg = read_config(models_dir)
    notes: list[str] = []
    order = {"auto": ["keras", "nb", "keyword"], "keras": ["keras"], "nb": ["nb"], "keyword": ["keyword"]}[backend]
    for kind in order:
        try:
            if kind == "keras" and (models_dir / cfg.get("model_file", "intent_model.keras")).exists():
                clf = KerasIntentModel(models_dir, cfg)
            elif kind == "nb" and (models_dir / "nb_pipeline.pkl").exists():
                clf = NBIntentModel(models_dir, cfg)
            elif kind == "keyword":
                clf = KeywordClassifier()
            else:
                continue
        except ImportError as exc:
            notes.append(f"Không nạp được mô hình {kind}: thiếu thư viện ({exc.name}).")
            continue
        clf.warnings = notes + (check_preprocess(cfg) if kind != "keyword" else [])
        return clf
    raise FileNotFoundError(f"Không có mô hình nào trong {models_dir}")


def load_threshold(models_dir: Path = MODELS_DIR, default: float = DEFAULT_THRESHOLD, classifier=None) -> float:
    """τ của mô hình đang dùng; nếu không biết mô hình thì đọc models/config.json."""
    if classifier is not None and getattr(classifier, "default_threshold", None) is not None:
        return float(classifier.default_threshold)
    models_dir = Path(models_dir)
    cfg = read_config(models_dir)
    if "threshold" in cfg:
        return float(cfg["threshold"])
    old = models_dir / "threshold_config.json"
    if old.exists():
        return float(json.loads(old.read_text(encoding="utf-8")).get("confidence_threshold", default))
    return default
