"""Tiền xử lý câu hỏi tiếng Việt – DÙNG CHUNG cho huấn luyện (C3) và chatbot (C4).

Các bước: Unicode NFC → chữ thường → chuẩn hóa vị trí dấu thanh → teencode/viết tắt
→ bỏ ký tự thừa → tách từ (pyvi / underthesea, nếu không có thì tách theo khoảng trắng).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Chuẩn hóa dấu kiểu cũ → kiểu mới (đặt dấu trên nguyên âm chính): hòa → hoà, thủy → thuỷ.
# Chỉ đổi theo chiều này vì chiều ngược lại làm hỏng các từ như "hoàng".
_TONE_MAP = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_RE = re.compile("|".join(_TONE_MAP))

# Teencode / viết tắt hay gặp trên fanpage. Bổ sung dần từ dữ liệu thật.
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khum": "không",
    "đc": "được", "dc": "được", "đk": "điều kiện", "bn": "bao nhiêu", "bnhiu": "bao nhiêu",
    "hp": "học phí", "ktx": "ký túc xá", "ctdt": "chương trình đào tạo", "ctđt": "chương trình đào tạo",
    "cntt": "công nghệ thông tin", "sp": "sư phạm", "tc": "tín chỉ", "sv": "sinh viên",
    "ad": "admin", "a": "anh", "e": "em", "j": "gì", "z": "vậy", "v": "vậy", "r": "rồi",
    "mn": "mọi người", "ntn": "như thế nào", "thpt": "trung học phổ thông",
}

_KEEP = re.compile(r"[^0-9a-zà-ỹđ\s]")  # giữ chữ (có dấu), số, khoảng trắng


def normalize(text: str) -> str:
    """Chuẩn hóa nhưng CHƯA tách từ. Giữ mã tổ hợp (a00, d01) và số."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _TONE_RE.sub(lambda m: _TONE_MAP[m.group(0)], text)
    text = _KEEP.sub(" ", text)
    words = [TEENCODE.get(w, w) for w in text.split()]
    return " ".join(words)


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from pyvi import ViTokenizer  # type: ignore
        return ViTokenizer.tokenize
    except Exception:
        pass
    try:
        from underthesea import word_tokenize  # type: ignore
        return lambda s: word_tokenize(s, format="text")
    except Exception:
        return lambda s: s  # không có thư viện → giữ nguyên (ghi rõ trong báo cáo nếu dùng)


def preprocess(text: str) -> str:
    """Chuẩn hóa + tách từ, các âm tiết của một từ nối bằng '_' (vd: 'học_phí')."""
    return _tokenizer()(normalize(text)).strip()


if __name__ == "__main__":
    for s in ["Hp ngành CNTT bn 1 tc vậy ad?", "Điểm chuẩn Sư phạm Tin học năm 2024", "KTX có còn chỗ ko ạ"]:
        print(f"{s!r:45} → {preprocess(s)!r}")
