"""Tiền xử lý câu hỏi tiếng Việt – DÙNG CHUNG cho huấn luyện (C3) và chatbot (C4).

Nội dung các hàm normalize()/segment()/preprocess() GIỐNG NGUYÊN VĂN Cell 3 của
notebooks/N31_De03_Train.ipynb (notebook đã huấn luyện models/intent_model.keras).
Sửa ở một nơi thì phải sửa cả nơi kia và huấn luyện lại, nếu không độ chính xác lúc chạy sẽ tụt.

    normalize(text)  : NFC → chữ thường → vị trí dấu → bỏ ký tự đặc biệt → teencode   (chưa tách từ)
                       → chatbot dùng để trích thực thể (ngành, tổ hợp, điểm, năm…)
    preprocess(text) : normalize() + tách từ underthesea ("học phí" → "học_phí")
                       → đầu vào của mô hình phân loại ý định
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Chuẩn hóa vị trí dấu thanh kiểu cũ → kiểu mới: hòa → hoà, thủy → thuỷ.
# Chỉ đổi theo chiều này vì chiều ngược lại làm hỏng các từ như "hoàng".
_TONE_MAP = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_RE = re.compile("|".join(_TONE_MAP))

# Teencode / viết tắt – một từ. KHÔNG thêm "a", "b", "v" vì trùng tên khối thi (khối A, B, V).
TEENCODE = {
    # phủ định, khẳng định
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khum": "không",
    "đc": "được", "dc": "được",
    "ok": "được rồi", "oke": "được rồi", "okela": "được rồi",
    # đại từ, hư từ
    "mn": "mọi người", "mk": "mình", "mik": "mình", "e": "em", "ng": "người",
    "bk": "biết", "bik": "biết", "bít": "biết",
    "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnh": "bao nhiêu",
    "ntn": "như thế nào", "ntno": "như thế nào",
    "j": "gì", "z": "vậy", "vs": "với", "vd": "ví dụ",
    "cx": "cũng", "cg": "cũng", "r": "rồi", "rùi": "rồi", "oy": "rồi",
    "nha": "nhé", "nhe": "nhé",
    "nhg": "nhưng", "nhưg": "nhưng", "nhug": "nhưng", "trog": "trong",
    "ad": "admin", "info": "thông tin",
    # tuyển sinh, đào tạo
    "đk": "điều kiện", "dk": "điều kiện",
    "đkxt": "đăng ký xét tuyển", "xt": "xét tuyển",
    "nv": "nguyện vọng", "nv1": "nguyện vọng 1", "nv2": "nguyện vọng 2", "nv3": "nguyện vọng 3",
    "ts": "tuyển sinh", "sv": "sinh viên", "gv": "giáo viên", "hs": "học sinh",
    "đh": "đại học", "dh": "đại học", "cđ": "cao đẳng", "cd": "cao đẳng",
    "thpt": "trung học phổ thông", "thcs": "trung học cơ sở",
    "thptqg": "trung học phổ thông quốc gia",
    "đgnl": "đánh giá năng lực", "dgnl": "đánh giá năng lực",
    "tc": "tín chỉ", "hp": "học phí", "hb": "học bổng", "ktx": "ký túc xá",
    "ctdt": "chương trình đào tạo", "ctđt": "chương trình đào tạo",
    "cntt": "công nghệ thông tin", "sp": "sư phạm",
    "hcmue": "đại học sư phạm thành phố hồ chí minh",
    "sptp": "đại học sư phạm thành phố hồ chí minh",
    "đhsp": "đại học sư phạm", "dhsp": "đại học sư phạm",
    "tphcm": "thành phố hồ chí minh", "hcm": "thành phố hồ chí minh", "hn": "hà nội",
}
# Chính tả i/y sau k, l, m: đưa về dạng chữ y – dạng có trong dữ liệu huấn luyện
# ("kí túc xá" → "ký túc xá", "học kì" → "học kỳ", "vật lí" → "vật lý", "kĩ năng" → "kỹ năng").
CHINH_TA_IY = {"kí": "ký", "kì": "kỳ", "kỉ": "kỷ", "kĩ": "kỹ", "lí": "lý", "mĩ": "mỹ"}
TEENCODE.update(CHINH_TA_IY)
# Teencode hai từ (xét trước teencode một từ). "tp.hcm" sau khi bỏ dấu chấm thành "tp hcm".
TEENCODE_2 = {
    "tp hcm": "thành phố hồ chí minh",
    "đh sp": "đại học sư phạm",
}
# Cụm ba từ gõ không dấu (xét trước cụm hai từ).
TEENCODE_3 = {
    "ky tuc xa": "ký túc xá",
    "ki tuc xa": "ký túc xá",
}

_SPECIAL_RE = re.compile(r"[^\w\s]|_")  # giữ chữ (có dấu), số, khoảng trắng; bỏ cả "_" người dùng gõ


def normalize(text: str) -> str:
    """Chuẩn hóa, CHƯA tách từ: NFC → chữ thường → vị trí dấu → bỏ ký tự đặc biệt → teencode.

    Giữ nguyên mã tổ hợp (a00, d01) và số. Chatbot dùng hàm này để trích thực thể.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TONE_RE.sub(lambda m: _TONE_MAP[m.group(0)], text)
    text = _SPECIAL_RE.sub(" ", text)
    tokens = text.split()
    out, i = [], 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            three = " ".join(tokens[i:i + 3])
            if three in TEENCODE_3:
                out.append(TEENCODE_3[three])
                i += 3
                continue
        if i + 1 < len(tokens):
            two = tokens[i] + " " + tokens[i + 1]
            if two in TEENCODE_2:
                out.append(TEENCODE_2[two])
                i += 2
                continue
        out.append(TEENCODE.get(tokens[i], tokens[i]))
        i += 1
    return " ".join(out)


USE_WORD_SEGMENT = True


@lru_cache(maxsize=1)
def _word_tokenize():
    """Nạp underthesea một lần (nạp muộn để các mô-đun chỉ cần normalize() không phải chờ)."""
    try:
        from underthesea import word_tokenize  # type: ignore
        return word_tokenize
    except ImportError:
        return None


def word_segmenter() -> str:
    """Tên bộ tách từ đang dùng – so với "word_segmenter" trong models/config.json."""
    return "underthesea" if USE_WORD_SEGMENT and _word_tokenize() is not None else "none"


def segment(text: str) -> str:
    """Tách từ, các âm tiết của một từ nối bằng '_' (vd: 'học_phí')."""
    tok = _word_tokenize() if USE_WORD_SEGMENT else None
    if tok is not None and text:
        return tok(text, format="text")
    return text


def preprocess(text: str) -> str:
    """Pipeline đầy đủ cho mô hình: normalize() rồi tách từ."""
    return segment(normalize(text)).strip()


def preprocess_batch(texts):
    return [preprocess(t) for t in texts]


if __name__ == "__main__":
    for s in ["Khối A gồm những môn nào?", "hp 1 tc bn tiền vậy ad", "đk nhận hb là gì",
              "Cho mk hỏi ktx có máy lạnh ko?", "tp.hcm có cơ sở 2 ko"]:
        print(f"{s!r:40} → {normalize(s)!r:55} → {preprocess(s)!r}")
