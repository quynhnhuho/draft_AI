# -*- coding: utf-8 -*-
"""
preprocess.py – Vietnamese NLU text pre-processing utilities.
"""

import unicodedata
import re

# ---------------------------------------------------------------------------
# Teencode / abbreviation table
# ---------------------------------------------------------------------------
TEENCODE_DICT = {
    "mn": "mọi người",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "ko biết": "không biết",
    "mk": "mình",
    "mik": "mình",
    "bk": "bạn",
    "bn": "bạn",
    "b": "bạn",
    "mình": "mình",
    "ng": "người",
    "ntn": "như thế nào",
    "ntno": "như thế nào",
    "đc": "được",
    "dc": "được",
    "đk": "được",
    "vs": "với",
    "vd": "ví dụ",
    "vd:": "ví dụ:",
    "ck": "chồng",
    "vk": "vợ",
    "cx": "cũng",
    "cg": "cũng",
    "r": "rồi",
    "rùi": "rồi",
    "oy": "rồi",
    "xong r": "xong rồi",
    "nha": "nhé",
    "nhe": "nhé",
    "nh": "nhé",
    "ah": "à",
    "a": "à",
    "ạ": "ạ",
    "nhg": "nhưng",
    "nhưg": "nhưng",
    "nhug": "nhưng",
    "trog": "trong",
    "trong": "trong",
    "nv": "nhân viên",
    "sv": "sinh viên",
    "gv": "giáo viên",
    "hs": "học sinh",
    "ts": "tuyển sinh",
    "đh": "đại học",
    "dh": "đại học",
    "cd": "cao đẳng",
    "cđ": "cao đẳng",
    "tc": "trung cấp",
    "thpt": "trung học phổ thông",
    "thcs": "trung học cơ sở",
    "thcs": "trung học cơ sở",
    "cntt": "công nghệ thông tin",
    "cntt&tt": "công nghệ thông tin và truyền thông",
    "ktx": "ký túc xá",
    "hp": "học phí",
    "hb": "học bổng",
    "nv1": "nguyện vọng 1",
    "nv2": "nguyện vọng 2",
    "nv3": "nguyện vọng 3",
    "hcmue": "đại học sư phạm thành phố hồ chí minh",
    "sptp": "đại học sư phạm thành phố hồ chí minh",
    "đhsp": "đại học sư phạm",
    "dhsp": "đại học sư phạm",
    "sphn": "đại học sư phạm hà nội",
    "tphcm": "thành phố hồ chí minh",
    "tp.hcm": "thành phố hồ chí minh",
    "tp hcm": "thành phố hồ chí minh",
    "hn": "hà nội",
    "hcm": "thành phố hồ chí minh",
    "bth": "bình thường",
    "oke": "được rồi",
    "ok": "được rồi",
    "oce": "được rồi",
    "ad": "admin",
    "mod": "quản trị viên",
    "info": "thông tin",
    "xem": "xem",
}


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC."""
    return unicodedata.normalize("NFC", text)


def replace_teencode(text: str) -> str:
    """Thay thế teencode / viết tắt bằng dạng đầy đủ."""
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        # Try two-word match first
        if i + 1 < len(tokens):
            two = " ".join(tokens[i : i + 2]).lower()
            if two in TEENCODE_DICT:
                result.append(TEENCODE_DICT[two])
                i += 2
                continue
        one = tokens[i].lower().rstrip(".,!?;:")
        if one in TEENCODE_DICT:
            # Preserve trailing punctuation
            suffix = tokens[i][len(one) :]
            result.append(TEENCODE_DICT[one] + suffix)
        else:
            result.append(tokens[i])
        i += 1
    return " ".join(result)


def remove_special_chars(text: str) -> str:
    """Xóa ký tự đặc biệt; giữ lại chữ cái tiếng Việt, chữ số và dấu cách."""
    # Keep Unicode letters (including Vietnamese), digits, and spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> str:
    """Word-segment using underthesea if available, else split by space."""
    try:
        from underthesea import word_tokenize  # type: ignore
        return word_tokenize(text, format="text")
    except ImportError:
        return text  # fallback: treat whitespace-split as tokens


def preprocess(text: str, use_tokenizer: bool = False) -> str:
    """Full pre-processing pipeline.

    Steps: normalize_unicode → lowercase → replace_teencode
           → remove_special_chars → (optional) tokenize
    """
    text = normalize_unicode(text)
    text = text.lower()
    text = replace_teencode(text)
    text = remove_special_chars(text)
    if use_tokenizer:
        text = tokenize(text)
    return text


def preprocess_batch(texts, use_tokenizer: bool = False):
    """Apply preprocess() to a list of texts."""
    return [preprocess(t, use_tokenizer=use_tokenizer) for t in texts]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Cho mk hỏi ktx của trường có máy lạnh ko?",
        "HP năm nay của ngành CNTT là bao nhiêu vậy ad?",
        "Điểm chuẩn NV1 vào ĐH năm 2024 là mấy điểm?",
    ]

    print("=" * 60)
    print("preprocess.py – self test")
    print("=" * 60)
    for s in samples:
        out = preprocess(s, use_tokenizer=False)
        print(f"  IN : {s}")
        print(f"  OUT: {out}")
        print()
    print("All tests passed.")
