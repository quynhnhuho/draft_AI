# AI-assisted: Claude (Anthropic) – rà soát, sửa lỗi và tái cấu trúc; thuật toán do nhóm viết và đã kiểm tra lại.
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field

from ..nlu.preprocess import normalize

SO_THICH = {
    "thich_lap_trinh": ["lập trình", "code", "viết code", "phần mềm", "làm app", "web"],
    "thich_du_lieu": ["dữ liệu", "data", "trí tuệ nhân tạo", "học máy", "phân tích dữ liệu"],
    "thich_thiet_ke_hoc_lieu": ["thiết kế học liệu", "học liệu", "bài giảng điện tử", "thiết kế bài giảng",
                                "e learning", "elearning"],
    "thich_day_hoc": ["dạy học", "đi dạy", "giảng dạy", "dạy"],
}
DINH_HUONG = {
    "lap_trinh_vien": ["lập trình viên", "kỹ sư phần mềm", "developer"],
    "phan_tich_du_lieu": ["phân tích dữ liệu", "data analyst", "chuyên viên dữ liệu", "khoa học dữ liệu"],
    "giao_vien": ["giáo viên", "nhà giáo", "thầy giáo", "cô giáo"],
    "chuyen_vien_edtech": ["edtech", "chuyên viên công nghệ giáo dục", "chuyên viên edtech"],
}
TEN_SO_THICH = {"thich_lap_trinh": "lập trình", "thich_du_lieu": "dữ liệu",
                "thich_thiet_ke_hoc_lieu": "thiết kế học liệu", "thich_day_hoc": "dạy học"}
TEN_DINH_HUONG = {"lap_trinh_vien": "lập trình viên", "phan_tich_du_lieu": "phân tích dữ liệu",
                  "giao_vien": "giáo viên", "chuyen_vien_edtech": "chuyên viên EdTech"}

_MON_ALIAS = {
    "tiếng anh": "Anh", "anh văn": "Anh", "ngoại ngữ": "Anh", "tieng anh": "Anh", "anh": "Anh",
    "toán": "Toan", "toan": "Toan",
    "vật lý": "Ly", "vật lí": "Ly", "vat ly": "Ly", "lý": "Ly", "lí": "Ly", "ly": "Ly", "li": "Ly",
    "hóa học": "Hoa", "hoá học": "Hoa", "hoa hoc": "Hoa", "hóa": "Hoa", "hoá": "Hoa", "hoa": "Hoa",
    "sinh học": "Sinh", "sinh hoc": "Sinh", "sinh": "Sinh",
    "tin học": "Tin", "tin hoc": "Tin", "tin": "Tin",
}
_DIEM_MON = re.compile(
    r"(?<!\w)(" + "|".join(sorted(map(re.escape, _MON_ALIAS), key=len, reverse=True)) + r")(?!\w)"
    r"\s*(?:[:=\-]|là|được|đạt)?\s*(\d{1,2}(?:[.,]\d{1,3})?)(?![\d.,]*\d)")
_KHU_VUC = re.compile(r"\b(?:kv|khu vực)\s?([123])(?:\s?(nt|nông thôn))?\b")

_TO_HOP = re.compile(r"\b([abcdx])\s?0?(\d{1,2})\b")
_YEAR = re.compile(r"\b(20[12]\d)\b")
_SCORE = re.compile(r"(?<![\d.])(\d{1,2}(?:[.,]\d{1,2})?)(?![\d.])")
_COURSE = re.compile(r"\b[A-Za-z]{2,5}\s?\d{3,4}\b")
_HK1 = re.compile(r"\b(?:hk|học kỳ|kỳ)\s?1\b|\bhk1\b")


@dataclass
class Entities:
    nganh: list[str] = field(default_factory=list)
    to_hop: list[str] = field(default_factory=list)
    nam: int | None = None
    diem: float | None = None
    ma_hp: list[str] = field(default_factory=list)
    nhac_hk1: bool = False
    so_thich: list[str] = field(default_factory=list)
    dinh_huong: list[str] = field(default_factory=list)
    phuong_thuc: str | None = None
    khu_vuc: str | None = None
    diem_mon: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


def _has(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _find_majors(t: str, majors: dict) -> list[str]:
    found: list[tuple[int, str]] = []
    for code, info in majors.items():
        names = {normalize(a) for a in info.get("aliases", [])} | {normalize(info["ten"])}
        pos = [f" {t} ".find(f" {n} ") for n in names if _has(t, n)]
        if pos:
            found.append((min(pos), code))
    return [c for _, c in sorted(found)]


def parse_score(text: str) -> float | None:
    t = _YEAR.sub(" ", text.lower())
    t = _TO_HOP.sub(" ", t)
    for m in _SCORE.finditer(t):
        v = float(m.group(1).replace(",", "."))
        if 0 < v <= 30:
            return v
    return None


def parse_diem_mon(text: str) -> dict[str, float]:
    t = unicodedata.normalize("NFC", text).lower()
    return {_MON_ALIAS[m.group(1)]: float(m.group(2).replace(",", ".")) for m in _DIEM_MON.finditer(t)}


def parse_phuong_thuc(text: str) -> str | None:
    t = normalize(text)
    if any(_has(t, w) for w in ("đgnl", "dgnl", "đánh giá năng lực", "năng lực", "học bạ", "hoc ba")):
        return "DGNL"
    if any(_has(t, w) for w in ("trung học phổ thông", "thi tốt nghiệp", "tốt nghiệp", "thi thpt")):
        return "THPT"
    return None


def parse_khu_vuc(text: str) -> str | None:
    t = normalize(text)
    if (m := _KHU_VUC.search(t)):
        return "KV2-NT" if m.group(1) == "2" and m.group(2) else f"KV{m.group(1)}"
    if _has(t, "nông thôn"):
        return "KV2-NT"
    return None


def extract(text: str, majors: dict, courses: dict | None = None) -> Entities:
    t = normalize(text)
    e = Entities(nganh=_find_majors(t, majors))

    e.to_hop = sorted({f"{m.group(1).upper()}{int(m.group(2)):02d}" for m in _TO_HOP.finditer(t)})
    if (y := _YEAR.search(t)):
        e.nam = int(y.group(1))
    elif _has(t, "năm ngoái") or _has(t, "năm trước"):
        e.nam = dt.date.today().year - 1
    elif _has(t, "năm nay"):
        e.nam = dt.date.today().year

    if any(_has(t, w) for w in ("điểm", "được", "đạt", "thi")):
        e.diem = parse_score(text)

    if courses:
        ma = {m.upper().replace(" ", "") for m in _COURSE.findall(text)}
        if isinstance(courses, dict):
            ma |= {code for code, hp in courses.items()
                   if getattr(hp, "ten", None) and _has(t, normalize(hp.ten))}
        e.ma_hp = sorted(m for m in ma if m in courses)
    e.nhac_hk1 = bool(_HK1.search(t))

    e.so_thich = [k for k, kws in SO_THICH.items() if any(_has(t, normalize(w)) for w in kws)]
    e.dinh_huong = [k for k, kws in DINH_HUONG.items() if any(_has(t, normalize(w)) for w in kws)]
    e.phuong_thuc = parse_phuong_thuc(text)
    e.khu_vuc = parse_khu_vuc(text)
    e.diem_mon = parse_diem_mon(text)
    return e
