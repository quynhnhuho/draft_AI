"""Trích xuất thực thể bằng luật + từ điển: ngành, tổ hợp, năm, điểm, mã học phần, sở thích, định hướng."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from ..nlu.preprocess import normalize

SO_THICH = {
    "lap_trinh": ["lập trình", "code", "viết code", "phần mềm", "làm app", "web"],
    "day_hoc": ["dạy học", "đi dạy", "giảng dạy", "dạy"],
    "du_lieu": ["dữ liệu", "data", "ai", "trí tuệ nhân tạo", "học máy"],
    "mang_bao_mat": ["mạng", "bảo mật", "an ninh mạng", "hệ thống"],
    "thiet_ke_game": ["thiết kế", "đồ hoạ", "đồ họa", "game"],
}
DINH_HUONG = {
    "giao_vien": ["giáo viên", "nhà giáo", "đi dạy"],
    "doanh_nghiep": ["công ty", "doanh nghiệp", "lập trình viên", "kỹ sư", "đi làm"],
    "nghien_cuu": ["nghiên cứu", "thạc sĩ", "học lên", "tiến sĩ"],
    "khoi_nghiep": ["khởi nghiệp", "startup"],
}

_TO_HOP = re.compile(r"\b([abcdx])\s?0?(\d{1,2})\b")
_YEAR = re.compile(r"\b(20[12]\d)\b")
_SCORE = re.compile(r"(?<![\d.])(\d{1,2}(?:[.,]\d{1,2})?)(?![\d.])")
_COURSE = re.compile(r"\b[A-Za-z]{2,4}\d{3}\b")


@dataclass
class Entities:
    nganh: list[str] = field(default_factory=list)   # mã ngành trong KB: "cntt", "sp_tin"
    to_hop: list[str] = field(default_factory=list)  # "A00", "D01"
    nam: int | None = None
    diem: float | None = None
    ma_hp: list[str] = field(default_factory=list)
    so_thich: list[str] = field(default_factory=list)
    dinh_huong: str | None = None

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
    """Điểm xét tuyển (thang 30). Bỏ qua số trông như năm hoặc mã tổ hợp."""
    t = _YEAR.sub(" ", text.lower())  # không dùng normalize(): nó xóa dấu "." trong "24.5"
    t = _TO_HOP.sub(" ", t)
    for m in _SCORE.finditer(t):
        v = float(m.group(1).replace(",", "."))
        if 0 < v <= 30:
            return v
    return None


def extract(text: str, majors: dict, course_ids: set[str] | None = None) -> Entities:
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

    if course_ids:
        e.ma_hp = sorted({m.upper() for m in _COURSE.findall(text) if m.upper() in course_ids})

    e.so_thich = [k for k, kws in SO_THICH.items() if any(_has(t, normalize(w)) for w in kws)]
    e.dinh_huong = next((k for k, kws in DINH_HUONG.items() if any(_has(t, normalize(w)) for w in kws)), None)
    return e
