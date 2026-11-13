"""Cơ sở tri thức: truy vấn fact theo (loại, ngành, năm) và định dạng câu trả lời CÓ NGUỒN + NĂM."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"


@dataclass
class QueryResult:
    facts: list[dict] = field(default_factory=list)
    requested_year: int | None = None
    year_fallback: bool = False   # True = không có năm được hỏi, đã dùng năm gần nhất


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta: dict = raw.get("meta", {})
        self.majors: dict = raw.get("nganh", {})
        self.facts: list[dict] = raw.get("facts", [])

    def major_name(self, code: str | None) -> str:
        return self.majors.get(code, {}).get("ten", code or "")

    def query(self, loai: str, nganh: str | None = None, nam: int | None = None) -> QueryResult:
        cands = [f for f in self.facts if f.get("loai") == loai]
        if nganh:
            cands = [f for f in cands if f.get("nganh") in (nganh, None)]
        res = QueryResult(requested_year=nam)
        if not cands:
            return res
        if nam is not None:
            exact = [f for f in cands if f.get("nam") == nam]
            if exact:
                res.facts = exact
                return res
            res.year_fallback = True
        # Mặc định: năm mới nhất cho từng ngành
        latest: dict = {}
        for f in cands:
            key = f.get("nganh")
            if key not in latest or f.get("nam", 0) > latest[key][0].get("nam", 0):
                latest[key] = [f]
            elif f.get("nam") == latest[key][0].get("nam"):
                latest[key].append(f)
        res.facts = [f for group in latest.values() for f in group]
        return res

    def contact_text(self) -> str:
        c = self.meta.get("lien_he_tu_van", {})
        return (f"- Email: {c.get('email', '')}\n- Điện thoại: {c.get('dien_thoai', '')}\n"
                f"- Fanpage: {c.get('fanpage', '')}")


def cite(fact: dict) -> str:
    """Dòng trích nguồn BẮT BUỘC cho mọi câu trả lời số liệu."""
    line = f"📌 Nguồn: {fact.get('nguon') or 'chưa ghi nguồn'} – năm áp dụng {fact.get('nam', '?')}"
    if fact.get("url"):
        line += f" ([link]({fact['url']}))"
    if not fact.get("da_kiem_chung", False):
        line += "  \n⚠️ Số liệu mẫu/mô phỏng, chưa đối chiếu với nguồn chính thức."
    return line


def format_facts(facts: list[dict]) -> str:
    return "\n\n".join(f"{f['noi_dung']}\n\n{cite(f)}" for f in facts)
