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
        # Mục chính sách áp dụng nhiều năm (vd. Nghị định 116 – có "ap_dung_tu_nam") luôn được kèm theo
        # số liệu của năm được hỏi, không tham gia chọn "năm mới nhất".
        luon = [f for f in cands if f.get("ap_dung_tu_nam") is not None
                and (nam is None or nam >= f["ap_dung_tu_nam"])]
        cands = [f for f in cands if f.get("ap_dung_tu_nam") is None]
        if not cands:
            res.facts = luon
            return res
        if nam is not None:
            exact = [f for f in cands if f.get("nam") == nam]
            if exact:
                res.facts = exact + luon
                return res
            res.year_fallback = True
            # Không có năm được hỏi → lấy năm GẦN NHẤT với năm đó (cùng khoảng cách thì lấy năm sau)
            gan = min({f.get("nam", 0) for f in cands}, key=lambda y: (abs(y - nam), -y))
            res.facts = [f for f in cands if f.get("nam") == gan] + luon
            return res
        # Mặc định: năm mới nhất cho từng ngành
        latest: dict = {}
        for f in cands:
            key = f.get("nganh")
            if key not in latest or f.get("nam", 0) > latest[key][0].get("nam", 0):
                latest[key] = [f]
            elif f.get("nam") == latest[key][0].get("nam"):
                latest[key].append(f)
        res.facts = [f for group in latest.values() for f in group] + luon
        return res

    def lien_he(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Các kênh liên hệ (Khoa, tuyển sinh của Trường): [(tên đơn vị, [(nhãn, giá trị), …]), …].

        Bỏ qua giá trị còn để mẫu dạng "<...>". Dùng chung cho thanh bên web và câu chuyển cán bộ.
        """
        out = []
        for khoa, mac_dinh in (("lien_he_tu_van", self.meta.get("khoa", "Khoa")),
                               ("lien_he_tuyen_sinh", "Tư vấn tuyển sinh của Trường")):
            c = self.meta.get(khoa, {})
            dong = [(nhan, str(c[k])) for k, nhan in (("dia_chi", "Địa chỉ"), ("dien_thoai", "Điện thoại"),
                                                      ("email", "Email"), ("fanpage", "Fanpage"))
                    if c.get(k) and not str(c[k]).startswith("<")]
            if dong:
                out.append((c.get("ten") or mac_dinh, dong))
        return out

    def contact_text(self) -> str:
        """Kênh liên hệ dạng markdown cho câu trả lời chuyển cán bộ."""
        nhom = [f"**{ten}**\n" + "\n".join(f"- {nhan}: {gt}" for nhan, gt in dong) for ten, dong in self.lien_he()]
        return "\n\n".join(nhom) or "- (Nhóm chưa cập nhật kênh liên hệ trong data/knowledge_base.json)"


def cite(fact: dict) -> str:
    """Dòng trích nguồn BẮT BUỘC cho mọi câu trả lời số liệu."""
    if fact.get("ap_dung_tu_nam"):
        nam = f"áp dụng từ khóa tuyển sinh {fact['ap_dung_tu_nam']}–{fact['ap_dung_tu_nam'] + 1}"
    else:
        nam = f"năm áp dụng {fact.get('nam', '?')}"
    line = f"Nguồn: {fact.get('nguon') or 'chưa ghi nguồn'} – {nam}"
    if fact.get("url"):
        line += f" ([link]({fact['url']}))"
    if not fact.get("da_kiem_chung", False):
        line += "  \nLưu ý: số liệu mẫu, nhóm chưa cập nhật nguồn chính thức."
    return line


def format_facts(facts: list[dict]) -> str:
    return "\n\n".join(f"{f['noi_dung']}\n\n{cite(f)}" for f in facts)
