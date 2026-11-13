"""C1 – Mô hình hóa bài toán lập lộ trình học (Phần 2 – 2.1).

Trạng thái  : State(done: frozenset[str], ky: int)
Toán tử     : chọn tập học phần đăng ký trong kỳ `ky` (≤ tín chỉ tối đa, thỏa tiên quyết, có mở)
Đích        : targets ⊆ done
Chi phí     : 1 học kỳ / bước
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "courses.json"


@dataclass(frozen=True)
class Course:
    ma: str
    ten: str
    tin_chi: int
    tien_quyet: tuple[str, ...] = ()
    hoc_ky_mo: tuple[int, ...] = (1, 2)
    mo_phong: bool = False


@dataclass(frozen=True)
class State:
    done: frozenset[str]
    ky: int  # học kỳ tuyệt đối: 1, 2, 3, ... ; loại kỳ = (ky - 1) % hoc_ky_moi_nam + 1


@dataclass
class PlanProblem:
    courses: dict[str, Course]
    start: State
    targets: frozenset[str]
    max_credits: int = 20
    terms_per_year: int = 3

    def term_type(self, ky: int) -> int:
        return (ky - 1) % self.terms_per_year + 1

    def is_goal(self, s: State) -> bool:
        return self.targets <= s.done

    def eligible(self, s: State) -> list[str]:
        """Học phần có thể đăng ký ở kỳ s.ky: chưa học, có mở, đã đạt mọi tiên quyết."""
        t = self.term_type(s.ky)
        return [m for m, c in self.courses.items()
                if m not in s.done and t in c.hoc_ky_mo and set(c.tien_quyet) <= s.done]

    def relevant(self) -> set[str]:
        """targets ∪ mọi tổ tiên tiên quyết của chúng – chỉ cần xét những học phần này."""
        out, stack = set(), list(self.targets)
        while stack:
            m = stack.pop()
            if m in out or m not in self.courses:
                continue
            out.add(m)
            stack.extend(self.courses[m].tien_quyet)
        return out

    def remaining_credits(self, s: State) -> int:
        return sum(self.courses[m].tin_chi for m in self.targets - s.done)

    def successors(self, s: State):
        """Sinh (tập_đăng_ký, trạng_thái_con).

        TODO (C1): cài đặt – gợi ý:
          - chỉ xét eligible(s) ∩ relevant()
          - sinh các tập con có tổng tín chỉ ≤ max_credits; tối ưu: chỉ sinh tập TỐI ĐẠI
            (trạng thái có done lớn hơn ở cùng kỳ luôn trội – giữ tối ưu, giảm số nút)
          - luôn cho phép tập rỗng nếu không có học phần nào mở (để chuyển sang kỳ sau)
        """
        raise NotImplementedError("PlanProblem.successors chưa được cài đặt (C1)")


@dataclass
class PlanResult:
    semesters: list[list[str]] = field(default_factory=list)  # semesters[i] = mã HP học ở kỳ start.ky + i
    algorithm: str = ""
    expanded: int = 0        # số nút mở rộng
    generated: int = 0       # số nút sinh ra
    time_ms: float = 0.0

    @property
    def so_ky(self) -> int:
        return len(self.semesters)


def load_courses(path: Path = DATA) -> tuple[dict[str, Course], dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    courses = {c["ma"]: Course(c["ma"], c["ten"], c["tin_chi"], tuple(c.get("tien_quyet", [])),
                               tuple(c.get("hoc_ky_mo", [1, 2])), c.get("mo_phong", False))
               for c in raw["courses"]}
    return courses, raw.get("meta", {})
