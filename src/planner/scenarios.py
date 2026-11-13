"""C1 – Ba tình huống bắt buộc + bảng so sánh thuật toán.

    python -m src.planner.scenarios

TODO (C1): cố định đầu vào thật cho 3 tình huống sau khi có ≥25 học phần.
"""
from __future__ import annotations

from .model import PlanProblem, State, load_courses
from .search import ALGORITHMS


def build_scenarios():
    courses, meta = load_courses()
    all_ids = frozenset(courses)
    max_tc = meta.get("tin_chi_toi_da_moi_ky", 20)
    vuot = meta.get("tin_chi_toi_da_hoc_vuot", 25)
    return {
        "Sinh viên năm nhất": PlanProblem(courses, State(frozenset(), 1), all_ids, max_tc),
        # TODO: chọn 2 học phần bị rớt có nhiều học phần phụ thuộc (vd. IT003, IT004)
        "Rớt 2 học phần": PlanProblem(courses, State(frozenset({"IT001", "MA001", "IT005"}), 3), all_ids, max_tc),
        "Học vượt": PlanProblem(courses, State(frozenset(), 1), all_ids, vuot),
    }


def main():
    for name, prob in build_scenarios().items():
        for algo, fn in ALGORITHMS.items():
            try:
                r = fn(prob)
                print(f"{name:20} | {algo:10} | {r.so_ky} kỳ | mở rộng {r.expanded} nút | {r.time_ms:.1f} ms")
            except NotImplementedError as e:
                print(f"{name:20} | {algo:10} | {e}")


if __name__ == "__main__":
    main()
