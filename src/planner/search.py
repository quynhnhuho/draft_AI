"""C1 – Thuật toán tìm lộ trình ít học kỳ nhất (Phần 2 – 2.2). TỰ CÀI ĐẶT.

Mọi hàm nhận PlanProblem, trả PlanResult (xem model.py) để chatbot và bảng so sánh dùng chung.
"""
from __future__ import annotations

from .model import PlanProblem, PlanResult, State


def heuristic(problem: PlanProblem, s: State) -> int:
    """h = ⌈tín chỉ còn lại của targets / tín chỉ tối đa mỗi kỳ⌉  (chấp nhận được).

    TODO (C1): cài đặt và chứng minh tính chấp nhận được trong báo cáo.
    """
    raise NotImplementedError("heuristic chưa được cài đặt (C1)")


def bfs(problem: PlanProblem) -> PlanResult:
    """TODO (C1): hàng đợi FIFO + visited; mỗi tầng = 1 học kỳ → lời giải đầu tiên là tối ưu.
    Lưu cha của mỗi nút để truy vết lộ trình.
    """
    raise NotImplementedError("BFS chưa được cài đặt (C1)")


def astar(problem: PlanProblem) -> PlanResult:
    """TODO (C1): heapq theo f = g + h, tie-break theo h; g = số kỳ đã dùng."""
    raise NotImplementedError("A* chưa được cài đặt (C1)")


def branch_and_bound(problem: PlanProblem) -> PlanResult:
    """TODO (C1): DFS quay lui; best khởi tạo bằng lời giải tham lam;
    cắt nhánh khi g + h >= best; duyệt con theo h tăng dần.
    """
    raise NotImplementedError("Nhánh cận chưa được cài đặt (C1)")


ALGORITHMS = {"BFS": bfs, "A*": astar, "Nhánh cận": branch_and_bound}


def plan(problem: PlanProblem, algorithm: str = "A*") -> PlanResult:
    """Điểm vào duy nhất mà chatbot (C4) gọi."""
    return ALGORITHMS[algorithm](problem)
