"""C2 – Suy diễn tiến và suy diễn lùi (Phần 3 – 3.4). TỰ CÀI ĐẶT.

Chatbot (C4) gọi đúng hai hàm dưới đây và dựa vào cấu trúc kết quả đã định.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rules import RULES, Rule


@dataclass
class ForwardResult:
    facts: set[str]                                   # tập sự kiện sau suy diễn
    fired: list[Rule] = field(default_factory=list)   # luật đã bắn, theo thứ tự (vết suy diễn)

    @property
    def recommendations(self) -> list[str]:
        """Mã ngành được gợi ý, ví dụ ["cntt"] từ sự kiện "goi_y_cntt"."""
        return sorted(f.removeprefix("goi_y_") for f in self.facts if f.startswith("goi_y_"))


@dataclass
class BackwardResult:
    goal: str
    proven: bool
    trace: list[str] = field(default_factory=list)    # các bước chứng minh (để giải thích)
    missing: set[str] = field(default_factory=set)    # sự kiện lá còn thiếu → chatbot hỏi tiếp


def forward_chaining(facts: set[str], rules: list[Rule] = RULES) -> ForwardResult:
    """TODO (C2): lặp – tìm luật có if_all ⊆ facts và then ∉ facts → thêm then, ghi vào fired;
    dừng khi không còn luật nào bắn được.
    """
    raise NotImplementedError("Suy diễn tiến chưa được cài đặt (C2)")


def backward_chaining(goal: str, facts: set[str], rules: list[Rule] = RULES) -> BackwardResult:
    """TODO (C2): chứng minh goal (vd. "du_dk_cntt"):
      - goal ∈ facts → đúng
      - với mỗi luật có then == goal: chứng minh đệ quy mọi tiền đề (dùng tập đang_xét chống lặp)
      - sự kiện lá không có luật nào sinh ra và không có trong facts → đưa vào missing
    """
    raise NotImplementedError("Suy diễn lùi chưa được cài đặt (C2)")
