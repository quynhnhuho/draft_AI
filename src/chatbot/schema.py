"""Kiểu dữ liệu dùng chung của chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotReply:
    text: str
    intent: str | None = None
    confidence: float = 0.0
    top_k: list[tuple[str, float]] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # các fact đã dùng (có nguon, nam)
    handoff: bool = False                               # True = đã chuyển cán bộ tư vấn
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueState:
    last_major: str | None = None          # nhớ ngữ cảnh: "thế học phí ngành đó?"
    active_flow: str | None = None         # luồng nhiều lượt đang chạy
    slots: dict[str, Any] = field(default_factory=dict)
    pending_slot: str | None = None        # slot đang chờ người dùng trả lời
    turns: int = 0

    def reset_flow(self) -> None:
        self.active_flow, self.slots, self.pending_slot = None, {}, None
