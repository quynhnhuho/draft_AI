"""C2 – Cơ sở luật tư vấn chọn ngành (Phần 3 – 3.3).

Sự kiện là chuỗi ký hiệu: "to_hop=A00", "so_thich=lap_trinh", "dinh_huong=giao_vien",
"diem_dat_cntt" (sinh bởi facts.py), ...
Luật: IF tất cả if_all ∈ facts THEN thêm `then`.

TODO (C2): bổ sung đủ ≥15 luật theo đề án tuyển sinh (ghi năm) + luật rút từ cây ID3.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    if_all: tuple[str, ...]
    then: str
    giai_thich: str = ""
    nguon: str = ""


# Mã ngành dùng trong sự kiện kết luận: goi_y_<ma>, du_dk_<ma>  (ma ∈ {"cntt", "sp_tin"})
RULES: list[Rule] = [
    Rule("R1", ("to_hop=A00",), "to_hop_hop_le_cntt", "A00 là tổ hợp xét tuyển ngành CNTT", "MẪU – Đề án TS 2025"),
    Rule("R2", ("to_hop_hop_le_cntt", "diem_dat_cntt"), "du_dk_cntt", "Tổ hợp hợp lệ và điểm ≥ điểm chuẩn gần nhất"),
    Rule("R3", ("so_thich=lap_trinh",), "phu_hop_cntt", "Thích lập trình phù hợp ngành CNTT"),
    Rule("R4", ("du_dk_cntt", "phu_hop_cntt"), "goi_y_cntt", "Đủ điều kiện và phù hợp sở thích"),
    # TODO (C2): R5 ... R15+  (tổ hợp từng ngành, điều kiện riêng sư phạm, sở thích, định hướng, luật ID3)
]
