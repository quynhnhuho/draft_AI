# Đánh giá end-to-end Chức năng 4

Bộ phân loại: **keras (Conv1D)** · τ = 0.92 · kịch bản: `tests/scenarios.yaml` (17 hội thoại, 44 lượt)

| Chỉ số | Kết quả |
|---|---|
| Lượt đạt mọi kỳ vọng | 42/44 (95%) |
| Đúng ý định | 22/24 (92%) |
| Câu trả lời có số liệu kèm nguồn + năm (mọi lượt đưa ra fact) | 18/18 (100%) |
| Chuyển cán bộ đúng (trên các câu cần chuyển) | 7/7 (100%) |
| Chuyển cán bộ thừa (trên các câu không cần chuyển) | 1/24 (4%) |
| Thời gian phản hồi trung bình / lớn nhất | 13 ms / 273 ms |

Ý định được quyết định bởi (các lượt không nằm giữa luồng nhiều lượt): `mo_hinh` 16, `luat_uu_tien` 2, `luat_bo_tro` 3, `chan_pham_vi` 3, `duoi_nguong` 5

## Chi tiết từng lượt

| # | Hội thoại | Câu người dùng | Ý định | Tin cậy | Quyết định bởi | Chuyển CB | Kết quả |
|---|---|---|---|---|---|---|---|
| 1 | Chào hỏi và cảm ơn | chào bạn nha | `chao_hoi` | 1.00 | mo_hinh |  | đạt |
| 2 | Chào hỏi và cảm ơn | cảm ơn ad nhiều | `chao_hoi` | 0.95 | mo_hinh |  | đạt |
| 3 | Điểm chuẩn có năm, rồi đổi năm | Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu? | `diem_chuan` | 1.00 | mo_hinh |  | đạt |
| 4 | Điểm chuẩn có năm, rồi đổi năm | thế điểm chuẩn năm 2025 của ngành đó thì sao | `diem_chuan` | 1.00 | mo_hinh |  | đạt |
| 5 | Điểm chuẩn không nêu năm → năm mới nhất | sư phạm tin lấy bao nhiêu điểm vậy ạ | `diem_chuan` | 1.00 | mo_hinh |  | đạt |
| 6 | Năm không có dữ liệu → nói rõ, đưa năm gần nhất | điểm chuẩn CNTT năm 2021 | `diem_chuan` | 1.00 | mo_hinh |  | đạt |
| 7 | Tổ hợp xét tuyển | ngành công nghệ giáo dục xét tuyển những tổ hợp môn nào | `to_hop_xet_tuyen` | 1.00 | mo_hinh |  | đạt |
| 8 | Học phí – viết tắt, rồi hỏi tiếp theo ngữ cảnh | hp ngành cntt 1 năm bn tiền | `hoc_phi` | 1.00 | mo_hinh |  | đạt |
| 9 | Học phí – viết tắt, rồi hỏi tiếp theo ngữ cảnh | điểm chuẩn sp tin năm 2025 | `diem_chuan` | 0.99 | mo_hinh |  | đạt |
| 10 | Học phí – viết tắt, rồi hỏi tiếp theo ngữ cảnh | còn học phí ngành đó thì sao ạ | `hoc_phi` | 1.00 | mo_hinh |  | đạt |
| 11 | Học bổng, ký túc xá, thủ tục | trường có học bổng cho sinh viên giỏi không ạ | `hoc_bong` | 1.00 | mo_hinh |  | đạt |
| 12 | Học bổng, ký túc xá, thủ tục | ktx của trường đăng ký thế nào | `ky_tuc_xa` | 0.96 | mo_hinh |  | đạt |
| 13 | Học bổng, ký túc xá, thủ tục | nhập học cần mang theo những giấy tờ gì | `chuyen_can_bo` | 0.76 | duoi_nguong | có | LỖI: ý định 'chuyen_can_bo' ≠ ['thu_tuc_nhap_hoc']; chuyển cán bộ thừa |
| 14 | Chương trình đào tạo và so sánh ngành | ngành CNTT học những môn gì trong 4 năm | `chuong_trinh_dao_tao` | 0.98 | mo_hinh |  | đạt |
| 15 | Chương trình đào tạo và so sánh ngành | CNTT với sư phạm tin khác nhau ở điểm nào | `diem_chuan` | 1.00 | mo_hinh |  | LỖI: ý định 'diem_chuan' ≠ ['chuong_trinh_dao_tao', 'tu_van_chon_nganh']; thiếu 'Sư phạm Tin học' |
| 16 | Việc làm | học công nghệ thông tin ra trường có dễ xin việc không | `viec_lam` | 0.98 | mo_hinh |  | đạt |
| 17 | Kiểm tra đủ điều kiện – điền sẵn từ câu hỏi (suy diễn lùi) | Em thi THPT được Toán 8, Anh 7, Lý 7, KV3, có đủ điều kiện vào CNTT không? | `tu_van_chon_nganh` | 0.61 | luat_uu_tien |  | đạt |
| 18 | Kiểm tra đủ điều kiện – hỏi từng bước | với điểm của em thì có đậu sư phạm tin không ạ | `kiem_tra_dieu_kien` | 1.00 | luat_uu_tien |  | đạt |
| 19 | Kiểm tra đủ điều kiện – hỏi từng bước | thpt | `kiem_tra_dieu_kien` | 1.00 | (luồng) |  | đạt |
| 20 | Kiểm tra đủ điều kiện – hỏi từng bước | Toán 7, Anh 6.2, Lý 6 | `kiem_tra_dieu_kien` | 1.00 | (luồng) |  | đạt |
| 21 | Kiểm tra đủ điều kiện – hỏi từng bước | KV1 | `tu_van_chon_nganh` | 1.00 | (luồng) |  | đạt |
| 22 | Tư vấn chọn ngành (suy diễn tiến) | em đang phân vân không biết nên chọn ngành nào của khoa | `tu_van_nganh` | 0.75 | luat_bo_tro |  | đạt |
| 23 | Tư vấn chọn ngành (suy diễn tiến) | học bạ + đgnl | `tu_van_nganh` | 1.00 | (luồng) |  | đạt |
| 24 | Tư vấn chọn ngành (suy diễn tiến) | 8 | `tu_van_nganh` | 1.00 | (luồng) |  | đạt |
| 25 | Tư vấn chọn ngành (suy diễn tiến) | Anh 8.5, Tin 9 | `tu_van_nganh` | 1.00 | (luồng) |  | đạt |
| 26 | Tư vấn chọn ngành (suy diễn tiến) | kv2 | `tu_van_nganh` | 1.00 | (luồng) |  | đạt |
| 27 | Tư vấn chọn ngành (suy diễn tiến) | lập trình, thiết kế học liệu | `tu_van_nganh` | 1.00 | (luồng) |  | đạt |
| 28 | Tư vấn chọn ngành (suy diễn tiến) | edtech | `tu_van_chon_nganh` | 1.00 | (luồng) |  | đạt |
| 29 | Lộ trình học – tân sinh viên | em là tân sinh viên, cho em xin lộ trình học các kỳ | `lo_trinh` | 0.99 | mo_hinh |  | đạt |
| 30 | Lộ trình học – tân sinh viên | không | `lo_trinh_hoc_tap` | 1.00 | (luồng) |  | đạt |
| 31 | Lộ trình học – bị rớt môn, hỏi từng bước | em rớt một môn thì học kỳ tới nên đăng ký học phần nào | `lo_trinh` | 0.64 | luat_bo_tro |  | đạt |
| 32 | Lộ trình học – bị rớt môn, hỏi từng bước | 3 | `lo_trinh` | 1.00 | (luồng) |  | đạt |
| 33 | Lộ trình học – bị rớt môn, hỏi từng bước | hk1 | `lo_trinh` | 1.00 | (luồng) |  | đạt |
| 34 | Lộ trình học – bị rớt môn, hỏi từng bước | không | `lo_trinh` | 1.00 | (luồng) |  | đạt |
| 35 | Lộ trình học – bị rớt môn, hỏi từng bước | có | `lo_trinh_hoc_tap` | 1.00 | (luồng) |  | đạt |
| 36 | Hủy luồng giữa chừng | tư vấn chọn ngành giúp em | `tu_van_nganh` | 0.63 | luat_bo_tro |  | đạt |
| 37 | Hủy luồng giữa chừng | hủy | `huy_luong` | 1.00 | (luồng) |  | đạt |
| 38 | Ngoài phạm vi – chủ đề khác | cuối tuần này Sài Gòn có mưa không | `chuyen_can_bo` | 0.35 | duoi_nguong | có | đạt |
| 39 | Ngoài phạm vi – chủ đề khác | chỉ mình cách nấu canh chua cá lóc | `chuyen_can_bo` | 0.67 | duoi_nguong | có | đạt |
| 40 | Ngoài phạm vi – chủ đề khác | giá vàng hôm nay bao nhiêu một chỉ | `chuyen_can_bo` | 0.62 | duoi_nguong | có | đạt |
| 41 | Ngoài phạm vi – chủ đề khác | viết giúp em đoạn code python sắp xếp nổi bọt | `chuyen_can_bo` | 0.36 | duoi_nguong | có | đạt |
| 42 | Ngoài phạm vi – ngành/trường khác (mô hình tự tin nhưng không có dữ liệu) | điểm chuẩn ngành sư phạm ngữ văn năm 2025 | `chuyen_can_bo` | 1.00 | chan_pham_vi | có | đạt |
| 43 | Ngoài phạm vi – ngành/trường khác (mô hình tự tin nhưng không có dữ liệu) | học phí ngành tâm lý học bao nhiêu ạ | `chuyen_can_bo` | 0.99 | chan_pham_vi | có | đạt |
| 44 | Ngoài phạm vi – ngành/trường khác (mô hình tự tin nhưng không có dữ liệu) | điểm chuẩn CNTT của đại học bách khoa năm nay | `chuyen_can_bo` | 1.00 | chan_pham_vi | có | đạt |
