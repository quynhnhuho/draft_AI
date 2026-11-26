# Chức năng 4 – Ứng dụng chatbot (CLO 3, CLO 4)

```bash
pip install -r requirements.txt
python app.py                              # giao diện web → http://127.0.0.1:5000
pytest -q                                  # 51 test (unit + web + kịch bản end-to-end + demo)
python -m src.chatbot.danh_gia --out docs/C4_DANH_GIA.md   # đánh giá end-to-end, ghi báo cáo
```

## 1. Kiến trúc

```
trình duyệt (templates/index.html, static/js/app.js)
   │  POST /api/chat · /api/tu-van (C2) · /api/lo-trinh (C1) · /api/danh-gia
   ▼
app.py (Flask) ──► ChatEngine.respond()                src/chatbot/engine.py
   ▲                      │
                          │
   đang trong luồng? ─có─► flows.continue_flow()        src/chatbot/flows.py   (gõ "hủy" để thoát)
                          │ không
                          ▼
              entities.extract()  +  normalize()        src/chatbot/entities.py, src/nlu/preprocess.py
                          │
              classifier.predict() → (intent, conf, top-3)   src/nlu/classifier.py  ← models/ (C3)
                          │
   ① chặn phạm vi: ngành / trường khác ─► chuyển cán bộ      src/chatbot/pham_vi.py
   ② luật ưu tiên: ngành của Khoa + "đủ điều kiện/đậu…" ─► suy diễn lùi (C2)
   ③ conf ≥ τ ─► định tuyến theo ý định của mô hình
   ④ conf < τ ─► luật bổ trợ (xác nhận ý định trong top-3) hoặc chuyển cán bộ + logs/handoff.csv
                          │
   _route(): KB (8 ý định) · chào hỏi · C2 (suy diễn tiến/lùi) · C1 (lộ trình, A* + nhánh cận)
```

Bộ phân loại được nạp **một lần** khi khởi động `app.py` (không nạp lại mỗi lượt chat) theo thứ tự: Keras (`models/intent_model.keras`,
cần TensorFlow) → Naive Bayes (`models/nb_pipeline.pkl`) → phân loại từ khóa (chỉ để phát triển). τ lấy
theo đúng mô hình đang dùng từ `models/config.json` (Conv1D 0.92, NB 0.16). Nếu máy chạy không có
underthesea mà mô hình được huấn luyện với underthesea, `/api/trang-thai` và demo.py in cảnh báo (lỗi 7.10).

### Giao diện web (theo mockup Figma "HCMUE FIT Chat")

| Vùng | Nội dung |
|---|---|
| Header | logo Trường + Khoa, "HCMUE FIT Chat", trạng thái "Đang hoạt động" |
| Thanh bên | Cuộc hội thoại mới · CÔNG CỤ TƯ VẤN (biểu mẫu tư vấn chọn ngành – C2, lập lộ trình học – C1) · GẦN ĐÂY (hội thoại của trình duyệt này) · Chế độ kiểm tra (ý định, độ tin cậy, top-3, thực thể, ai quyết định; chỉnh τ) · Liên hệ Khoa CNTT và Tư vấn tuyển sinh của Trường |
| Vùng chat | tin nhắn bot (nền xanh nhạt) / người dùng (nền #124874), dòng **Nguồn … – năm áp dụng …** tách riêng dưới mỗi số liệu, nút Hữu ích / Chưa đúng |
| Ô nhập | câu hỏi gợi ý (chỉ giữ câu đã kiểm thử trả lời được), Enter để gửi |
| Điện thoại (≤ 900px) | header 2 hàng + nút Menu mở thanh bên, chip cuộn ngang |

Giao diện chỉ hiển thị: HTML câu trả lời do `src/web/markdown_lite.py` tạo từ văn bản của engine, mọi
nội dung người dùng được escape. Hội thoại lưu trong bộ nhớ máy chủ theo `client_id` của trình duyệt.

## 2. Định tuyến (mục 7.2)

| Ý định | Xử lý | Nguồn trả lời |
|---|---|---|
| diem_chuan, to_hop_xet_tuyen, hoc_phi, hoc_bong, ky_tuc_xa, thu_tuc_nhap_hoc, chuong_trinh_dao_tao, viec_lam | `answer_from_kb` – lọc theo loại, ngành (nhớ ngữ cảnh), năm; không nêu năm → năm mới nhất; năm không có → nói rõ + năm gần nhất | `data/knowledge_base.json` |
| chuong_trinh_dao_tao / tu_van_chon_nganh có ≥ 2 ngành + "khác/so sánh" | `compare_majors` – so sánh mô tả các ngành | cơ sở tri thức |
| chao_hoi | `greet` | câu mẫu |
| tu_van_chon_nganh | có ngành + "đủ điều kiện/đậu/vào được" → **suy diễn lùi**; ngược lại → **suy diễn tiến**; thiếu thông tin → luồng hỏi tiếp | C2 |
| lo_trinh_hoc_tap | luồng hỏi: học kỳ → môn đã đạt → môn rớt → học vượt → `dieu_phoi.lap_lo_trinh()` | C1 |

Mọi câu trả lời có số liệu in `Nguồn: … – năm áp dụng …`. Điểm chuẩn, tổ hợp lấy từ fanpage tuyển sinh của
Trường được nhóm xác nhận là nguồn chính thức (`da_kiem_chung=true`, không hiện cảnh báo); các mục còn là
mẫu (`da_kiem_chung=false`) có thêm dòng "Lưu ý: số liệu mẫu, nhóm chưa cập nhật nguồn chính thức".

## 3. Ba cơ chế bổ sung cho mô hình (và lý do)

| Cơ chế | Khi nào | Vì sao cần | Đã đo |
|---|---|---|---|
| **Chặn phạm vi** (`pham_vi.py`) | câu nhắc "ngành X" không thuộc Khoa, hoặc tên trường khác | mô hình tự tin ~1.0 cho "điểm chuẩn ngành Sư phạm Toán" → nếu không chặn, bot trả số liệu CNTT (sai) | 0/660 câu trong `intents.csv` bị chặn nhầm; chặn 10/82 câu `out_of_scope.csv` |
| **Luật ưu tiên** đủ điều kiện | ngành của Khoa + cụm "đủ điều kiện vào/đủ điểm vào/có đậu/vào được…" | dữ liệu C3 chưa có câu loại này, mô hình đoán `viec_lam` | trùng 2/660 câu (đều là "em được 27đ có đủ điểm vào CNTT không") |
| **Luật bổ trợ** (`RULE_CUES`, `RULE_CUES_RO` trong `engine.py`) | conf < τ, ý định nằm trong top-3 và câu có cụm đặc trưng: "lộ trình", "rớt môn", "nên chọn ngành nào" (C1/C2); "học phí", "tín chỉ", "việc làm", "ra trường làm"… (học phí, CTĐT, việc làm). Cụm rất rõ "sinh hoạt phí", "nghị định 116", "hỗ trợ/miễn học phí" được nhận kể cả ngoài top-3 | dữ liệu C3 ít câu về ngành CNGD, sinh hoạt phí, số tín chỉ nên mô hình hay dưới τ ở các câu này; luật không thay mô hình, chỉ xác nhận | 10/10 câu thử về thông tin mới được trả lời (trước đó 6/10); 99 câu test không đổi; 0 câu ngoài phạm vi bị trả lời nhầm do luật |

Chế độ debug hiện **ý định, độ tin cậy, top-3, thực thể, ai quyết định ý định** (`mo_hinh`,
`luat_uu_tien`, `luat_bo_tro`, `chan_pham_vi`, `duoi_nguong`) và thời gian xử lý.

## 4. Chuyển cán bộ và nhật ký (mục 7.7)

- `logs/handoff.csv`: thời gian, câu hỏi, ý định dự đoán, độ tin cậy, top-3, lý do
  (`duoi_nguong`, `nganh_khac`, `truong_khac`, `thieu_du_lieu`) – hàng đợi cho cán bộ và dữ liệu bổ
  sung tập huấn luyện vòng sau.
- `logs/feedback.csv`: nút Hữu ích / Chưa đúng dưới mỗi câu trả lời (thiết kế không dùng icon).
- `logs/conversations.jsonl`: mọi lượt (ý định, độ tin cậy, top-3, nguồn ý định, thời gian).
- Kênh liên hệ lấy từ `meta.lien_he_tu_van` (Khoa CNTT – phòng B.507) và `meta.lien_he_tuyen_sinh` (Trang Thông
  tin Tuyển sinh của Trường – phòng A.101) trong `knowledge_base.json`; hiển thị ở thanh bên và trong mọi câu
  chuyển cán bộ. Giá trị còn để mẫu `<…>` không được hiển thị.

## 5. Đánh giá end-to-end (mục 7.9)

`tests/scenarios.yaml`: 17 hội thoại, 44 lượt (câu do nhóm viết, không lấy từ tập huấn luyện), gồm
ngữ cảnh nhiều lượt, 3 luồng C1/C2, hủy luồng, ngoài phạm vi, ngành/trường khác. Kết quả đầy đủ:
[`C4_DANH_GIA.md`](C4_DANH_GIA.md).

| Chỉ số | Keras Conv1D (τ 0.92) | Naive Bayes (τ 0.16) |
|---|---|---|
| Lượt đạt mọi kỳ vọng | 42/44 (95%) | 41/44 (93%) |
| Đúng ý định | 22/24 (92%) | 21/24 (88%) |
| Câu có số liệu kèm nguồn + năm | 18/18 (100%) | 17/17 (100%) |
| Chuyển cán bộ đúng | 7/7 (100%) | 7/7 (100%) |
| Chuyển cán bộ thừa | 1/24 (4%) | 1/24 (4%) |
| Thời gian phản hồi trung bình | ~15–25 ms (không tính lần nạp mô hình) | ~8 ms |

Hai lượt sai của Keras là lỗi mô hình (giữ nguyên để báo cáo trung thực): "nhập học cần mang theo những
giấy tờ gì" (0.76 < τ → chuyển cán bộ thừa) và "CNTT với sư phạm tin khác nhau ở điểm nào" (`diem_chuan` 1.00).

Còn lại cho nhóm: thử với 3–5 người dùng thật (dùng nút Hữu ích / Chưa đúng + `logs/feedback.csv`) → mục "Đánh giá
người dùng" trong báo cáo.

## 6. Việc nhóm cần làm trước khi nộp

- [x] Kênh liên hệ thật của Khoa và Trường trong `data/knowledge_base.json`.
- [x] Học phí CNTT, SP Tin học (QĐ 2465/QĐ-ĐHSP), Công nghệ giáo dục (QĐ 2849/QĐ-ĐHSP) cho các năm học
  2024–2027; hỗ trợ học phí + sinh hoạt phí sinh viên sư phạm (Nghị định 116/2020/NĐ-CP, trường `ap_dung_tu_nam`:
  luôn kèm theo học phí SP Tin học của mọi năm); Công nghệ giáo dục không thuộc diện NĐ 116.
- [x] Ký túc xá: mức thu (QĐ 2892/QĐ-ĐHSP ngày 13/08/2025) và gia hạn hợp đồng, đóng tiền năm học 2025–2026
  (thông báo của Ký túc xá ngày 13/08/2025).
- [x] Học bổng (Quy trình xét HBKKHT 2023), CTĐT + việc làm 3 ngành (CTĐT CNTT – QĐ 2619/QĐ-ĐHSP; Cẩm nang
  tuyển sinh 2025 của Khoa), thủ tục nhập học 2026 và đăng ký KTX cho tân sinh viên (KH 539/KH-ĐHSP) – không còn mục mẫu.
- [ ] Bổ sung nếu có: mức tiền và điều kiện điểm của học bổng KKHT; số tín chỉ CTĐT Sư phạm Tin học và Công nghệ
  giáo dục; nội dung sửa đổi của Nghị định 60/2025/NĐ-CP (KH 539 nhắc tới) đối với mức hỗ trợ của NĐ 116.
- [ ] C3: quy trình gán nhãn + kappa; thêm câu "đủ điều kiện vào ngành X", câu ngắn, câu `viec_lam`,
  `thu_tuc_nhap_hoc`; huấn luyện lại bằng notebook rồi chép `models.zip` vào `models/` và chạy `pytest -q`.
- [ ] Thử với người dùng thật, quay video demo.

## 7. Hình giao diện (cho báo cáo)

`docs/hinh/c4/`: `web_trang_chu.png` (màn hình đầu), `web_diem_chuan_co_nguon.png` (số liệu kèm nguồn + năm),
`web_c2_suy_dien_lui.png` (kiểm tra đủ điều kiện – chuỗi luật), `web_c1_lo_trinh.png` (biểu mẫu lộ trình,
có môn học lại), `web_chuyen_can_bo.png` (ngành khác → chuyển cán bộ), `web_dien_thoai.png` (bản điện thoại).
