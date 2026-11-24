# Chatbot tư vấn tuyển sinh – Khoa Công nghệ thông tin

Chatbot hiểu câu hỏi tiếng Việt của thí sinh/sinh viên, trả lời từ cơ sở tri thức đã kiểm chứng
(kèm **nguồn + năm áp dụng**), tư vấn chọn ngành bằng hệ luật và gợi ý lộ trình đăng ký học phần.
Không dùng API mô hình ngôn ngữ lớn làm lõi xử lý.

📖 **Định hướng toàn bộ đồ án:** [`docs/HUONG_DAN_DO_AN.md`](docs/HUONG_DAN_DO_AN.md)

## Chạy thử

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Trạng thái các chức năng

| Chức năng | Thư mục | Trạng thái |
|---|---|---|
| C1 – Lộ trình học (BFS, A*, nhánh cận) | `src/planner/` | ✅ Hoàn thành – [`docs/C1_LO_TRINH.md`](docs/C1_LO_TRINH.md) |
| C2 – Tư vấn ngành (luật, suy diễn tiến/lùi, ID3) | `src/advisor/` | ✅ 33 luật + suy diễn tiến/lùi + kiểm chứng ID3 bằng scikit-learn – [`docs/C2_TU_VAN_NGANH.md`](docs/C2_TU_VAN_NGANH.md) |
| C3 – Phân loại ý định (NB, Keras) | `src/nlu/`, `notebooks/`, `models/` | ✅ 660 câu / 11 ý định; Conv1D test macro F1 0.89, τ = 0.92 – [`docs/C3_PHAN_LOAI_Y_DINH.md`](docs/C3_PHAN_LOAI_Y_DINH.md) |
| C4 – Ứng dụng chatbot | `app.py`, `src/chatbot/` | ✅ Dùng mô hình Keras của C3, gọi C1 + C2, chặn ngành/trường khác, chuyển cán bộ, 👍/👎; end-to-end 42/44 lượt đạt – [`docs/C4_CHATBOT.md`](docs/C4_CHATBOT.md) |

C4 tự động dùng mô hình tốt nhất có trong `models/`: `intent_model.keras` (C3, cần TensorFlow) →
`nb_pipeline.pkl` → bộ phân loại từ khóa tạm thời. Mô hình được huấn luyện với **underthesea** nên phải
cài thư viện này (đã có trong `requirements.txt`). Huấn luyện lại: chạy `notebooks/N31_De03_Train.ipynb`
trên Colab, giải nén `models.zip` vào `models/`, rồi `pytest -q`.
Kiểm chứng cây quyết định C2: `python -m src.advisor.kiem_chung_sklearn`.
Đánh giá end-to-end chatbot: `python -m src.chatbot.danh_gia`.

Lộ trình học (Chức năng 1): hỏi chatbot hoặc dùng form "Lộ trình học" ở thanh bên; chatbot tự chọn
thuật toán (A* chính, nhánh cận dự phòng). BFS được cài đặt theo đề nhưng không dùng trong chatbot.

## Việc cần làm tiếp

- [ ] Điền `data/knowledge_base.json` bằng số liệu thật (nguồn, url, năm) và **kênh liên hệ thật** của Khoa – nhiều mục vẫn là MẪU
- [ ] Xác nhận cột `hoc_ky_mo` trong `data/hoc_phan.csv` lấy từ CTĐT hay mô phỏng; điền tên thành viên vào 3 file thuật toán C1
- [x] `data/intents.csv` 660 câu / 11 ý định, `data/out_of_scope.csv` 82 câu
- [ ] C3: quy trình gán nhãn + Cohen's kappa, cột `nguon`, khai báo câu do LLM sinh (nếu có)
- [x] C2: hệ luật 33 luật, suy diễn tiến, suy diễn lùi
- [x] C2: khảo sát `data/survey.csv` (15 mẫu), kiểm chứng ID3 bằng scikit-learn
- [ ] C2: thay nguồn trong `data/tuyen_sinh_2026.json` bằng Đề án TS 2026 chính thức
- [x] Huấn luyện NB + LSTM + Conv1D, chọn ngưỡng τ (notebook C3)
- [ ] Thử chatbot với 3–5 người dùng thật, quay video demo
