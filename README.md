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
| C2 – Tư vấn ngành (luật, suy diễn tiến/lùi, ID3) | `src/advisor/` | ✅ 33 luật + suy diễn tiến/lùi – [`docs/C2_TU_VAN_NGANH.md`](docs/C2_TU_VAN_NGANH.md); ID3 `TODO` |
| C3 – Phân loại ý định (NB, Keras) | `src/nlu/` | 🧱 Tiền xử lý + script huấn luyện mẫu; cần bộ dữ liệu ≥600 câu |
| C4 – Ứng dụng chatbot | `app.py`, `src/chatbot/` | ✅ Chạy được: phân loại **từ khóa tạm thời**, gọi C1 (tự chọn thuật toán) và C2 (hỏi đáp nhiều lượt do suy diễn lùi điều khiển) |

C4 tự động dùng mô hình tốt nhất có trong `models/`: `intent_model.keras` (C3) → `nb_pipeline.pkl` →
bộ phân loại từ khóa tạm thời. Kịch bản C2: `python -m src.advisor.kich_ban`.

App có 2 trang (chọn ở sidebar): **💬 Chatbot** cho thí sinh/sinh viên và **📊 So sánh thuật toán (C1)**
cho giảng viên/hội đồng. Lộ trình học chạy ở dòng lệnh: `python -m src.planner.cli`.

## Việc cần làm tiếp

- [ ] Điền `data/knowledge_base.json` bằng số liệu thật (nguồn, url, năm) – hiện **toàn bộ là MẪU**
- [ ] Xác nhận cột `hoc_ky_mo` trong `data/hoc_phan.csv` lấy từ CTĐT hay mô phỏng; điền tên thành viên vào 3 file thuật toán C1
- [ ] `data/intents.csv` ≥600 câu (mẫu: `data/intents_sample.csv`), `data/out_of_scope.csv`, `data/survey.csv`
- [x] C2: hệ luật 33 luật, suy diễn tiến, suy diễn lùi
- [ ] C2: thay nguồn trong `data/tuyen_sinh_2026.json` bằng Đề án TS 2026 chính thức; cài `src/advisor/id3.py`
- [ ] Huấn luyện `python -m src.nlu.train_keras --data data/intents.csv`, chọn ngưỡng τ
