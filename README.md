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
| C2 – Tư vấn ngành (luật, suy diễn tiến/lùi, ID3) | `src/advisor/` | 🧱 Khung: 4 luật mẫu, `facts.py` xong; suy diễn + ID3 `TODO` |
| C3 – Phân loại ý định (NB, Keras) | `src/nlu/` | 🧱 Tiền xử lý + script huấn luyện mẫu; cần bộ dữ liệu ≥600 câu |
| C4 – Ứng dụng chatbot | `app.py`, `src/chatbot/` | ✅ Chạy được: phân loại **từ khóa tạm thời**, gọi C1 (tự chọn thuật toán), báo "mô-đun chưa cài đặt" cho C2 |

C4 tự động dùng mô hình tốt nhất có trong `models/`: `intent_model.keras` (C3) → `nb_pipeline.pkl` →
bộ phân loại từ khóa tạm thời. Khi C2 được cài đặt, chatbot tự gọi được mà không cần sửa mã C4.

App có 2 trang (chọn ở sidebar): **💬 Chatbot** cho thí sinh/sinh viên và **📊 So sánh thuật toán (C1)**
cho giảng viên/hội đồng. Lộ trình học chạy ở dòng lệnh: `python -m src.planner.cli`.

## Việc cần làm tiếp

- [ ] Điền `data/knowledge_base.json` bằng số liệu thật (nguồn, url, năm) – hiện **toàn bộ là MẪU**
- [ ] Xác nhận cột `hoc_ky_mo` trong `data/hoc_phan.csv` lấy từ CTĐT hay mô phỏng; điền tên thành viên vào 3 file thuật toán C1
- [ ] `data/intents.csv` ≥600 câu (mẫu: `data/intents_sample.csv`), `data/out_of_scope.csv`, `data/survey.csv`
- [ ] Cài đặt các hàm `TODO` trong `src/advisor/`
- [ ] Huấn luyện `python -m src.nlu.train_keras --data data/intents.csv`, chọn ngưỡng τ
