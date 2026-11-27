# HCMUE FIT Chat – Chatbot tư vấn tuyển sinh Khoa Công nghệ Thông tin

Đồ án Trí tuệ nhân tạo – Nhóm 31, Đề 03. Chatbot hiểu câu hỏi tiếng Việt của thí sinh/sinh viên,
trả lời từ cơ sở tri thức **kèm nguồn và năm áp dụng**, tư vấn chọn ngành bằng hệ luật và gợi ý lộ trình
đăng ký học phần. Toàn bộ thuật toán AI là mã Python của nhóm; giao diện web (Flask + HTML/CSS/JS) chỉ gọi
và hiển thị kết quả. **Không dùng API mô hình ngôn ngữ lớn, không cần khóa API, chạy được không cần GPU.**

| Chức năng | Nội dung | Mã nguồn | Tài liệu |
|---|---|---|---|
| C1 | Lập lộ trình học tối ưu: BFS, A*, quay lui – nhánh cận | `src/planner/` | [`docs/C1_LO_TRINH.md`](docs/C1_LO_TRINH.md) |
| C2 | Tư vấn chọn ngành: 33 luật, suy diễn tiến / lùi, ID3 | `src/advisor/` | [`docs/C2_TU_VAN_NGANH.md`](docs/C2_TU_VAN_NGANH.md) |
| C3 | Phân loại ý định: tiền xử lý tiếng Việt, Naive Bayes, Keras (LSTM, Conv1D) | `src/nlu/`, `notebooks/`, `models/` | [`docs/C3_PHAN_LOAI_Y_DINH.md`](docs/C3_PHAN_LOAI_Y_DINH.md) |
| C4 | Ứng dụng chatbot web, tích hợp C1–C3 | `app.py`, `src/chatbot/`, `src/web/`, `templates/`, `static/` | [`docs/C4_CHATBOT.md`](docs/C4_CHATBOT.md) |

## 1. Thành viên

| STT | Họ và tên | MSSV | Phụ trách |
|---|---|---|---|
| 1 | `<Họ tên>` | `<MSSV>` | `<…>` |
| 2 | `<Họ tên>` | `<MSSV>` | `<…>` |
| 3 | `<Họ tên>` | `<MSSV>` | `<…>` |
| 4 | `<Họ tên>` | `<MSSV>` | `<…>` |

> Nhóm điền bảng trên và dòng "Thành viên phụ trách" ở đầu `src/planner/bfs.py`, `a_sao.py`, `nhanh_can.py`.

## 2. Môi trường đã chạy thử

| Hệ điều hành | Python | Kết quả |
|---|---|---|
| Ubuntu 24.04, chỉ CPU | 3.13.18 | `pytest` 75/75 · `python demo.py` ~5 giây (lần đầu nạp TensorFlow ~30 giây) · `python app.py` |
| Ubuntu 24.04, chỉ CPU | 3.11.18 | như trên |
| Ubuntu 24.04, chỉ CPU | 3.10.20 | như trên (Keras 3.12, nạp mô hình bằng chế độ tương thích – xem mục 7) |

Mọi thư viện trong `requirements.txt` đều có bản cài sẵn cho **Python 3.10–3.13 trên Windows, Linux, macOS**
(đã kiểm tra bằng `pip download`); nhóm chưa chạy thử trực tiếp trên Windows/macOS.

## 3. Cài đặt

```bash
python -m venv .venv
```

Kích hoạt môi trường – Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Cần Internet một lần để cài thư viện. Mô hình đã huấn luyện có sẵn trong `models/` – **không cần huấn luyện lại**.

## 4. Chạy chương trình

**Demo không giao diện – Chức năng 1, 2, 3 trên dữ liệu mẫu** (in lộ trình + chi phí + số nút mở rộng,
kết luận suy diễn tiến/lùi, độ đo mô hình; khoảng 5–40 giây trên CPU):

```bash
python demo.py
```

Từng chức năng riêng:

```bash
python demo.py c1
```

```bash
python demo.py c2
```

```bash
python demo.py c3
```

**Chức năng 4 – chatbot web**, rồi mở trình duyệt tại http://127.0.0.1:5000:

```bash
python app.py
```

Tùy chọn: `python app.py --port 8000` (đổi cổng), `python app.py --nb` (dùng baseline Naive Bayes nếu máy
không cài được TensorFlow).

**Các lệnh khác**

| Việc | Lệnh |
|---|---|
| C2 – ID3 hai tầng, cây scikit-learn, rút luật, lưu hình | `python -m src.advisor.kiem_chung_sklearn` |
| C3 – huấn luyện lại (Colab hoặc Jupyter) | mở `notebooks/N31_De03_Train.ipynb`, chạy hết; giải nén `models.zip` vào `models/` |
| C4 – đánh giá end-to-end 44 lượt hội thoại | `python -m src.chatbot.danh_gia` |
| Toàn bộ kiểm thử | `pytest -q` |

## 5. Cấu trúc thư mục

```
app.py                  Chức năng 4 – máy chủ Flask (giao diện web)
demo.py                 chạy C1–C3 không giao diện, in kết quả
requirements.txt        thư viện, ghi rõ phiên bản
src/planner/            C1: model.py, bfs.py, a_sao.py, nhanh_can.py, dieu_phoi.py
src/advisor/            C2: rules.py, inference.py (suy diễn tiến/lùi), tu_van.py, kiem_chung_sklearn.py
src/nlu/                C3: preprocess.py, classifier.py (+ script thử nghiệm cũ train_*.py)
src/chatbot/            C4: engine.py (định tuyến, ngưỡng, chuyển cán bộ), flows.py, entities.py, kb.py, pham_vi.py
src/web/                C4: markdown_lite.py (câu trả lời → HTML đã escape)
templates/, static/     giao diện HTML/CSS/JS theo mockup Figma
models/                 mô hình Keras đã huấn luyện (intent_model.keras) + tokenizer, nhãn, ngưỡng, số liệu
data/                   dữ liệu đầy đủ (mục 6)
data/sample/            dữ liệu mẫu đủ để chạy demo.py
notebooks/              notebook huấn luyện C3
docs/                   tài liệu từng chức năng, hình cho báo cáo
tests/                  pytest + kịch bản hội thoại tests/scenarios.yaml
```

## 6. Dữ liệu – nguồn và giấy phép

| File | Nội dung | Nguồn | Giấy phép / ghi chú |
|---|---|---|---|
| `data/hoc_phan.csv` | 28 học phần (75 TC) | Chương trình đào tạo ban hành kèm Quyết định 2619/QĐ-ĐHSP ngày 04/09/2024, Trường ĐHSP TP.HCM | Văn bản công khai của Trường; nhóm trích để học tập. Mã, tên, tín chỉ, học phần học trước khớp CTĐT; cột `hoc_ky_mo` (Lẻ/Chẵn) là **mô phỏng** dựa trên kế hoạch đào tạo (2 học phần COMP1016, COMP1332 khác kế hoạch CTĐT) |
| `data/tuyen_sinh_2026.json` | Tổ hợp, ngưỡng, điểm chuẩn 2026 (tham số C2) | Nhóm tổng hợp từ fanpage tuyển sinh của Trường (link trong file) | Trường không ban hành công văn riêng; nhóm xác nhận fanpage là nguồn chính thức (`da_kiem_chung: true`) |
| `data/knowledge_base.json` | Cơ sở tri thức của chatbot (44 mục, mỗi mục có `nguon`, `nam`) | Fanpage tuyển sinh, Cổng Chính phủ (điểm chuẩn, tổ hợp); QĐ 2465, QĐ 2849 (học phí); NĐ 116/2020 (hỗ trợ sinh viên sư phạm); QĐ 2892, thông báo KTX (ký túc xá); KH 539/KH-ĐHSP (nhập học 2026); Quy trình xét học bổng KKHT (2023); CTĐT CNTT (QĐ 2619) và Cẩm nang tuyển sinh 2025 của Khoa (CTĐT, việc làm) | Văn bản công khai của Trường/Chính phủ; mọi mục đã kiểm chứng |
| `data/survey.csv` | 15 mẫu khảo sát, 4 thuộc tính (ID3) | Nhóm tự khảo sát | Dữ liệu ẩn danh của nhóm, dùng cho học tập |
| `data/intents.csv` | 660 câu, 11 ý định | Nhóm tự viết / thu thập | Dữ liệu của nhóm, dùng cho học tập. Khai báo câu do LLM gợi ý (nếu có) trong báo cáo |
| `data/out_of_scope.csv` | 82 câu ngoài phạm vi | Nhóm tự viết | Như trên |
| `data/sample/` | Dữ liệu mẫu cho `demo.py`: bản sao `hoc_phan.csv`, `survey.csv`; 99 câu test và 41 câu ngoài phạm vi **không dùng khi huấn luyện**; `tinh_huong.json` (hồ sơ mẫu do nhóm đặt) | Trích từ các file trên | Như trên |
| `static/img/logo_*.png`, `static/favicon.ico`, `static/img/favicon-*.png` | Logo Trường và Khoa (favicon thu nhỏ từ logo Khoa) | Bộ nhận diện HCMUE (qua mockup Figma của nhóm) | Thuộc sở hữu của Trường ĐHSP TP.HCM; chỉ dùng trong đồ án |

Dữ liệu đầy đủ đều nhỏ (< 1 MB) nên được nộp kèm, không cần link tải. Font Inter (giấy phép SIL OFL) được
tải từ Google Fonts; khi không có mạng, giao diện tự dùng font hệ thống.
Thư viện chính: TensorFlow/Keras (Apache-2.0), scikit-learn (BSD-3), Flask (BSD-3), underthesea (GPL-3.0).

## 7. Lưu ý khi chấm

- **Python 3.10:** Keras mới nhất cho Python 3.10 là 3.12 (requirements.txt tự chọn theo phiên bản Python).
  Mô hình được lưu bằng Keras mới hơn nên `src/nlu/classifier.py` tự bỏ trường rỗng `quantization_config`
  trên bản sao tạm khi nạp; trọng số và kết quả không đổi (`python demo.py c3` in "KHỚP").
- Lần đầu chạy `demo.py` / `app.py` mất 10–30 giây để nạp TensorFlow; các lần trả lời sau ~10–30 ms.
- `logs/` (hội thoại, câu chuyển cán bộ, đánh giá Hữu ích / Chưa đúng) được tạo khi chạy web, không nộp.
- Tạo file nộp không kèm `venv/`, `__pycache__/`, `logs/`:

```bash
git archive -o Nhom31_De03.zip HEAD
```

## 8. Video demo

`<link video demo>`
