# Chức năng 3 – Phân loại ý định (Phần 5: 5.2, 5.3)

> Notebook huấn luyện: [`notebooks/N31_De03_Train.ipynb`](../notebooks/N31_De03_Train.ipynb) (chạy trên Colab).
> Mô hình đang dùng trong chatbot: `models/` (sinh ra từ notebook, mỗi file < 2 MB nên được commit).
> Hình cho báo cáo: [`docs/hinh/c3/`](hinh/c3/). Toàn bộ số liệu: `models/metrics_c3.json`.

## 1. Dữ liệu

| Tập | Số câu | Ghi chú |
|---|---|---|
| `data/intents.csv` | 660 | 11 ý định × 60 câu (cân bằng); không có câu trùng sau tiền xử lý |
| `data/out_of_scope.csv` | 82 | câu ngoài phạm vi – chỉ dùng để chọn ngưỡng τ, **không** là một lớp |

11 ý định: `diem_chuan`, `to_hop_xet_tuyen`, `hoc_phi`, `hoc_bong`, `ky_tuc_xa`, `chuong_trinh_dao_tao`,
`viec_lam`, `thu_tuc_nhap_hoc`, `chao_hoi` + 2 ý định gọi mô-đun: `tu_van_chon_nganh` (C2), `lo_trinh_hoc_tap` (C1).

Chia stratified **70/15/15** (`random_state=42`): train 462 · val 99 · test 99. Câu ngoài phạm vi chia đôi:
41 câu để chọn τ (OOS_val), 41 câu để báo cáo (OOS_test).

**Còn thiếu theo yêu cầu đề** (nhóm bổ sung): cột `nguon`, `nguoi_gan_1`, `nguoi_gan_2`; quy trình gán
nhãn + Cohen's kappa (`docs/quy_trinh_gan_nhan.md`); khai báo tỷ lệ câu do LLM sinh (nếu có).

## 2. Tiền xử lý (`src/nlu/preprocess.py` = Cell 3 của notebook)

NFC → chữ thường → chuẩn hóa vị trí dấu (hòa → hoà) → bỏ ký tự đặc biệt → teencode/viết tắt
(`hp`→học phí, `ktx`→ký túc xá, `bn`→bao nhiêu, `đk`→điều kiện, `tc`→tín chỉ; **không** đổi `a`, `b` để giữ
"khối A/B") → tách từ **underthesea** (`học_phí`, `công_nghệ_thông_tin`).

Một hàm duy nhất cho cả huấn luyện và chatbot. Kiểm chứng: nạp mô hình trong `models/` bằng
`src/nlu/classifier.py` và chạy lại trên tập test tái tạo **đúng** accuracy trong `metrics_c3.json`
(Conv1D 0.8889, NB 0.8283) – xem `tests/test_c3_c4.py::test_mo_hinh_tai_tao_dung_so_lieu_test`.

## 3. Mô hình

| Mô hình | Kiến trúc |
|---|---|
| Naive Bayes (baseline) | TF-IDF (1–2 gram) + MultinomialNB, `alpha` chọn trên val = 1.0 |
| LSTM | Embedding(64, `mask_zero`) → BiLSTM(64) → BiLSTM(32) → Dense(64) → softmax |
| LSTM v2 | Embedding(64, `mask_zero`) → SpatialDropout → BiLSTM(64) → LayerNorm → BiGRU(32) → Dense → softmax |
| **Conv1D (mô hình chính)** | Embedding(64) → Conv1D(128,3) → MaxPool → Conv1D(64,3) → GlobalMaxPool → Dense(64) → softmax |

Huấn luyện: Adam, `sparse_categorical_crossentropy`, EarlyStopping theo `val_loss` (patience 6,
khôi phục trọng số tốt nhất), `MAX_LEN = 50`, padding sau.

> **Lưu ý tái lập:** lần huấn luyện tạo ra `models/` hiện tại **không cố định seed của TensorFlow**
> (`seed = 42` trong `config.json` chỉ áp dụng cho phép chia dữ liệu). Chạy lại notebook sẽ cho trọng số
> và số liệu hơi khác; phép chia train/val/test thì giữ nguyên.

## 4. Kết quả (cùng tập test 99 câu)

| Mô hình | Val macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|
| Naive Bayes | 0.854 | 0.828 | 0.831 |
| LSTM | 0.781 | 0.768 | 0.776 |
| LSTM v2 | 0.888 | 0.879 | 0.881 |
| **Conv1D** | **0.928** | **0.889** | **0.894** |

Mô hình chính = macro F1 trên **val** cao nhất (không chọn bằng tập test) → Conv1D.

F1 từng ý định (test):

| Ý định | NaiveBayes | LSTM | LSTM_v2 | Conv1D |
|---|---|---|---|---|
| chao_hoi | 0.94 | 0.94 | 0.94 | 0.94 |
| chuong_trinh_dao_tao | 0.88 | 0.78 | 0.78 | 0.82 |
| diem_chuan | 0.73 | 0.73 | 0.89 | 0.94 |
| hoc_bong | 0.88 | 0.64 | 0.80 | 1.00 |
| hoc_phi | 0.67 | 0.82 | 0.86 | 1.00 |
| ky_tuc_xa | 0.94 | 0.88 | 0.89 | 1.00 |
| lo_trinh_hoc_tap | 0.94 | 0.75 | 0.89 | 0.89 |
| thu_tuc_nhap_hoc | 0.94 | 0.71 | 0.94 | 0.71 |
| to_hop_xet_tuyen | 0.75 | 0.71 | 1.00 | 1.00 |
| tu_van_chon_nganh | 0.95 | 0.94 | 0.94 | 0.89 |
| viec_lam | 0.53 | 0.64 | 0.76 | 0.64 |
| **macro** | 0.83 | 0.78 | 0.88 | 0.89 |

Hình: ma trận nhầm lẫn `hinh/c3/confusion_matrices.png`, đường học `hinh/c3/*_history.png`,
so sánh `hinh/c3/model_comparison.png`, `hinh/c3/f1_per_intent_comparison.png`.

## 5. Ngưỡng tin cậy τ

Quét τ trên val ∪ OOS_val với hai tỷ lệ: **trả lời đúng** câu trong phạm vi và **chặn được** câu ngoài
phạm vi; chọn τ có trung bình điều hòa lớn nhất (`hinh/c3/confidence_threshold.png`).

| Mô hình | τ | Test: trả lời đúng | Test: chặn OOS | Độ chính xác của các câu bot chịu trả lời |
|---|---|---|---|---|
| Naive Bayes | 0.16 | 71.7% | 68.3% | 75.5% |
| LSTM | 0.77 | 58.6% | 56.1% | 67.4% |
| LSTM v2 | 0.87 | 74.7% | 80.5% | 88.1% |
| **Conv1D** | **0.92** | **78.8%** | **75.6%** | **88.6%** |

τ = 0.92 ghi trong `models/config.json`; chatbot đọc giá trị này (slider ở sidebar để demo).

## 6. Hạn chế (đưa vào báo cáo)

- Ngưỡng τ không chặn được câu về **ngành/trường khác** (mô hình tự tin ~1.0 vì câu giống `diem_chuan`)
  → C4 chặn thêm bằng thực thể (`src/chatbot/pham_vi.py`), xem [`C4_CHATBOT.md`](C4_CHATBOT.md).
- Dữ liệu chưa có câu dạng **"em có đủ điều kiện vào ngành X không"** → mô hình đoán `viec_lam`;
  C4 dùng luật ưu tiên (mục 7.2 hướng dẫn). Vòng sau nên thêm ~20 câu loại này vào `tu_van_chon_nganh`.
- Câu rất ngắn / câu so sánh hai ngành đôi khi sai với độ tin cậy cao (vd. "CNTT với sư phạm tin khác
  nhau ở điểm nào" → `diem_chuan` 1.00). Câu bị chuyển cán bộ được ghi ở `logs/handoff.csv` để bổ sung
  vào tập huấn luyện vòng sau.
- `viec_lam` và `thu_tuc_nhap_hoc` có F1 thấp nhất (0.64, 0.71) – cần thêm câu đa dạng.
