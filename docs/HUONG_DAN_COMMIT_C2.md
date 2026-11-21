# Hướng dẫn commit Chức năng 2 lên repo nhóm

Đưa Chức năng 2 lên repo nhóm theo đúng thứ tự phụ thuộc; mỗi thành viên tự commit phần của mình
(lịch sử git là minh chứng ai làm phần nào). Bản đầy đủ có giải thích từng file: `docs/C2_mo_ta_cai_dat.docx`.

## 1. Các bước

| Bước | Phần | File | Dùng mã của | Sau bước này |
|---|---|---|---|---|
| 1 | Tham số và tính điểm | `data/tuyen_sinh_2026.json`, `src/advisor/tham_so.py`, `src/advisor/diem.py` | – | Chatbot chạy như cũ |
| 2 | Hệ luật 33 luật | `src/advisor/rules.py` | 1 | Chatbot chạy như cũ |
| 3 | Dữ kiện và suy diễn tiến/lùi | `src/advisor/facts.py`, `src/advisor/inference.py` | 1, 2 | **Chatbot lỗi import** cho đến bước 5 |
| 4 | Lớp tư vấn | `src/advisor/tu_van.py` | 1, 2, 3 | Gọi được `tu_van()`, `kiem_tra_dieu_kien()`; chatbot vẫn lỗi |
| 5 | Nối chatbot | `src/chatbot/entities.py`, `flows.py`, `engine.py`, `kb.py`, `data/knowledge_base.json`, `app.py`, `tests/test_engine.py` | 1–4 | Chatbot chạy lại, `pytest` qua 12/12 |
| 6 | Khảo sát và kiểm chứng ID3 | `data/survey.csv`, `src/advisor/kiem_chung_sklearn.py`, `docs/hinh/cay_quyet_dinh_sklearn.png` | – | Không ảnh hưởng chatbot |
| 7 | Tài liệu | `docs/C2_TU_VAN_NGANH.md`, `docs/C2_mo_ta_cai_dat.docx`, `docs/HUONG_DAN_COMMIT_C2.md`, `README.md` | – | Không đổi hành vi |

Phụ thuộc: `tuyen_sinh_2026.json` ← `tham_so.py` ← `diem.py`, `rules.py` ← `facts.py` ← `inference.py`
← `tu_van.py` ← chatbot. Bước 3 bỏ hàm `latest_cutoffs` mà `engine.py` cũ còn dùng nên **bước 3, 4, 5 phải
merge cùng một lần**. Bước 6 độc lập, làm lúc nào cũng được.

## 2. Nguyên tắc

- Mỗi người `git config user.name/user.email` bằng tên của chính mình.
- Không commit thẳng lên `main`: mỗi bước một nhánh, một Pull Request; nhóm trưởng merge.
- Không dùng `git add .`; không đưa `__pycache__/`, `.venv/`, `venv/` lên.
- Giữ dòng `# AI-assisted: …` ở đầu mỗi file.

## 3. Lệnh mẫu (bước 2)

```bash
git switch main
git pull
git switch -c c2-he-luat
git add src/advisor/rules.py
git commit -m "C2: hệ luật 33 luật"
git push -u origin c2-he-luat
```

## 4. Kiểm tra nhanh (thư mục gốc repo)

Sau bước 4:

```bash
python -B -c "from src.advisor.tu_van import kiem_tra_dieu_kien as k; print(k({'phuong_thuc': 'THPT', 'khu_vuc': 'KV1', 'diem_Toan': 7, 'diem_Anh': 6.2, 'diem_Ly': 6}, 'SPTH').ket_luan)"
```

→ `Không` (DXT 19.95 thiếu 0.05 điểm so với ngưỡng 20.00)

Sau bước 5:

```bash
python -m pytest -q
```

Sau bước 6:

```bash
python -m src.advisor.kiem_chung_sklearn
```

→ in IG hai tầng (gốc A3 – Lĩnh vực), cây scikit-learn, 4 luật rút trích và lưu `docs/hinh/cay_quyet_dinh_sklearn.png`.
