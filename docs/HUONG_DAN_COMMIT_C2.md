# Hướng dẫn commit Chức năng 2 lên repo nhóm

Phần A chỉ gồm mã của Chức năng 2, **chưa nối vào chatbot (Chức năng 4)**; mỗi bước chạy được độc lập.
Phần B (nối chatbot) làm sau, khi nhóm tích hợp Chức năng 4. Giải thích từng file: `docs/C2_mo_ta_cai_dat.docx`.

## Phần A – Chức năng 2

| Bước | File | Commit message | Dùng mã của |
|---|---|---|---|
| 1 | `data/tuyen_sinh_2026.json`, `src/advisor/__init__.py`, `src/advisor/tham_so.py`, `src/advisor/diem.py` | `C2: tham số tuyển sinh 2026 và công thức tính điểm` | – |
| 2 | `src/advisor/rules.py` | `C2: biểu diễn luật và 33 luật tư vấn chọn ngành` | 1 |
| 3 | `src/advisor/facts.py` | `C2: kiểm tra dữ kiện đầu vào và chuyển thành sự kiện` | 1 |
| 4 | `src/advisor/inference.py` | `C2: tự cài đặt suy diễn tiến và suy diễn lùi` | 2, 3 |
| 5 | `src/advisor/tu_van.py` | `C2: lớp tư vấn – gợi ý ngành và kiểm tra điều kiện xét tuyển` | 1–4 |
| 6 | `data/survey.csv`, `src/advisor/kiem_chung_sklearn.py`, `docs/hinh/cay_quyet_dinh_sklearn.png` | `C2: khảo sát và kiểm chứng cây quyết định bằng scikit-learn` | – |
| 7 | `docs/C2_TU_VAN_NGANH.md`, `docs/C2_mo_ta_cai_dat.docx`, `docs/HUONG_DAN_COMMIT_C2.md` | `C2: tài liệu` | – |

Phụ thuộc: `tuyen_sinh_2026.json` ← `tham_so.py` ← `diem.py`, `rules.py`, `facts.py` ← `inference.py` ← `tu_van.py`.
Bước 6 độc lập, làm lúc nào cũng được.

**Lưu ý:** nếu repo đang có `src/chatbot/engine.py` bản khung cũ (có dòng
`from ..advisor.facts import build_facts, latest_cutoffs`), bước 3 sẽ làm chatbot lỗi import cho đến khi làm
Phần B. Khi đó gộp bước 3–5 và Phần B vào cùng một Pull Request. Nếu repo chưa có chatbot thì không ảnh hưởng.

### Kiểm tra sau từng bước (thư mục gốc repo; `-B` để không sinh `__pycache__`)

Bước 1 → `19.95` (KV1, tổng 19.2 < 22.5 nên cộng đủ 0.75):

```bash
python -B -c "from src.advisor.diem import diem_xet_tuyen; print(diem_xet_tuyen(19.2, 'KV1'))"
```

Bước 2 → `33 luật, 3 tầng`:

```bash
python -B -c "from src.advisor.rules import RULES, phan_tang; print(len(RULES), 'luật,', max(phan_tang(RULES).values()) + 1, 'tầng')"
```

Bước 5 → `Không 0.05` (DXT 19.95, thiếu 0.05 điểm so với ngưỡng SPTH 20.00):

```bash
python -B -c "from src.advisor.tu_van import kiem_tra_dieu_kien as k; r = k({'phuong_thuc': 'THPT', 'khu_vuc': 'KV1', 'diem_Toan': 7, 'diem_Anh': 6.2, 'diem_Ly': 6, 'diem_Hoa': None, 'diem_Sinh': None, 'diem_Tin': None}, 'SPTH'); print(r.ket_luan, r.lui.thieu_diem)"
```

Môn không thi phải ghi `None`. Nếu bỏ trống (không có khóa), suy diễn lùi coi là "chưa hỏi" và trả về
"Chưa đủ thông tin".

Bước 6 → in IG hai tầng (gốc A3 – Lĩnh vực), cây scikit-learn, 4 luật, `15/15` và lưu hình:

```bash
python -m src.advisor.kiem_chung_sklearn
```

## Phần B – Nối vào chatbot (làm sau, cùng Chức năng 4)

| File | Commit message |
|---|---|
| `src/chatbot/entities.py` | `C2: nhận phương thức, khu vực, điểm từng môn; sửa nhận nhầm ngành Công nghệ giáo dục` |
| `src/chatbot/flows.py` | `C2: hội thoại nhiều lượt, câu hỏi tiếp theo do suy diễn lùi chọn` |
| `src/chatbot/engine.py` | `C2: chatbot gọi tư vấn ngành và kiểm tra điều kiện xét tuyển` |
| `src/chatbot/kb.py` | `C2: dòng nguồn và lưu ý số liệu chưa đối chiếu nguồn chính thức` |
| `data/knowledge_base.json` | `C2: thêm ngành Công nghệ giáo dục, điểm chuẩn và tổ hợp 2024-2026` |
| `app.py` | `C2: form tư vấn chọn ngành trên giao diện` |
| `tests/test_engine.py` | `C2: test hội thoại tư vấn ngành và kiểm tra điều kiện` |

Các file này import lẫn nhau (`engine.py` cần `flows.py`, `entities.py`), nên commit cùng một Pull Request rồi
chạy `python -B -m pytest -q` (12/12).

## Nguyên tắc

- Mỗi người `git config user.name/user.email` bằng tên của chính mình.
- Không commit thẳng lên `main`: mỗi bước một nhánh, một Pull Request; nhóm trưởng merge.
- Không dùng `git add .`; chỉ thêm đúng file của bước. Không đưa `__pycache__/`, `.venv/`, `venv/` lên.
- Giữ dòng `# AI-assisted: …` ở đầu mỗi file.

Lệnh mẫu (bước 2):

```bash
git switch main
git pull
git switch -c c2-he-luat
git add src/advisor/rules.py
git commit -m "C2: biểu diễn luật và 33 luật tư vấn chọn ngành"
git push -u origin c2-he-luat
```
