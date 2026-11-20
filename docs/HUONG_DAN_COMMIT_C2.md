# Hướng dẫn commit Chức năng 2 theo thứ tự

Chuỗi 7 commit dưới đây đi từ `main` (commit `4864067`) đến đúng trạng thái của nhánh
`claude/festive-volta-pxxl0d`. Thứ tự theo phụ thuộc: commit sau dùng mã của commit trước.
Kết quả "sau commit" là kết quả **đã chạy thật** khi dựng lại chuỗi commit trên một bản sao của `main`.

## Tóm tắt

| # | Commit | Tệp | Dùng mã của | Sau commit |
|---|---|---|---|---|
| 1 | C2: tham số tuyển sinh 2026 và công thức tính điểm | `.gitignore`, `data/tuyen_sinh_2026.json`, `src/advisor/tham_so.py`, `src/advisor/diem.py` | – | 28/28 test qua, chatbot chạy như cũ |
| 2 | C2: hệ luật 33 luật | `src/advisor/rules.py` | 1 | 28/28 test qua, chatbot chạy như cũ |
| 3 | C2: dữ kiện đầu vào, suy diễn tiến và suy diễn lùi | `src/advisor/facts.py`, `src/advisor/inference.py` | 1, 2 | **Chatbot lỗi import** (chờ commit 5) |
| 4 | C2: lớp tư vấn, kịch bản minh họa và kiểm thử | `src/advisor/tu_van.py`, `src/advisor/kich_ban.py`, `tests/test_advisor.py` | 1, 2, 3 | 20/20 test C2 qua, kịch bản chạy; chatbot vẫn lỗi |
| 5 | C2: tích hợp vào chatbot | `src/chatbot/entities.py`, `src/chatbot/flows.py`, `src/chatbot/engine.py`, `src/chatbot/kb.py`, `data/knowledge_base.json`, `tests/test_engine.py` | 1–4 | **51/51 test qua**, chatbot chạy lại |
| 6 | C4: giao diện tư vấn ngành mới, bỏ biểu tượng | `app.py` | 1, 5 | 51/51, form C2 trên giao diện dùng được |
| 7 | C2: tài liệu | `docs/C2_TU_VAN_NGANH.md`, `docs/C2_mo_ta_cai_dat.docx`, `docs/HUONG_DAN_DO_AN.md`, `docs/HUONG_DAN_COMMIT_C2.md`, `README.md` | – | Không đổi hành vi |

**Quan trọng:** commit 3 thay `facts.py` mới, bỏ hàm `latest_cutoffs` mà `engine.py` cũ còn import, nên từ
commit 3 đến commit 4 chatbot và `tests/test_engine.py` lỗi; commit 5 sửa lại. Vì vậy **đẩy 7 commit cùng
lúc (một pull request)**, không gộp riêng commit 3 hoặc 4 vào `main`.

## Chi tiết từng commit

### Commit 1 – nền tảng số liệu

```
C2: tham số tuyển sinh 2026 và công thức tính điểm

- data/tuyen_sinh_2026.json: tổ hợp, mức ưu tiên khu vực, hệ số ĐGNL, ngưỡng, điểm chuẩn, chỉ tiêu, nguồn
- tham_so.py: đọc tham số thành Decimal; nguong(), diem_chuan(), đổi mã ngành
- diem.py: tổng điểm THPT/ĐGNL, điểm ưu tiên có ngưỡng 22.5, DXT làm tròn 2 chữ số
- .gitignore: bỏ qua venv/, env/
```

- Ảnh hưởng: chỉ **thêm** tệp mới, chưa tệp nào cũ dùng tới. Mọi thứ chạy như `main`.
- Commit sau cần nó: `rules.py` lấy ngưỡng/điểm chuẩn; `facts.py` gọi các hàm tính điểm.

### Commit 2 – hệ luật

```
C2: hệ luật 33 luật

- Biểu diễn luật: atom có biến, phủ định Not, điều kiện số Test (DiemLiet, DatMuc)
- 33 luật chia nhóm a (R1-R18), b (R19-R28), c (R29-R33)
- Kiểm tra luật an toàn và phân tầng vị từ cho phủ định
```

- Ảnh hưởng: thay `rules.py` khung cũ. `inference.py` cũ chỉ import `RULES`, `Rule` (vẫn còn) nên
  chatbot vẫn chạy và vẫn báo "mô-đun chưa cài đặt" cho C2.
- Cần commit 1 (`tham_so`). Commit 3 cần nó (bộ suy diễn đọc `RULES`, `phan_tang`).

### Commit 3 – bộ suy diễn

```
C2: dữ kiện đầu vào, suy diễn tiến và suy diễn lùi

- facts.py: kiểm tra đầu vào, chuyển dữ kiện thành sự kiện, phân biệt chưa hỏi và không thi, DuLieu
- inference.py: suy diễn tiến theo tầng đến điểm bất động
- inference.py: suy diễn lùi kiểu Prolog (hợp nhất, quay lui, phủ định, chống lặp bằng tập tổ tiên),
  trả về Có / Không / Chưa đủ thông tin, điều kiện thất bại, số điểm còn thiếu, dữ kiện cần hỏi
```

- Ảnh hưởng: **chatbot cũ lỗi** `ImportError: cannot import name 'latest_cutoffs'`; `tests/test_engine.py`
  lỗi; giao diện không mở được. `tests/test_planner.py` (C1) vẫn qua 18/18.
- Cần commit 1, 2. Commit 4 cần nó.

### Commit 4 – lớp dịch vụ và kiểm thử

```
C2: lớp tư vấn, kịch bản minh họa và kiểm thử

- tu_van.py: tu_van() (suy diễn tiến, xếp hạng, R32/R33), kiem_tra_dieu_kien() (suy diễn lùi), định dạng trả lời
- kich_ban.py: 8 kịch bản cho báo cáo (python -m src.advisor.kich_ban)
- test_advisor.py: 20 test, gồm đối chiếu suy diễn tiến và lùi trên 300 hồ sơ ngẫu nhiên
```

- Ảnh hưởng: `tests/test_advisor.py` 20/20 qua, `python -m src.advisor.kich_ban` chạy được. Chatbot vẫn lỗi
  như commit 3.
- Cần commit 1–3. Commit 5 cần nó (chatbot gọi `tu_van`, `kiem_tra_dieu_kien`).

### Commit 5 – tích hợp chatbot

```
C2: tích hợp vào chatbot

- entities.py: trích phương thức, khu vực, điểm từng môn, sở thích và định hướng theo tên dữ kiện của luật
- flows.py: luồng kiểm tra điều kiện do suy diễn lùi chọn câu hỏi tiếp theo; luồng tư vấn ngành
- engine.py: gọi tu_van / kiem_tra_dieu_kien, trả lời kèm nguồn năm 2026
- knowledge_base.json: thêm ngành Công nghệ giáo dục
- engine.py, flows.py, entities.py, kb.py: bỏ chú thích, thay biểu tượng bằng chữ, thêm ghi chú hỗ trợ AI
- test_engine.py: test hội thoại kiểm tra điều kiện và tư vấn ngành
```

- Ảnh hưởng: **51/51 test qua**, chatbot chạy lại và trả lời được C2 trong khung chat.
  Riêng form C2 ở thanh bên (app.py cũ) còn gửi trường cũ (tổ hợp, tổng điểm) nên chỉ hiện
  "Thông tin chưa hợp lệ: Thiếu phương thức xét tuyển", không bị lỗi.
- Cần commit 1–4. Commit 6 cần nó (`TEN_SO_THICH`, `TEN_DINH_HUONG` trong `entities.py`).

### Commit 6 – giao diện

```
C4: giao diện tư vấn ngành mới, bỏ biểu tượng

- Form Chức năng 2: phương thức, khu vực, điểm từng môn hoặc ĐGNL + học bạ, sở thích, định hướng
- Bỏ biểu tượng ở tiêu đề, nút, bảng so sánh thuật toán
```

- Ảnh hưởng: form C2 trên thanh bên trả về gợi ý ngành và kết quả kiểm tra điều kiện. 51/51 test qua.
- Cần commit 1 (`tham_so`) và 5 (`entities`, `engine`).

### Commit 7 – tài liệu

```
C2: tài liệu

- C2_TU_VAN_NGANH.md, C2_mo_ta_cai_dat.docx: mô tả cài đặt, ví dụ, kết quả kịch bản
- HUONG_DAN_COMMIT_C2.md: thứ tự commit
- README.md, HUONG_DAN_DO_AN.md: cập nhật trạng thái Chức năng 2
```

- Ảnh hưởng: không đổi hành vi chương trình.

## Lệnh thực hiện

Từ `main` mới nhất, tạo nhánh riêng rồi lấy đúng tệp của từng commit từ nhánh đã làm sẵn:

```bash
git switch main
```

```bash
git pull
```

```bash
git switch -c c2-tu-van-nganh
```

```bash
git fetch origin claude/festive-volta-pxxl0d
```

Commit 1 (các commit sau làm giống hệt, chỉ đổi danh sách tệp và nội dung commit theo bảng trên):

```bash
git checkout origin/claude/festive-volta-pxxl0d -- .gitignore data/tuyen_sinh_2026.json src/advisor/tham_so.py src/advisor/diem.py
```

```bash
git commit -m "C2: tham số tuyển sinh 2026 và công thức tính điểm"
```

Kiểm tra sau mỗi commit:

```bash
python -m pytest -q
```

Sau commit 7, `git diff origin/claude/festive-volta-pxxl0d` phải rỗng. Đẩy cả 7 commit một lần rồi tạo một pull request:

```bash
git push -u origin c2-tu-van-nganh
```

Nếu nhiều thành viên cùng làm C2, mỗi người đặt `git config user.name` / `user.email` của mình trước khi
commit phần được giao (ví dụ: commit 1–2 người làm hệ luật, commit 3 người làm suy diễn, commit 4–5 người
làm tích hợp), giống cách làm ở Chức năng 1.
