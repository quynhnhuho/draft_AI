# Hướng dẫn commit Chức năng 1 lên repo nhóm (repo hiện chỉ có `src/`)

Mục tiêu: đưa Chức năng 1 lên repo nhóm **theo đúng thứ tự phụ thuộc**, để sau **mỗi** commit repo vẫn
chạy được, và **mỗi thành viên tự commit thuật toán của mình** (lịch sử git là minh chứng cho giảng viên).

## 1. Phân công

| Người | Phụ trách | File |
|---|---|---|
| Nhóm trưởng | Phần khung dùng chung, test, tài liệu | `data/hoc_phan.csv`, `src/planner/model.py`, `dieu_phoi.py`, `kich_ban.py`, `cli.py`, `tests/test_planner.py`, … |
| `<thành viên 1>` | BFS | `src/planner/bfs.py` |
| `<thành viên 2>` | A* | `src/planner/a_sao.py` |
| `<thành viên 3>` | Quay lui – nhánh cận | `src/planner/nhanh_can.py` |

## 2. Thứ tự commit (tổng quan)

Phụ thuộc giữa các file:
`hoc_phan.csv` ← `model.py` ← `dieu_phoi.py` ← `kich_ban.py`, `cli.py`, `tests/` ;
3 file thuật toán **chỉ** cần `model.py` và **không** phụ thuộc nhau.

| Bước | Ai | Commit những file | Kiểm tra sau commit | Kết quả mong đợi |
|---|---|---|---|---|
| 1 | Nhóm trưởng | `.gitignore`, `data/hoc_phan.csv`, `src/__init__.py`, `src/planner/__init__.py`, `src/planner/model.py` | lệnh kiểm tra ở mục 4 | `28 HP; HK1: 7 HP` |
| 2 | Nhóm trưởng | `src/planner/dieu_phoi.py`, `kich_ban.py`, `cli.py` | `python -m src.planner.kich_ban` | in "(chưa có thuật toán nào…)" – **đúng như mong đợi** |
| 3 | Nhóm trưởng | `pytest.ini`, `tests/test_planner.py`, `requirements.txt` | `python -m pytest tests/test_planner.py -q` | `7 passed, 11 skipped` |
| 4 | Thành viên 1 | `src/planner/bfs.py` | như bước 3 | `11 passed, 7 skipped` |
| 5 | Thành viên 2 | `src/planner/a_sao.py` | như bước 3 | `16 passed, 2 skipped` |
| 6 | Thành viên 3 | `src/planner/nhanh_can.py` | như bước 3 | `18 passed` |
| 7 | Nhóm trưởng | `docs/C1_LO_TRINH.md`, `docs/HUONG_DAN_COMMIT_C1.md` | – | – |
| 8 | Nhóm trưởng *(chỉ khi repo đã có chatbot)* | xem mục 6 | `python -m pytest -q`, `streamlit run app.py` | không có test `failed` |

- Bước 4, 5, 6 làm **theo thứ tự nào cũng được**, có thể làm song song (mỗi người một nhánh). Số
  `passed/skipped` ở bảng là cho thứ tự BFS → A* → nhánh cận; thứ tự khác sẽ ra số khác nhưng luôn
  **0 failed**. "skipped" = test của thuật toán chưa commit, tự bỏ qua – không phải lỗi.
- Bảng trên đã được chạy thử trên một repo mới chỉ có `src/`.

## 3. Chuẩn bị (mỗi người làm một lần)

Mở terminal **ở thư mục gốc của repo** (thư mục chứa `src/`), đặt tên **của chính mình** để commit đứng
tên đúng người:

```bash
git config user.name "Họ Tên"
git config user.email "email-github-cua-ban@example.com"
```

Nguyên tắc:
- **Không commit thẳng lên `main`.** Mỗi bước làm trên một nhánh riêng rồi mở Pull Request (PR).
- **Không dùng `git add .`** – chỉ `git add` đúng các file của bước đó (tránh đưa nhầm `__pycache__/`,
  `.venv/`, file của người khác).
- Lấy file từ gói Chức năng 1 (repo `draft_AI`, nhánh `claude/youthful-franklin-gc94bi`) và chép vào
  **đúng đường dẫn** như trong bảng.

## 4. Các bước chi tiết

### Bước 1 – Mô hình và dữ liệu (nhóm trưởng)

Nếu repo chưa có `src/__init__.py` và `src/planner/__init__.py` thì tạo file **rỗng** với đúng tên đó
(thiếu là bị lỗi `ModuleNotFoundError: No module named 'src'`). `.gitignore` chặn `__pycache__/`,
`logs/`, `models/`, `.venv/`.

```bash
git switch main
git pull
git switch -c c1-mo-hinh
git add .gitignore data/hoc_phan.csv src/__init__.py src/planner/__init__.py src/planner/model.py
git commit -m "C1: dữ liệu học phần và mô hình bài toán dùng chung"
git push -u origin c1-mo-hinh
```

Kiểm tra trước khi push:

```bash
python -c "from src.planner.model import doc_du_lieu, hk1_co_dinh; ds = doc_du_lieu(); print(len(ds), 'HP; HK1:', len(hk1_co_dinh(ds)), 'HP')"
```

Mở PR `c1-mo-hinh` → `main`, gộp (merge) xong mới sang bước 2.

> Nếu `src/planner/` còn file cũ của bản khung đầu (`search.py`, `scenarios.py`) thì xóa trong commit này:
> `git rm src/planner/search.py src/planner/scenarios.py`.

### Bước 2 – Bộ điều phối, kịch bản, dòng lệnh (nhóm trưởng)

```bash
git switch main
git pull
git switch -c c1-dieu-phoi
git add src/planner/dieu_phoi.py src/planner/kich_ban.py src/planner/cli.py
git commit -m "C1: bộ điều phối kết hợp 3 thuật toán, kịch bản so sánh, chạy dòng lệnh"
git push -u origin c1-dieu-phoi
```

Kiểm tra: `python -m src.planner.kich_ban` in "(chưa có thuật toán nào trong src/planner/)" – đúng, vì
chưa ai commit thuật toán. Mở PR và merge.

### Bước 3 – Test (nhóm trưởng)

`pytest.ini` giúp pytest tìm thấy gói `src`. Nếu repo chưa có `requirements.txt` thì thêm (ít nhất
có dòng `pytest`).

```bash
git switch main
git pull
git switch -c c1-test
git add pytest.ini tests/test_planner.py requirements.txt
git commit -m "C1: test tự động cho mô hình và thuật toán"
git push -u origin c1-test
```

Kiểm tra: `python -m pytest tests/test_planner.py -q` → `7 passed, 11 skipped`. Mở PR và merge.

### Bước 4, 5, 6 – Mỗi thành viên commit thuật toán của mình

Ví dụ cho BFS (A* thay `bfs` bằng `a_sao`, nhánh cận thay bằng `nhanh_can`):

1. Lấy bản `main` mới nhất (đã có bước 1–3) và tạo nhánh riêng:

   ```bash
   git switch main
   git pull
   git switch -c c1-bfs
   ```

2. Chép file `bfs.py` vào `src/planner/`, mở file và **điền tên vào dòng `Thành viên phụ trách`**.
   Giữ nguyên dòng *"Ghi chú hỗ trợ AI"*.
3. Kiểm tra:

   ```bash
   python -m pytest tests/test_planner.py -q
   python -m src.planner.bfs
   ```

4. Commit **chỉ file của mình** và đẩy lên:

   ```bash
   git add src/planner/bfs.py
   git commit -m "C1: cài đặt thuật toán BFS"
   git push -u origin c1-bfs
   ```

5. Mở PR `c1-bfs` → `main`, nhờ nhóm trưởng review và merge.

Không sửa `model.py` trong nhánh thuật toán – cả 3 thuật toán dùng chung. Cần sửa thì báo nhóm, làm
một nhánh/PR riêng.

### Bước 7 – Tài liệu (nhóm trưởng)

```bash
git switch main
git pull
git switch -c c1-tai-lieu
git add docs/C1_LO_TRINH.md docs/HUONG_DAN_COMMIT_C1.md
git commit -m "C1: tài liệu thiết kế, kết quả và hướng dẫn commit"
git push -u origin c1-tai-lieu
```

Sau bước 7 kiểm tra tổng: `python -m src.planner.kich_ban` in đủ 5 dòng (BFS, A\*, A\* (h2), Nhánh cận,
Nhánh cận (h2)) cho mỗi tình huống với **cùng số học kỳ**.

## 5. Không dùng dòng lệnh? Làm trên web GitHub

1. Vào repo → chọn thư mục đích (ví dụ `src/planner`) → **Add file → Upload files** (hoặc
   **Create new file** và gõ cả đường dẫn, ví dụ `data/hoc_phan.csv`, GitHub tự tạo thư mục).
2. Ở phần commit, ghi message như bảng trên và **chọn "Create a new branch for this commit and start a
   pull request"** – **không** chọn "Commit directly to the main branch".
3. Tạo PR và merge. Commit sẽ đứng tên tài khoản GitHub đang đăng nhập.

## 6. Bước 8 – Nếu repo đã có chatbot (Chức năng 4)

Chỉ làm khi repo nhóm đã có `src/chatbot/` và `app.py` từ bản khung trước. Các file này được cập nhật
để chatbot gọi Chức năng 1 (không cho người dùng chọn thuật toán, thêm trang So sánh thuật toán):

```bash
git switch main
git pull
git switch -c c1-tich-hop-chatbot
git add src/chatbot/engine.py src/chatbot/entities.py src/chatbot/flows.py app.py tests/test_engine.py data/knowledge_base.json requirements.txt
git commit -m "C4: tích hợp lập lộ trình học (C1) vào chatbot"
git push -u origin c1-tich-hop-chatbot
```

Kiểm tra: `python -m pytest -q` không có `failed`; `streamlit run app.py` → hỏi "em là tân sinh viên,
lộ trình học thế nào" → chatbot hỏi học vượt → trả lời bảng lộ trình 5 học kỳ kèm nguồn.

## 7. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy lệnh không ở thư mục gốc repo, hoặc thiếu `src/__init__.py` / `src/planner/__init__.py` / `pytest.ini` | `cd` về thư mục gốc; tạo file `__init__.py` rỗng |
| `FileNotFoundError: ... data/hoc_phan.csv` | Chưa commit/chép `data/hoc_phan.csv` hoặc sai vị trí | File phải nằm ở `data/hoc_phan.csv` ngay dưới thư mục gốc |
| `git status` hiện `__pycache__/` | Thiếu `.gitignore` | Commit `.gitignore`; nếu đã lỡ commit: `git rm -r --cached src/**/__pycache__` |
| Lỡ commit thẳng lên `main` | Quên tạo nhánh | Báo nhóm trưởng; hoàn tác bằng `git revert <mã commit>` trên `main` rồi `git cherry-pick <mã commit>` sang nhánh đúng – không force-push `main` |
| Test báo `skipped` | Thuật toán tương ứng chưa được commit | Bình thường; hết `skipped` khi đủ 3 file |
| Commit đứng tên người khác | Chưa `git config user.name/user.email` trên máy mình | Làm mục 3 trước khi commit |
