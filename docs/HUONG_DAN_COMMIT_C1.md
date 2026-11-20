# Hướng dẫn commit Chức năng 1 lên repo nhóm

Đưa Chức năng 1 lên repo nhóm theo đúng thứ tự phụ thuộc; **mỗi thành viên tự commit thuật toán
của mình** (lịch sử git là minh chứng ai làm phần nào).

## 1. Các file và người upload

| Bước | Người | File | Đặt ở |
|---|---|---|---|
| 1 | Nhóm trưởng | `__init__.py` (rỗng) – tạo thư mục `planner`; `.gitignore` nếu chưa có | `src/planner/`, gốc repo |
| 2 | Nhóm trưởng | `hoc_phan.csv`, `model.py` | `data/`, `src/planner/` |
| 3 | Thành viên 1, 2, 3 | `bfs.py`, `a_sao.py`, `nhanh_can.py` (mỗi người một file) | `src/planner/` |
| 4 | Nhóm trưởng | `dieu_phoi.py` – kết nối 3 thuật toán | `src/planner/` |
| 5 | Nhóm trưởng *(khi repo đã có chatbot)* | `src/chatbot/engine.py`, `flows.py`, `entities.py`, `app.py` | như cũ |

Phụ thuộc: `hoc_phan.csv` ← `model.py` ← 3 thuật toán ← `dieu_phoi.py` ← chatbot.
Bước 3: ba bạn làm **song song, thứ tự nào cũng được** – các file không phụ thuộc nhau.

## 2. Nguyên tắc

- Mỗi người `git config user.name/user.email` bằng tên **của chính mình** (hoặc đăng nhập GitHub đúng
  tài khoản khi upload trên web).
- **Không commit thẳng lên `main`**: mỗi bước một nhánh, một Pull Request; nhóm trưởng merge.
- **Không dùng `git add .`** – chỉ thêm đúng file của bước đó; không đưa `__pycache__/`, `.venv/` lên.
- Trước khi upload thuật toán: điền tên vào dòng `Thành viên phụ trách` ở đầu file, giữ dòng
  *"Ghi chú hỗ trợ AI"*. Không sửa `model.py` trong nhánh thuật toán.

## 3. Lệnh mẫu (ví dụ thành viên 1 – BFS)

```bash
git switch main
git pull
git switch -c c1-bfs
git add src/planner/bfs.py
git commit -m "C1: cài đặt thuật toán BFS"
git push -u origin c1-bfs
```

Rồi tạo Pull Request trên GitHub. Các bước khác làm tương tự với file và tên nhánh tương ứng
(`c1-planner`, `c1-mo-hinh`, `c1-a-sao`, `c1-nhanh-can`, `c1-dieu-phoi`, `c1-chatbot`).

## 4. Kiểm tra nhanh (ở thư mục gốc repo, `-B` để không sinh `__pycache__`)

Sau bước 2:

```bash
python -B -c "from src.planner.model import doc_du_lieu, hk1_co_dinh; ds = doc_du_lieu(); print(len(ds), 'HP; HK1:', len(hk1_co_dinh(ds)), 'HP')"
```

→ `28 HP; HK1: 7 HP`

Sau khi upload thuật toán của mình (đổi `bfs` thành `a_sao` hoặc `nhanh_can`):

```bash
python -B -c "from src.planner.model import doc_du_lieu, tao_bai_toan; from src.planner.bfs import bfs; print(bfs(tao_bai_toan(doc_du_lieu(), set(), set(), 1)).so_ky, 'kỳ')"
```

→ `5 kỳ` (sinh viên năm nhất)

Sau bước 4:

```bash
python -B -c "from src.planner.model import doc_du_lieu, tao_bai_toan; from src.planner.dieu_phoi import lap_lo_trinh; kq = lap_lo_trinh(tao_bai_toan(doc_du_lieu(), set(), set(), 1), doi_chieu=True); print(kq.ket_qua.thuat_toan, kq.ket_qua.so_ky, 'kỳ; khớp BFS:', kq.khop_doi_chieu)"
```

→ `A* 5 kỳ; khớp BFS: True`

Sau bước 5: `streamlit run app.py` → hỏi "em là tân sinh viên, lộ trình học thế nào" → trả lời "không"
→ chatbot trả về lộ trình 5 học kỳ kèm nguồn.
