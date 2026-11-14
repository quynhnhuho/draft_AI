# Hướng dẫn thành viên commit thuật toán Chức năng 1

Mỗi thành viên commit **file thuật toán của mình** để giảng viên thấy rõ ai làm phần nào (lịch sử git
là minh chứng). Ba file độc lập, chỉ dùng chung `src/planner/model.py`.

| Thành viên | File | Lệnh chạy riêng |
|---|---|---|
| `<thành viên 1>` | `src/planner/bfs.py` | `python -m src.planner.bfs` |
| `<thành viên 2>` | `src/planner/a_sao.py` | `python -m src.planner.a_sao` |
| `<thành viên 3>` | `src/planner/nhanh_can.py` | `python -m src.planner.nhanh_can` |

## Thứ tự

1. **Nhóm trưởng** đưa phần khung lên repo nhóm trước: `data/hoc_phan.csv`, `src/planner/model.py`,
   `dieu_phoi.py`, `kich_ban.py`, `cli.py`, `tests/test_planner.py` (chưa có 3 file thuật toán – chatbot
   vẫn chạy, báo "mô-đun chưa cài đặt", test thuật toán tự skip).
2. **Từng thành viên** thêm file của mình theo các bước dưới, mỗi người một nhánh và một pull request.

## Các bước cho mỗi thành viên

Cấu hình tên **của chính mình** (một lần, để commit đứng tên đúng người):

```bash
git config user.name "Họ Tên"
```

```bash
git config user.email "email-github-cua-ban@example.com"
```

Lấy code mới nhất và tạo nhánh riêng (ví dụ cho BFS):

```bash
git switch main
```

```bash
git pull
```

```bash
git switch -c c1-bfs
```

Chép file được giao vào `src/planner/`, **điền tên vào dòng `Thành viên phụ trách`** ở đầu file, rồi kiểm tra:

```bash
python -m pytest tests/test_planner.py -q
```

```bash
python -m src.planner.kich_ban
```

Commit và đẩy lên:

```bash
git add src/planner/bfs.py
```

```bash
git commit -m "C1: cài đặt thuật toán BFS"
```

```bash
git push -u origin c1-bfs
```

Sau đó tạo pull request trên GitHub để nhóm trưởng gộp vào `main`.

## Lưu ý

- **Không tự sửa `model.py`** trong nhánh thuật toán: cả 3 thuật toán dùng chung. Cần sửa thì báo nhóm và
  sửa trong một commit riêng.
- Giữ dòng *"Ghi chú hỗ trợ AI"* ở đầu file và khai báo trong báo cáo (yêu cầu của đề).
- Khi cả 3 file đã có: `python -m pytest -q` phải qua hết, trang **📊 So sánh thuật toán** trong app hiện đủ
  5 dòng (BFS, A\*, A\* (h2), Nhánh cận, Nhánh cận (h2)) với cùng số học kỳ.
