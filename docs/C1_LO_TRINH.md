# Chức năng 1 – Lập lộ trình học tối ưu

> Phần 2 – mục 2.1 (mô hình hóa), 2.2 (BFS, A*, quay lui – nhánh cận). Tài liệu mô tả các file,
> giả định, cách 3 thuật toán phối hợp và cách chatbot gọi Chức năng 1.

## 1. Các file

| File | Nội dung | Người phụ trách |
|---|---|---|
| `data/hoc_phan.csv` | 28 học phần (75 TC): mã, tên, tín chỉ, học phần học trước, kỳ mở | cả nhóm |
| `src/planner/model.py` | Mô hình **dùng chung**: `HocPhan`, `BaiToan` (trạng thái, toán tử, heuristic h), `KetQua`, kiểm tra dữ liệu nhập | cả nhóm |
| `src/planner/bfs.py` | Thuật toán **BFS** | `<thành viên 1>` |
| `src/planner/a_sao.py` | Thuật toán **A\*** | `<thành viên 2>` |
| `src/planner/nhanh_can.py` | **Quay lui – nhánh cận** + lời giải tham lam | `<thành viên 3>` |
| `src/planner/dieu_phoi.py` | **Kết nối 3 thuật toán** – điểm vào duy nhất chatbot gọi | cả nhóm |

Kết nối với chatbot (file của Chức năng 4, phần dành cho Chức năng 1):

| File | Phần của Chức năng 1 |
|---|---|
| `src/chatbot/engine.py` | `make_plan()`: kiểm tra dữ liệu nhập, gọi `dieu_phoi.lap_lo_trinh()`, in bảng lộ trình + nguồn + giả định |
| `src/chatbot/flows.py` | Luồng hỏi `lo_trinh`: học kỳ sắp đăng ký → học phần đã đạt → bị rớt → có học vượt |
| `src/chatbot/entities.py` | Nhận học phần theo mã hoặc tên, hiểu "hk1" |
| `app.py` | Form "🗓️ Lộ trình học (Chức năng 1)" ở thanh bên |

Ba file thuật toán **chỉ import từ `model.py`**, không phụ thuộc nhau, nên mỗi thành viên commit file
của mình độc lập. File nào chưa có, bộ điều phối tự bỏ qua; chưa có file nào thì chatbot báo
"mô-đun chưa cài đặt".

## 2. Dữ liệu và giả định

**Nguồn:** Danh mục học phần trích từ chương trình đào tạo ban hành kèm theo Quyết định số
2619/QĐ-ĐHSP ngày 04/09/2024 của Hiệu trưởng Trường Đại học Sư phạm Thành phố Hồ Chí Minh.

Giả định nhóm thống nhất (`model.GIA_DINH`, chatbot hiển thị kèm mỗi lộ trình):

1. **Không có học kỳ hè**: mỗi năm 2 học kỳ; học kỳ lẻ là HK1, học kỳ chẵn là HK2 của năm học.
2. **HK1 do Trường xếp cố định**: 7 học phần đầu tiên trong `hoc_phan.csv` (18 TC).
3. **Học trước = chỉ cần đã học** (đạt hoặc rớt đều được). Học phần rớt vẫn phải học lại.
4. **Tối đa 18 TC/kỳ; 22 TC/kỳ khi học vượt.**
5. Mọi học phần đăng ký trong lộ trình được giả định là đạt.
6. Kỳ mở (Lẻ/Chẵn) lấy theo cột `hoc_ky_mo` (ghi rõ trong báo cáo lấy từ CTĐT hay mô phỏng).

## 3. Mô hình hóa (mục 2.1)

- **Trạng thái** `(đã_đạt, học_kỳ)`; "đã học" = `đã_đạt ∪ rớt`.
- **Đích**: đạt toàn bộ nhóm học phần mục tiêu (mặc định cả 28 học phần).
- **Toán tử**: chọn tập học phần trong kỳ: chưa đạt, mở trong kỳ, đã học mọi học phần học trước,
  `Σ TC ≤ MAX_TC`. Kỳ do Trường xếp chỉ có **một** toán tử. Chi phí = 1 kỳ.
- **Chỉ sinh tập tối đại** (không thêm được môn nào nữa): đạt nhiều môn hơn không bao giờ làm lộ trình
  dài ra ⇒ không mất lời giải tối ưu, giảm mạnh số nút.
- **Gộp trạng thái** theo `(đã_đạt, học_kỳ mod 2)`: cùng tập đã đạt, cùng loại kỳ thì tương lai như nhau.
- **Heuristic** `h(n) = ⌈TC còn phải học / MAX_TC⌉`: *chấp nhận được* (mỗi kỳ học tối đa MAX_TC tín
  chỉ) và *nhất quán* (mỗi bước h giảm tối đa 1).

## 4. Ba thuật toán và cách kết nối

| Thuật toán | Ý tưởng | Vai trò trong chatbot |
|---|---|---|
| **BFS** | Xét lần lượt từng tầng (mỗi tầng = 1 kỳ); tầng đầu tiên chạm đích là ít kỳ nhất | **Đối chiếu** (chế độ debug) – chắc chắn đúng nhưng tốn nút/bộ nhớ |
| **A\*** | Ưu tiên nút có `f = g + h` nhỏ nhất; h chấp nhận được nên lời giải tối ưu | **Thuật toán chính** |
| **Nhánh cận** | Có sẵn lời giải tham lam (mỗi kỳ nhiều TC nhất) làm cận trên; cắt nhánh có `g + h ≥ tốt nhất` | **Dự phòng** khi A\* chậm – dừng lúc nào cũng có lời giải |

**Người dùng không chọn thuật toán.** `dieu_phoi.lap_lo_trinh()` tự quyết:

```
A* (60% thời gian cho phép, mặc định tổng 3 giây)
  ├─ xong      → trả lời: lộ trình ít học kỳ nhất
  └─ hết giờ   → Nhánh cận (thời gian còn lại)
                  ├─ xong    → lộ trình tối ưu
                  └─ hết giờ → lộ trình tốt nhất đã tìm + cảnh báo "có thể chưa tối ưu"
Chế độ debug: chạy thêm BFS → "Đối chiếu BFS: … ✔ khớp"
```

## 5. Chatbot gọi Chức năng 1

```
"Em rớt 2 môn thì nên đăng ký học phần nào?"
  → ý định lo_trinh_hoc_tap  → flows.py hỏi: học kỳ → đã đạt → rớt → học vượt
  → engine.make_plan()       → kiểm tra dữ liệu nhập → dieu_phoi.lap_lo_trinh()
  → bảng lộ trình theo học kỳ (môn học lại, kỳ Trường xếp) + nguồn QĐ 2619 + giả định
```

Cũng có thể lập lộ trình bằng form "Lộ trình học" ở thanh bên của app (`streamlit run app.py`).
