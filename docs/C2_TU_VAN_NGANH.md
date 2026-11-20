# Chức năng 2 – Tư vấn chọn ngành bằng hệ luật

> Phần 3 – mục 3.3 (hệ luật), 3.4 (suy diễn tiến / lùi). Phần 4 – mục 4.1 (ID3) **chưa làm** (`src/advisor/id3.py`).
> Tập luật: "Tập luật C2 (v2)" – 33 luật, năm áp dụng **2026**. Ngành: CNTT (7480201), SPTH (7140210), CNGD (7140103).

## 1. Cấu trúc mã nguồn

| File | Nội dung |
|---|---|
| `data/tuyen_sinh_2026.json` | Bảng tham số 2026: tổ hợp, mức ưu tiên, hệ số ĐGNL, ngưỡng, điểm chuẩn, chỉ tiêu, **nguồn** |
| `src/advisor/tham_so.py` | Đọc file JSON trên (Decimal), hàm `nguong()`, `diem_chuan()`, đổi mã ngành KB ↔ luật |
| `src/advisor/diem.py` | Công thức: tổng điểm THPT/ĐGNL, **điểm ưu tiên có ngưỡng 22.5**, DXT làm tròn 2 chữ số |
| `src/advisor/facts.py` | Kiểm tra đầu vào, `build_facts()` (dữ kiện → sự kiện), `DuLieu` (ngữ cảnh số), dữ kiện cần hỏi |
| `src/advisor/rules.py` | Biểu diễn luật (atom, biến `?T`, `Not`, điều kiện số `Test`), **33 luật**, kiểm tra an toàn, **phân tầng** |
| `src/advisor/inference.py` | **Tự cài đặt** `forward_chaining()` và `backward_chaining()` |
| `src/advisor/tu_van.py` | Lớp dịch vụ cho chatbot: `tu_van()`, `kiem_tra_dieu_kien()`, xếp hạng, R32/R33, định dạng trả lời |
| `src/advisor/kich_ban.py` | 8 kịch bản minh họa cho báo cáo |
| `tests/test_advisor.py` | 20 test, gồm test **suy diễn tiến ⇔ suy diễn lùi** trên 300 hồ sơ ngẫu nhiên |

```bash
python -m src.advisor.kich_ban          # 8 kịch bản: kết quả suy diễn tiến + lùi
python -m src.advisor.kich_ban --vet    # kèm nhật ký suy diễn lùi
python -m src.advisor.kich_ban --luat   # in 33 luật và tầng
python -m pytest tests/test_advisor.py -q
```

## 2. Biểu diễn tri thức

- **Sự kiện** là tuple: `("du_dk", "CNTT")` ≡ `du_dk(CNTT)`; `("thich_lap_trinh",)`.
- **Luật** `Rule(id, nhom, dieu_kien, ket_luan, giai_thich, nguon)`; điều kiện gồm 3 loại:
  - atom dương, có thể chứa biến: `tohop_xet(?T)`;
  - `Not(atom)` – phủ định "không chứng minh được"; biến chưa gán trong `Not` nghĩa là ¬∃ (R32, R33);
  - `Test` – điều kiện số gọi hàm tính (`DiemLiet(?T)`, `DatMuc(?T, X, "nguong"|"chuan")`).
- **Dữ kiện chưa biết ≠ dữ kiện không có**: khóa vắng trong `user` = chưa hỏi; giá trị `None` = không thi.

Hai chỗ viết lại tương đương so với tập luật trên giấy:

1. R11–R16 "∃T: tohop_xet(T) ∧ DXT_best ≥ ngưỡng" ⇔ "tohop_xet(?T) ∧ DXT(?T) ≥ ngưỡng" (max ≥ a ⇔ ∃ phần tử ≥ a).
2. `kha_nang(X)=Cao/Trung bình` → hai vị từ `kha_nang_cao(X)`, `kha_nang_tb(X)`; R18 = `du_dk(X) ∧ ¬kha_nang_cao(X)`
   (tương đương DXT_best < điểm chuẩn). Nhờ vậy không vị từ nào phủ định chính nó ⇒ hệ luật phân tầng được.

## 3. Suy diễn tiến – "danh sách ngành gợi ý"

1. `phan_tang()` tính tầng của mỗi vị từ: tầng(kết luận) ≥ tầng(điều kiện dương), > tầng(điều kiện phủ định).
   Kết quả: tầng 0 (tổ hợp, điểm liệt, sở thích, nghề) → tầng 1 (tohop_xet, du_dk, kha_nang_cao, goi_y) →
   tầng 2 (kha_nang_tb, R32, R33).
2. Chạy từng tầng đến điểm bất động: duyệt luật, tìm **mọi phép gán** thỏa điều kiện (`_khop`, khớp mẫu
   trái → phải), kết luận mới → thêm vào bộ nhớ làm việc + ghi vết `LanKichHoat`.
3. `tu_van.xep_hang()` sắp các `goi_y(X, mức)` theo
   `diem_phu_hop = [nhom_nganh] + [khop_manh] + 2·[nghe_phu_hop] + [kha_nang_cao]`, hòa thì ưu tiên khả năng Cao,
   rồi DXT_best − điểm chuẩn.

## 4. Suy diễn lùi – "Em có đủ điều kiện xét tuyển ngành X không?"

Mục tiêu `du_dk(X)`. Tìm kiếm theo chiều sâu kiểu Prolog (generator, có quay lui):

- mục tiêu có trong dữ kiện → đúng;
- thử từng luật có kết luận **hợp nhất** được với mục tiêu (đổi tên biến trước), chứng minh đệ quy từng điều kiện;
- `¬p`: thử chứng minh p – được thì sai, không được (và không thiếu dữ kiện) thì đúng;
- điều kiện số: thiếu dữ kiện (vd. khu vực) → ghi vào `missing`; sai → ghi lý do + **số điểm còn thiếu**;
- sự kiện lá không có trong dữ kiện và **chưa hỏi** → `missing` (chatbot hỏi tiếp);
- chống lặp vô hạn bằng tập **tổ tiên** truyền theo tham số.

Kết quả: **Có** (kèm chuỗi luật), **Không** (kèm điều kiện thất bại, số điểm thiếu), **Chưa đủ thông tin** (kèm dữ kiện cần hỏi).

## 5. Tích hợp chatbot (C4)

- Luồng `kiem_tra_dieu_kien`: **câu hỏi tiếp theo do suy diễn lùi quyết định** – sau mỗi câu trả lời, chạy lại
  `backward_chaining`, lấy dữ kiện còn thiếu đầu tiên → hỏi (phương thức → điểm các môn → khu vực).
- Luồng `tu_van_nganh`: hỏi đủ phương thức, điểm, khu vực, sở thích, định hướng → `forward_chaining`.
- Câu kích hoạt được điền sẵn: "Em thi THPT Toán 8, Anh 7, Lý 7, KV2 có đủ điều kiện vào CNTT không?" trả lời ngay.
- Bật "Chế độ debug" ở sidebar để xem vết suy diễn tiến / nhật ký suy diễn lùi đầy đủ.

## 6. Việc còn lại

- Thay link nguồn trong `data/tuyen_sinh_2026.json` bằng Đề án tuyển sinh 2026 chính thức, đặt `da_kiem_chung: true`;
  kiểm tra lại ngưỡng CNTT (đang trùng điểm chuẩn).
- `data/knowledge_base.json` vẫn là điểm chuẩn **MẪU** (trả lời câu hỏi "điểm chuẩn bao nhiêu") – chưa đồng bộ với số 2026.
- ID3 (mục 4.1) và luật rút từ cây: `src/advisor/id3.py`.
