# Chức năng 2 – Tư vấn chọn ngành bằng hệ luật

> Phần 3 – mục 3.3 (hệ luật), 3.4 (suy diễn tiến / lùi); Phần 4 – mục 4.1 (ID3, kiểm chứng scikit-learn).
> Tập luật v2 – 33 luật, năm áp dụng **2026**. Ngành: CNTT (7480201), SPTH (7140210), CNGD (7140103).
> Mô tả đầy đủ từng file, thứ tự commit và ảnh hưởng: `docs/C2_mo_ta_cai_dat.docx`.

## 1. Mã nguồn

| File | Nội dung |
|---|---|
| `data/tuyen_sinh_2026.json` | Tham số 2026: tổ hợp, mức ưu tiên, hệ số ĐGNL, ngưỡng, điểm chuẩn, nguồn |
| `src/advisor/tham_so.py` | Đọc tham số (Decimal); `nguong()`, `diem_chuan()`, đổi mã ngành |
| `src/advisor/diem.py` | Tổng điểm THPT/ĐGNL, điểm ưu tiên có ngưỡng 22.5, DXT làm tròn 2 chữ số |
| `src/advisor/facts.py` | Kiểm tra đầu vào, dữ kiện → sự kiện, phân biệt chưa hỏi / không thi, `DuLieu` |
| `src/advisor/rules.py` | Biểu diễn luật (biến `?T`, `Not`, điều kiện số), 33 luật, phân tầng |
| `src/advisor/inference.py` | Tự cài đặt `forward_chaining()` và `backward_chaining()` |
| `src/advisor/tu_van.py` | Điểm vào cho chatbot: `tu_van()`, `kiem_tra_dieu_kien()`, xếp hạng, trả lời |
| `data/survey.csv` | 15 mẫu khảo sát, 4 thuộc tính A1–A4, nhãn Y |
| `src/advisor/kiem_chung_sklearn.py` | IG hai tầng đầu (đối chiếu tính tay), cây scikit-learn, rút luật, lưu hình |

```bash
python -m src.advisor.kiem_chung_sklearn
python -m pytest -q
```

## 2. Biểu diễn tri thức

- Sự kiện là tuple: `("du_dk", "CNTT")` ≡ `du_dk(CNTT)`; tham số bắt đầu bằng `?` là biến.
- Điều kiện của luật: atom dương, `Not(atom)` (phủ định theo kiểu không chứng minh được; biến chưa gán là ¬∃),
  `Test` (điều kiện số gọi hàm tính).
- Khóa vắng trong dữ liệu người dùng = chưa hỏi; giá trị `None` = không thi.
- R11–R16 viết "tohop_xet(?T) ∧ DXT(?T) ≥ ngưỡng" (tương đương DXT_best ≥ ngưỡng); `kha_nang` tách thành
  `kha_nang_cao` / `kha_nang_tb` để hệ luật phân tầng được.

## 3. Suy diễn tiến

Phân tầng vị từ (tầng kết luận > tầng vị từ bị phủ định), chạy từng tầng đến điểm bất động, ghi vết luật đã
kích hoạt. Gợi ý xếp theo `[nhom_nganh] + [khop_manh] + 2·[nghe_phu_hop] + [kha_nang_cao]`; không có gợi ý
thì báo R32 hoặc R33 kèm lý do.

## 4. Suy diễn lùi

Mục tiêu `du_dk(X)`. Tìm theo chiều sâu kiểu Prolog: hợp nhất kết luận luật với mục tiêu, chứng minh lần lượt
điều kiện, quay lui khi cần, chống lặp bằng tập tổ tiên. Kết quả: Có (chuỗi luật) / Không (điều kiện thất bại,
số điểm còn thiếu) / Chưa đủ thông tin (dữ kiện cần hỏi – chatbot dùng để chọn câu hỏi tiếp theo).

## 5. Việc còn lại

- Thay nguồn trong `data/tuyen_sinh_2026.json` bằng Đề án tuyển sinh 2026 chính thức, đặt `da_kiem_chung: true`.
- Điểm chuẩn và tổ hợp 2024–2026 trong `data/knowledge_base.json` đã khớp với `data/tuyen_sinh_2026.json`
  (có test kiểm tra); khi đổi nguồn chính thức cần sửa cả hai file.
