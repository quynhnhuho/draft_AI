# Chức năng 1 – Lập lộ trình học tối ưu

> Phần 2 – mục 2.1 (mô hình hóa), 2.2 (BFS, A*, quay lui – nhánh cận). Tài liệu này mô tả thiết kế,
> giả định, cách chatbot dùng 3 thuật toán và kết quả chạy trên dữ liệu thật để đưa vào báo cáo.

## 1. Cấu trúc mã nguồn

| File | Nội dung | Người phụ trách |
|---|---|---|
| `data/hoc_phan.csv` | 28 học phần (75 TC): mã, tên, tín chỉ, học phần học trước, kỳ mở | cả nhóm |
| `src/planner/model.py` | Mô hình **dùng chung**: `HocPhan`, `BaiToan` (trạng thái, toán tử, h, h2), `KetQua`, kiểm tra đầu vào/lộ trình | cả nhóm |
| `src/planner/bfs.py` | Thuật toán **BFS** | `<thành viên 1>` |
| `src/planner/a_sao.py` | Thuật toán **A\*** | `<thành viên 2>` |
| `src/planner/nhanh_can.py` | **Quay lui – nhánh cận** + lời giải tham lam | `<thành viên 3>` |
| `src/planner/dieu_phoi.py` | **Bộ điều phối**: kết hợp 3 thuật toán, điểm vào duy nhất cho chatbot | cả nhóm |
| `src/planner/kich_ban.py` | 3 tình huống bắt buộc + bảng so sánh cho báo cáo | cả nhóm |
| `src/planner/cli.py` | Chạy ở dòng lệnh, nhập thông tin sinh viên từ bàn phím | cả nhóm |
| `tests/test_planner.py` | Test: ngữ nghĩa học trước, HK1 cố định, 3 thuật toán cùng số kỳ, heuristic chấp nhận được… | cả nhóm |

Ba file thuật toán **chỉ import từ `model.py`**, không phụ thuộc nhau, nên mỗi thành viên commit file
của mình độc lập (xem `docs/HUONG_DAN_COMMIT_C1.md`). File nào chưa có, bộ điều phối tự bỏ qua; nếu
chưa có file nào, chatbot báo "mô-đun chưa cài đặt" và test thuật toán tự skip.

```bash
python -m src.planner.cli                      # nhập thông tin, tự chọn thuật toán như chatbot
python -m src.planner.cli --thuat-toan tat_ca  # chạy cả 3 thuật toán
python -m src.planner.bfs                      # chạy riêng BFS (tương tự a_sao, nhanh_can)
python -m src.planner.kich_ban --md            # bảng so sánh 3 tình huống (Markdown)
python -m pytest tests/test_planner.py -q
```

## 2. Dữ liệu và giả định

**Nguồn:** Danh mục học phần trích từ chương trình đào tạo ban hành kèm theo Quyết định số
2619/QĐ-ĐHSP ngày 04/09/2024 của Hiệu trưởng Trường Đại học Sư phạm Thành phố Hồ Chí Minh.

Giả định nhóm thống nhất (khai báo trong `model.GIA_DINH`, chatbot hiển thị kèm mỗi lộ trình):

1. **Không có học kỳ hè**: mỗi năm 2 học kỳ; học kỳ lẻ là HK1, học kỳ chẵn là HK2 của năm học.
2. **HK1 do Trường xếp cố định**: 7 học phần đầu tiên trong `hoc_phan.csv` (18 TC). Sinh viên không
   tự chọn thêm/bớt. Học phần HK1 bị rớt thì sinh viên tự đăng ký học lại ở kỳ sau.
3. **Học trước = chỉ cần đã học** (đạt hoặc rớt đều được). Học phần rớt vẫn phải học lại để đạt.
4. **Tối đa 18 TC/kỳ; 22 TC/kỳ khi học vượt.**
5. Mọi học phần đăng ký trong lộ trình được giả định là đạt.
6. Kỳ mở (Lẻ/Chẵn) lấy theo cột `hoc_ky_mo`. ⚠️ Nhóm cần ghi rõ trong báo cáo cột này lấy từ CTĐT
   hay do nhóm suy ra (nếu suy ra thì là **dữ liệu mô phỏng**).

## 3. Mô hình hóa (mục 2.1)

- **Trạng thái** `(đã_đạt, học_kỳ)`. "Đã học" = `đã_đạt ∪ rớt` (tập rớt cố định của bài toán).
- **Trạng thái đầu** `(học phần đã đạt, học kỳ sắp đăng ký)`; **đích**: đạt toàn bộ nhóm học phần mục tiêu.
- **Toán tử**: chọn tập học phần `A` trong kỳ: chưa đạt, mở trong kỳ, mọi học phần học trước đã học,
  `Σ TC(A) ≤ MAX_TC`. Kỳ do Trường xếp chỉ có **một** toán tử là tập học phần đã xếp. Chi phí = 1 kỳ.
- **Học phần còn phải học** (`can_hoc`) = mục tiêu chưa đạt ∪ các học phần học trước *chưa từng học*
  (bao đóng) – cho phép mục tiêu là một nhóm học phần bất kỳ.
- **Chỉ sinh tập tối đại** (không thêm được môn nào nữa mà vẫn ≤ MAX_TC): trạng thái đạt nhiều hơn ở
  cùng học kỳ luôn *trội* (không có ràng buộc tối thiểu/đồng thời) ⇒ không mất lời giải tối ưu, giảm
  mạnh số nút. Khi liệt kê có cắt nhánh: nếu kể cả lấy hết môn còn lại mà môn nhỏ nhất đã bỏ vẫn vừa
  thì nhánh đó không thể tối đại.
- **Gộp trạng thái** theo khóa `(đã_đạt, học_kỳ mod 2)`: cùng tập đã đạt và cùng tính chẵn/lẻ thì
  tương lai như nhau (kỳ do Trường xếp có khóa riêng).
- **Thứ tự**: ứng viên sắp theo thứ tự trong CTĐT (`stt`) ⇒ lộ trình tự nhiên (môn đại cương không bị
  đẩy xuống cuối) và lời giải tốt được sinh trước (A* trong tình huống 1 chỉ mở 5 nút).

**Heuristic**

- `h(n) = ⌈TC còn phải học / MAX_TC⌉` (theo đề). *Chấp nhận được*: mỗi kỳ học tối đa MAX_TC tín chỉ nên
  cần ít nhất ngần ấy kỳ. *Nhất quán*: mỗi bước giảm TC còn lại tối đa MAX_TC ⇒ h giảm tối đa 1 = chi phí bước.
- `h2(n) = max(h_lẻ/chẵn, h_chuỗi)` (mở rộng, so sánh trong báo cáo):
  - `h_lẻ/chẵn`: môn kỳ lẻ cần `a = ⌈TC_lẻ / MAX⌉` kỳ lẻ, môn kỳ chẵn cần `b = ⌈TC_chẵn / MAX⌉` kỳ chẵn ⇒
    ít nhất `max(2a−1, 2b)` kỳ nếu đang ở kỳ lẻ (`max(2b−1, 2a)` nếu đang ở kỳ chẵn).
  - `h_chuỗi`: chuỗi học trước dài nhất còn lại, có tính chờ tới kỳ mở môn.
  - Cả hai là lời giải chính xác của bài toán *nới lỏng* (bỏ học trước / bỏ giới hạn TC) ⇒ chấp nhận được.

## 4. Thuật toán (mục 2.2) và vai trò trong chatbot

| Thuật toán | Cài đặt | Vai trò trong chatbot |
|---|---|---|
| **BFS** | FIFO, kiểm tra đích khi sinh nút, mỗi tầng = 1 kỳ | **Chuẩn đối chiếu** (debug, trang so sánh, test) – không dùng trả lời vì tốn bộ nhớ khi bài toán lớn |
| **A\*** | heap theo `f = g + h`, hòa ưu tiên `g` lớn, kiểm tra đích khi lấy ra | **Bộ giải chính** – có giới hạn thời gian |
| **Nhánh cận** | DFS quay lui; cận trên ban đầu = **lời giải tham lam** (mỗi kỳ chọn tập nhiều TC nhất); cắt khi `g + h ≥ tốt nhất`; cắt trùng lặp; con có h nhỏ xét trước | **Bộ giải dự phòng** khi A* hết giờ – luôn có lời giải nên dừng lúc nào cũng trả lời được ("anytime") |

**Người dùng không chọn thuật toán.** `dieu_phoi.lap_lo_trinh()`:

```
A* (60% thời gian, mặc định tổng 3 giây)
  ├─ xong        → trả lời: lộ trình ít học kỳ nhất ✅
  └─ hết giờ     → Nhánh cận (thời gian còn lại)
                    ├─ xong    → tối ưu ✅
                    └─ hết giờ → lời giải tốt nhất đã tìm + cảnh báo "có thể chưa tối ưu" ⚠️
Chế độ debug: chạy thêm BFS → "Đối chiếu BFS: 5 kỳ ✔ khớp"
```

Chatbot chỉ hỏi bằng ngôn ngữ thường: học kỳ sắp đăng ký → học phần đã đạt (mã/tên, "hk1") →
học phần bị rớt → có muốn học vượt không. Trang **📊 So sánh thuật toán** trong app chạy cả 3 thuật toán
(+ biến thể h2) cho hội đồng xem.

## 5. Kết quả trên dữ liệu thật

Chạy `python -m src.planner.kich_ban --md` (thời gian thay đổi theo máy). Mọi thuật toán cho **cùng số
học kỳ** và mọi lộ trình được `kiem_tra_lo_trinh` xác nhận hợp lệ.

**1. Sinh viên năm nhất** – HK1 Trường xếp, 18 TC/kỳ → **5 kỳ**

| Thuật toán | Số kỳ | Nút mở rộng | Nút sinh | Ghi chú |
|---|---|---|---|---|
| BFS | 5 | 1075 | 1880 | |
| A* | 5 | 5 | 276 | |
| Nhánh cận | 5 | 5 | 276 | cận trên tham lam 6, cắt theo cận 271 |
| A* (h2) / Nhánh cận (h2) | 5 | 5 | 276 | |

**2. Sinh viên rớt 2 học phần** – đầu HK3, rớt COMP1010 (HK1) và COMP1016 (HK2) → **4 kỳ** (xong HK6, trễ 1 kỳ)

| Thuật toán | Số kỳ | Nút mở rộng | Nút sinh | Ghi chú |
|---|---|---|---|---|
| BFS | 4 | 247 | 463 | |
| A* | 4 | 247 | 463 | h yếu: không thấy chuỗi học trước |
| Nhánh cận | 4 | 246 | 462 | cận trên tham lam 4 (đã tối ưu), cắt theo cận 217 |
| A* (h2) | 4 | **4** | 38 | |

**3a. Học vượt từ năm nhất** (22 TC từ HK2) → **5 kỳ = không học vượt**

| Thuật toán | Số kỳ | Nút mở rộng | Nút sinh | Ghi chú |
|---|---|---|---|---|
| BFS | 5 | 327 | 488 | |
| A* | 5 | 303 | 464 | |
| Nhánh cận | 5 | 302 | 463 | cận trên tham lam 5, cắt theo cận 162 |
| A* (h2) | 5 | **5** | 166 | |

**3b. Năm 2 rớt COMP1016, học vượt để bù** → **3 kỳ** (không học vượt: 4 kỳ)

| Thuật toán | Số kỳ | Nút mở rộng | Nút sinh |
|---|---|---|---|
| BFS / A* / A* (h2) | 3 | 3 | 3 |
| Nhánh cận | 3 | 2 | 2 |

**Nhận xét để viết báo cáo**

- **Hiệu ứng dây chuyền khi rớt**: tình huống 2 xong ở HK6 thay vì HK5; COMP1803 (cần COMP1013, mở kỳ
  chẵn) bị đẩy sang kỳ chẵn cuối.
- **Học vượt không phải lúc nào cũng rút ngắn** (3a): nút thắt là *chuỗi học trước + kỳ mở*
  (POLI2001 → POLI2003 → POLI2005 → POLI2004 xen kẽ lẻ/chẵn; COMP1701 → COMP1314 đều mở kỳ lẻ và
  COMP1701 không thuộc HK1), không phải số tín chỉ. Khi nút thắt là tín chỉ (3b), học vượt rút được 1 kỳ.
- **Khi nào heuristic có ích**: ở 2 và 3a, `h` (chỉ dựa trên tín chỉ) gần như không giúp A* hơn BFS;
  `h2` (có kỳ lẻ/chẵn + chuỗi học trước) giảm số nút xuống bằng số kỳ.
- **Nhánh cận + tham lam**: ở 2, 3a, 3b lời giải tham lam đã tối ưu nên nhánh cận chủ yếu *chứng minh*
  không có lời giải tốt hơn (cắt theo cận); ở 1, tham lam cho 6 kỳ và nhánh cận hạ xuống 5.
- **Thứ tự sinh con** ảnh hưởng mạnh số nút của A*/nhánh cận (sắp theo `stt`: tình huống 1 còn 5 nút).

## 6. Kiểm thử và minh chứng vai trò của 3 thuật toán

Mục tiêu: chứng minh **bằng số liệu** vì sao chatbot dùng A* làm bộ giải chính, nhánh cận làm dự
phòng khi A* chậm, BFS làm chuẩn đối chiếu. Đánh giá theo 4 tiêu chí, mỗi tình huống có 3 bảng:

| Tiêu chí | Đo bằng | Kết luận rút ra |
|---|---|---|
| 1. Đúng | Số kỳ so với BFS (chuẩn), `kiem_tra_lo_trinh` | Cả 3 đúng ⇒ được phép dùng A*/nhánh cận thay BFS |
| 2. Hiệu quả | Nút mở rộng/sinh, thời gian TB, bộ nhớ đỉnh (Bảng 1) | A* tốt nhất ⇒ **bộ giải chính** |
| 3. Khi bị giới hạn thời gian | Có lời giải hay không ở 0 → 20 ms (Bảng 2) | Chỉ nhánh cận luôn có lời giải ⇒ **dự phòng** |
| 4. Ổn định | 300 sinh viên ngẫu nhiên (Bảng 4) | Kết luận không chỉ đúng với vài tình huống chọn sẵn |

Bảng 3 kiểm tra bộ điều phối thật sự chọn đúng: bình thường → A*; A* hết giờ → nhánh cận; debug →
đối chiếu BFS khớp.

### Cách tạo minh chứng

```bash
python -m pytest tests/test_planner.py tests/test_danh_gia.py -v
python -m src.planner.danh_gia --out bao_cao_danh_gia.md
```

- Lệnh 1: chụp màn hình danh sách `PASSED` (hoặc thêm `--html=bao_cao_test.html --self-contained-html`
  sau khi cài `pytest-html`). Test trong `tests/test_danh_gia.py` chỉ kiểm tra các kết luận **không phụ
  thuộc tốc độ máy**: A*, nhánh cận luôn khớp BFS trên 100 sinh viên ngẫu nhiên; A* mở ít nút hơn hoặc
  bằng BFS; hết giờ (0 ms) thì chỉ nhánh cận còn lộ trình hợp lệ; bộ điều phối chọn đúng vai trò.
- Lệnh 2: sinh đủ 4 loại bảng (Markdown) để dán vào báo cáo; đầu file ghi thời điểm, phiên bản
  Python, hệ điều hành. Thời gian/bộ nhớ đổi theo máy – ghi rõ máy chạy trong báo cáo.

### Kết quả tham khảo (máy Linux x86_64, Python 3.13; 20 lần lặp; 300 sinh viên, seed 42)

| Tình huống | Số kỳ | Nút mở rộng BFS / A* / NC | Thời gian TB (ms) BFS / A* / NC | Bộ nhớ (KB) BFS / A* / NC | Giới hạn 1 ms: ai còn trả lời |
|---|---|---|---|---|---|
| 1. Năm nhất | 5 | 1075 / 5 / 5 | 14.1 / 1.3 / 2.8 | 2276 / 424 / 607 | chỉ nhánh cận (6 kỳ ⚠) |
| 2. Rớt 2 học phần | 4 | 247 / 247 / 246 | 2.7 / 3.7 / 4.6 | 676 / 612 / 467 | chỉ nhánh cận (4 kỳ ⚠) |
| 3a. Học vượt từ năm nhất | 5 | 327 / 303 / 302 | 4.2 / 5.9 / 6.7 | 791 / 685 / 582 | chỉ nhánh cận (5 kỳ ⚠) |
| 3b. Học vượt để bù môn rớt | 3 | 3 / 3 / 2 | 0.02 / 0.03 / 0.07 | 9 / 9 / 12 | chỉ nhánh cận (3 kỳ ⚠) |

300 sinh viên ngẫu nhiên: cùng số kỳ **300/300**, hợp lệ **300/300**, A* ít nút hơn hoặc bằng BFS
**300/300**, A* nhanh hơn hoặc bằng nhánh cận **253/300 (84%)**. Thời gian TB: BFS 12.2 ms, A* 1.8 ms,
nhánh cận 2.4 ms; lớn nhất: BFS 396 ms, A* 71 ms, nhánh cận 77 ms; số nút lớn nhất: BFS 23 872, A* 3 592.

### Lập luận đưa vào báo cáo

- **A* là bộ giải chính**: luôn tối ưu như BFS; trung bình nhanh hơn BFS ~6,7 lần, trường hợp xấu nhất
  71 ms so với 396 ms; nhanh hơn nhánh cận ở 84% trường hợp. *Trung thực*: ở tình huống 2 và 3a, h
  (chỉ dựa trên tín chỉ) không giúp được – A* mở gần bằng BFS và chậm hơn ~1 ms do chi phí hàng đợi ưu
  tiên; chênh lệch này không đáng kể với người dùng, còn ở các ca khó A* hơn hẳn. Heuristic mở rộng h2
  khắc phục được (xem mục 5).
- **Nhánh cận là dự phòng**: có lời giải tham lam ngay từ đầu nên khi bị giới hạn thời gian vẫn trả
  lời được (BFS, A* thì chưa); lời giải có thể chưa tối ưu (năm nhất: 6 kỳ thay vì 5) nên chatbot gắn
  cảnh báo ⚠. Không dùng làm bộ giải chính vì chậm hơn A* ở phần lớn trường hợp.
- **BFS là chuẩn đối chiếu**: chắc chắn tối ưu, cài đặt đơn giản nên đáng tin để so, nhưng tốn nút và bộ
  nhớ nhất (gấp ~5 lần A* ở tình huống năm nhất) nên không dùng trả lời người dùng.
