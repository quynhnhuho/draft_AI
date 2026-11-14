# Định hướng đồ án: Chatbot tư vấn tuyển sinh – Khoa CNTT

> Tài liệu này là "bản đồ" cho cả nhóm: kiến trúc tổng thể, hợp đồng (interface) giữa các mô-đun,
> các bước làm từng chức năng, cách đánh giá và cách viết báo cáo. Đọc mục 1–3 trước, sau đó mỗi
> người đọc kỹ mục của chức năng mình phụ trách. **Chức năng 4 (mục 7) là nơi ghép mọi thứ lại** —
> nên dựng khung của nó ngay từ tuần đầu để các mô-đun khác cắm vào dần.

---

## 1. Kiến trúc tổng thể

```
                    ┌──────────────────────────────────────────────────────────┐
  Người dùng  ───►  │  app.py  (Streamlit: khung chat + sidebar form + debug)  │
                    └──────────────────────────┬───────────────────────────────┘
                                               │ text, DialogueState
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  src/chatbot/engine.py   ChatEngine.respond()            │
                    │   1. Đang trong luồng nhiều lượt? → tiếp tục luồng       │
                    │   2. Tiền xử lý + trích thực thể (ngành, tổ hợp, năm...) │
                    │   3. Nhận ý định  (C3)  → (intent, confidence)           │
                    │   4. confidence < τ  → CHUYỂN CÁN BỘ (ghi log)           │
                    │   5. Định tuyến theo intent → handler                    │
                    └───┬───────────────┬────────────────┬─────────────────────┘
                        │               │                │
            ┌───────────▼───┐  ┌────────▼────────┐  ┌────▼──────────────────┐
            │ Cơ sở tri thức│  │ C2: Tư vấn ngành│  │ C1: Lộ trình học      │
            │ kb.py +       │  │ src/advisor/    │  │ src/planner/          │
            │ knowledge_    │  │ suy diễn tiến / │  │ BFS / A* / nhánh cận  │
            │ base.json     │  │ suy diễn lùi    │  │                       │
            │ (có nguồn+năm)│  │ + luật từ ID3   │  │                       │
            └───────────────┘  └─────────────────┘  └───────────────────────┘
                        ▲
            ┌───────────┴─────────────────────────────┐
            │ C3: src/nlu/  tiền xử lý tiếng Việt,     │
            │ NB baseline, Keras LSTM/GRU/Conv1D       │
            │ → models/intent_model.keras + labels.json│
            └──────────────────────────────────────────┘
```

**Nguyên tắc cốt lõi:** mỗi chức năng là một mô-đun Python độc lập, có hàm công khai rõ ràng,
chạy được bằng dòng lệnh và có notebook/script riêng để sinh số liệu cho báo cáo. Chatbot (C4)
chỉ *gọi* các hàm đó, không chứa logic thuật toán.

## 2. Cấu trúc thư mục

```
draft_AI/
├── app.py                         # C4 – giao diện Streamlit
├── requirements.txt
├── data/
│   ├── knowledge_base.json        # C4 – cơ sở tri thức (mỗi số liệu có nguồn + năm)
│   ├── hoc_phan.csv               # C1 – 28 học phần (QĐ 2619/QĐ-ĐHSP, 2024)
│   ├── survey.csv                 # C2 – ≥14 mẫu khảo sát, ≥4 thuộc tính
│   ├── intents.csv                # C3 – ≥600 câu, ≥8 ý định (file mẫu: intents_sample.csv)
│   └── out_of_scope.csv           # C3 – câu ngoài phạm vi để chọn ngưỡng τ
├── src/
│   ├── planner/                   # C1
│   │   ├── model.py               #   mô hình dùng chung: HocPhan, BaiToan, KetQua, h, h2
│   │   ├── bfs.py                 #   BFS            (thành viên 1)
│   │   ├── a_sao.py               #   A*             (thành viên 2)
│   │   ├── nhanh_can.py           #   nhánh cận      (thành viên 3)
│   │   ├── dieu_phoi.py           #   kết hợp 3 thuật toán – chatbot chỉ gọi file này
│   │   ├── kich_ban.py            #   tình huống + bảng so sánh cho báo cáo
│   │   └── cli.py                 #   chạy dòng lệnh
│   ├── advisor/                   # C2
│   │   ├── rules.py               #   Rule + tập ≥15 luật
│   │   ├── facts.py               #   chuyển input số (điểm) → sự kiện ký hiệu
│   │   ├── inference.py           #   forward_chaining(), backward_chaining() ← TỰ CÀI ĐẶT
│   │   └── id3.py                 #   tính entropy/IG, so với sklearn, rút luật
│   ├── nlu/                       # C3
│   │   ├── preprocess.py          #   chuẩn hóa dấu, teencode, tách từ
│   │   ├── train_nb.py            #   baseline TF-IDF + MultinomialNB
│   │   ├── train_keras.py         #   mô hình chính
│   │   └── classifier.py          #   IntentClassifier.predict() – dùng chung cho C4
│   └── chatbot/                   # C4
│       ├── entities.py            #   trích ngành, tổ hợp, năm, điểm
│       ├── kb.py                  #   truy vấn cơ sở tri thức
│       ├── flows.py               #   hội thoại nhiều lượt (slot filling) cho C1, C2
│       └── engine.py              #   ChatEngine: định tuyến + ngưỡng + chuyển cán bộ
├── models/                        # artefact huấn luyện (.keras, .pkl, labels.json)
├── logs/                          # handoff.csv, conversations.jsonl
├── notebooks/                     # tính tay ID3, đánh giá mô hình, vẽ biểu đồ
├── tests/                         # pytest + kịch bản hội thoại end-to-end
└── docs/                          # tài liệu này, quy trình gán nhãn, báo cáo
```

## 3. Nguyên tắc chung cho cả nhóm

1. **Mọi số liệu tuyển sinh đều có `nguon` + `nam_ap_dung` (+ `url`, `ngay_truy_cap`).** Không có
   nguồn thì không đưa vào cơ sở tri thức. Nguồn hợp lệ: đề án tuyển sinh, thông báo điểm chuẩn,
   quyết định học phí, chương trình đào tạo đăng trên website chính thức của Trường.
2. **Dữ liệu mô phỏng phải ghi rõ** (trường `mo_phong: true` hoặc `da_kiem_chung: false`) và
   chatbot hiển thị cảnh báo khi trả lời bằng dữ liệu đó.
3. **Không dùng API LLM làm lõi.** Nếu dùng LLM để gợi ý thêm câu huấn luyện: ghi cột
   `nguon = "llm_sinh"` cho từng câu, có người kiểm tra lại từng câu, và khai báo trong báo cáo
   (tỷ lệ bao nhiêu %, đã sửa/bỏ bao nhiêu câu). Tốt nhất giữ tập **test** 100% câu do người viết
   hoặc thu thập thật.
4. **Thuật toán cốt lõi tự cài đặt** (BFS, A*, nhánh cận, suy diễn tiến/lùi, entropy/IG).
   Thư viện chỉ dùng để *kiểm chứng* (sklearn cho ID3) hoặc ở phần được phép (Keras, NB).
5. **Cố định `random_state`/seed**, lưu cấu hình và kết quả thí nghiệm để báo cáo tái lập được.
6. Git: mỗi chức năng một nhánh, commit nhỏ, không commit file model > 50 MB.

---

## 4. Chức năng 1 – Lập lộ trình học tối ưu (Phần 2: 2.1, 2.2) ✅

**Đã hoàn thành** – chi tiết thiết kế, giả định, kết quả: [`docs/C1_LO_TRINH.md`](C1_LO_TRINH.md);
hướng dẫn từng thành viên commit thuật toán của mình: [`docs/HUONG_DAN_COMMIT_C1.md`](HUONG_DAN_COMMIT_C1.md).

Tóm tắt:
- Dữ liệu `data/hoc_phan.csv`: 28 học phần, 75 TC – nguồn: CTĐT ban hành kèm Quyết định số 2619/QĐ-ĐHSP
  ngày 04/09/2024 của Hiệu trưởng Trường ĐHSP TP.HCM.
- Giả định: 18 TC/kỳ (22 khi học vượt), không có học kỳ hè, HK1 do Trường xếp (7 học phần đầu, 18 TC),
  học trước chỉ cần đã học (kể cả rớt).
- `model.py` dùng chung; mỗi thuật toán một file (`bfs.py`, `a_sao.py`, `nhanh_can.py`) để mỗi thành viên
  tự commit; `dieu_phoi.py` kết hợp 3 thuật toán cho chatbot.
- Chatbot **không cho người dùng chọn thuật toán**: A* là bộ giải chính, nhánh cận (khởi tạo bằng lời
  giải tham lam, dừng lúc nào cũng có lời giải) là dự phòng, BFS là chuẩn đối chiếu.
- 4 tình huống (năm nhất, rớt 2 học phần, học vượt từ năm nhất, học vượt để bù môn rớt): mọi thuật toán
  cho cùng số học kỳ; có thêm heuristic mở rộng h2 để so sánh.

---

## 5. Chức năng 2 – Tư vấn chọn ngành (Phần 3: 3.3, 3.4; Phần 4: 4.1)

### 5.1 Biểu diễn tri thức

- **Sự kiện** dạng ký hiệu: `to_hop=A00`, `so_thich=lap_trinh`, `dinh_huong=giao_vien`,
  `diem_dat_cntt`, …
- **Luật**: `Rule(id, if_all=[...], then="...", giai_thich="...", nguon="Đề án TS 2025, mục ...")`.
- **Điểm số → sự kiện**: một bước "xây dựng sự kiện" (`facts.py`) so điểm người dùng với điểm
  chuẩn/ngưỡng đảm bảo chất lượng **năm gần nhất trong cơ sở tri thức** rồi sinh `diem_dat_<ngành>`
  hoặc `diem_sat_nguong_<ngành>` (dưới 1 điểm). Như vậy luật vẫn thuần ký hiệu, dễ suy diễn.

### 5.2 Gợi ý nhóm ≥15 luật

| Nhóm | Ví dụ |
|---|---|
| Tổ hợp hợp lệ (5–6 luật) | `to_hop ∈ {tổ hợp của ngành X theo đề án}` → `to_hop_hop_le_X` |
| Điểm (3–4 luật) | `to_hop_hop_le_X ∧ diem_dat_X` → `du_dk_X` |
| Điều kiện riêng sư phạm (2 luật) | ngưỡng đảm bảo chất lượng đầu vào ngành sư phạm do Bộ GD&ĐT công bố; yêu cầu học lực lớp 12 (nếu xét học bạ) – **kiểm tra trong đề án năm áp dụng** |
| Sở thích/định hướng (4–5 luật) | `so_thich=day_hoc ∧ dinh_huong=giao_vien` → `phu_hop_sp_tin`; `so_thich=du_lieu` → `phu_hop_cntt` |
| Kết luận (2–3 luật) | `du_dk_X ∧ phu_hop_X` → `goi_y_X`; `phu_hop_X ∧ ¬du_dk_X` → `goi_y_X_phuong_thuc_khac` |
| Luật từ cây ID3 | rút từ `export_text` (mục 5.4) và gắn `nguon="ID3 khảo sát"` |

### 5.3 Suy diễn

- **Tiến (forward chaining)** – "danh sách ngành gợi ý": lặp tìm luật có `if_all ⊆ facts` và
  `then ∉ facts` → thêm vào, ghi vết (luật nào bắn, theo thứ tự). Dừng khi không còn luật mới.
  Trả về các sự kiện `goi_y_*` + **vết suy diễn** (để chatbot giải thích "vì sao").
- **Lùi (backward chaining)** – "Em có đủ điều kiện xét tuyển ngành X không?": mục tiêu `du_dk_X`.
  Đệ quy: đã có trong facts → đúng; có luật kết luận ra mục tiêu → chứng minh mọi tiền đề; chống
  lặp vô hạn bằng tập `dang_xet`. Sự kiện lá còn thiếu (chưa biết tổ hợp/điểm) → **trả về danh sách
  câu cần hỏi** để chatbot hỏi tiếp (đây là điểm nối đẹp với C4).
- Viết cả hai thành hàm thuần, có unit test với vài bộ facts cố định.

### 5.4 ID3

1. `data/survey.csv`: ≥14 mẫu (nên 20–30), ≥4 thuộc tính rời rạc, ví dụ: `mon_manh` (Toán/Tin/Văn-Anh),
   `so_thich` (lập trình/dạy học/dữ liệu), `thich_giao_tiep` (có/không), `dinh_huong`
   (doanh nghiệp/giáo viên/nghiên cứu) → nhãn `nganh_chon` (CNTT/SP Tin). Ghi rõ đối tượng khảo sát
   (bạn bè, sinh viên khóa dưới…) hoặc "mô phỏng".
2. **Tính tay** trong báo cáo: `H(S)`, `H(S|A)` và `IG(A)` cho cả 4 thuộc tính → chọn gốc; lặp lại
   cho từng nhánh tầng 2. Trình bày bảng, giữ 3–4 chữ số thập phân.
3. Viết `id3.py` tính lại đúng các con số đó (để đối chiếu với tính tay).
4. **sklearn:** `DecisionTreeClassifier(criterion="entropy")` + one-hot. Lưu ý sklearn dùng **chia
   nhị phân (CART)** nên cây có thể khác ID3 đa nhánh — nêu và giải thích sự khác biệt, nhưng thuộc
   tính gốc thường trùng. `export_text` → rút luật → đưa vào `rules.py`.

---

## 6. Chức năng 3 – Phân loại ý định (Phần 5: 5.2, 5.3)

### 6.1 Danh sách ý định đề xuất (11)

`diem_chuan`, `to_hop_xet_tuyen`, `hoc_phi`, `hoc_bong`, `ky_tuc_xa`, `chuong_trinh_dao_tao`,
`viec_lam`, `thu_tuc_nhap_hoc`, `chao_hoi` (chào/cảm ơn/tạm biệt) **+ 2 ý định để gọi mô-đun:**
`tu_van_chon_nganh` (C2) và `lo_trinh_hoc_tap` (C1).

> "Khác/ngoài phạm vi" **không nên là một lớp**: thu riêng 60–100 câu ngoài phạm vi vào
> `out_of_scope.csv` (hỏi thời tiết, ngành của khoa khác, câu vô nghĩa…) để **chọn ngưỡng τ**. Mô hình
> softmax buộc phải chọn một lớp, nên chính ngưỡng là cơ chế "không trả lời bừa".

### 6.2 Bộ dữ liệu ≥600 câu

- Cột: `text, intent, nguon (fanpage|faq|nhom_viet|bien_the|llm_sinh), nguoi_gan_1, nguoi_gan_2`.
- ≥60 câu/ý định, cân bằng tương đối. Nguồn: câu hỏi thật trên fanpage/nhóm tuyển sinh (ẩn danh),
  FAQ của Trường, mỗi thành viên tự viết, biến thể (không dấu, teencode "hp", "ktx", "đc", viết tắt
  "cntt", lỗi chính tả, câu dài/ngắn).
- **Quy trình gán nhãn** (viết thành `docs/quy_trinh_gan_nhan.md`): định nghĩa từng ý định + ví dụ
  biên; 2 người gán độc lập trên toàn bộ (hoặc ≥20%) → tính **Cohen's kappa** → thảo luận các câu
  lệch → chốt nhãn. Câu chứa 2 ý định: gán ý định chính, ghi chú.
- **Xóa trùng trước khi chia** (sau chuẩn hóa), chia stratified 70/15/15 → tránh rò rỉ.

### 6.3 Tiền xử lý (`src/nlu/preprocess.py` – đã có khung)

Unicode NFC → chữ thường → chuẩn hóa vị trí dấu (`hòa`→`hoà`, `thủy`→`thuỷ`) → teencode/viết tắt
→ bỏ ký tự thừa → tách từ (`pyvi` hoặc `underthesea`; ghép từ bằng `_`). Dùng **cùng một hàm** cho
huấn luyện và lúc chatbot chạy.

### 6.4 Mô hình

- **Baseline:** `TfidfVectorizer(ngram_range=(1,2))` + `MultinomialNB(alpha≈0.1–1)`; thêm char-ngram
  để chịu lỗi chính tả là một thí nghiệm hay.
- **Keras:** `TextVectorization` → `Embedding(vocab, 128)` → `Bidirectional(LSTM(64))` (hoặc GRU,
  hoặc `Conv1D(128,3)+GlobalMaxPool`) → `Dropout(0.3)` → `Dense(n_intent, softmax)`. EarlyStopping
  theo `val_loss`. Dữ liệu nhỏ → mô hình nhỏ, dropout cao; so sánh 2–3 kiến trúc.
- **Đánh giá:** `classification_report` (P/R/F1 từng ý định + macro), ma trận nhầm lẫn, đường
  học (loss/accuracy). So sánh NB vs Keras trên **cùng tập test**.

### 6.5 Ngưỡng tin cậy

- Điểm tin cậy = xác suất softmax lớn nhất (có thể thêm điều kiện chênh lệch top1–top2 ≥ δ).
- Quét τ từ 0.3 → 0.95 trên **val ∪ out_of_scope**: vẽ đồ thị *tỷ lệ trả lời đúng trong phạm vi*
  và *tỷ lệ chặn được câu ngoài phạm vi* theo τ → chọn τ cân bằng, ghi vào `models/config.json`.
- **Hợp đồng với C4** (`src/nlu/classifier.py`): `predict(text) -> (intent, confidence, top3)`.

---

## 7. Chức năng 4 – Ứng dụng chatbot (CLO 3, CLO 4) ★

Khung đã có trong repo: `app.py` + `src/chatbot/`. Chạy được ngay kể cả khi C1–C3 chưa xong
(dùng bộ phân loại từ khóa tạm thời và thông báo "mô-đun chưa cài đặt").

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 7.1 Luồng xử lý một lượt

```
text ─► [đang trong luồng nhiều lượt?] ──có──► flows.continue_flow()  (gõ "hủy" để thoát)
          │ không
          ▼
        preprocess ─► entities (ngành, tổ hợp, năm, điểm, mã HP)
          │
          ▼
        classifier.predict ─► (intent, conf)
          │
          ├─ conf < τ ─► "Câu hỏi này em xin chuyển cán bộ tư vấn…" + liên hệ + ghi logs/handoff.csv
          │
          ▼
        HANDLERS[intent](entities, state)  ─► BotReply(text, sources, intent, conf, handoff)
```

### 7.2 Bảng định tuyến

| Ý định | Handler | Nguồn trả lời |
|---|---|---|
| diem_chuan, to_hop_xet_tuyen, hoc_phi, hoc_bong, ky_tuc_xa, thu_tuc_nhap_hoc | `answer_from_kb` | `knowledge_base.json` – lọc theo `loai`, `nganh`, `nam` |
| chuong_trinh_dao_tao, viec_lam | `answer_from_kb` (+ mô tả ngành) | `knowledge_base.json` |
| chao_hoi | `greet` | câu mẫu |
| tu_van_chon_nganh | nếu câu có ngành cụ thể + "đủ điều kiện/đậu/vào được" → **suy diễn lùi**; ngược lại → **suy diễn tiến**. Thiếu thông tin → mở luồng hỏi tiếp | C2 |
| lo_trinh_hoc_tap | mở luồng hỏi: học kỳ sắp đăng ký → môn đã đạt → môn rớt → có học vượt không → `dieu_phoi.lap_lo_trinh()` (tự chọn thuật toán) | C1 |

### 7.3 Cơ sở tri thức (`data/knowledge_base.json`)

Mỗi "fact" là một bản ghi có nguồn:

```json
{ "id": "dc_cntt_2025_thpt", "loai": "diem_chuan", "nganh": "cntt", "nam": 2025,
  "noi_dung": "Điểm chuẩn ngành CNTT theo phương thức điểm thi THPT: 24.50",
  "gia_tri": 24.5, "phuong_thuc": "thpt", "to_hop": ["A00","A01","D01"],
  "nguon": "Thông báo điểm trúng tuyển đợt 1 năm 2025", "url": "https://…",
  "ngay_truy_cap": "2026-11-01", "da_kiem_chung": true }
```

- Handler **luôn** in dòng `📌 Nguồn: <nguon> – năm áp dụng <nam>`. Nếu `da_kiem_chung=false` thêm
  `⚠ Số liệu mẫu/mô phỏng, chưa đối chiếu nguồn chính thức`.
- Hỏi không nêu năm → trả **năm mới nhất** và nói rõ năm. Hỏi năm không có dữ liệu → nói không có,
  đưa năm gần nhất. Không bao giờ "đoán" số liệu.
- Mã ngành chuẩn của Bộ: CNTT `7480201`, Sư phạm Tin học `7140210` (kiểm tra lại trong đề án).
- Học phí sư phạm: sinh viên sư phạm được hỗ trợ học phí + sinh hoạt phí theo Nghị định 116/2020/NĐ-CP
  (và văn bản sửa đổi nếu có) → **kiểm tra văn bản hiện hành** trước khi đưa vào.

### 7.4 Trích xuất thực thể (`entities.py`) – luật + từ điển, không cần học máy

- **Ngành:** từ điển bí danh (`cntt`, `công nghệ thông tin`, `it`, `sp tin`, `sư phạm tin`…) lấy từ
  `aliases` trong cơ sở tri thức.
- **Tổ hợp:** regex `\b[A-DX]\d{2}\b` (A00, D01…).
- **Năm:** `\b20[12]\d\b`. **Điểm:** số thực 0–30 đứng sau "điểm/được/đạt".
- **Học phần:** mã (COMP1010) hoặc tên đầy đủ trong `hoc_phan.csv`; "hk1" = toàn bộ học kỳ 1 do Trường xếp.

### 7.5 Quản lý hội thoại (`DialogueState`)

- `last_major`, `last_year`: nhớ ngữ cảnh → "Thế học phí ngành đó?" dùng lại ngành vừa nói.
- `active_flow` + `slots`: luồng nhiều lượt (slot filling). Mỗi slot có câu hỏi, hàm phân tích câu
  trả lời, và hỏi lại khi không hiểu. Gõ "hủy"/"thoát" để thoát luồng.
- Streamlit: lưu state trong `st.session_state` để không mất khi trang chạy lại.

### 7.6 Tích hợp C1 và C2 — hai cách song song

1. **Trong khung chat** (ý định `tu_van_chon_nganh`, `lo_trinh_hoc_tap`) → luồng hỏi từng bước.
2. **Form ở sidebar** (chọn tổ hợp, nhập điểm, tick sở thích / chọn môn đã đạt) → bấm nút → kết quả
   hiện trong chat. Form giúp demo nhanh và chắc chắn, chat thể hiện "hiểu ngôn ngữ".

Kết quả C2 hiển thị **vết suy diễn** (R3 → R7 → R12…) để chứng minh hệ luật; kết quả C1 hiển thị
bảng học kỳ × học phần, tổng tín chỉ mỗi kỳ, nguồn CTĐT và giả định. Người dùng **không chọn thuật toán**;
ở chế độ debug mới hiện thuật toán đã dùng, số nút và đối chiếu BFS. Trang **📊 So sánh thuật toán**
trong app chạy cả 3 thuật toán cho hội đồng.

### 7.7 Ngưỡng & chuyển cán bộ

- τ đọc từ `models/config.json` (từ C3), có slider ở sidebar để demo.
- Câu bị chặn: trả lời lịch sự + kênh liên hệ thật (email/điện thoại/fanpage Khoa) + ghi
  `logs/handoff.csv` (thời gian, câu hỏi, top-3 ý định, độ tin cậy). File này vừa là "hàng đợi" cho cán
  bộ, vừa là dữ liệu để bổ sung tập huấn luyện vòng sau → nêu trong báo cáo như vòng cải tiến.
- Nút 👍/👎 sau mỗi câu trả lời → `logs/feedback.csv` (tùy chọn, ghi điểm cộng).

### 7.8 Giao diện

- Khung chính: `st.chat_message` / `st.chat_input`, các nút gợi ý câu hỏi nhanh.
- Sidebar: form tư vấn ngành, form lộ trình, slider τ, công tắc "chế độ debug" (hiện ý định, độ tin
  cậy, top-3, thực thể — rất hữu ích khi demo trước hội đồng).
- Mỗi câu trả lời số liệu: phần "Nguồn" ở cuối.

### 7.9 Kiểm thử & đánh giá chức năng 4

- `tests/test_engine.py`: unit test định tuyến, ngưỡng, trích thực thể, ngữ cảnh.
- **Kịch bản end-to-end** (`tests/scenarios.yaml`, 30–50 lượt): câu hỏi → ý định mong đợi → câu trả lời
  phải chứa (số liệu/nguồn/năm) → có chuyển cán bộ hay không. Báo cáo: tỷ lệ đúng ý định, tỷ lệ câu
  trả lời có nguồn (phải 100% với câu số liệu), tỷ lệ chuyển cán bộ đúng/sai, thời gian phản hồi.
- Thử với 3–5 người dùng thật (bạn cùng lớp), thu nhận xét → mục "Đánh giá người dùng".

### 7.10 Lỗi thường gặp cần tránh

- Tiền xử lý lúc chạy khác lúc huấn luyện → độ chính xác demo tụt mạnh.
- Câu trả lời số liệu thiếu năm/nguồn (bị trừ điểm theo yêu cầu đề).
- Load model Keras mỗi lượt chat → chậm; dùng `@st.cache_resource`.
- Trả lời bừa khi câu ngoài phạm vi → phải có τ và tập out-of-scope để chứng minh.
- Thuật toán C1 bùng nổ tổ hợp khi demo → giới hạn số học phần liên quan, dùng tập tối đại, có timeout.

---

## 8. Dàn ý báo cáo (ánh xạ yêu cầu)

| Chương | Nội dung | Mục đề |
|---|---|---|
| 1 | Giới thiệu bài toán, mục tiêu, phạm vi, nguồn dữ liệu (năm) | — |
| 2 | Lộ trình học: mô hình hóa không gian trạng thái, BFS, A* (chứng minh h chấp nhận được), nhánh cận, 3 tình huống, so sánh | Phần 2 – 2.1, 2.2 |
| 3 | Hệ luật: biểu diễn, ≥15 luật (bảng), suy diễn tiến, suy diễn lùi, ví dụ vết suy diễn | Phần 3 – 3.3, 3.4 |
| 4 | ID3: bảng khảo sát, tính tay 2 tầng, kiểm chứng sklearn, tập luật rút ra | Phần 4 – 4.1 |
| 5 | Phân loại ý định: dữ liệu + quy trình gán nhãn + kappa, tiền xử lý, NB, Keras, P/R/F1, ngưỡng | Phần 5 – 5.2, 5.3 |
| 6 | Ứng dụng chatbot: kiến trúc, cơ sở tri thức, định tuyến, hội thoại, giao diện, đánh giá end-to-end | CLO 3, CLO 4 |
| 7 | Kết luận, hạn chế, hướng phát triển; **khai báo sử dụng công cụ AI** | — |
| PL | Bảng học phần, danh sách luật, mẫu dữ liệu, kịch bản test, phân công | — |

## 9. Kế hoạch gợi ý (8 tuần, nhóm 3–4 người)

| Tuần | Việc chính |
|---|---|
| 1 | Chốt trường/khoa, thu thập đề án tuyển sinh + CTĐT + học phí (ghi nguồn, năm). Dựng khung C4 (đã có). Viết định nghĩa ý định + quy trình gán nhãn. |
| 2 | `hoc_phan.csv` ≥25 HP (đã xong); `knowledge_base.json` đầy đủ; bắt đầu thu câu hỏi (mục tiêu 300). |
| 3 | C1: model + BFS + A*. C2: luật + suy diễn tiến. C3: đủ 600 câu, gán nhãn đôi, kappa. |
| 4 | C1: nhánh cận + 3 tình huống. C2: suy diễn lùi + khảo sát + ID3 tính tay. C3: tiền xử lý + NB. |
| 5 | C3: Keras + đánh giá + ngưỡng. C2: sklearn + rút luật. Tích hợp C1, C2 vào chatbot. |
| 6 | Tích hợp model Keras vào C4, kịch bản end-to-end, sửa lỗi, thử với người dùng. |
| 7 | Viết báo cáo, vẽ hình/biểu đồ, quay video demo dự phòng. |
| 8 | Rà soát, tập thuyết trình, đóng gói nộp. |

**Phân vai:** A – C1 + phần 2 báo cáo; B – C2 + phần 3, 4; C – C3 + dữ liệu; D (hoặc cả nhóm) – C4,
cơ sở tri thức, tích hợp, kiểm thử. Ai cũng góp câu hỏi cho tập dữ liệu C3.

## 10. Checklist nộp bài

- [ ] ≥25 học phần, đánh dấu phần mô phỏng; BFS, A*, nhánh cận tự cài; 3 tình huống + bảng so sánh
- [ ] ≥15 luật có nguồn (đề án năm …); suy diễn tiến + lùi tự cài; có vết suy diễn
- [ ] Bảng khảo sát ≥14 mẫu × ≥4 thuộc tính; tính tay ID3 2 tầng; kiểm chứng sklearn; luật rút trích
- [ ] ≥600 câu, ≥8 ý định, quy trình gán nhãn + kappa; khai báo câu do LLM sinh (nếu có)
- [ ] Tiền xử lý tiếng Việt; NB baseline; Keras; P/R/F1 từng ý định; ngưỡng + tập ngoài phạm vi
- [ ] Chatbot: nhận ý định bằng Keras, trả lời từ cơ sở tri thức, gọi C1 + C2, **mọi số liệu có nguồn + năm**, chuyển cán bộ khi không chắc
- [ ] Không dùng API LLM trong lõi xử lý; README hướng dẫn cài đặt/chạy; video demo
