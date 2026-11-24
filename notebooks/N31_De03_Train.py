# -*- coding: utf-8 -*-
"""N31_De03_Train – Chức năng 3: Phân loại ý định (bản đã chỉnh sửa C, D, E)

Chạy trên Google Colab (mỗi khối "# Cell ..." là một ô), hoặc chạy cục bộ:
đặt intents.csv và out_of_scope.csv cạnh file này hoặc trong thư mục data/.

NHỮNG THAY ĐỔI SO VỚI BẢN CŨ
C – Tiền xử lý
  * Sửa teencode bị hiểu sai nghĩa trong miền tuyển sinh: bỏ "a", "b" (để "khối A",
    "khối B" giữ nguyên), "bn" → bao nhiêu, "đk" → điều kiện, "tc" → tín chỉ,
    "nv" → nguyện vọng, "bk" → biết; bỏ "nh" (mơ hồ) và khóa trùng "thcs".
  * Thêm bước chuẩn hóa vị trí dấu thanh: hòa → hoà, thủy → thuỷ (cho normalize()).
    Sau đó underthesea tự đưa mọi kiểu dấu về một dạng chung, nên "hoà"/"hòa" luôn ra
    cùng một token.
  * Bỏ ký tự đặc biệt TRƯỚC khi thay teencode, nên "hp?", "(ktx)", "tp.hcm" đều được nhận.
  * Bật tách từ bằng underthesea ("học phí" → "học_phí"). Tokenizer Keras đặt filters=""
    để không xóa dấu "_" (mặc định Keras xóa "_" và tách từ ghép trở lại thành âm tiết).
  * Tách thành normalize() (chưa tách từ – chatbot dùng để trích thực thể) và preprocess()
    (normalize + tách từ – dùng cho mô hình). Hai hàm này cần chép NGUYÊN VĂN sang
    src/nlu/preprocess.py để lúc chatbot chạy xử lý câu giống hệt lúc huấn luyện.
D – Huấn luyện
  * Cố định seed cho cả TensorFlow (tf.keras.utils.set_random_seed) → kết quả tái lập được.
  * Xóa câu trùng (sau tiền xử lý) TRƯỚC khi chia; chia stratified 70/15/15 train/val/test.
  * Tokenizer và TF-IDF chỉ học trên tập train; có tập val riêng (validation_data),
    không dùng validation_split.
  * LSTM: mask_zero=True để bỏ qua phần đệm 0 (bản cũ LSTM v2 thiếu mask). Bỏ input_length
    (Keras 3 báo lỗi thời). Mô hình nhỏ hơn, dropout cao do dữ liệu ít.
  * Mọi mô hình cùng loss sparse_categorical_crossentropy, EarlyStopping theo val_loss.
  * Chọn alpha của Naive Bayes trên tập val.
  * Không ghi đè lstm_best.keras bằng LSTM v2 – giữ riêng để so sánh minh bạch.
  * Thêm ma trận nhầm lẫn cho từng mô hình; báo cáo macro F1 (dữ liệu cân bằng).
  * Chọn mô hình bằng tập val (KHÔNG chọn bằng tập test); tập test chỉ để báo cáo.
E – Ngưỡng tin cậy τ
  * Chia out_of_scope.csv 50/50: một nửa để chọn τ, một nửa để báo cáo.
  * Quét τ trên val ∪ OOS_val với 2 chỉ số đúng yêu cầu đề: tỷ lệ TRẢ LỜI ĐÚNG câu trong
    phạm vi và tỷ lệ CHẶN được câu ngoài phạm vi; τ tối ưu = điểm có trung bình điều hòa
    của hai tỷ lệ lớn nhất. (Bản cũ coi mọi câu trong phạm vi được trả lời là đúng, dù
    mô hình đoán sai nhãn, và chỉ quét trên LSTM v1.)
  * Quét cho TẤT CẢ mô hình (dải τ 0.10 → 0.99) để so sánh khả năng chặn câu ngoài phạm vi.
  * Mô hình chính = mô hình có macro F1 trên tập val cao nhất; τ là ngưỡng tối ưu của
    chính mô hình đó. (Không chọn mô hình theo điểm cân bằng vì OOS_val chỉ ~40 câu, quá nhiễu.)
  * Đánh giá τ đã chọn trên test ∪ OOS_test.
  * Ghi config.json (τ, mô hình chính, MAX_LEN, cách tiền xử lý) và threshold_config.json
    (khóa confidence_threshold – tương thích src/nlu/classifier.py hiện tại).
"""

# Cell 1 – Cài đặt thư viện (chỉ cần trên Colab)
!pip install -q underthesea tensorflow scikit-learn matplotlib seaborn
print('Cài đặt hoàn tất')

# Cell 2 – Upload dữ liệu (Colab) hoặc tìm file cục bộ
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print('Upload intents.csv và out_of_scope.csv')
    uploaded = files.upload()
    print('Files uploaded:', list(uploaded.keys()))
else:
    print('Không chạy trên Colab – tìm intents.csv / out_of_scope.csv ở thư mục hiện tại hoặc data/')

# Cell 3 – Import, seed và hàm tiền xử lý (MỤC C)
import os
import re
import json
import glob
import pickle
import random
import shutil
import zipfile
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)  # cố định seed cho Python, NumPy và TensorFlow

# ---------------------------------------------------------------------------
# PHẦN NÀY CHÉP NGUYÊN VĂN SANG src/nlu/preprocess.py
# ---------------------------------------------------------------------------
# Chuẩn hóa vị trí dấu thanh kiểu cũ → kiểu mới: hòa → hoà, thủy → thuỷ.
# Chỉ đổi theo chiều này vì chiều ngược lại làm hỏng các từ như "hoàng".
_TONE_MAP = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_RE = re.compile("|".join(_TONE_MAP))

# Teencode / viết tắt – một từ. KHÔNG thêm "a", "b", "v" vì trùng tên khối thi (khối A, B, V).
TEENCODE = {
    # phủ định, khẳng định
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khum": "không",
    "đc": "được", "dc": "được",
    "ok": "được rồi", "oke": "được rồi", "okela": "được rồi",
    # đại từ, hư từ
    "mn": "mọi người", "mk": "mình", "mik": "mình", "e": "em", "ng": "người",
    "bk": "biết", "bik": "biết", "bít": "biết",
    "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnh": "bao nhiêu",
    "ntn": "như thế nào", "ntno": "như thế nào",
    "j": "gì", "z": "vậy", "vs": "với", "vd": "ví dụ",
    "cx": "cũng", "cg": "cũng", "r": "rồi", "rùi": "rồi", "oy": "rồi",
    "nha": "nhé", "nhe": "nhé",
    "nhg": "nhưng", "nhưg": "nhưng", "nhug": "nhưng", "trog": "trong",
    "ad": "admin", "info": "thông tin",
    # tuyển sinh, đào tạo
    "đk": "điều kiện", "dk": "điều kiện",
    "đkxt": "đăng ký xét tuyển", "xt": "xét tuyển",
    "nv": "nguyện vọng", "nv1": "nguyện vọng 1", "nv2": "nguyện vọng 2", "nv3": "nguyện vọng 3",
    "ts": "tuyển sinh", "sv": "sinh viên", "gv": "giáo viên", "hs": "học sinh",
    "đh": "đại học", "dh": "đại học", "cđ": "cao đẳng", "cd": "cao đẳng",
    "thpt": "trung học phổ thông", "thcs": "trung học cơ sở",
    "thptqg": "trung học phổ thông quốc gia",
    "đgnl": "đánh giá năng lực", "dgnl": "đánh giá năng lực",
    "tc": "tín chỉ", "hp": "học phí", "hb": "học bổng", "ktx": "ký túc xá",
    "ctdt": "chương trình đào tạo", "ctđt": "chương trình đào tạo",
    "cntt": "công nghệ thông tin", "sp": "sư phạm",
    "hcmue": "đại học sư phạm thành phố hồ chí minh",
    "sptp": "đại học sư phạm thành phố hồ chí minh",
    "đhsp": "đại học sư phạm", "dhsp": "đại học sư phạm",
    "tphcm": "thành phố hồ chí minh", "hcm": "thành phố hồ chí minh", "hn": "hà nội",
}
# Teencode hai từ (xét trước teencode một từ). "tp.hcm" sau khi bỏ dấu chấm thành "tp hcm".
TEENCODE_2 = {
    "tp hcm": "thành phố hồ chí minh",
    "đh sp": "đại học sư phạm",
}

_SPECIAL_RE = re.compile(r"[^\w\s]|_")  # giữ chữ (có dấu), số, khoảng trắng; bỏ cả "_" người dùng gõ


def normalize(text: str) -> str:
    """Chuẩn hóa, CHƯA tách từ: NFC → chữ thường → vị trí dấu → bỏ ký tự đặc biệt → teencode.

    Giữ nguyên mã tổ hợp (a00, d01) và số. Chatbot dùng hàm này để trích thực thể.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TONE_RE.sub(lambda m: _TONE_MAP[m.group(0)], text)
    text = _SPECIAL_RE.sub(" ", text)
    tokens = text.split()
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            two = tokens[i] + " " + tokens[i + 1]
            if two in TEENCODE_2:
                out.append(TEENCODE_2[two])
                i += 2
                continue
        out.append(TEENCODE.get(tokens[i], tokens[i]))
        i += 1
    return " ".join(out)


USE_WORD_SEGMENT = True  # ghi vào config.json để lúc chạy chatbot biết có tách từ hay không
try:
    from underthesea import word_tokenize as _uts_word_tokenize
    WORD_SEGMENTER = "underthesea"
except ImportError:  # không có thư viện → không tách từ (phải ghi rõ trong báo cáo)
    _uts_word_tokenize = None
    WORD_SEGMENTER = "none"
    print('[cảnh báo] Chưa cài underthesea – bỏ qua bước tách từ')


def segment(text: str) -> str:
    """Tách từ, các âm tiết của một từ nối bằng '_' (vd: 'học_phí')."""
    if USE_WORD_SEGMENT and _uts_word_tokenize is not None and text:
        return _uts_word_tokenize(text, format="text")
    return text


def preprocess(text: str) -> str:
    """Pipeline đầy đủ cho mô hình: normalize() rồi tách từ."""
    return segment(normalize(text)).strip()


def preprocess_batch(texts):
    return [preprocess(t) for t in texts]
# ---------------------------------------------------------------------------
# HẾT PHẦN CHÉP SANG src/nlu/preprocess.py
# ---------------------------------------------------------------------------

print('Tự kiểm tra tiền xử lý:')
for s in ['Khối A gồm những môn nào?', 'hp 1 tc bn tiền vậy ad', 'đk nhận hb là gì',
          'Cho mk hỏi ktx có máy lạnh ko?', 'Ngành Giáo dục Hòa nhập, thủy sản', 'tp.hcm có cơ sở 2 ko']:
    print(f'  {s!r:42} → {preprocess(s)!r}')

# Cell 4 – Load dữ liệu, xóa trùng và EDA
def find_file(name):
    for p in (name, os.path.join('data', name), os.path.join('..', 'data', name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Không tìm thấy {name}')


def load_csv(name):
    d = pd.read_csv(find_file(name), encoding='utf-8-sig')  # utf-8-sig: bỏ ký tự BOM đầu file
    if 'label' not in d.columns and 'intent' in d.columns:  # chấp nhận cả cột "intent"
        d = d.rename(columns={'intent': 'label'})
    d = d.dropna(subset=['text']).copy()
    d['text'] = d['text'].astype(str).str.strip()
    d = d[d['text'] != '']
    d['text_clean'] = preprocess_batch(d['text'].tolist())
    return d


df = load_csv('intents.csv').dropna(subset=['label'])
df['label'] = df['label'].astype(str).str.strip()
n_raw = len(df)

# Câu giống hệt nhau sau tiền xử lý nhưng khác nhãn → cần nhóm xem lại
conflict = df.groupby('text_clean')['label'].nunique()
conflict = conflict[conflict > 1]
if len(conflict):
    print(f'[cảnh báo] {len(conflict)} câu trùng nhưng KHÁC nhãn – cần xem lại khi gán nhãn:')
    print(df[df['text_clean'].isin(conflict.index)][['text', 'label']].to_string())

df = df.drop_duplicates(subset='text_clean').reset_index(drop=True)
print('=== Thông tin dataset ===')
print(f'Tổng số mẫu: {n_raw} → sau khi xóa trùng: {len(df)} (bỏ {n_raw - len(df)})')
print(f'Số nhãn: {df["label"].nunique()}')
label_counts = df['label'].value_counts()
print(label_counts)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('Set2', n_colors=len(label_counts))
axes[0].bar(range(len(label_counts)), label_counts.values, color=colors)
axes[0].set_xticks(range(len(label_counts)))
axes[0].set_xticklabels(label_counts.index, rotation=45, ha='right')
axes[0].set_title('Phân phối nhãn trong dataset', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Nhãn ý định')
axes[0].set_ylabel('Số lượng mẫu')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Tỉ lệ phần trăm nhãn', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

df['n_token'] = df['text_clean'].str.split().str.len()
print('\nĐộ dài câu sau tiền xử lý (số token):')
print(df['n_token'].describe().round(2))

# Cell 5 – Chia dữ liệu 70/15/15 và tokenizer (MỤC D)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 5000
MAX_LEN = 50        # PHẢI trùng với MAX_LEN trong src/nlu/classifier.py (được ghi vào config.json)
EMBED_DIM = 64
TEST_SIZE = 0.15
VAL_SIZE = 0.15

le = LabelEncoder()
y_all = le.fit_transform(df['label'])
label_names = list(le.classes_)
num_classes = len(label_names)
print(f'Nhãn ({num_classes}): {label_names}')

idx = np.arange(len(df))
idx_trainval, idx_test = train_test_split(idx, test_size=TEST_SIZE, stratify=y_all,
                                          random_state=SEED)
idx_train, idx_val = train_test_split(idx_trainval, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                      stratify=y_all[idx_trainval], random_state=SEED)

X_train_text = df['text_clean'].values[idx_train]
X_val_text = df['text_clean'].values[idx_val]
X_test_text = df['text_clean'].values[idx_test]
y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
print(f'Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}')

# Câu ngoài phạm vi: một nửa để chọn τ (oos_val), một nửa để báo cáo (oos_test)
df_oos = load_csv('out_of_scope.csv').drop_duplicates(subset='text_clean').reset_index(drop=True)
df_oos = df_oos[~df_oos['text_clean'].isin(set(df['text_clean']))]  # bỏ câu trùng với dữ liệu trong phạm vi
oos_val_text, oos_test_text = train_test_split(df_oos['text_clean'].values, test_size=0.5,
                                               random_state=SEED)
print(f'Out-of-scope: {len(df_oos)} câu → OOS_val {len(oos_val_text)} | OOS_test {len(oos_test_text)}')

# Tokenizer Keras: chỉ học trên tập TRAIN; filters="" để giữ "_" của từ ghép
keras_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>', filters='', lower=False)
keras_tokenizer.fit_on_texts(X_train_text)
VOCAB_SIZE = min(MAX_WORDS, len(keras_tokenizer.word_index) + 1)


def to_seq(texts):
    return pad_sequences(keras_tokenizer.texts_to_sequences(list(texts)),
                         maxlen=MAX_LEN, padding='post', truncating='post')


X_train_seq, X_val_seq, X_test_seq = to_seq(X_train_text), to_seq(X_val_text), to_seq(X_test_text)

# Tất cả các tập dùng để đánh giá – mọi mô hình được chấm trên cùng các tập này
EVAL_TEXTS = {'val': X_val_text, 'test': X_test_text,
              'oos_val': oos_val_text, 'oos_test': oos_test_text}

print(f'Vocabulary (train): {len(keras_tokenizer.word_index)} | VOCAB_SIZE dùng: {VOCAB_SIZE}')
print(f'Tỉ lệ câu train dài hơn MAX_LEN={MAX_LEN}: '
      f'{np.mean([len(t.split()) > MAX_LEN for t in X_train_text]):.1%}')
print(f'X_train_seq shape: {X_train_seq.shape}')

# Cell 6 – Baseline Naive Bayes (TF-IDF + MultinomialNB)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

results = {}  # tên mô hình → kết quả; dùng cho mọi cell phía sau


def register_model(name, predict_proba):
    """Chấm một mô hình trên val/test/OOS và lưu vào results.

    predict_proba: hàm nhận danh sách câu ĐÃ preprocess, trả ma trận xác suất (n, num_classes).
    """
    proba = {k: np.asarray(predict_proba(v)) for k, v in EVAL_TEXTS.items()}
    pred_val = proba['val'].argmax(1)
    pred_test = proba['test'].argmax(1)
    report = classification_report(y_test, pred_test, labels=range(num_classes),
                                   target_names=label_names, output_dict=True, zero_division=0)
    results[name] = {
        'predict_proba': predict_proba,
        'proba': proba,
        'val_acc': accuracy_score(y_val, pred_val),
        'val_macro_f1': f1_score(y_val, pred_val, average='macro'),
        'test_acc': accuracy_score(y_test, pred_test),
        'test_macro_f1': f1_score(y_test, pred_test, average='macro'),
        'test_weighted_f1': f1_score(y_test, pred_test, average='weighted'),
        'report': report,
        'y_pred': pred_test,
    }
    r = results[name]
    print(f'\n--- {name}: val macro F1 {r["val_macro_f1"]:.4f} | '
          f'test acc {r["test_acc"]:.4f} | test macro F1 {r["test_macro_f1"]:.4f} ---')
    print(classification_report(y_test, pred_test, labels=range(num_classes),
                                target_names=label_names, digits=4, zero_division=0))


def build_nb(alpha):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_WORDS, sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w+\b')),  # giữ cả token 1 ký tự (số "1", "2")
        ('nb', MultinomialNB(alpha=alpha)),
    ])


print('=== Naive Bayes Baseline – chọn alpha trên tập val ===')
nb_scores = {}
for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
    p = build_nb(alpha).fit(X_train_text, y_train)
    nb_scores[alpha] = f1_score(y_val, p.predict(X_val_text), average='macro')
    print(f'  alpha={alpha:<5} val macro F1 = {nb_scores[alpha]:.4f}')
NB_ALPHA = max(nb_scores, key=nb_scores.get)
print(f'→ chọn alpha = {NB_ALPHA}')

nb_pipeline = build_nb(NB_ALPHA).fit(X_train_text, y_train)
register_model('NaiveBayes', lambda texts: nb_pipeline.predict_proba(list(texts)))

with open('nb_pipeline.pkl', 'wb') as f:
    pickle.dump(nb_pipeline, f)
print('Đã lưu: nb_pipeline.pkl')

# Cell 7 – Hàm huấn luyện chung cho Keras + Model LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, GRU, Dense, Dropout, Bidirectional,
                                     LayerNormalization, SpatialDropout1D,
                                     Conv1D, MaxPooling1D, GlobalMaxPooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

print('TensorFlow version:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))

histories = {}


def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title(f'{name} – Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True, alpha=.3)
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title(f'{name} – Loss'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_history.png', dpi=150)
    plt.show()


def train_keras(name, build_fn, epochs=60, batch_size=32):
    """Huấn luyện một mô hình Keras trên train, dừng sớm theo val_loss, lưu <name>_best.keras."""
    tf.keras.utils.set_random_seed(SEED)  # mỗi mô hình bắt đầu từ cùng một seed
    model = build_fn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, MAX_LEN))
    print(f'\n--- Kiến trúc {name} ---')
    model.summary()

    ckpt = f'{name.lower()}_best.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0),
        ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    history = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
    histories[name] = history
    plot_history(history, name)
    register_model(name, lambda texts, m=model: m.predict(to_seq(texts), verbose=0))
    print(f'Đã lưu: {ckpt}')
    return model


def build_lstm():
    # Dữ liệu nhỏ → mô hình nhỏ, dropout cao. mask_zero=True: LSTM bỏ qua phần đệm 0,
    # nên câu ngắn không bị "chìm" trong hàng chục token đệm.
    return Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])


model_lstm = train_keras('LSTM', build_lstm)

# Cell 7B – Model LSTM v2 (BiLSTM + BiGRU)
def build_lstm_v2():
    # Không dùng recurrent_dropout: nó tắt cuDNN nên chạy chậm trên GPU Colab.
    return Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
        LayerNormalization(),
        Bidirectional(GRU(32, dropout=0.3)),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])


model_v2 = train_keras('LSTM_v2', build_lstm_v2)

# Cell 8 – Model Conv1D
def build_conv1d():
    # Conv1D không hỗ trợ mask → không đặt mask_zero; MAX_LEN cố định nên phần đệm nhất quán
    # giữa huấn luyện và lúc chạy.
    return Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])


model_conv1d = train_keras('Conv1D', build_conv1d)

# Cell 9 – So sánh các mô hình (chọn bằng VAL, báo cáo trên TEST)
model_names = list(results)
summary_df = pd.DataFrame({
    m: {'Val macro F1': results[m]['val_macro_f1'],
        'Test accuracy': results[m]['test_acc'],
        'Test macro F1': results[m]['test_macro_f1'],
        'Test weighted F1': results[m]['test_weighted_f1']}
    for m in model_names}).T.round(4)
print('=== BẢNG TỔNG HỢP HIỆU SUẤT (cùng tập test) ===')
print(summary_df.to_string())

for metric, title in [('f1-score', 'F1'), ('precision', 'PRECISION'), ('recall', 'RECALL')]:
    t = pd.DataFrame({m: [results[m]['report'][l][metric] for l in label_names]
                      for m in model_names}, index=label_names).round(4)
    print(f'\n=== {title} TỪNG Ý ĐỊNH (tập test) ===')
    print(t.to_string())

# Ma trận nhầm lẫn của từng mô hình
n = len(model_names)
fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
axes = np.atleast_1d(axes)
for ax, m in zip(axes, model_names):
    cm = confusion_matrix(y_test, results[m]['y_pred'], labels=range(num_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_title(f'{m} (test macro F1 {results[m]["test_macro_f1"]:.3f})')
    ax.set_xlabel('Dự đoán'); ax.set_ylabel('Thực tế')
    ax.tick_params(axis='x', rotation=60); ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# Biểu đồ accuracy và macro F1
palette = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ['Test macro F1', 'Test accuracy']):
    vals = summary_df[col].values
    bars = ax.bar(model_names, vals, color=palette[:n], width=0.5)
    ax.set_ylim(0, 1.1); ax.set_title(col, fontsize=13, fontweight='bold')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
plt.suptitle('So sánh hiệu suất các mô hình (tập test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# F1 từng ý định
x = np.arange(num_classes)
width = 0.8 / n
fig, ax = plt.subplots(figsize=(15, 6))
for i, m in enumerate(model_names):
    scores = [results[m]['report'][l]['f1-score'] for l in label_names]
    ax.bar(x - 0.4 + width * (i + 0.5), scores, width, label=m, color=palette[i])
ax.set_xticks(x); ax.set_xticklabels(label_names, rotation=35, ha='right')
ax.set_ylabel('F1-score'); ax.set_ylim(0, 1.15); ax.grid(axis='y', linestyle='--', alpha=.5)
ax.set_title('F1-score từng ý định (tập test)', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('f1_per_intent_comparison.png', dpi=200)
plt.show()

# Cell 10 – Chọn ngưỡng tin cậy τ (MỤC E)
TAUS = np.round(np.arange(0.10, 0.995, 0.01), 2)


def threshold_rates(P_in, y_in, P_oos, tau):
    """Trả về (tỉ lệ trả lời ĐÚNG trong phạm vi, tỉ lệ có trả lời trong phạm vi, tỉ lệ CHẶN OOS)."""
    conf_in, pred_in = P_in.max(1), P_in.argmax(1)
    answered = conf_in >= tau
    return (float(np.mean(answered & (pred_in == y_in))),
            float(np.mean(answered)),
            float(np.mean(P_oos.max(1) < tau)))


def balance(correct, blocked):
    """Trung bình điều hòa: chỉ cao khi CẢ HAI tỉ lệ cùng cao."""
    return 2 * correct * blocked / (correct + blocked) if correct + blocked > 0 else 0.0


thr_rows, curves = [], {}
for m in model_names:
    P = results[m]['proba']
    rows = [(t, *threshold_rates(P['val'], y_val, P['oos_val'], t)) for t in TAUS]
    curve = pd.DataFrame(rows, columns=['tau', 'correct', 'answered', 'blocked'])
    curve['balance'] = [balance(c, b) for c, b in zip(curve['correct'], curve['blocked'])]
    curves[m] = curve
    best = curve.loc[curve['balance'].idxmax()]
    tau = float(best['tau'])
    c_t, a_t, b_t = threshold_rates(P['test'], y_test, P['oos_test'], tau)
    # Độ chính xác của câu trả lời: trong các câu bot CHỊU trả lời (cả trong và ngoài phạm vi),
    # bao nhiêu phần trăm là trả lời đúng ý định.
    n_ok = c_t * len(y_test)
    n_answered = a_t * len(y_test) + (1 - b_t) * len(oos_test_text)
    thr_rows.append({
        'Mô hình': m, 'τ': tau,
        'Val: đúng': best['correct'], 'Val: chặn OOS': best['blocked'], 'Val: cân bằng': best['balance'],
        'Test: đúng': c_t, 'Test: chặn OOS': b_t, 'Test: cân bằng': balance(c_t, b_t),
        'Test: độ chính xác câu trả lời': n_ok / n_answered if n_answered else 0.0,
    })
    results[m]['threshold'] = tau

thr_df = pd.DataFrame(thr_rows).set_index('Mô hình').round(4)
print('=== NGƯỠNG τ TỐI ƯU CHO TỪNG MÔ HÌNH (chọn trên val ∪ OOS_val, báo cáo trên test ∪ OOS_test) ===')
print(thr_df.to_string())

# Mô hình chính: macro F1 trên VAL cao nhất (không nhìn tập test); hòa thì lấy điểm cân bằng cao hơn
best_model_name = max(model_names, key=lambda m: (round(results[m]['val_macro_f1'], 4),
                                                  thr_df.loc[m, 'Val: cân bằng']))
best_thresh = results[best_model_name]['threshold']
print(f'\n→ Mô hình chính: {best_model_name} | τ = {best_thresh:.2f}')
print(f'  Trên test: trả lời đúng {thr_df.loc[best_model_name, "Test: đúng"]:.1%} câu trong phạm vi, '
      f'chặn {thr_df.loc[best_model_name, "Test: chặn OOS"]:.1%} câu ngoài phạm vi')

# Đồ thị: đường quét τ của mô hình chính + đường cân bằng của mọi mô hình + phân phối độ tin cậy
c = curves[best_model_name]
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(c['tau'], c['correct'], label='Trả lời đúng (trong phạm vi)', color='green', lw=2)
axes[0].plot(c['tau'], c['answered'], label='Có trả lời (trong phạm vi)', color='green', ls=':')
axes[0].plot(c['tau'], c['blocked'], label='Chặn được (ngoài phạm vi)', color='red', lw=2)
axes[0].plot(c['tau'], c['balance'], label='Cân bằng (TB điều hòa)', color='blue', ls='--')
axes[0].axvline(best_thresh, color='purple', ls=':', lw=2, label=f'τ = {best_thresh:.2f}')
axes[0].set_title(f'Quét τ – {best_model_name} (val ∪ OOS_val)', fontweight='bold')
axes[0].set_xlabel('τ'); axes[0].set_ylabel('Tỉ lệ'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

for i, m in enumerate(model_names):
    axes[1].plot(curves[m]['tau'], curves[m]['balance'], label=m, color=palette[i])
axes[1].set_title('Điểm cân bằng theo τ – mọi mô hình', fontweight='bold')
axes[1].set_xlabel('τ'); axes[1].legend(); axes[1].grid(alpha=.3)

P = results[best_model_name]['proba']
conf_in = np.concatenate([P['val'].max(1), P['test'].max(1)])
conf_oos = np.concatenate([P['oos_val'].max(1), P['oos_test'].max(1)])
axes[2].hist(conf_in, bins=20, range=(0, 1), alpha=.6, color='blue', label='Trong phạm vi')
axes[2].hist(conf_oos, bins=20, range=(0, 1), alpha=.6, color='red', label='Ngoài phạm vi')
axes[2].axvline(best_thresh, color='purple', ls='--', lw=2, label=f'τ = {best_thresh:.2f}')
axes[2].set_title(f'Phân phối độ tin cậy – {best_model_name}', fontweight='bold')
axes[2].set_xlabel('Xác suất lớn nhất'); axes[2].set_ylabel('Số câu'); axes[2].legend(); axes[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig('confidence_threshold.png', dpi=150)
plt.show()

# Câu ngoài phạm vi vẫn lọt qua ngưỡng – đưa vào phần "hạn chế" của báo cáo
P_oos_all = np.vstack([P['oos_val'], P['oos_test']])
oos_all = np.concatenate([oos_val_text, oos_test_text])
leak = np.where(P_oos_all.max(1) >= best_thresh)[0]
print(f'\nCâu ngoài phạm vi KHÔNG chặn được với τ={best_thresh:.2f}: {len(leak)}/{len(oos_all)}')
for i in leak[:15]:
    print(f'  {P_oos_all[i].max():.2f}  {label_names[P_oos_all[i].argmax()]:<22} {oos_all[i][:90]}')

# Cell 10B – Thử dự đoán như chatbot
def predict_intent(text, model_name=None, tau=None):
    model_name = model_name or best_model_name
    tau = results[model_name]['threshold'] if tau is None else tau
    proba = np.asarray(results[model_name]['predict_proba']([preprocess(text)]))[0]
    order = proba.argsort()[::-1][:3]
    top3 = [(label_names[i], round(float(proba[i]), 4)) for i in order]
    intent = top3[0][0] if top3[0][1] >= tau else 'chuyen_can_bo'
    return intent, top3


for s in ['Điểm chuẩn ngành CNTT năm nay là bao nhiêu?', 'Trường có ký túc xá không?',
          'Học phí mỗi kỳ là bao nhiêu tiền?', 'hp cntt bn', 'xin chào', 'cảm ơn ad nha',
          'Thời tiết hôm nay thế nào?', 'Tỉ giá đô la hôm nay?']:
    intent, top3 = predict_intent(s)
    print(f'{s!r:48} → {intent:<22} {top3}')

# Cell 11 – Lưu cấu hình, số liệu và tải mô hình về
labels_list = label_names
with open('labels.json', 'w', encoding='utf-8') as f:
    json.dump(labels_list, f, ensure_ascii=False, indent=2)
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(keras_tokenizer.to_json())

# Mô hình chính có thêm một bản tên cố định để chatbot nạp
best_file = 'nb_pipeline.pkl' if best_model_name == 'NaiveBayes' else f'{best_model_name.lower()}_best.keras'
if best_file.endswith('.keras'):
    shutil.copy(best_file, 'intent_model.keras')
    best_file = 'intent_model.keras'

config = {
    'threshold': float(best_thresh),
    'best_model': best_model_name,
    'model_file': best_file,
    'max_len': MAX_LEN,
    'padding': 'post',
    'use_word_segment': bool(USE_WORD_SEGMENT and WORD_SEGMENTER != 'none'),
    'word_segmenter': WORD_SEGMENTER,
    'labels': labels_list,
    'seed': SEED,
    'split': {'train': len(idx_train), 'val': len(idx_val), 'test': len(idx_test),
              'oos_val': len(oos_val_text), 'oos_test': len(oos_test_text)},
    'thresholds_per_model': {m: float(results[m]['threshold']) for m in model_names},
}
with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
# Tương thích với src/nlu/classifier.py hiện tại (đọc khóa confidence_threshold)
with open('threshold_config.json', 'w', encoding='utf-8') as f:
    json.dump({'confidence_threshold': float(best_thresh)}, f, indent=2)

# Toàn bộ số liệu để viết báo cáo
metrics = {
    'n_samples': {'raw': n_raw, 'dedup': len(df), 'oos': len(df_oos)},
    'nb_alpha': NB_ALPHA,
    'models': {m: {k: float(results[m][k]) for k in
                   ('val_acc', 'val_macro_f1', 'test_acc', 'test_macro_f1', 'test_weighted_f1', 'threshold')}
               for m in model_names},
    'report_test': {m: results[m]['report'] for m in model_names},
    'threshold_table': thr_df.reset_index().to_dict(orient='records'),
    'best_model': best_model_name,
}
with open('metrics_c3.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(config, ensure_ascii=False, indent=2))

model_files = ['intent_model.keras', 'lstm_best.keras', 'lstm_v2_best.keras', 'conv1d_best.keras',
               'nb_pipeline.pkl', 'tokenizer.json', 'labels.json', 'config.json',
               'threshold_config.json', 'metrics_c3.json']
with zipfile.ZipFile('models.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in model_files:
        if os.path.exists(fn):
            z.write(fn)
            print(f'   {fn}')
        else:
            print(f'   (bỏ qua) {fn} không tồn tại')

if IN_COLAB:
    files.download('models.zip')
print('Đã tạo models.zip – giải nén vào thư mục models/ của repo')

# Cell 12 – Tải hình về
png_files = sorted(glob.glob('*.png'))
print(f'Tìm thấy {len(png_files)} file PNG: {png_files}')
if png_files:
    with zipfile.ZipFile('images.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in png_files:
            z.write(fn)
    if IN_COLAB:
        files.download('images.zip')
    print('Đã tạo images.zip')
