"""Chặn câu hỏi về NGÀNH / TRƯỜNG KHÁC bằng thực thể (bổ sung cho ngưỡng τ của C3).

Ngưỡng τ không chặn được câu như "điểm chuẩn ngành Sư phạm Toán": mô hình nhận đúng ý định
diem_chuan với độ tin cậy ~1.0, nhưng cơ sở tri thức chỉ có 3 ngành của Khoa CNTT. Nếu không chặn,
chatbot sẽ trả lời bằng số liệu của ngành CNTT, tức là trả lời SAI.

Luật (trên câu đã normalize()):
  * "ngành <X>" mà X không phải ngành của Khoa và không phải từ chung ("ngành nào", "ngành học"…)
    → ngành khác, chỉ chặn khi câu KHÔNG nhắc ngành nào của Khoa;
  * "trường / đại học <tên trường khác>" → trường khác, chặn kể cả khi có nhắc ngành của Khoa
    ("điểm chuẩn CNTT của Bách khoa" không được trả lời bằng số liệu của HCMUE).
"""
from __future__ import annotations

import re

from ..nlu.preprocess import normalize

# Từ đứng ngay sau "ngành" nhưng không phải tên ngành
_TU_CHUNG = set("""
nào gì này đó ấy kia khác học của em mình bên trường năm thì có là mà ạ vậy nhé nha không hot phù như
được hoặc và sang cũ ở bắt liên đại nên vừa trong để tại mới đang sẽ đã cho với hay hợp tốt nghề chính
phụ đào tuyển xét thứ yêu thích muốn lấy các những một hai ba ra vào dự đúng nữa luôn thế sao
giúp đi ơi hả nhỉ chứ chưa rồi vì nhưng khi nếu từ đến tới bao mấy theo về trên dưới cùng hiện sắp vô lên
tiếp thêm đây đấy nhiều ít rất quá lắm hơn nhất ổn dễ khó giỏi gần xa nặng nhẹ kiểu loại ngoài nằm thuộc
gồm tên mã nhóm đó bạn anh chị cô thầy tôi tụi chúng hiểu biết hỏi xin hướng
""".split())

# Ngành của Khoa có tên bắt đầu bằng một từ chung với ngành khác: chỉ nhận khi đi kèm đúng đuôi
_DAU_NGANH = {
    ("công", "nghệ"): {("thông", "tin"), ("giáo", "dục")},   # công nghệ sinh học, công nghệ thực phẩm… là ngành khác
    ("sư", "phạm"): {("tin",)},                              # sư phạm toán, sư phạm ngữ văn… là ngành khác
}

# Trường khác hay bị hỏi (đã normalize: chữ thường, có dấu)
TRUONG_KHAC = (
    "bách khoa", "y dược", "y khoa", "kinh tế", "luật", "ngoại thương", "nguyễn tất thành", "sài gòn",
    "tôn đức thắng", "văn lang", "hoa sen", "fpt", "rmit", "công nghiệp", "khoa học tự nhiên",
    "khoa học xã hội", "nhân văn", "quốc gia", "quốc tế", "mở", "ngân hàng", "tài chính", "kiến trúc",
    "nông lâm", "giao thông vận tải", "sư phạm kỹ thuật", "sư phạm hà nội", "sư phạm huế",
    "sư phạm đà nẵng", "cần thơ", "đà nẵng", "huế", "công nghệ thông tin", "hutech", "uit", "uef",
    "văn hiến", "hồng bàng", "gia định", "lạc hồng", "thủ dầu một", "an giang", "duy tân",
)
# Tên riêng đủ đặc trưng, nhận cả khi không có chữ "trường/đại học" đứng trước
TRUONG_KHAC_RIENG = ("bách khoa", "y dược", "ngoại thương", "nguyễn tất thành", "tôn đức thắng", "văn lang",
                     "hoa sen", "fpt", "rmit", "hutech", "uit", "uef", "sư phạm kỹ thuật", "sư phạm hà nội")


def _alt(names) -> str:
    return "|".join(sorted(map(re.escape, names), key=len, reverse=True))


_TRUONG_RE = re.compile(r"\b(?:trường|đại học|đh)\s+(?:đại học\s+)?(" + _alt(TRUONG_KHAC) + r")\b"
                        r"|\b(" + _alt(TRUONG_KHAC_RIENG) + r")\b")
_NGANH_RE = re.compile(r"\bngành((?:\s+\S+){1,6})")  # tối đa 6 từ sau "ngành" (tên ngành dài nhất)


def _nganh_cua_khoa(majors: dict) -> list[tuple[str, ...]]:
    ten = set()
    for info in majors.values():
        for a in list(info.get("aliases", [])) + [info["ten"]]:
            n = normalize(a)
            if n.startswith("ngành "):
                continue
            ten.add(tuple(n.split()))
    return sorted(ten, key=len, reverse=True)


def _cum_ten(w: tuple[str, ...]) -> str:
    """Tối đa 4 từ, dừng ở từ chung đầu tiên: ('tâm','lý','học','tại') → 'tâm lý học'."""
    out = [w[0]]
    for x in w[1:4]:
        if x in _TU_CHUNG and x != "học":   # giữ "học" trong "tâm lý học", "công nghệ sinh học"
            break
        out.append(x)
    return " ".join(out)


def nganh_ngoai_khoa(t: str, majors: dict) -> tuple[str, tuple[int, int]] | None:
    """(cụm từ ngành lạ đầu tiên, vị trí trong t), vd ('sư phạm toán', (14, 31)); không có thì None."""
    cua_khoa = _nganh_cua_khoa(majors)
    for m in _NGANH_RE.finditer(t):
        w = tuple(m.group(1).split())
        if not w or w[0] in _TU_CHUNG or w[0].isdigit():
            continue
        if any(w[:len(a)] == a for a in cua_khoa):
            continue
        dau = w[:2]
        if dau in _DAU_NGANH:
            duoi = w[2:]
            if not duoi or duoi[0] in _TU_CHUNG or any(duoi[:len(d)] == d for d in _DAU_NGANH[dau]):
                continue          # "ngành sư phạm" / "ngành công nghệ" đứng một mình: coi là của Khoa
        return _cum_ten(w), m.span()
    return None


def truong_khac(t: str) -> str | None:
    m = _TRUONG_RE.search(t)
    return (m.group(1) or m.group(2)) if m else None


def kiem_tra_pham_vi(t: str, majors: dict, tim_nganh) -> tuple[str, str] | None:
    """(mã lý do, câu giải thích) nếu câu hỏi về trường/ngành ngoài phạm vi, ngược lại None.

    tim_nganh(câu) → danh sách ngành của Khoa nhắc trong câu. Ngành lạ chỉ bị chặn khi phần CÒN LẠI
    của câu (đã bỏ cụm ngành lạ) không nhắc ngành nào của Khoa – vì bí danh "ngành công nghệ" của
    CNTT cũng khớp trong "ngành công nghệ sinh học".
    """
    if (tr := truong_khac(t)):
        return "truong_khac", f"Câu hỏi có nhắc đến trường **{tr}** – mình chỉ có thông tin của Trường ĐHSP TP.HCM."
    if (hit := nganh_ngoai_khoa(t, majors)) and not tim_nganh(t[:hit[1][0]] + " " + t[hit[1][1]:]):
        ng = hit[0]
        ten = ", ".join(info["ten"] for info in majors.values())
        return "nganh_khac", (f"Ngành **{ng}** không thuộc phạm vi tư vấn của mình – mình chỉ có thông tin "
                              f"các ngành của Khoa CNTT: {ten}.")
    return None
