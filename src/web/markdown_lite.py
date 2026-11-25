"""Đổi câu trả lời dạng markdown của ChatEngine sang HTML để hiển thị trên web.

Chỉ hỗ trợ đúng những gì engine sinh ra: đoạn văn, **đậm**, *nghiêng* / _nghiêng_, `mã`, [liên kết](http…),
danh sách "- " / "1. " (thụt 2 dấu cách = cấp 2), bảng "| … |", khối ``` … ```, xuống dòng cứng "  ".
Dòng bắt đầu bằng "Nguồn:" / "Lưu ý:" được gắn class riêng để làm nổi bật nguồn + năm áp dụng.

Mọi văn bản được escape TRƯỚC khi chèn thẻ → câu hỏi của người dùng không thể chèn HTML/JS.
"""
from __future__ import annotations

import html
import re

_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])[*_](?![\s*_])(.+?)(?<![\s*_])[*_](?![\w*])")
_OL = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_UL = re.compile(r"^(\s*)[-*]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def inline(text: str) -> str:
    """Định dạng trong dòng; text CHƯA escape."""
    codes: list[str] = []

    def giu_code(m: re.Match) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    t = _CODE.sub(giu_code, text)
    links: list[str] = []

    def giu_link(m: re.Match) -> str:
        links.append(f'<a href="{html.escape(m.group(2), quote=True)}" target="_blank" rel="noopener noreferrer">'
                     f"{html.escape(m.group(1))}</a>")
        return f"\x01{len(links) - 1}\x01"

    t = _LINK.sub(giu_link, t)
    t = html.escape(t, quote=False)
    t = _BOLD.sub(r"<strong>\1</strong>", t)
    t = _ITAL.sub(r"<em>\1</em>", t)
    t = re.sub(r"\x01(\d+)\x01", lambda m: links[int(m.group(1))], t)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], t)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(rows: list[str]) -> str:
    head, body = _cells(rows[0]), [_cells(r) for r in rows[2:]]
    out = ['<div class="table-wrap"><table><thead><tr>']
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _list(lines: list[str]) -> str:
    """Danh sách (có cấp 2 khi thụt ≥ 2 dấu cách)."""
    out: list[str] = []
    stack: list[tuple[int, str]] = []          # (độ thụt, thẻ)
    for ln in lines:
        m = _OL.match(ln) or _UL.match(ln)
        indent = len(m.group(1))
        tag = "ol" if _OL.match(ln) else "ul"
        content = m.group(3) if tag == "ol" else m.group(2)
        while stack and indent < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or indent > stack[-1][0]:
            out.append(f"<{tag}>")
            stack.append((indent, tag))
        else:
            out.append("</li>")
        out.append(f"<li>{inline(content)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def _paragraph(lines: list[str]) -> str:
    """Đoạn văn; tách riêng các dòng "Nguồn:" / "Lưu ý:" để tô khác."""
    out, buf = [], []

    def flush():
        if buf:
            out.append("<p>" + "<br>".join(inline(x.rstrip()) for x in buf) + "</p>")
            buf.clear()

    for ln in lines:
        s = ln.strip()
        if s.startswith("Nguồn:") or s.startswith("📌"):
            flush()
            out.append(f'<p class="source">{inline(s)}</p>')
        elif s.startswith("Lưu ý:") or s.startswith("⚠"):
            flush()
            out.append(f'<p class="note">{inline(s)}</p>')
        else:
            buf.append(ln)
    flush()
    return "".join(out)


def to_html(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.strip().startswith("```"):                      # khối mã
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(_table(lines[i:j]))
            i = j
            continue
        if _OL.match(ln) or _UL.match(ln):
            j = i
            while j < len(lines) and lines[j].strip() and (_OL.match(lines[j]) or _UL.match(lines[j])):
                j += 1
            out.append(_list(lines[i:j]))
            i = j
            continue
        j = i
        while (j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("```")
               and not (_OL.match(lines[j]) or _UL.match(lines[j]))
               and not (lines[j].lstrip().startswith("|") and j + 1 < len(lines)
                        and _TABLE_SEP.match(lines[j + 1].strip()))):
            j += 1
        out.append(_paragraph(lines[i:j]))
        i = j
    return "".join(out)
