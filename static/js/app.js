/* HCMUE FIT Chat – mã giao diện. Mọi xử lý AI nằm ở máy chủ Python (app.py → src/). */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ------------------------------------------------------------------ trạng thái
  function storageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function storageSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* trình duyệt chặn lưu */ } }
  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
  }
  let clientId = storageGet("hcmue_fit_client");
  if (!clientId) { clientId = newId(); storageSet("hcmue_fit_client", clientId); }

  const state = { conversationId: null, busy: false };
  const column = $("#message-column");
  const scroller = $("#messages");
  const input = $("#input");
  const sendBtn = $("#send");
  const recentBox = $("#recent");
  const EMPTY_RECENT = recentBox.innerHTML;       // "Chưa có cuộc hội thoại nào" (kèm icon) trong template

  function now() {
    const d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function scrollBottom() { scroller.scrollTop = scroller.scrollHeight; }

  async function api(path, body) {
    const opt = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.assign({ client_id: clientId, conversation_id: state.conversationId }, body)),
    };
    const res = await fetch(path, opt);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ("Lỗi máy chủ " + res.status));
    return data;
  }

  // ------------------------------------------------------------------ hiển thị tin nhắn
  function addUser(text, time) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<p class="msg-meta">Bạn  ·  ${escapeHtml(time || now())}</p><div class="bubble"></div>`;
    $(".bubble", el).textContent = text;
    column.appendChild(el);
    scrollBottom();
  }

  function addBot(reply, opts = {}) {
    const el = document.createElement("div");
    el.className = "msg bot" + (reply.handoff ? " handoff" : "");
    // reply.html do máy chủ tạo từ markdown đã escape (src/web/markdown_lite.py)
    el.innerHTML = `<p class="msg-meta">Trợ lý Tuyển sinh  ·  ${escapeHtml(reply.time || now())}</p>
      <div class="bubble">${reply.html}</div>`;
    if (!opts.welcome) {
      const tools = document.createElement("div");
      tools.className = "msg-tools";
      tools.innerHTML = `<button class="fb-btn" type="button" data-v="tot">Hữu ích</button>
        <button class="fb-btn" type="button" data-v="chua_tot">Chưa đúng</button>`;
      $$(".fb-btn", tools).forEach(b => {
        if (reply.feedback === b.dataset.v) b.classList.add("chosen");
        b.addEventListener("click", () => sendFeedback(reply, b, tools));
      });
      el.appendChild(tools);
    }
    column.appendChild(el);
    scrollBottom();
  }

  async function sendFeedback(reply, btn, tools) {
    try {
      await api("/api/danh-gia", { index: reply.index, value: btn.dataset.v });
      $$(".fb-btn", tools).forEach(b => b.classList.toggle("chosen", b === btn));
      reply.feedback = btn.dataset.v;
    } catch (e) { /* bỏ qua: đánh giá không bắt buộc */ }
  }

  function showWelcome() {
    $$(".msg, .typing", column).forEach(n => n.remove());
    addBot({ html: $("#welcome").innerHTML, time: now() }, { welcome: true });
  }

  function typing(on) {
    let t = $(".typing", column);
    if (on && !t) {
      t = document.createElement("p");
      t.className = "typing";
      t.textContent = "Trợ lý Tuyển sinh đang soạn câu trả lời…";
      column.appendChild(t);
      scrollBottom();
    } else if (!on && t) t.remove();
  }

  function setBusy(b) {
    state.busy = b;
    sendBtn.disabled = b;
  }

  function afterReply(data, withUser) {
    const list = Array.isArray(data) ? data : [data];
    list.forEach(d => {
      state.conversationId = d.conversation.id;
      if (withUser) addUser(d.user.text, d.user.time);
      addBot(d.reply);
    });
    loadRecent();
  }

  async function ask(text) {
    text = (text || "").trim();
    if (!text || state.busy) return;
    closeMenu();
    addUser(text);
    input.value = "";
    autoSize();
    setBusy(true);
    typing(true);
    try {
      afterReply(await api("/api/chat", { message: text }));
    } catch (e) {
      addBot({ html: `<p class="note">Không gửi được câu hỏi: ${escapeHtml(e.message)}</p>`, time: now() },
        { welcome: true });
    } finally {
      typing(false);
      setBusy(false);
      input.focus();
    }
  }

  // ------------------------------------------------------------------ danh sách hội thoại
  function formatTime(iso) {
    const d = new Date(iso);
    if (isNaN(d)) return "";
    const today = new Date();
    if (d.toDateString() === today.toDateString()) return "Hôm nay";
    const y = new Date(today); y.setDate(today.getDate() - 1);
    if (d.toDateString() === y.toDateString()) return "Hôm qua";
    return d.toLocaleDateString("vi-VN");
  }

  async function loadRecent() {
    let list = [];
    try { list = await api("/api/hoi-thoai?client_id=" + encodeURIComponent(clientId)); } catch (e) { return; }
    const box = $("#recent");
    box.innerHTML = "";
    if (!list.length) {
      box.innerHTML = EMPTY_RECENT;
      return;
    }
    list.slice(0, 8).forEach(c => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "conv-item" + (c.id === state.conversationId ? " active" : "");
      b.innerHTML = '<span class="conv-title"></span><span class="conv-time"></span>';
      $(".conv-title", b).textContent = c.title;
      $(".conv-time", b).textContent = c.id === state.conversationId ? "Đang xem" : formatTime(c.updated);
      b.addEventListener("click", () => openConversation(c.id));
      box.appendChild(b);
    });
  }

  async function openConversation(id) {
    try {
      const c = await api(`/api/hoi-thoai/${encodeURIComponent(id)}?client_id=${encodeURIComponent(clientId)}`);
      state.conversationId = c.id;
      showWelcome();
      c.messages.forEach(m => m.role === "user" ? addUser(m.text, m.time) : addBot(m));
      loadRecent();
      closeMenu();
    } catch (e) { /* hội thoại đã hết hạn trên máy chủ */ loadRecent(); }
  }

  function newConversation() {
    state.conversationId = null;
    showWelcome();
    loadRecent();
    closeMenu();
    input.focus();
  }

  // ------------------------------------------------------------------ ô nhập
  function autoSize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 132) + "px";
  }
  input.addEventListener("input", autoSize);
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); }
  });
  $("#composer-form").addEventListener("submit", e => { e.preventDefault(); ask(input.value); });
  const mq = window.matchMedia("(max-width: 900px)");
  function placeholder() {
    input.placeholder = mq.matches ? "Nhập câu hỏi về tuyển sinh…" : "Nhập câu hỏi của bạn về tuyển sinh…";
  }
  (mq.addEventListener ? mq.addEventListener("change", placeholder) : mq.addListener(placeholder));
  placeholder();

  $$("[data-query]").forEach(b => b.addEventListener("click", () => ask(b.dataset.query)));

  // Dải câu hỏi gợi ý: lăn chuột dọc → cuộn ngang mượt
  const chips = $(".chips");
  if (chips) {
    chips.addEventListener("wheel", e => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX) && chips.scrollWidth > chips.clientWidth) {
        e.preventDefault();
        chips.scrollBy({ left: e.deltaY, behavior: "smooth" });
      }
    }, { passive: false });
  }
  $$('[data-action="new"]').forEach(b => b.addEventListener("click", newConversation));

  // ------------------------------------------------------------------ menu (điện thoại)
  const menuBtn = $("#menu-btn");
  const backdrop = $("#backdrop");
  function closeMenu() {
    document.body.classList.remove("menu-open");
    backdrop.hidden = true;
    menuBtn.setAttribute("aria-expanded", "false");
  }
  menuBtn.addEventListener("click", () => {
    const open = !document.body.classList.contains("menu-open");
    document.body.classList.toggle("menu-open", open);
    backdrop.hidden = !open;
    menuBtn.setAttribute("aria-expanded", String(open));
  });
  backdrop.addEventListener("click", closeMenu);

  // ------------------------------------------------------------------ biểu mẫu C1, C2
  $$("[data-dialog]").forEach(b => b.addEventListener("click", () => {
    closeMenu();
    const d = document.getElementById(b.dataset.dialog);
    $(".form-error", d).hidden = true;
    d.showModal();
    const body = $(".dialog-body", d);
    if (body) body.scrollTop = 0;
  }));
  // Đóng hộp thoại: nút ×, nút Đóng (method="dialog"), phím Esc (mặc định) hoặc bấm ra vùng tối bên ngoài.
  $$("dialog").forEach(d => {
    $$("[data-close]", d).forEach(b => b.addEventListener("click", () => d.close()));
    let downOutside = false;            // chỉ đóng khi cả nhấn và thả chuột đều ở ngoài (tránh kéo chọn chữ)
    const outside = e => {
      const r = d.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    d.addEventListener("pointerdown", e => { downOutside = e.target === d && outside(e); });
    d.addEventListener("click", e => {
      if (downOutside && e.target === d && outside(e)) d.close();
      downOutside = false;
    });
  });

  const formTV = $("#form-tu-van");
  function togglePt() {
    const pt = formTV.phuong_thuc.value;
    $$("[data-pt]", formTV).forEach(g => { g.hidden = g.dataset.pt !== pt; });
  }
  $$('input[name="phuong_thuc"]', formTV).forEach(r => r.addEventListener("change", togglePt));
  togglePt();

  async function submitForm(form, path, collect) {
    const err = $(".form-error", form);
    const btn = $("[data-submit]", form);
    err.hidden = true;
    btn.disabled = true;
    try {
      const data = await api(path, collect(form));
      form.closest("dialog").close();
      afterReply(data, true);
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
    } finally { btn.disabled = false; }
  }

  formTV.addEventListener("submit", e => {
    if (e.submitter && e.submitter.value === "cancel") return;
    e.preventDefault();
    submitForm(formTV, "/api/tu-van", f => {
      const diem = {};
      $$('input[name^="diem_"]', f).forEach(i => {
        if (i.name !== "diem_dgnl") diem[i.name.slice(5)] = i.value;
      });
      return {
        phuong_thuc: f.phuong_thuc.value, khu_vuc: f.khu_vuc.value, diem,
        diem_dgnl: f.diem_dgnl.value, hb1: f.hb1.value, mon_hb2: f.mon_hb2.value, hb2: f.hb2.value,
        so_thich: $$('input[name="so_thich"]:checked', f).map(i => i.value),
        dinh_huong: $$('input[name="dinh_huong"]:checked', f).map(i => i.value),
        kiem_tra_nganh: f.kiem_tra_nganh.value,
      };
    });
  });

  const formLT = $("#form-lo-trinh");
  $("[data-hk1]", formLT).addEventListener("click", () => {
    (window.HK1 || []).forEach(ma => { const s = $(`select[data-ma="${ma}"]`, formLT); if (s) s.value = "dat"; });
    if (Number(formLT.hoc_ky.value) < 2) formLT.hoc_ky.value = 2;
  });
  formLT.addEventListener("submit", e => {
    if (e.submitter && e.submitter.value === "cancel") return;
    e.preventDefault();
    submitForm(formLT, "/api/lo-trinh", f => ({
      hoc_ky: f.hoc_ky.value, hoc_vuot: f.hoc_vuot.checked,
      da_dat: $$("select[data-ma]", f).filter(s => s.value === "dat").map(s => s.dataset.ma),
      rot: $$("select[data-ma]", f).filter(s => s.value === "rot").map(s => s.dataset.ma),
    }));
  });

  // ------------------------------------------------------------------ khởi động
  showWelcome();
  loadRecent();
})();
