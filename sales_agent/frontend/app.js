(function () {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const emptyState = document.getElementById("empty-state");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());

  const DOMAIN_COLORS = {
    product: "#4fd1c5",
    sales: "#7c6cff",
    support: "#f6ad55",
    profile: "#a78bfa",
  };

  /* ---------------- tabs ---------------- */
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      document.getElementById(btn.dataset.tab + "-view").classList.add("active");
      if (btn.dataset.tab === "compare" && !window.__compareLoaded) loadCompare();
    });
  });

  /* ---------------- chat ---------------- */
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;
    appendUser(text);
    inputEl.value = "";
    autoResize();
    hideEmpty();

    const typing = appendTyping();
    setBusy(true);
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ user_message: text, session_id: sessionId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      typing.remove();
      appendAssistant(data);
    } catch (err) {
      typing.remove();
      appendError(err.message || String(err));
    } finally {
      setBusy(false);
      inputEl.focus();
    }
  }

  function appendUser(text) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="avatar">🙂</div><div class="bubble"></div>`;
    el.querySelector(".bubble").textContent = text;
    messagesEl.appendChild(el);
    scrollBottom();
  }

  function appendTyping() {
    const el = document.createElement("div");
    el.className = "msg assistant typing";
    el.innerHTML = `<div class="avatar">◈</div><div class="bubble"><span></span><span></span><span></span></div>`;
    messagesEl.appendChild(el);
    scrollBottom();
    return el;
  }

  function appendAssistant(data) {
    const el = document.createElement("div");
    el.className = "msg assistant";
    el.innerHTML = `<div class="avatar">◈</div><div class="bubble"></div>`;
    const bubble = el.querySelector(".bubble");

    const text = document.createElement("div");
    text.className = "response-text";
    text.textContent = data.response || "(no response)";
    bubble.appendChild(text);

    bubble.appendChild(buildMeta(data));
    messagesEl.appendChild(el);
    scrollBottom();
  }

  function buildMeta(data) {
    const meta = document.createElement("div");
    meta.className = "meta";

    if (data.agent === "gateway_blocked") {
      meta.appendChild(badge("blocked", "blocked"));
    } else if (data.path) {
      meta.appendChild(badge(data.path, data.path));
    }

    if (data.agent && data.agent !== "gateway_blocked") {
      meta.appendChild(badge(data.agent.replace(/_/g, " "), "agent"));
    }

    (data.domains || []).forEach((d) => {
      const chip = document.createElement("span");
      chip.className = "chip-domain";
      chip.style.color = DOMAIN_COLORS[d] || "var(--muted)";
      chip.style.borderColor = DOMAIN_COLORS[d] || "var(--border)";
      chip.textContent = d;
      meta.appendChild(chip);
    });

    if (data.lead_score != null) {
      const lead = document.createElement("span");
      lead.className = "lead-score";
      lead.innerHTML = `
        <span class="score-pill">${data.lead_score}</span>
        <span class="score-bar"><span style="width:${data.lead_score}%"></span></span>
        <span style="color:var(--muted);font-weight:500">lead score</span>`;
      meta.appendChild(lead);
    }

    const sum = data.trace_summary || {};
    meta.appendChild(
      badge(`${sum.nodes || 0} nodes · ${sum.total_input_tokens || 0}in/${sum.total_output_tokens || 0}out tok · ${(sum.wallclock_ms || 0).toFixed(0)}ms wall`, "agent")
    );

    if (data.trace && data.trace.length) {
      const toggle = document.createElement("button");
      toggle.className = "trace-toggle";
      toggle.textContent = "▸ trace";
      toggle.addEventListener("click", () => {
        toggle.textContent = traceEl.classList.toggle("open") ? "▾ trace" : "▸ trace";
      });
      meta.appendChild(toggle);

      const traceEl = document.createElement("div");
      traceEl.className = "trace";
      const rows = data.trace
        .map(
          (t) =>
            `<div class="trace-row"><span class="dot"></span><span class="trace-node">${esc(t.node)}</span>
             <span class="trace-note">${esc(t.note || "")}</span>
             <span class="trace-meta">${t.input_tokens || 0}in/${t.output_tokens || 0}out · ${(t.latency_ms || 0).toFixed(1)}ms</span></div>`
        )
        .join("");
      traceEl.innerHTML = rows;
      bubble.appendChild(traceEl);
    }

    return meta;
  }

  function appendError(msg) {
    const el = document.createElement("div");
    el.className = "error-banner";
    el.textContent = "⚠ " + msg;
    messagesEl.appendChild(el);
    scrollBottom();
  }

  function badge(label, kind) {
    const b = document.createElement("span");
    b.className = "badge " + kind;
    b.textContent = label;
    return b;
  }

  function hideEmpty() {
    if (emptyState) emptyState.style.display = "none";
  }
  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  function setBusy(b) {
    sendBtn.disabled = b;
    inputEl.disabled = b;
  }
  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ---------------- compare ---------------- */
  async function loadCompare() {
    window.__compareLoaded = true;
    const tbody = document.querySelector("#compare-table tbody");
    const highlights = document.getElementById("compare-highlights");
    tbody.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">Loading…</td></tr>`;
    try {
      const res = await fetch("/compare");
      const data = await res.json();
      tbody.innerHTML = "";
      data.rows.forEach((r, i) => {
        const tr = document.createElement("tr");
        if (i === 0) tr.className = "highlight-row";
        tr.innerHTML = `
          <td>${esc(r.label)}</td>
          <td>${esc(r.path)}</td>
          <td>${r.nodes}</td>
          <td>${r.in_tok}</td>
          <td>${r.out_tok}</td>
          <td>${r.lat_ms.toFixed(1)}</td>`;
        tbody.appendChild(tr);
      });

      const s = data.savings;
      highlights.innerHTML = `
        <div class="hl-card"><div class="k">Token savings vs always-Supervisor</div>
          <div class="v good">${s.token_savings > 0 ? "−" : ""}${Math.abs(s.token_savings)}</div></div>
        <div class="hl-card"><div class="k">Node calls saved on a simple request</div>
          <div class="v good">${s.node_savings > 0 ? "−" : ""}${Math.abs(s.node_savings)}</div></div>
        <div class="hl-card"><div class="k">Parallel fan-out token delta</div>
          <div class="v neutral">${s.parallel_token_delta}</div></div>
        <div class="hl-card"><div class="k">Parallel vs sequential latency</div>
          <div class="v ${s.parallel_latency_delta_ms <= 0 ? "good" : "bad"}">${s.parallel_latency_delta_ms} ms</div></div>`;
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="6" style="color:var(--blocked)">Failed to load: ${esc(err.message)}</td></tr>`;
    }
  }

  /* ---------------- events ---------------- */
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  inputEl.addEventListener("input", autoResize);

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      inputEl.value = chip.textContent;
      autoResize();
      sendMessage();
    });
  });

  inputEl.focus();
})();