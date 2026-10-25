const $ = (sel) => document.querySelector(sel);
const MAX_IMAGES = 4;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal markdown: **bold**, bullets, headings, paragraphs.
function renderMarkdown(md) {
  const out = [];
  let inList = false;
  for (const raw of esc(md).split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^([-*•]|\d+\.)\s+(.*)$/);
    if (bullet) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${bullet[2]}</li>`);
      continue;
    }
    if (inList) { out.push("</ul>"); inList = false; }
    if (!line) continue;
    out.push(`<p>${line.replace(/^#+\s*/, "")}</p>`);
  }
  if (inList) out.push("</ul>");
  return out.join("").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

const DISCLAIMER = `<p class="legend">Bu analiz yerel ve küçük bir yapay zekâ modeliyle (Qwen2.5-VL) üretilmiştir; yüzde/olasılık içermez,
  hatalı bulgu içerebilir. Kesin tanı değildir. Kırmızı bayraklar (idrar/gaita kontrol bozukluğu, eyer anestezisi, ilerleyici güç kaybı)
  klinik olarak ayrıca değerlendirilmelidir.</p>`;

let guidelinesCache = null;
async function citedGuidelines(text) {
  if (!guidelinesCache) guidelinesCache = await (await fetch("/api/guidelines")).json();
  const ids = new Set([...text.matchAll(/\[([A-Z0-9-]+)\]/g)].map((m) => m[1]));
  return guidelinesCache.filter((g) => ids.has(g.id));
}

function startResults(files) {
  const thumbs = [...files].map((f) => `<img src="${URL.createObjectURL(f)}" alt="">`).join("");
  $("#results").innerHTML = `<div class="card summary"><div class="vision-imgs">${thumbs}</div>
    <div id="live"><p class="muted">Model resmi inceliyor…</p></div><p id="timing" class="legend"></p></div>
    <div id="guides"></div>${DISCLAIMER}`;
}

async function finishResults(text) {
  const used = await citedGuidelines(text);
  if (!used.length) return;
  $("#guides").innerHTML = `<div class="card"><details><summary>Dayanılan kılavuz maddeleri (${used.length})</summary><ul class="guides">${
    used.map((g) => `<li><b>[${esc(g.id)}]</b> ${esc(g.text)} <span class="muted">— ${esc(g.source)}</span></li>`).join("")
  }</ul></details></div>`;
}

function showError(msg) {
  const err = $("#error");
  err.textContent = msg;
  err.hidden = !msg;
}

$("#images").addEventListener("change", (e) => {
  const files = [...e.target.files];
  showError(files.length > MAX_IMAGES ? `En fazla ${MAX_IMAGES} resim seçebilirsiniz.` : "");
  $("#thumbs").innerHTML = "";
  files.slice(0, MAX_IMAGES).forEach((f) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    $("#thumbs").appendChild(img);
  });
});

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = $("#images").files;
  if (!files.length) return showError("Lütfen en az bir resim seçin.");
  if (files.length > MAX_IMAGES) return showError(`En fazla ${MAX_IMAGES} resim seçebilirsiniz.`);
  const btn = $("#submit");
  showError("");
  btn.disabled = true;
  const t0 = performance.now();
  let firstAt = null;
  const tick = setInterval(() => { btn.textContent = `Analiz ediliyor… ${((performance.now() - t0) / 1000).toFixed(0)} sn`; }, 250);
  startResults(files);
  let text = "";
  try {
    const resp = await fetch("/api/analyze-images", { method: "POST", body: new FormData(e.target) });
    if (!resp.ok) throw new Error((await resp.json()).detail || "Bilinmeyen hata");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (firstAt === null) firstAt = performance.now();
      text += decoder.decode(value, { stream: true });
      $("#live").innerHTML = renderMarkdown(text);
    }
    const total = (performance.now() - t0) / 1000;
    $("#timing").textContent = `İlk yanıt ${((firstAt - t0) / 1000).toFixed(1)} sn · toplam ${total.toFixed(1)} sn`;
    await finishResults(text);
  } catch (ex) {
    showError(ex.message);
  } finally {
    clearInterval(tick);
    btn.disabled = false;
    btn.textContent = "Analiz et";
  }
});
