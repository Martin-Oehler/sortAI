"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let queueItems = [];   // from /api/queue
let logEntries = [];   // from /api/log
let focusedId = null;  // currently focused row id

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  await Promise.all([fetchQueue(), fetchLog()]);
  renderAll();
  connectSSE();
}

// ── Data fetching ──────────────────────────────────────────────────────────
async function fetchQueue() {
  try {
    const r = await fetch('/api/queue');
    queueItems = await r.json();
  } catch (e) { console.error('fetchQueue', e); }
}

async function fetchLog() {
  try {
    const r = await fetch('/api/log');
    logEntries = await r.json();
  } catch (e) { console.error('fetchLog', e); }
}

// ── SSE ────────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/api/events');
  es.addEventListener('queue_updated', async () => {
    await fetchQueue();
    renderAll();
  });
  es.addEventListener('log_updated', async () => {
    await fetchLog();
    renderHistory();
  });
  es.onopen = () => document.getElementById('conn-dot').classList.add('connected');
  es.onerror = () => {
    document.getElementById('conn-dot').classList.remove('connected');
    setTimeout(connectSSE, 3000);
    es.close();
  };
}

// ── Render ─────────────────────────────────────────────────────────────────
function renderAll() {
  renderReview();
  renderHistory();
}

function renderReview() {
  const reviewItems = queueItems.filter(i => i.status === 'pending' || i.status === 'reprocessing');
  const pendingCount = queueItems.filter(i => i.status === 'pending').length;
  const badge = document.getElementById('pending-badge');
  badge.textContent = 'Needs Review: ' + pendingCount;
  badge.classList.toggle('visible', pendingCount > 0);

  const list = document.getElementById('review-list');
  if (reviewItems.length === 0) {
    list.innerHTML = '<div id="review-empty">No items awaiting review.</div>';
    return;
  }
  list.innerHTML = reviewItems.map(item => {
    const isReprocessing = item.status === 'reprocessing';
    return `
    <div class="row${focusedId === item.id ? ' focused' : ''}${isReprocessing ? ' reprocessing' : ''}"
         id="row-${item.id}"
         onclick="selectRow('${item.id}', 'queue')">
      ${isReprocessing ? '<span class="spinner"></span>' : '<span class="row-icon">📄</span>'}
      <div class="row-body">
        <div class="row-filename" title="${esc(item.original_filename)}">${esc(item.original_filename)}</div>
        <div class="row-dest">→ ${esc(item.proposed_folder)}/${esc(item.proposed_filename)}</div>
        ${isReprocessing
          ? '<span class="reprocessing-label">Re-processing…</span>'
          : `<div class="row-actions">
          <button class="btn-accept" onclick="event.stopPropagation();acceptItem('${item.id}')">Accept ✓</button>
          <button class="btn-reject" onclick="event.stopPropagation();rejectItem('${item.id}')">Reject ✗</button>
        </div>`}
      </div>
      <button class="row-menu-btn" title="Options" onclick="event.stopPropagation();openContextMenu(event,'${item.id}','queue',null)">⋯</button>
    </div>`;
  }).join('');
}

function renderHistory() {
  const filter = document.getElementById('filter-select').value;

  // Build merged history: rejected queue items + all log entries
  const rejected = queueItems
    .filter(i => i.status === 'rejected')
    .map(i => ({
      id: i.id,
      type: 'queue',
      timestamp: i.timestamp,
      filename: i.original_filename,
      dest: '_rejected/' + i.original_filename,
      status: 'rejected',
      summary: i.summary,
    }));

  const logged = logEntries.map((e, idx) => ({
    id: 'log-' + idx,
    type: 'log',
    logIdx: idx,
    timestamp: e.timestamp || '',
    filename: basename(e.original_path || ''),
    dest: e.error ? '' : relPath(e.new_path || '', e.archive_root || ''),
    status: e.error ? 'error' : 'accepted',
    summary: e.error ? (e.error_reason || '') : (e.summary || ''),
  }));

  const all = [...rejected, ...logged].sort((a, b) => b.timestamp.localeCompare(a.timestamp));

  const filtered = filter === 'all' ? all : all.filter(r => r.status === filter);

  const list = document.getElementById('history-list');
  if (filtered.length === 0) {
    list.innerHTML = '<div style="padding:0.6rem 0.75rem;color:#888;font-style:italic;font-size:0.8rem">No entries.</div>';
    return;
  }

  list.innerHTML = filtered.map(r => {
    const icon = r.status === 'accepted' ? '✓' : r.status === 'rejected' ? '✗' : '⚠';
    const iconColor = r.status === 'accepted' ? '#27ae60' : r.status === 'rejected' ? '#e74c3c' : '#e67e22';
    const ts = (r.timestamp || '').slice(0, 16).replace('T', ' ');
    const fileUrl = r.type === 'queue' ? `/files/queue/${r.id}` : `/files/log/${r.logIdx}`;
    return `<div class="row${focusedId === r.id ? ' focused' : ''}"
         id="row-${r.id}"
         onclick="selectRow('${r.id}', '${r.type}', '${fileUrl}')">
      <span class="row-icon" style="color:${iconColor}">${icon}</span>
      <div class="row-body">
        <div class="row-filename" title="${esc(r.filename)}">${esc(r.filename)}</div>
        <div class="row-dest">${r.dest ? esc(r.dest) : '<span style="color:#e67e22">'+esc(r.summary.slice(0,80))+'</span>'}</div>
        <div class="row-summary" style="color:#999">${esc(ts)}</div>
      </div>
      <button class="row-menu-btn" title="Options" onclick="event.stopPropagation();openContextMenu(event,'${r.id}','${r.type}',${r.type === 'log' ? r.logIdx : 'null'})">⋯</button>
    </div>`;
  }).join('');
}

// ── Actions ────────────────────────────────────────────────────────────────
function selectRow(id, type, fileUrl) {
  focusedId = id;
  renderAll();

  document.getElementById('llm-panel').classList.remove('visible');

  const url = fileUrl || (type === 'queue' ? `/files/queue/${id}` : null);
  if (!url) return;
  const frame = document.getElementById('pdf-frame');
  const hint = document.getElementById('preview-hint');
  frame.src = url;
  frame.classList.add('visible');
  hint.style.display = 'none';
}

// ── Context menu ───────────────────────────────────────────────────────────
function openContextMenu(event, id, type, logIdx) {
  const menu = document.getElementById('context-menu');
  const rect = event.currentTarget.getBoundingClientRect();
  menu.style.top = (rect.bottom + 4) + 'px';
  menu.style.left = rect.left + 'px';
  menu.dataset.id = id;
  menu.dataset.type = type;
  menu.dataset.logIdx = logIdx ?? '';
  menu.classList.add('visible');
  event.stopPropagation();
}

async function revealFromMenu() {
  const menu = document.getElementById('context-menu');
  const id = menu.dataset.id;
  const type = menu.dataset.type;
  const logIdx = menu.dataset.logIdx;
  menu.classList.remove('visible');
  try {
    const r = await fetch('/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, type, log_idx: logIdx !== '' ? parseInt(logIdx) : null })
    });
    if (!r.ok) { const e = await r.json(); showToast('Reveal failed: ' + (e.detail || r.status), true); }
  } catch (e) { showToast('Reveal error: ' + e, true); }
}

function retriggerFromMenu() {
  const menu = document.getElementById('context-menu');
  const overlay = document.getElementById('retrigger-overlay');
  overlay.dataset.id = menu.dataset.id;
  overlay.dataset.type = menu.dataset.type;
  overlay.dataset.logIdx = menu.dataset.logIdx;
  menu.classList.remove('visible');
  document.getElementById('retrigger-hint').value = '';
  overlay.classList.add('visible');
  document.getElementById('retrigger-hint').focus();
}

function retriggerCancel() {
  document.getElementById('retrigger-overlay').classList.remove('visible');
}

async function retriggerSubmit() {
  const overlay = document.getElementById('retrigger-overlay');
  const hint = document.getElementById('retrigger-hint').value.trim();
  const id = overlay.dataset.id;
  const type = overlay.dataset.type;
  const logIdx = overlay.dataset.logIdx;
  overlay.classList.remove('visible');
  try {
    const r = await fetch('/api/reprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, type, log_idx: logIdx !== '' ? parseInt(logIdx) : null, hint })
    });
    if (r.ok) {
      showToast('Re-processing started — the queue will update when done.');
    } else {
      const e = await r.json();
      showToast('Re-process failed: ' + (e.detail || r.status), true);
    }
  } catch (e) { showToast('Re-process error: ' + e, true); }
}

function inspectFromMenu() {
  const menu = document.getElementById('context-menu');
  const id = menu.dataset.id;
  const type = menu.dataset.type;
  const logIdx = menu.dataset.logIdx !== '' ? parseInt(menu.dataset.logIdx) : null;
  menu.classList.remove('visible');
  inspectInteractions(id, type, logIdx);
}

function inspectInteractions(id, type, logIdx) {
  let interactions = [];
  if (type === 'queue') {
    const item = queueItems.find(i => i.id === id);
    interactions = item?.interactions || [];
  } else {
    const entry = logEntries[logIdx];
    interactions = entry?.interactions || [];
  }
  const panel = document.getElementById('llm-panel');
  const frame = document.getElementById('pdf-frame');
  const hint = document.getElementById('preview-hint');
  hint.style.display = 'none';
  frame.classList.remove('visible');
  panel.innerHTML = renderInteractionsHtml(interactions);
  panel.classList.add('visible');
}

function renderInteractionsHtml(interactions) {
  if (!interactions || interactions.length === 0)
    return '<div class="llm-empty">No LLM interaction data available.</div>';

  const stageOrder = ['summarize', 'navigate', 'choose_filename'];
  const stageLabels = { summarize: 'Stage 1: Summarize', navigate: 'Stage 2: Navigate to Folder', choose_filename: 'Stage 3: Choose Filename' };
  const groups = {};
  for (const ix of interactions) {
    (groups[ix.stage] = groups[ix.stage] || []).push(ix);
  }
  const stages = stageOrder.filter(s => groups[s]).concat(
    Object.keys(groups).filter(s => !stageOrder.includes(s))
  );
  return stages.map(stage => {
    const label = stageLabels[stage] || stage;
    const steps = groups[stage];
    const stepsHtml = steps.map((ix, i) => `
      <div class="llm-step">
        ${steps.length > 1 ? `<div class="llm-step-label">Step ${ix.step ?? i + 1}</div>` : ''}
        ${ix.reasoning ? `<div class="llm-field-label">Reasoning</div><div class="llm-field-value">${esc(ix.reasoning)}</div>` : ''}
        <div class="llm-field-label">Answer</div>
        <div class="llm-field-value">${esc(ix.answer)}</div>
        <details class="llm-prompt-toggle">
          <summary>Show prompt</summary>
          <pre>${esc(ix.prompt)}</pre>
        </details>
      </div>`).join('');
    return `<div class="llm-stage"><div class="llm-stage-title">${esc(label)}</div>${stepsHtml}</div>`;
  }).join('');
}

document.addEventListener('click', () => {
  document.getElementById('context-menu').classList.remove('visible');
});

async function acceptItem(id) {
  try {
    const r = await fetch(`/api/accept/${id}`, { method: 'POST' });
    if (!r.ok) { const e = await r.json(); showToast('Accept failed: ' + (e.detail || r.status), true); }
  } catch (e) { showToast('Accept error: ' + e, true); }
  await fetchQueue();
  renderAll();
}

async function rejectItem(id) {
  try {
    const r = await fetch(`/api/reject/${id}`, { method: 'POST' });
    if (!r.ok) { const e = await r.json(); showToast('Reject failed: ' + (e.detail || r.status), true); }
  } catch (e) { showToast('Reject error: ' + e, true); }
  await fetchQueue();
  renderAll();
}

async function acceptAll() {
  const pending = queueItems.filter(i => i.status === 'pending');
  for (const item of pending) {
    await fetch(`/api/accept/${item.id}`, { method: 'POST' });
  }
  await fetchQueue();
  await fetchLog();
  renderAll();
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
function allFocusableRows() {
  const pending = queueItems.filter(i => i.status === 'pending');
  const filter = document.getElementById('filter-select').value;
  const rejected = queueItems.filter(i => i.status === 'rejected')
    .map(i => ({ id: i.id, type: 'queue', timestamp: i.timestamp }));
  const logged = logEntries.map((e, idx) => ({
    id: 'log-' + idx,
    type: 'log',
    logIdx: idx,
    timestamp: e.timestamp || '',
    status: e.error ? 'error' : 'accepted',
  }));
  const history = [...rejected, ...logged].sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  const filteredHistory = filter === 'all' ? history : history.filter(r => r.status === filter);
  return [...pending.map(i => ({ id: i.id, type: 'queue' })), ...filteredHistory];
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    document.getElementById('retrigger-overlay').classList.remove('visible');
    document.getElementById('context-menu').classList.remove('visible');
    return;
  }
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'BUTTON' || e.target.tagName === 'TEXTAREA') return;
  const rows = allFocusableRows();
  if (rows.length === 0) return;
  const idx = focusedId ? rows.findIndex(r => r.id === focusedId) : -1;

  if (e.key === 'j' || e.key === 'J') {
    const next = rows[Math.min(idx + 1, rows.length - 1)];
    selectRow(next.id, next.type);
    scrollToRow(next.id);
    e.preventDefault();
  } else if (e.key === 'k' || e.key === 'K') {
    const prev = rows[Math.max(idx - 1, 0)];
    selectRow(prev.id, prev.type);
    scrollToRow(prev.id);
    e.preventDefault();
  } else if ((e.key === 'y' || e.key === 'Y') && focusedId) {
    const item = queueItems.find(i => i.id === focusedId && i.status === 'pending');
    if (item) acceptItem(focusedId);
  } else if ((e.key === 'n' || e.key === 'N') && focusedId) {
    const item = queueItems.find(i => i.id === focusedId && i.status === 'pending');
    if (item) rejectItem(focusedId);
  }
});

function scrollToRow(id) {
  const el = document.getElementById('row-' + id);
  if (el) el.scrollIntoView({ block: 'nearest' });
}

// ── Utilities ──────────────────────────────────────────────────────────────
function showToast(message, isError = false) {
  const t = document.getElementById('toast');
  t.textContent = message;
  t.classList.toggle('error', isError);
  t.classList.add('visible');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('visible'), 3000);
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function basename(p) {
  return p.replace(/\\/g, '/').split('/').pop() || p;
}

function relPath(newPath, archiveRoot) {
  if (!newPath) return '';
  const np = newPath.replace(/\\/g, '/');
  const ar = (archiveRoot || '').replace(/\\/g, '/').replace(/\/?$/, '/');
  if (ar && np.startsWith(ar)) return np.slice(ar.length);
  return basename(np);
}

// ── Draggable divider ──────────────────────────────────────────────────────
(function () {
  const divider = document.getElementById('divider');
  const left = document.getElementById('left');

  const saved = localStorage.getItem('leftPanelWidth');
  if (saved) left.style.width = saved + 'px';

  divider.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = left.offsetWidth;
    divider.classList.add('dragging');

    function onMove(e) {
      const w = Math.min(Math.max(startWidth + e.clientX - startX, 200), window.innerWidth * 0.7);
      left.style.width = w + 'px';
    }

    function onUp() {
      divider.classList.remove('dragging');
      localStorage.setItem('leftPanelWidth', left.offsetWidth);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
})();

// ── Boot ───────────────────────────────────────────────────────────────────
init();
