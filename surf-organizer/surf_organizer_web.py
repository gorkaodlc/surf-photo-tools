#!/usr/bin/env python3
"""
Surf Photo Organizer — Web UI (Flask + SSE + Face Detection)
Organiza fotos JPG + ARW de surf por surfista.

Uso:
    uv add flask opencv-contrib-python-headless
    uv run surf_organizer_web.py

Se abrirá en http://localhost:5050
"""

import base64
import json
import os
import shutil
import sys
import time
import threading
import webbrowser
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, render_template_string

app = Flask(__name__)

progress_queues: dict[str, Queue] = {}

# ── Detectores OpenCV ─────────────────────────────────────────────────
HOG_DETECTOR = cv2.HOGDescriptor()
HOG_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Surf Photo Organizer</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3347; --text: #e4e4e7; --text2: #9ca3af;
    --accent: #3b82f6; --accent-hover: #2563eb;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: var(--text2); font-size: 0.9rem; margin-bottom: 28px; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  .card h2 { font-size: 1.1rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .card h2 .icon { font-size: 1.3rem; }

  .field { margin-bottom: 14px; }
  .field label { display: block; font-size: 0.85rem; color: var(--text2); margin-bottom: 5px; }
  .input-row { display: flex; gap: 8px; }
  .input-row input { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 14px; color: var(--text); font-size: 0.9rem; outline: none; transition: border .2s; }
  .input-row input:focus { border-color: var(--accent); }
  .btn { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; font-size: 0.9rem;
    font-weight: 600; transition: all .15s; }
  .btn-browse { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-browse:hover { border-color: var(--accent); }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  .btn-danger { background: var(--red); color: #fff; }
  .btn-success { background: var(--green); color: #fff; }
  .btn-sm { padding: 6px 12px; font-size: 0.8rem; border-radius: 6px; }
  .actions { display: flex; gap: 10px; margin-top: 8px; }

  /* ── Preview ── */
  .preview-section { display: none; }
  .preview-section.visible { display: block; }
  .surfer-block { background: var(--surface2); border-radius: 10px; padding: 16px; margin-bottom: 12px; border: 1px solid var(--border); }
  .surfer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .surfer-header-left { display: flex; align-items: center; gap: 12px; }
  .surfer-avatar { width: 96px; height: 96px; border-radius: 50%; object-fit: cover; border: 3px solid var(--accent);
    background: var(--surface); display: flex; align-items: center; justify-content: center; font-size: 2.2rem; overflow: hidden; flex-shrink: 0;
    box-shadow: 0 0 16px #3b82f644; cursor: pointer; position: relative; transition: transform .15s, box-shadow .15s; }
  .surfer-avatar:hover { transform: scale(1.08); box-shadow: 0 0 24px #3b82f688; }
  .surfer-avatar img { width: 100%; height: 100%; object-fit: cover; }
  .surfer-avatar .avatar-hint { position: absolute; bottom: -2px; right: -2px; background: var(--accent); color: #fff;
    width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; border: 2px solid var(--surface2); pointer-events: none; }
  .surfer-avatar.loading { opacity: .5; pointer-events: none; }
  .surfer-avatar.loading::after { content: ""; position: absolute; width: 28px; height: 28px; border: 3px solid transparent;
    border-top-color: var(--accent); border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .avatar-counter { font-size: 0.7rem; color: var(--text2); margin-top: 2px; text-align: center; }
  .surfer-header h3 { font-size: 1rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .badge { font-size: 0.75rem; padding: 2px 10px; border-radius: 99px; background: var(--accent); color: #fff; }
  .badge.warn { background: var(--yellow); color: #000; }
  .photo-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; }
  .photo-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px;
    background: var(--surface); font-size: 0.85rem; border: 1px solid transparent; transition: border .15s; }
  .photo-item:hover { border-color: var(--border); }
  .photo-item input[type=checkbox] { accent-color: var(--accent); width: 16px; height: 16px; }
  .photo-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .raw-tag.yes { font-size: 0.7rem; padding: 1px 6px; border-radius: 4px; background: #16a34a33; color: var(--green); }
  .raw-tag.no { font-size: 0.7rem; padding: 1px 6px; border-radius: 4px; background: #ef444433; color: var(--red); }

  /* ── Progress ── */
  .progress-section { display: none; }
  .progress-section.visible { display: block; }
  .progress-bar-outer { width: 100%; height: 28px; background: var(--surface2); border-radius: 8px; overflow: hidden; margin-bottom: 12px; }
  .progress-bar-inner { height: 100%; background: linear-gradient(90deg, var(--accent), #6366f1); width: 0%;
    transition: width .3s; display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 600; color: #fff; min-width: 40px; }
  .log-box { background: #000; border-radius: 8px; padding: 14px; height: 260px; overflow-y: auto;
    font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.6; border: 1px solid var(--border); }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.info { color: var(--text2); }
  .log-line.ok { color: var(--green); }
  .log-line.warn { color: var(--yellow); }
  .log-line.err { color: var(--red); }
  .log-line.done { color: var(--accent); font-weight: 600; }
  .summary-box { margin-top: 16px; padding: 16px; background: var(--surface2); border-radius: 8px; display: none; }
  .summary-box.visible { display: block; }
  .summary-box h3 { margin-bottom: 8px; }

  /* ── Modal ── */
  .modal-bg { display: none; position: fixed; inset: 0; background: #000a; z-index: 100; align-items: center; justify-content: center; }
  .modal-bg.open { display: flex; }
  .modal { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; width: 520px;
    max-height: 70vh; display: flex; flex-direction: column; }
  .modal-header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .modal-header h3 { font-size: 1rem; }
  .modal-close { background: none; border: none; color: var(--text2); cursor: pointer; font-size: 1.3rem; }
  .modal-body { padding: 12px; overflow-y: auto; flex: 1; }
  .dir-current { padding: 8px 12px; font-size: 0.8rem; color: var(--text2); background: var(--surface2);
    border-radius: 6px; margin-bottom: 8px; word-break: break-all; }
  .dir-item { padding: 10px 14px; cursor: pointer; border-radius: 6px; font-size: 0.9rem;
    display: flex; align-items: center; gap: 8px; transition: background .1s; }
  .dir-item:hover { background: var(--surface2); }
  .modal-footer { padding: 12px 20px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; gap: 8px; }
  .new-folder-row { display: flex; gap: 8px; margin-bottom: 10px; }
  .new-folder-row input { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 12px; color: var(--text); font-size: 0.85rem; outline: none; }
  .new-folder-row input:focus { border-color: var(--accent); }
  .new-folder-row input::placeholder { color: var(--text2); }
</style>
</head>
<body>
<div class="container">
  <h1>🏄 Surf Photo Organizer</h1>
  <p class="subtitle">Organiza tus fotos JPG + ARW por surfista</p>

  <!-- STEP 1 -->
  <div class="card" id="step-paths">
    <h2><span class="icon">📂</span> Directorios</h2>
    <div class="field">
      <label>Carpeta de JPGs (con subcarpetas por surfista)</label>
      <div class="input-row">
        <input id="inp-jpg" type="text" placeholder="/ruta/a/jpg">
        <button class="btn btn-browse" onclick="openBrowser('inp-jpg')">Explorar</button>
      </div>
    </div>
    <div class="field">
      <label>Carpeta de RAWs (todos los .ARW sueltos)</label>
      <div class="input-row">
        <input id="inp-raw" type="text" placeholder="/ruta/a/raw">
        <button class="btn btn-browse" onclick="openBrowser('inp-raw')">Explorar</button>
      </div>
    </div>
    <div class="field">
      <label>Carpeta de salida</label>
      <div class="input-row">
        <input id="inp-out" type="text" placeholder="/ruta/a/salida">
        <button class="btn btn-browse" onclick="openBrowser('inp-out')">Explorar</button>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" id="btn-scan" onclick="scan()">🔍 Analizar</button>
    </div>
  </div>

  <!-- STEP 2 -->
  <div class="card preview-section" id="step-preview">
    <h2><span class="icon">📋</span> Vista previa</h2>
    <div id="preview-content"></div>
    <div class="actions" style="margin-top: 16px;">
      <button class="btn btn-success" id="btn-confirm" onclick="startCopy()">✅ Confirmar y copiar</button>
      <button class="btn btn-browse" onclick="hidePreview()">Cancelar</button>
    </div>
  </div>

  <!-- STEP 3 -->
  <div class="card progress-section" id="step-progress">
    <h2><span class="icon">⚙️</span> Progreso</h2>
    <div class="progress-bar-outer"><div class="progress-bar-inner" id="pbar">0%</div></div>
    <div class="log-box" id="log-box"></div>
    <div class="summary-box" id="summary-box">
      <h3>✅ Completado</h3>
      <div id="summary-content"></div>
    </div>
  </div>
</div>

<!-- Browse Modal -->
<div class="modal-bg" id="modal-browse">
  <div class="modal">
    <div class="modal-header">
      <h3>Seleccionar carpeta</h3>
      <button class="modal-close" onclick="closeBrowser()">✕</button>
    </div>
    <div class="modal-body">
      <div class="dir-current" id="browse-current"></div>
      <div class="new-folder-row">
        <input id="new-folder-name" type="text" placeholder="Nombre de nueva carpeta...">
        <button class="btn btn-sm btn-primary" onclick="createFolder()">➕ Crear</button>
      </div>
      <div id="browse-list"></div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-browse" onclick="closeBrowser()">Cancelar</button>
      <button class="btn btn-primary" onclick="selectDir()">Seleccionar esta carpeta</button>
    </div>
  </div>
</div>

<script>
let scanData = null;
let browseTarget = null;
let browseCurrent = "";
let avatarState = {};  // { surferName: { photos: [...], currentIdx: 0 } }

// ── Explorador ────────────────────────────────────────────
async function openBrowser(inputId) {
  browseTarget = inputId;
  const start = document.getElementById(inputId).value || (navigator.platform.includes("Win") ? "C:\\" : "/");
  await loadDir(start);
  document.getElementById("modal-browse").classList.add("open");
  document.getElementById("new-folder-name").value = "";
}
function closeBrowser() { document.getElementById("modal-browse").classList.remove("open"); }
function selectDir() {
  document.getElementById(browseTarget).value = browseCurrent;
  closeBrowser();
}
async function loadDir(dirPath) {
  const r = await fetch("/api/browse", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({path: dirPath}) });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }
  browseCurrent = data.current;
  document.getElementById("browse-current").textContent = data.current;
  const list = document.getElementById("browse-list");
  list.innerHTML = "";
  if (data.parent) {
    const el = document.createElement("div");
    el.className = "dir-item";
    el.innerHTML = "⬆️ ..";
    el.onclick = () => loadDir(data.parent);
    list.appendChild(el);
  }
  data.dirs.forEach(d => {
    const el = document.createElement("div");
    el.className = "dir-item";
    el.innerHTML = `📁 ${d.name}`;
    el.onclick = () => loadDir(d.path);
    list.appendChild(el);
  });
}
async function createFolder() {
  const name = document.getElementById("new-folder-name").value.trim();
  if (!name) { alert("Escribe un nombre para la carpeta"); return; }
  const r = await fetch("/api/mkdir", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({parent: browseCurrent, name: name}) });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById("new-folder-name").value = "";
  await loadDir(data.path);
}

// ── Escaneo ───────────────────────────────────────────────
async function scan() {
  const jpg = document.getElementById("inp-jpg").value.trim();
  const raw = document.getElementById("inp-raw").value.trim();
  const out = document.getElementById("inp-out").value.trim();
  if (!jpg || !raw || !out) { alert("Completa los tres directorios"); return; }

  document.getElementById("btn-scan").disabled = true;
  document.getElementById("btn-scan").textContent = "⏳ Analizando...";

  const r = await fetch("/api/scan", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({jpg_dir: jpg, raw_dir: raw, out_dir: out}) });
  const data = await r.json();

  document.getElementById("btn-scan").disabled = false;
  document.getElementById("btn-scan").textContent = "🔍 Analizar";

  if (data.error) { alert(data.error); return; }
  scanData = data;
  renderPreview(data);
  loadAvatars(data);
}

function renderPreview(data) {
  const container = document.getElementById("preview-content");
  container.innerHTML = "";
  data.surfers.forEach(s => {
    const block = document.createElement("div");
    block.className = "surfer-block";
    const noRaw = s.photos.filter(p => !p.has_raw).length;
    block.innerHTML = `
      <div class="surfer-header">
        <div class="surfer-header-left">
          <div>
            <div class="surfer-avatar" id="avatar-${s.name}" onclick="cycleAvatar('${s.name}')" title="Clic para cambiar foto">
              🏄
              <span class="avatar-hint">↻</span>
            </div>
            <div class="avatar-counter" id="avatar-counter-${s.name}"></div>
          </div>
          <h3>${s.name} <span class="badge">${s.photos.length} fotos</span>
            ${noRaw ? `<span class="badge warn">${noRaw} sin RAW</span>` : ""}</h3>
        </div>
        <div>
          <button class="btn btn-sm btn-browse" onclick="toggleAll('${s.name}', true)">Todos</button>
          <button class="btn btn-sm btn-browse" onclick="toggleAll('${s.name}', false)">Ninguno</button>
        </div>
      </div>
      <div class="photo-list" id="photos-${s.name}">
        ${s.photos.map(p => `
          <label class="photo-item">
            <input type="checkbox" checked data-surfer="${s.name}" data-photo="${p.jpg}" data-raw="${p.raw || ""}">
            <span class="name">${p.jpg}</span>
            <span class="raw-tag ${p.has_raw ? "yes" : "no"}">${p.has_raw ? "RAW ✔" : "sin RAW"}</span>
          </label>
        `).join("")}
      </div>`;
    container.appendChild(block);
  });
  document.getElementById("step-preview").classList.add("visible");
  document.getElementById("step-preview").scrollIntoView({behavior: "smooth"});
}

async function loadAvatars(data) {
  const jpg = document.getElementById("inp-jpg").value.trim();
  for (const s of data.surfers) {
    const photos = s.photos.map(p => p.jpg);
    avatarState[s.name] = { photos, currentIdx: 0 };
    await loadAvatarFrom(s.name, jpg, photos, 0);
  }
}

async function loadAvatarFrom(surferName, jpgDir, photos, startIdx) {
  const el = document.getElementById(`avatar-${surferName}`);
  const counter = document.getElementById(`avatar-counter-${surferName}`);
  if (!el) return;

  el.classList.add("loading");

  try {
    const r = await fetch("/api/face-thumbnail", { method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({jpg_dir: jpgDir, surfer: surferName, photos: photos, start_idx: startIdx}) });
    const result = await r.json();
    if (result.thumbnail) {
      el.innerHTML = `<img src="data:image/jpeg;base64,${result.thumbnail}" alt="${surferName}"><span class="avatar-hint">↻</span>`;
      avatarState[surferName].currentIdx = result.used_idx;
      if (counter) counter.textContent = `${result.used_idx + 1} / ${photos.length}`;
    } else {
      el.innerHTML = `🏄<span class="avatar-hint">↻</span>`;
      if (counter) counter.textContent = "sin detección";
    }
  } catch(e) {
    el.innerHTML = `🏄<span class="avatar-hint">↻</span>`;
  }
  el.classList.remove("loading");
}

async function cycleAvatar(surferName) {
  const state = avatarState[surferName];
  if (!state || state.photos.length <= 1) return;
  const nextIdx = (state.currentIdx + 1) % state.photos.length;
  const jpg = document.getElementById("inp-jpg").value.trim();
  await loadAvatarFrom(surferName, jpg, state.photos, nextIdx);
}

function toggleAll(surfer, state) {
  document.querySelectorAll(`input[data-surfer="${surfer}"]`).forEach(cb => cb.checked = state);
}
function hidePreview() { document.getElementById("step-preview").classList.remove("visible"); }

// ── Copia con SSE ─────────────────────────────────────────
function startCopy() {
  const selected = [];
  document.querySelectorAll("#preview-content input[type=checkbox]:checked").forEach(cb => {
    selected.push({ surfer: cb.dataset.surfer, jpg: cb.dataset.photo, raw: cb.dataset.raw || null });
  });
  if (!selected.length) { alert("No hay fotos seleccionadas"); return; }

  document.getElementById("btn-confirm").disabled = true;
  document.getElementById("step-progress").classList.add("visible");
  document.getElementById("step-progress").scrollIntoView({behavior: "smooth"});
  document.getElementById("summary-box").classList.remove("visible");
  document.getElementById("log-box").innerHTML = "";

  const jpg = document.getElementById("inp-jpg").value.trim();
  const raw = document.getElementById("inp-raw").value.trim();
  const out = document.getElementById("inp-out").value.trim();

  fetch("/api/copy", { method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({jpg_dir: jpg, raw_dir: raw, out_dir: out, selected}) })
    .then(r => r.json()).then(d => {
      if (d.stream_id) { listenProgress(d.stream_id); }
    });
}

function listenProgress(streamId) {
  const es = new EventSource(`/api/progress/${streamId}`);
  const logBox = document.getElementById("log-box");
  const pbar = document.getElementById("pbar");

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "log") {
      const line = document.createElement("div");
      line.className = `log-line ${msg.level || "info"}`;
      line.textContent = msg.text;
      logBox.appendChild(line);
      logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "progress") {
      pbar.style.width = msg.pct + "%";
      pbar.textContent = msg.pct + "%";
    } else if (msg.type === "done") {
      es.close();
      pbar.style.width = "100%";
      pbar.textContent = "100%";
      document.getElementById("btn-confirm").disabled = false;
      const box = document.getElementById("summary-box");
      document.getElementById("summary-content").innerHTML =
        `<p>📁 Surfistas: ${msg.surfers} · 📸 JPGs: ${msg.jpgs} · 🎞️ ARWs: ${msg.arws}</p>`;
      box.classList.add("visible");
    }
  };
}
</script>
</body>
</html>
"""


# ── API: Explorador de directorios ────────────────────────────────────
@app.route("/api/browse", methods=["POST"])
def api_browse():
    data = request.json
    target = data.get("path", "/")
    p = Path(target)
    if not p.is_dir():
        return jsonify(error=f"No es un directorio: {target}")
    try:
        dirs = sorted(
            [{"name": d.name, "path": str(d)} for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda x: x["name"].lower()
        )
    except PermissionError:
        return jsonify(error="Sin permisos para leer este directorio")
    parent = str(p.parent) if p != p.parent else None
    return jsonify(current=str(p), parent=parent, dirs=dirs)


# ── API: Crear carpeta ────────────────────────────────────────────────
@app.route("/api/mkdir", methods=["POST"])
def api_mkdir():
    data = request.json
    parent = Path(data.get("parent", "/"))
    name = data.get("name", "").strip()

    if not name:
        return jsonify(error="Nombre vacío")
    if any(c in name for c in ['/', '\\', '..', '\0']):
        return jsonify(error="Nombre de carpeta no válido")
    if not parent.is_dir():
        return jsonify(error=f"Directorio padre no existe: {parent}")

    new_dir = parent / name
    try:
        new_dir.mkdir(parents=False, exist_ok=True)
    except PermissionError:
        return jsonify(error="Sin permisos para crear carpeta aquí")
    except Exception as e:
        return jsonify(error=str(e))

    return jsonify(path=str(new_dir))


# ── API: Escaneo ──────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    raw_dir = Path(data["raw_dir"])

    if not jpg_dir.is_dir():
        return jsonify(error=f"Carpeta JPG no existe: {jpg_dir}")
    if not raw_dir.is_dir():
        return jsonify(error=f"Carpeta RAW no existe: {raw_dir}")

    raw_index = {}
    for f in raw_dir.iterdir():
        if f.suffix.lower() == ".arw":
            raw_index[f.stem.lower()] = f.name

    surfers = []
    for d in sorted(jpg_dir.iterdir()):
        if not d.is_dir():
            continue
        photos = []
        seen = set()
        for jpg in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if jpg.suffix.lower() != ".jpg":
                continue
            key = jpg.name.lower()
            if key in seen:
                continue
            seen.add(key)
            raw_name = raw_index.get(jpg.stem.lower())
            photos.append({"jpg": jpg.name, "raw": raw_name, "has_raw": raw_name is not None})
        if photos:
            surfers.append({"name": d.name, "photos": photos})

    return jsonify(surfers=surfers)


# ── API: Detección facial + thumbnail ─────────────────────────────────
def _make_square_crop(img, cx, cy, radius):
    """Recorta un cuadrado centrado en (cx, cy) con el radio dado."""
    h, w = img.shape[:2]
    radius = max(radius, 20)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    # Reajustar centro si tocamos bordes
    crop_w, crop_h = x2 - x1, y2 - y1
    side = min(crop_w, crop_h)
    x1 = max(0, x2 - side) if x1 == 0 else x1
    y1 = max(0, y2 - side) if y1 == 0 else y1
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)


def _try_hog_person(img, small, scale):
    """Intento 1: HOG full-body person detector."""
    boxes, weights = HOG_DETECTOR.detectMultiScale(
        small, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    if len(boxes) == 0:
        return None
    best_idx = int(np.argmax(weights))
    x, y, bw, bh = boxes[best_idx]
    # Centro del torso superior en coordenadas originales
    cx = int((x + bw / 2) / scale)
    cy = int((y + bh * 0.35) / scale)
    # Radio = mitad del ancho del cuerpo detectado (buen balance)
    radius = int((bw / scale) * 0.7)
    return _make_square_crop(img, cx, cy, radius)


def _try_saliency(img, small, scale):
    """Intento 2: Detección por saliencia — centrado en la región más llamativa."""
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = saliency.computeSaliency(small)
    if not ok:
        return None
    sal_map = (sal_map * 255).astype(np.uint8)
    _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    min_area = small.shape[0] * small.shape[1] * 0.005
    if cv2.contourArea(biggest) < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(biggest)
    # Centro de la región en coordenadas originales
    cx = int((x + bw / 2) / scale)
    cy = int((y + bh / 2) / scale)
    # Radio = lado mayor de la región (sin zoom extra)
    radius = int(max(bw, bh) / scale / 2)
    return _make_square_crop(img, cx, cy, radius)


def _center_crop(img):
    """Intento 3: Crop del tercio central de la imagen."""
    h, w = img.shape[:2]
    radius = min(h, w) // 6
    return _make_square_crop(img, w // 2, h // 2, radius)


@app.route("/api/face-thumbnail", methods=["POST"])
def api_face_thumbnail():
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    surfer = data["surfer"]
    photos = data.get("photos", [])
    start_idx = data.get("start_idx", 0)

    surfer_dir = jpg_dir / surfer
    if not surfer_dir.is_dir():
        return jsonify(thumbnail=None)

    n = len(photos)
    if n == 0:
        return jsonify(thumbnail=None)

    # Reordenar: empezar desde start_idx y recorrer circularmente
    ordered = [photos[(start_idx + i) % n] for i in range(n)]
    ordered_indices = [(start_idx + i) % n for i in range(n)]

    # Intento con HOG y saliencia
    for i, photo_name in enumerate(ordered):
        photo_path = surfer_dir / photo_name
        if not photo_path.exists():
            continue
        try:
            img = cv2.imread(str(photo_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            target_long = 640
            scale = target_long / max(h, w)
            if scale < 1:
                small = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                small = img
                scale = 1.0

            crop = _try_hog_person(img, small, scale)
            if crop is not None:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jsonify(thumbnail=base64.b64encode(buf).decode(),
                               source=photo_name, method="hog", used_idx=ordered_indices[i])

            crop = _try_saliency(img, small, scale)
            if crop is not None:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jsonify(thumbnail=base64.b64encode(buf).decode(),
                               source=photo_name, method="saliency", used_idx=ordered_indices[i])

        except Exception:
            continue

    # Fallback: center crop de la foto en start_idx
    photo_path = surfer_dir / photos[start_idx]
    if photo_path.exists():
        try:
            img = cv2.imread(str(photo_path))
            if img is not None:
                crop = _center_crop(img)
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jsonify(thumbnail=base64.b64encode(buf).decode(),
                               source=photos[start_idx], method="center", used_idx=start_idx)
        except Exception:
            pass

    return jsonify(thumbnail=None)


# ── API: Copiar ───────────────────────────────────────────────────────
@app.route("/api/copy", methods=["POST"])
def api_copy():
    data = request.json
    stream_id = f"s-{int(time.time()*1000)}"
    q = Queue()
    progress_queues[stream_id] = q
    thread = threading.Thread(target=do_copy, args=(data, q, stream_id), daemon=True)
    thread.start()
    return jsonify(stream_id=stream_id)


def do_copy(data, q: Queue, stream_id: str):
    jpg_dir = Path(data["jpg_dir"])
    raw_dir = Path(data["raw_dir"])
    out_dir = Path(data["out_dir"])
    selected = data["selected"]
    total = len(selected)
    stats = {"jpg": 0, "arw": 0, "surfers": set()}

    for i, item in enumerate(selected):
        surfer = item["surfer"]
        jpg_name = item["jpg"]
        raw_name = item.get("raw")

        dest_raw = out_dir / surfer / "raw"
        dest_jpg = out_dir / surfer / "jpg_originales"
        dest_edit = out_dir / surfer / "editadas"
        dest_raw.mkdir(parents=True, exist_ok=True)
        dest_jpg.mkdir(parents=True, exist_ok=True)
        dest_edit.mkdir(parents=True, exist_ok=True)

        stats["surfers"].add(surfer)

        src_jpg = jpg_dir / surfer / jpg_name
        if src_jpg.exists():
            shutil.copy2(src_jpg, dest_jpg / jpg_name)
            stats["jpg"] += 1
            q.put({"type": "log", "level": "ok", "text": f"✔ {surfer}/{jpg_name}"})
        else:
            q.put({"type": "log", "level": "err", "text": f"✘ JPG no encontrado: {surfer}/{jpg_name}"})

        if raw_name:
            src_raw = raw_dir / raw_name
            if src_raw.exists():
                shutil.copy2(src_raw, dest_raw / raw_name)
                stats["arw"] += 1
                q.put({"type": "log", "level": "ok", "text": f"✔ {surfer}/{raw_name}"})
            else:
                q.put({"type": "log", "level": "warn", "text": f"⚠ RAW no encontrado: {raw_name}"})
        else:
            q.put({"type": "log", "level": "warn", "text": f"⚠ Sin RAW para {jpg_name}"})

        pct = int(((i + 1) / total) * 100)
        q.put({"type": "progress", "pct": pct})

    q.put({"type": "done", "surfers": len(stats["surfers"]), "jpgs": stats["jpg"], "arws": stats["arw"]})
    q.put(None)
    time.sleep(5)
    progress_queues.pop(stream_id, None)


# ── SSE ───────────────────────────────────────────────────────────────
@app.route("/api/progress/<stream_id>")
def api_progress(stream_id):
    q = progress_queues.get(stream_id)
    if not q:
        return Response("data: {}\n\n", mimetype="text/event-stream")
    def generate():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n🏄 Surf Photo Organizer → http://localhost:{port}\n")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)