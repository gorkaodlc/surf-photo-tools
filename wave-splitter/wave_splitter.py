#!/usr/bin/env python3
"""
Wave Splitter — Agrupa fotos de surf por ola/secuencia automáticamente.
Usa timestamps EXIF para detectar ráfagas y separarlas en carpetas.

Uso:
    uv add flask pillow opencv-contrib-python-headless ultralytics
    uv run wave_splitter.py

Se abrirá en http://127.0.0.1:5070
"""

import base64
import io
import json
import math
import os
import shutil
import socket
import subprocess
import platform
import sys
import time
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, render_template_string
from PIL import Image, ImageOps, ImageFilter
from PIL.ExifTags import Base as ExifBase
from ultralytics import YOLO

app = Flask(__name__)

progress_queues: dict[str, Queue] = {}

# ══════════════════════════════════════════════════════════════════════
# EXIF helpers
# ══════════════════════════════════════════════════════════════════════

def get_exif_datetime(filepath: Path) -> datetime | None:
    try:
        with Image.open(filepath) as img:
            exif = img.getexif()
            if not exif:
                return None
            for tag_id in (36867, 36868, 306):
                val = exif.get(tag_id)
                if val:
                    val = val.strip().split(".")[0]
                    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                        try:
                            return datetime.strptime(val, fmt)
                        except ValueError:
                            continue
    except Exception:
        pass
    return None


# Ruta al watermark PNG (sin fondo)
WATERMARK_PATH = Path(__file__).parent / "assets" / "watermark.png"
_watermark_cache: Image.Image | None = None


def _get_watermark() -> Image.Image | None:
    """Carga y cachea el watermark PNG con canal alpha."""
    global _watermark_cache
    if _watermark_cache is not None:
        return _watermark_cache
    if not WATERMARK_PATH.exists():
        return None
    try:
        wm = Image.open(WATERMARK_PATH).convert("RGBA")
        _watermark_cache = wm
        return wm
    except Exception:
        return None


def _apply_watermark(img: Image.Image) -> Image.Image:
    """
    Aplica el watermark en diagonal desde esquina superior-izquierda
    a inferior-derecha, escalado grande y con opacidad ~40%.
    """
    wm_orig = _get_watermark()
    if wm_orig is None:
        return img

    img = img.convert("RGBA")
    iw, ih = img.size

    # Diagonal exacta de la imagen → el watermark tendrá ese ancho
    # para que sus extremos toquen las esquinas TL y BR
    diag = math.sqrt(iw ** 2 + ih ** 2)
    angle_deg = math.degrees(math.atan2(ih, iw))  # ángulo de la diagonal

    # Escalar watermark al largo de la diagonal
    wm = wm_orig.copy()
    ratio = diag / wm.width
    wm = wm.resize((int(diag), max(1, int(wm.height * ratio))), Image.LANCZOS)

    # PIL rota en sentido antihorario → negamos para rotar en sentido horario (TL→BR)
    wm = wm.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

    # Difuminar ligeramente
    wm = wm.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Reducir opacidad al 40%
    r, g, b, a = wm.split()
    a = a.point(lambda x: int(x * 0.40))
    wm = Image.merge("RGBA", (r, g, b, a))

    # Centrar en la imagen
    paste_x = (iw - wm.width) // 2
    paste_y = (ih - wm.height) // 2
    img.paste(wm, (paste_x, paste_y), wm)

    return img.convert("RGB")


def make_thumbnail_b64(filepath: Path, size: int = 300) -> str | None:
    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size), Image.LANCZOS)
            img = _apply_watermark(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=15)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def make_watermarked_preview(filepath: Path, out_path: Path, size: int = 1200) -> bool:
    """
    Genera una previsualización con marca de agua y la guarda en out_path.
    Calidad reducida (baja resolución + compresión alta).
    """
    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((size, size), Image.LANCZOS)
            img = _apply_watermark(img)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=55)
        return True
    except Exception:
        return False


def scan_photos(jpg_dir: Path, raw_dir: Path, jpg_ext: str = ".jpg", raw_ext: str = ".arw"):
    raw_index = {}
    if raw_ext != "none" and raw_dir.is_dir():
        for f in raw_dir.iterdir():
            if f.suffix.lower() == raw_ext.lower():
                raw_index[f.stem.lower()] = f.name

    photos = []
    no_exif = []

    jpg_files = sorted(
        [f for f in jpg_dir.iterdir() if f.suffix.lower() == jpg_ext.lower()],
        key=lambda x: x.name.lower()
    )

    for jpg in jpg_files:
        dt = get_exif_datetime(jpg)
        raw_name = raw_index.get(jpg.stem.lower())
        entry = {
            "jpg": jpg.name,
            "raw": raw_name,
            "has_raw": raw_name is not None,
            "datetime": dt,
            "datetime_str": dt.strftime("%H:%M:%S") if dt else None,
            "datetime_full": dt.isoformat() if dt else None,
        }
        if dt:
            photos.append(entry)
        else:
            no_exif.append(entry)

    photos.sort(key=lambda p: p["datetime"])
    return photos, no_exif


# ══════════════════════════════════════════════════════════════════════
# Detección de surfista (YOLO + fallbacks)
# ══════════════════════════════════════════════════════════════════════

_yolo_model = None
_detection_cache: dict[str, dict] = {}

PERSON_CLASS = 0
SURFBOARD_CLASS = 37


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("📦 Cargando modelo YOLO (primera vez descarga ~6MB)...")
        _yolo_model = YOLO("yolov8n.pt")
        print("✅ Modelo YOLO cargado")
    return _yolo_model


def _make_square_crop(img, cx, cy, radius):
    h, w = img.shape[:2]
    radius = max(radius, 20)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    crop_w, crop_h = x2 - x1, y2 - y1
    side = min(crop_w, crop_h)
    if x1 == 0:
        x1 = max(0, x2 - side)
    if y1 == 0:
        y1 = max(0, y2 - side)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (300, 300), interpolation=cv2.INTER_AREA)


def _try_yolo(img):
    """
    Fase 1: YOLO — busca 'person' y 'surfboard'.
    Cuando hay varias personas, elige la más cercana al centro de la imagen
    ponderada por confianza. Esto evita seleccionar surfistas que están en
    primer plano (mayor confianza) pero no son el sujeto principal de la foto,
    que tiende a estar centrado en el encuadre.
    """
    model = _get_yolo()
    h, w = img.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    all_persons = []   # lista de (score, x1, y1, x2, y2)
    best_board = None
    best_board_conf = 0

    # Multi-escala: YOLO a varias resoluciones para surfistas lejos y cerca
    seen_persons: set[tuple] = set()
    for imgsz in (640, 960, 1280):
        results = model(img, imgsz=imgsz, conf=0.15, verbose=False)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == PERSON_CLASS:
                    # Deduplicar detecciones de distintas escalas por IoU aproximado
                    key = (x1 // 20, y1 // 20, x2 // 20, y2 // 20)
                    if key in seen_persons:
                        continue
                    seen_persons.add(key)

                    # Distancia normalizada del centro del bbox al centro de imagen
                    pcx = (x1 + x2) / 2.0
                    pcy = (y1 + y2) / 2.0
                    dist = ((pcx - img_cx) / w) ** 2 + ((pcy - img_cy) / h) ** 2
                    dist_norm = dist ** 0.5  # 0=centro, ~0.7=esquina

                    # Score: 60% confianza + 40% centralidad
                    # → prioriza detecciones seguras, centralidad como desempate
                    centrality = 1.0 - min(dist_norm / 0.7, 1.0)
                    score = 0.6 * conf + 0.4 * centrality
                    all_persons.append((score, x1, y1, x2, y2))

                elif cls == SURFBOARD_CLASS and conf > best_board_conf:
                    best_board = (x1, y1, x2, y2)
                    best_board_conf = conf

    # Prioridad 1: persona más centrada (mayor score)
    if all_persons:
        all_persons.sort(key=lambda t: t[0], reverse=True)
        _, x1, y1, x2, y2 = all_persons[0]
        cx = (x1 + x2) // 2
        cy = y1 + (y2 - y1) // 3   # torso/cabeza para el thumbnail
        bw, bh = x2 - x1, y2 - y1
        radius = int(max(bw, bh) * 0.6)
        bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
        return _make_square_crop(img, cx, cy, radius), bbox

    # Prioridad 2: tabla detectada (el surfista está encima/al lado)
    if best_board:
        x1, y1, x2, y2 = best_board
        cx = (x1 + x2) // 2
        bh = y2 - y1
        cy = y1 - bh // 4
        cy = max(0, cy)
        bw = x2 - x1
        radius = int(max(bw, bh) * 1.2)
        bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
        return _make_square_crop(img, cx, cy, radius), bbox

    return None, None


def _try_dual_saliency(img, small, scale):
    """Fase 2: Doble saliencia como fallback."""
    sh, sw = small.shape[:2]
    combined = np.zeros((sh, sw), dtype=np.float32)
    found = False

    for SalClass in (cv2.saliency.StaticSaliencySpectralResidual_create,
                     cv2.saliency.StaticSaliencyFineGrained_create):
        try:
            sal = SalClass()
            ok, sal_map = sal.computeSaliency(small)
            if ok and sal_map is not None:
                sal_map = sal_map.astype(np.float32)
                if sal_map.max() > 0:
                    sal_map /= sal_map.max()
                combined += sal_map
                found = True
        except Exception:
            continue

    if not found or combined.max() == 0:
        return None

    combined = (combined / combined.max() * 255).astype(np.uint8)
    _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < sh * sw * 0.003:
        return None

    x, y, bw, bh = cv2.boundingRect(biggest)
    cx = int((x + bw / 2) / scale)
    cy = int((y + bh / 2) / scale)
    radius = int(max(bw, bh) / scale / 2)
    return _make_square_crop(img, cx, cy, radius)


def _center_crop(img):
    """Fase 3: Center crop como último recurso."""
    h, w = img.shape[:2]
    radius = min(h, w) // 6
    return _make_square_crop(img, w // 2, h // 2, radius)


def get_photo_detection(filepath: Path) -> dict:
    """
    Pipeline de detección de surfista. Devuelve {crop: b64|None, bbox: [x1n,y1n,x2n,y2n]|None}.
    Resultado cacheado por ruta de archivo.
    """
    key = str(filepath)
    if key in _detection_cache:
        return _detection_cache[key]

    result: dict = {"crop": None, "bbox": None}
    try:
        img = cv2.imread(key)
        if img is None:
            _detection_cache[key] = result
            return result

        h, w = img.shape[:2]

        # Fase 1: YOLO
        crop_arr, bbox = _try_yolo(img)

        # Fase 2: Saliencia (sin bbox)
        if crop_arr is None:
            sc = 640 / max(h, w)
            if sc < 1:
                small = cv2.resize(img, (int(w * sc), int(h * sc)))
            else:
                small = img
                sc = 1.0
            crop_arr = _try_dual_saliency(img, small, sc)

        # Fase 3: Center crop
        if crop_arr is None:
            crop_arr = _center_crop(img)

        if crop_arr is not None:
            _, buf = cv2.imencode(".jpg", crop_arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result["crop"] = base64.b64encode(buf).decode()

        if bbox:
            result["bbox"] = list(bbox)

    except Exception:
        pass

    _detection_cache[key] = result
    return result


def detect_person_thumbnail(filepath: Path) -> str | None:
    return get_photo_detection(filepath)["crop"]


# ══════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wave Splitter</title>
<style>
  :root {
    --bg: #eef4f8; --surface: #ffffff; --surface2: #f0f7fa;
    --border: #ccdde8; --text: #1a2d3d; --text2: #5a7a8c;
    --accent: #44bad8; --accent-hover: #2fa8c8; --accent-light: #ddf2f9;
    --green: #3dba7e; --green-light: #e2f9ef;
    --red: #ff6a70; --red-light: #fff0f0;
    --yellow: #fed891; --yellow-dark: #8a6010;
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.06);
    --shadow: 0 3px 14px rgba(68,186,216,0.1), 0 1px 4px rgba(0,0,0,0.05);
    --radius: 14px; --radius-sm: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; margin-bottom: 4px; }
  h1 em { color: var(--accent); font-style: normal; }
  .subtitle { color: var(--text2); font-size: 0.88rem; margin-bottom: 32px; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 20px; box-shadow: var(--shadow); }
  .card h2 { font-size: 0.95rem; font-weight: 700; margin-bottom: 18px; display: flex; align-items: center; gap: 10px; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em; }
  .card h2::before { content: ""; display: inline-block; width: 4px; height: 16px; background: var(--accent); border-radius: 2px; flex-shrink: 0; }

  .field { margin-bottom: 16px; }
  .field label { display: block; font-size: 0.75rem; font-weight: 700; color: var(--text2); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.06em; }
  .input-row { display: flex; gap: 8px; align-items: stretch; }
  .input-row input[type=text] { flex: 1; background: var(--surface); border: 1.5px solid var(--border); border-radius: var(--radius-sm);
    padding: 10px 14px; color: var(--text); font-size: 0.9rem; outline: none; transition: border-color .2s, box-shadow .2s; box-shadow: var(--shadow-sm); }
  .input-row input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(68,186,216,0.15); }
  .input-row input[type=text]::placeholder { color: #aac4d4; }

  .btn { padding: 10px 18px; border-radius: var(--radius-sm); border: none; cursor: pointer; font-size: 0.86rem; font-weight: 700; transition: all .15s; letter-spacing: 0.01em; }
  .btn-browse { background: var(--surface); color: var(--text2); border: 1.5px solid var(--border); box-shadow: var(--shadow-sm); }
  .btn-browse:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-light); }
  .btn-primary { background: var(--accent); color: #fff; box-shadow: 0 2px 10px rgba(68,186,216,0.35); }
  .btn-primary:hover { background: var(--accent-hover); box-shadow: 0 4px 16px rgba(68,186,216,0.45); transform: translateY(-1px); }
  .btn-primary:active { transform: none; }
  .btn-primary:disabled { opacity: .45; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-success { background: var(--green); color: #fff; box-shadow: 0 2px 10px rgba(61,186,126,0.35); }
  .btn-success:hover { background: #2ea86b; transform: translateY(-1px); }
  .btn-sm { padding: 7px 14px; font-size: 0.78rem; border-radius: 6px; }
  .actions { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }

  /* Format selector pill */
  .fmt-select-wrap { position: relative; flex-shrink: 0; display: flex; align-items: stretch; }
  .fmt-select-wrap::after {
    content: "";
    position: absolute; right: 11px; top: 50%; transform: translateY(-50%);
    width: 10px; height: 6px; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2344bad8' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-size: contain;
  }
  .format-select {
    appearance: none; -webkit-appearance: none;
    background: var(--accent-light); border: 1.5px solid var(--accent);
    border-radius: 20px; padding: 0 34px 0 14px;
    color: var(--accent); font-size: 0.82rem; font-weight: 800;
    outline: none; cursor: pointer; letter-spacing: 0.02em;
    transition: background .15s, box-shadow .15s;
  }
  .format-select:hover { background: #c8eaf5; }
  .format-select:focus { box-shadow: 0 0 0 3px rgba(68,186,216,0.2); }

  .hidden { display: none; }
  .visible { display: block !important; }

  /* Slider */
  .slider-row { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; padding: 14px 18px; background: var(--surface2); border-radius: var(--radius-sm); border: 1px solid var(--border); }
  .slider-label { white-space: nowrap; font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.06em; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); cursor: pointer; }
  .slider-val { background: var(--accent); color: #fff; padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 800; min-width: 54px; text-align: center; box-shadow: 0 2px 6px rgba(68,186,216,0.3); }

  /* Stats */
  .scan-stats { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 16px; }
  .stat-chip { background: var(--accent-light); color: var(--accent); border: 1px solid rgba(68,186,216,0.3); border-radius: 20px; padding: 3px 12px; font-weight: 800; font-size: 0.78rem; }
  .stat-chip.warn { background: #fff9e0; color: var(--yellow-dark); border-color: rgba(254,216,145,0.5); }
  .stat-label { font-size: 0.82rem; color: var(--text2); }

  /* Wave blocks */
  .wave-block { background: var(--surface); border-radius: var(--radius-sm); padding: 14px 16px; margin-bottom: 10px; border: 1.5px solid var(--border); transition: all .2s; box-shadow: var(--shadow-sm); }
  .wave-block:hover { box-shadow: var(--shadow); border-color: rgba(68,186,216,0.45); }
  .wave-block.excluded { opacity: .35; }
  .wave-block.dragging { opacity: .25; }
  .wave-block.drag-over { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(68,186,216,0.2); }
  .wave-header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .wave-toggle { appearance: none; width: 40px; height: 22px; background: var(--green); border-radius: 99px; position: relative; cursor: pointer;
    flex-shrink: 0; transition: background .2s; border: none; outline: none; box-shadow: 0 1px 4px rgba(61,186,126,0.3); }
  .wave-toggle::after { content: ""; position: absolute; top: 3px; left: 3px; width: 16px; height: 16px;
    background: #fff; border-radius: 50%; transition: transform .2s; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
  .wave-toggle:not(:checked) { background: #ccdae2; box-shadow: none; }
  .wave-toggle:checked::after { transform: translateX(18px); }
  .wave-name-input { background: var(--surface2); border: 1.5px solid var(--border); border-radius: 6px; padding: 4px 10px;
    color: var(--text); font-size: 0.88rem; font-weight: 700; width: 160px; outline: none; transition: border-color .2s, box-shadow .2s; }
  .wave-name-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(68,186,216,0.1); }

  .wave-thumb { width: 110px; height: 110px; border-radius: 10px; object-fit: cover; background: var(--surface2); border: 2px solid var(--border); flex-shrink: 0;
    cursor: pointer; transition: transform .15s, border-color .15s, box-shadow .15s; box-shadow: var(--shadow-sm); }
  .wave-thumb:hover { transform: scale(1.05); border-color: var(--accent); box-shadow: 0 4px 14px rgba(68,186,216,0.25); }
  .wave-info { flex: 1; min-width: 0; }
  .wave-info h3 { font-size: 0.9rem; margin-bottom: 3px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .wave-info .meta { font-size: 0.77rem; color: var(--text2); margin-top: 2px; }
  .wave-info .meta b { color: var(--accent); font-weight: 700; }
  .wave-thumb-counter { font-size: 0.67rem; color: var(--text2); text-align: center; margin-top: 3px; }

  .badge { font-size: 0.72rem; padding: 2px 10px; border-radius: 99px; font-weight: 700; }
  .badge { background: var(--accent-light); color: var(--accent); border: 1px solid rgba(68,186,216,0.25); }
  .badge.warn { background: #fff8e0; color: var(--yellow-dark); border: 1px solid rgba(254,216,145,0.5); }

  .wave-photos { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 10px; }
  .wave-photo-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); cursor: pointer; transition: transform .15s, box-shadow .15s; opacity: .75; }
  .wave-photo-dot:hover { transform: scale(1.7); box-shadow: 0 0 6px var(--accent); opacity: 1; }
  .wave-photo-dot.no-raw { background: var(--yellow); }
  .wave-photo-dot.active { transform: scale(1.7); opacity: 1; box-shadow: 0 0 0 2.5px #fff, 0 0 0 4px var(--accent); }

  .no-exif-section { margin-top: 14px; padding: 14px 18px; background: var(--red-light); border: 1.5px solid rgba(255,106,112,0.3); border-radius: var(--radius-sm); }
  .no-exif-section h4 { color: var(--red); margin-bottom: 5px; font-size: 0.88rem; }
  .no-exif-section p { font-size: 0.82rem; color: var(--text2); }

  .progress-bar-outer { width: 100%; height: 26px; background: var(--surface2); border-radius: 20px; overflow: hidden; margin-bottom: 14px; border: 1px solid var(--border); }
  .progress-bar-inner { height: 100%; background: linear-gradient(90deg, var(--accent), #6dd8ee); width: 0%;
    transition: width .3s; display: flex; align-items: center; justify-content: center;
    font-size: 0.76rem; font-weight: 800; color: #fff; min-width: 42px; }
  .log-box { background: #18212e; border-radius: var(--radius-sm); padding: 14px; height: 240px; overflow-y: auto;
    font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace; font-size: 0.78rem; line-height: 1.7; border: 1px solid var(--border); }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.info { color: #6a8a9a; }
  .log-line.ok { color: var(--green); }
  .log-line.warn { color: #f0c040; }
  .log-line.err { color: var(--red); }
  .summary-box { margin-top: 14px; padding: 16px 20px; background: var(--green-light); border: 1.5px solid rgba(61,186,126,0.3); border-radius: var(--radius-sm); }
  .summary-box h3 { margin-bottom: 6px; color: var(--green); font-size: 0.95rem; }

  .modal-bg { display: none; position: fixed; inset: 0; background: rgba(20,40,55,0.45); backdrop-filter: blur(4px); z-index: 100; align-items: center; justify-content: center; }
  .modal-bg.open { display: flex; }
  .modal { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); width: 520px; max-height: 70vh; display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,0.12); }
  .modal-header { padding: 18px 22px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .modal-header h3 { font-size: 0.95rem; font-weight: 700; }
  .modal-close { background: none; border: none; color: var(--text2); cursor: pointer; font-size: 1.2rem; width: 30px; height: 30px; border-radius: 6px; display: flex; align-items: center; justify-content: center; transition: background .1s; }
  .modal-close:hover { background: var(--surface2); color: var(--text); }
  .modal-body { padding: 14px; overflow-y: auto; flex: 1; }
  .dir-current { padding: 8px 12px; font-size: 0.78rem; color: var(--text2); background: var(--surface2); border-radius: 6px; margin-bottom: 10px; word-break: break-all; border: 1px solid var(--border); }
  .dir-item { padding: 10px 14px; cursor: pointer; border-radius: 8px; font-size: 0.88rem; display: flex; align-items: center; gap: 8px; transition: background .1s; }
  .dir-item:hover { background: var(--accent-light); color: var(--accent); }
  .modal-footer { padding: 14px 22px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; gap: 8px; }
  .new-folder-row { display: flex; gap: 8px; margin-bottom: 10px; }
  .new-folder-row input { flex: 1; background: var(--surface); border: 1.5px solid var(--border); border-radius: 8px; padding: 9px 12px; color: var(--text); font-size: 0.85rem; outline: none; transition: border-color .2s; }
  .new-folder-row input:focus { border-color: var(--accent); }
  .new-folder-row input::placeholder { color: #aac4d4; }

  .drag-handle { color: #b8d0dc; font-size: 1.1rem; cursor: grab; padding: 4px 6px; border-radius: 4px; user-select: none; flex-shrink: 0; line-height: 1; transition: color .15s, background .15s; }
  .drag-handle:hover { color: var(--accent); background: var(--accent-light); }

  .raw-dir-row { transition: opacity .2s; }
  .raw-dir-row.disabled { opacity: .45; pointer-events: none; }

  .lightbox-bg { display: none; position: fixed; inset: 0; background: rgba(15,25,35,0.95); backdrop-filter: blur(8px); z-index: 300; align-items: center; justify-content: center; flex-direction: column; }
  .lightbox-bg.open { display: flex; }
  .lightbox-wrap { position: relative; display: inline-flex; max-width: 90vw; max-height: 82vh; }
  .lightbox-img { max-width: 90vw; max-height: 82vh; object-fit: contain; border-radius: 6px; transition: opacity .15s; display: block; box-shadow: 0 8px 40px rgba(0,0,0,0.6); }
  .lightbox-close { position: absolute; top: 16px; right: 20px; background: rgba(255,255,255,0.1); border: none; color: #fff; font-size: 1.3rem; cursor: pointer; width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: background .15s; }
  .lightbox-close:hover { background: rgba(255,255,255,0.2); }
  .lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.08); border: none; color: #fff; font-size: 3rem; cursor: pointer; opacity: .55; padding: 12px; line-height: 1; border-radius: 8px; transition: opacity .15s, background .15s; }
  .lightbox-nav:hover { opacity: 1; background: rgba(255,255,255,0.14); }
  .lightbox-prev { left: 8px; }
  .lightbox-next { right: 8px; }
  .lightbox-footer { display: flex; align-items: center; gap: 14px; margin-top: 14px; }
  .lightbox-caption { color: rgba(255,255,255,0.45); font-size: 0.78rem; }
  .lightbox-split-btn { padding: 7px 16px; background: var(--red); color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.8rem; font-weight: 700; transition: all .15s; }
  .lightbox-split-btn:hover { background: #e85a60; transform: translateY(-1px); }

  #split-menu { display: none; position: fixed; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 4px; z-index: 200; box-shadow: 0 8px 30px rgba(0,0,0,0.1); min-width: 180px; }
  .split-menu-item { padding: 9px 14px; cursor: pointer; border-radius: 7px; font-size: 0.84rem; font-weight: 500; transition: background .1s; }
  .split-menu-item:hover { background: var(--accent-light); color: var(--accent); }

</style>
</head>
<body>
<div class="container">
  <h1>Wave Splitter</h1>
  <p class="subtitle">Agrupa automáticamente tus fotos de surf por ola usando timestamps EXIF</p>

  <div class="card" id="step-paths">
    <h2>Directorios</h2>
    <div class="field">
      <label>Carpeta de fotos procesadas</label>
      <div class="input-row">
        <input id="inp-jpg" type="text" placeholder="/ruta/a/fotos" onkeydown="if(event.key==='Enter')scan()">
        <select id="sel-jpg-ext" class="format-select">
          <option value=".jpg">JPG</option>
          <option value=".jpeg">JPEG</option>
          <option value=".png">PNG</option>
          <option value=".tiff">TIFF</option>
          <option value=".heic">HEIC</option>
          <option value=".webp">WebP</option>
        </select>
        <button class="btn btn-browse" onclick="openBrowser('inp-jpg')">Explorar</button>
      </div>
    </div>
    <div class="field raw-dir-row" id="raw-dir-row">
      <label>Carpeta de RAW</label>
      <div class="input-row">
        <input id="inp-raw" type="text" placeholder="/ruta/a/raw" onkeydown="if(event.key==='Enter')scan()">
        <select id="sel-raw-ext" class="format-select" onchange="onRawExtChange()">
          <option value=".arw">ARW (Sony)</option>
          <option value=".cr3">CR3 (Canon)</option>
          <option value=".cr2">CR2 (Canon)</option>
          <option value=".nef">NEF (Nikon)</option>
          <option value=".raf">RAF (Fujifilm)</option>
          <option value=".dng">DNG</option>
          <option value=".orf">ORF (Olympus)</option>
          <option value=".rw2">RW2 (Panasonic)</option>
          <option value=".pef">PEF (Pentax)</option>
          <option value="none">Sin RAW</option>
        </select>
        <button class="btn btn-browse" onclick="openBrowser('inp-raw')">Explorar</button>
      </div>
    </div>
    <div class="field">
      <label>Carpeta de salida</label>
      <div class="input-row">
        <input id="inp-out" type="text" placeholder="/ruta/a/salida" onkeydown="if(event.key==='Enter')scan()">
        <button class="btn btn-browse" onclick="openBrowser('inp-out')">Explorar</button>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" id="btn-scan" onclick="scan()">🔍 Analizar fotos</button>
    </div>
  </div>

  <div class="card hidden" id="step-preview">
    <h2>Secuencias detectadas</h2>
    <div class="scan-stats" id="scan-stats"></div>
    <div class="slider-row">
      <label style="white-space:nowrap; font-size:0.85rem;">Umbral de separación:</label>
      <input type="range" id="gap-slider" min="2" max="60" value="10" oninput="document.getElementById('gap-val').textContent=this.value+'s'" onchange="regroup()">
      <div class="slider-val" id="gap-val">10s</div>
    </div>
    <div id="waves-container"></div>
    <div id="no-exif-container"></div>

    <div class="actions" style="margin-top: 16px;">
      <button class="btn btn-success" id="btn-confirm" onclick="startCopy()">✅ Confirmar y copiar</button>
      <button class="btn btn-browse" onclick="document.getElementById('step-preview').classList.remove('visible');document.getElementById('step-preview').classList.add('hidden');">Cancelar</button>
    </div>
  </div>

  <div class="card hidden" id="step-progress">
    <h2>Progreso</h2>
    <div class="progress-bar-outer"><div class="progress-bar-inner" id="pbar">0%</div></div>
    <div class="log-box" id="log-box"></div>
    <div class="summary-box hidden" id="summary-box">
      <h3>Completado</h3>
      <div id="summary-content"></div>
    </div>
  </div>
</div>

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

<div id="split-menu">
  <div class="split-menu-item" onclick="doSplit()">✂ Separar desde aquí</div>
</div>

<div class="lightbox-bg" id="lightbox" onclick="if(event.target===this)closeLightbox()">
  <button class="lightbox-close" onclick="closeLightbox()">✕</button>
  <button class="lightbox-nav lightbox-prev" onclick="navLightbox(-1)">‹</button>
  <div class="lightbox-wrap">
    <img class="lightbox-img" id="lightbox-img" src="" alt="">
  </div>
  <div class="lightbox-footer">
    <span class="lightbox-caption" id="lightbox-caption"></span>
    <button class="lightbox-split-btn" id="lightbox-split-btn" onclick="splitFromLightbox()" style="display:none">✂ Separar desde aquí</button>
  </div>
  <button class="lightbox-nav lightbox-next" onclick="navLightbox(1)">›</button>
</div>

<script>
const HOME_DIR = {{ home_dir | tojson }};
let browseTarget = null;
let browseCurrent = "";
let allPhotos = [];
let noExifPhotos = [];
let currentWaves = [];
let thumbCache = {};
let waveThumbIdx = {};

async function openBrowser(inputId) {
  browseTarget = inputId;
  const start = document.getElementById(inputId).value || HOME_DIR;
  await loadDir(start);
  document.getElementById("modal-browse").classList.add("open");
  document.getElementById("new-folder-name").value = "";
}
function closeBrowser() { document.getElementById("modal-browse").classList.remove("open"); }
function selectDir() { document.getElementById(browseTarget).value = browseCurrent; closeBrowser(); }

async function loadDir(dirPath) {
  const r = await fetch("/api/browse", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({path: dirPath}) });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }
  browseCurrent = data.current;
  document.getElementById("browse-current").textContent = data.current;
  const list = document.getElementById("browse-list");
  list.innerHTML = "";
  if (data.parent) {
    const el = document.createElement("div"); el.className = "dir-item";
    el.innerHTML = "⬆️ .."; el.onclick = () => loadDir(data.parent); list.appendChild(el);
  }
  data.dirs.forEach(d => {
    const el = document.createElement("div"); el.className = "dir-item";
    el.innerHTML = "📁 " + d.name; el.onclick = () => loadDir(d.path); list.appendChild(el);
  });
}

async function createFolder() {
  const name = document.getElementById("new-folder-name").value.trim();
  if (!name) { alert("Escribe un nombre"); return; }
  const r = await fetch("/api/mkdir", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({parent: browseCurrent, name}) });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById("new-folder-name").value = "";
  await loadDir(data.path);
}

async function scan() {
  const jpg = document.getElementById("inp-jpg").value.trim();
  const raw = document.getElementById("inp-raw").value.trim();
  if (!jpg || !raw) { alert("Completa las rutas de JPG y RAW"); return; }

  document.getElementById("btn-scan").disabled = true;
  document.getElementById("btn-scan").textContent = "⏳ Leyendo EXIF...";

  try {
    const jpgExt = document.getElementById("sel-jpg-ext").value;
    const rawExt = document.getElementById("sel-raw-ext").value;
    const r = await fetch("/api/scan", { method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({jpg_dir: jpg, raw_dir: raw, jpg_ext: jpgExt, raw_ext: rawExt}) });

    if (!r.ok) { alert("Error del servidor: " + r.status); return; }

    const text = await r.text();
    console.log("Respuesta /api/scan:", text);
    let data;
    try { data = JSON.parse(text); } catch(e) {
      alert("Error parseando respuesta. Revisa consola (Cmd+Option+J).");
      console.error("JSON parse error:", e, text);
      return;
    }

    if (data.error) { alert(data.error); return; }

    allPhotos = data.photos || [];
    noExifPhotos = data.no_exif || [];

    console.log("Fotos con EXIF:", allPhotos.length, "Sin EXIF:", noExifPhotos.length,
                "Archivos:", data.total_files, "Subcarpetas:", data.subdirs);

    if (allPhotos.length === 0 && noExifPhotos.length === 0) {
      alert("No se encontraron JPGs.\n\nArchivos: " + (data.total_files||0) +
            "\nSubcarpetas: " + (data.subdirs||0) +
            ((data.subdirs||0) > 0 ? "\n\n⚠ Hay subcarpetas. Apunta a la carpeta con los .JPG directamente." : ""));
      return;
    }
    if (allPhotos.length === 0 && noExifPhotos.length > 0) {
      alert(noExifPhotos.length + " JPGs encontrados pero sin EXIF. Irán a 'sin_clasificar'.");
    }

    thumbCache = {};
    waveThumbIdx = {};
    regroup();

    const panel = document.getElementById("step-preview");
    panel.classList.remove("hidden"); panel.classList.add("visible");
    panel.scrollIntoView({behavior: "smooth"});
  } catch(e) {
    alert("Error de conexión: " + e.message);
    console.error("Scan error:", e);
  } finally {
    document.getElementById("btn-scan").disabled = false;
    document.getElementById("btn-scan").textContent = "🔍 Analizar fotos";
  }
}

function regroup() {
  const gap = parseInt(document.getElementById("gap-slider").value);
  document.getElementById("gap-val").textContent = gap + "s";

  currentWaves = [];
  waveThumbIdx = {};
  if (allPhotos.length === 0) { renderWaves(); return; }

  let wave = [allPhotos[0]];
  for (let i = 1; i < allPhotos.length; i++) {
    const prev = new Date(allPhotos[i-1].datetime_full);
    const curr = new Date(allPhotos[i].datetime_full);
    if ((curr - prev) / 1000 > gap) {
      currentWaves.push(wave);
      wave = [allPhotos[i]];
    } else {
      wave.push(allPhotos[i]);
    }
  }
  if (wave.length) currentWaves.push(wave);

  currentWaves.forEach(function(w, i) {
    w.waveName = 'ola_' + String(i + 1).padStart(3, '0');
  });

  const totalPhotos = allPhotos.length + noExifPhotos.length;
  const withRaw = allPhotos.filter(p => p.has_raw).length + noExifPhotos.filter(p => p.has_raw).length;
  document.getElementById("scan-stats").innerHTML =
    "<span>" + totalPhotos + "</span> fotos · <span>" + currentWaves.length + "</span> secuencias · <span>" + withRaw + "</span> con RAW" +
    (noExifPhotos.length ? " · <span style='color:var(--yellow)'>" + noExifPhotos.length + "</span> sin EXIF" : "");

  renderWaves();
}

function renderWaves() {
  const container = document.getElementById("waves-container");
  container.innerHTML = "";

  currentWaves.forEach((wave, idx) => {
    const first = wave[0];
    const last = wave[wave.length - 1];
    const noRaw = wave.filter(p => !p.has_raw).length;
    const num = String(idx + 1).padStart(3, "0");

    const block = document.createElement("div");
    block.className = "wave-block";
    block.id = "wave-block-" + idx;
    block.innerHTML =
      '<div class="wave-header">' +
        '<div class="drag-handle" draggable="true" title="Arrastrar para unir con otra ola">⠿</div>' +
        '<input type="checkbox" class="wave-toggle" id="wave-toggle-' + idx + '" checked onchange="toggleWave(' + idx + ')" title="Incluir / excluir">' +
        '<div>' +
          '<img class="wave-thumb" id="thumb-' + idx + '" onclick="openLightbox(' + idx + ', 0)" ' +
            'src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="" title="Clic para ver foto completa">' +
          '<div class="wave-thumb-counter" id="thumb-counter-' + idx + '"></div>' +
        '</div>' +
        '<div class="wave-info">' +
          '<h3>' +
            '<input type="text" class="wave-name-input" id="wave-name-' + idx + '" value="' + (wave.waveName || ('ola_' + num)) + '" title="Editar nombre">' +
            ' <span class="badge">' + wave.length + ' fotos</span>' +
            (noRaw ? ' <span class="badge warn">' + noRaw + ' sin RAW</span>' : '') +
          '</h3>' +
          '<div class="meta">🕐 <b>' + first.datetime_str + '</b> — <b>' + last.datetime_str + '</b>' +
            ' · ' + ((new Date(last.datetime_full) - new Date(first.datetime_full)) / 1000).toFixed(1) + 's</div>' +
          '<div class="wave-photos" id="dots-' + idx + '">' +
            wave.map(function(p, pi) {
              return '<div class="wave-photo-dot ' + (p.has_raw ? '' : 'no-raw') + (pi === 0 ? ' active' : '') +
                '" title="' + p.jpg + '" onclick="showWavePhoto(' + idx + ',' + pi + ')" oncontextmenu="showSplitMenu(event,' + idx + ',' + pi + ')"></div>';
            }).join('') +
          '</div>' +
        '</div>' +
      '</div>';
    container.appendChild(block);
    setupDragHandlers(block, idx);

    // Foto central con detección, o la más cercana que la tenga
    loadBestWaveThumb(idx);
  });

  const noExifBox = document.getElementById("no-exif-container");
  noExifBox.innerHTML = "";
  if (noExifPhotos.length) {
    noExifBox.innerHTML =
      '<div class="no-exif-section">' +
        '<h4>⚠ ' + noExifPhotos.length + ' foto(s) sin datos EXIF</h4>' +
        '<p>Se copiarán a <b>sin_clasificar</b> para que no se pierda ninguna.</p>' +
        '<div style="margin-top:6px;font-size:0.8rem;color:var(--text2);">' + noExifPhotos.map(function(p){return p.jpg;}).join(", ") + '</div>' +
      '</div>';
  }
}

async function loadWaveThumb(waveIdx, photoIdx) {
  const wave = currentWaves[waveIdx];
  if (!wave || !wave[photoIdx]) return;
  const filename = wave[photoIdx].jpg;
  const el = document.getElementById("thumb-" + waveIdx);
  const counter = document.getElementById("thumb-counter-" + waveIdx);

  waveThumbIdx[waveIdx] = photoIdx;
  if (counter) counter.textContent = (photoIdx + 1) + " / " + wave.length;
  updateDotActive(waveIdx, photoIdx);

  if (thumbCache[filename]) {
    if (el) el.src = "data:image/jpeg;base64," + thumbCache[filename];
    return;
  }
  const jpg = document.getElementById("inp-jpg").value.trim();
  try {
    const r = await fetch("/api/thumbnail", { method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({jpg_dir: jpg, filename: filename}) });
    const data = await r.json();
    if (data.thumbnail) {
      thumbCache[filename] = data.thumbnail;
      if (el) el.src = "data:image/jpeg;base64," + data.thumbnail;
    }
  } catch(e) { console.error("Thumb error:", e); }
}

async function loadBestWaveThumb(waveIdx) {
  const wave = currentWaves[waveIdx];
  if (!wave || !wave.length) return;
  const jpg = document.getElementById("inp-jpg").value.trim();
  const el = document.getElementById("thumb-" + waveIdx);
  const counter = document.getElementById("thumb-counter-" + waveIdx);
  try {
    const r = await fetch("/api/wave-thumbnail", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({jpg_dir: jpg, photos: wave.map(function(p){ return p.jpg; })})
    });
    const data = await r.json();
    if (data.thumbnail) {
      var photoIdx = data.photo_idx;
      waveThumbIdx[waveIdx] = photoIdx;
      thumbCache[wave[photoIdx].jpg] = data.thumbnail;
      if (el) el.src = "data:image/jpeg;base64," + data.thumbnail;
      if (counter) counter.textContent = (photoIdx + 1) + " / " + wave.length;
      updateDotActive(waveIdx, photoIdx);
    }
  } catch(e) { console.error("Wave thumb error:", e); }
}

function cycleWaveThumb(waveIdx) {
  const wave = currentWaves[waveIdx];
  if (!wave || wave.length <= 1) return;
  var next = (waveThumbIdx[waveIdx] + 1) % wave.length;
  loadWaveThumb(waveIdx, next);
}

function showWavePhoto(waveIdx, photoIdx) {
  loadWaveThumb(waveIdx, photoIdx);
}

function updateDotActive(waveIdx, photoIdx) {
  var dots = document.getElementById("dots-" + waveIdx);
  if (!dots) return;
  dots.querySelectorAll(".wave-photo-dot").forEach(function(dot, i) {
    dot.classList.toggle("active", i === photoIdx);
  });
}

function toggleWave(idx) {
  var cb = document.getElementById("wave-toggle-" + idx);
  var block = document.getElementById("wave-block-" + idx);
  if (block) block.classList.toggle("excluded", !cb.checked);
}

function startCopy() {
  const out = document.getElementById("inp-out").value.trim();
  if (!out) { alert("Elige una carpeta de salida"); return; }

  const wavesData = [];
  currentWaves.forEach(function(w, i) {
    var toggle = document.getElementById("wave-toggle-" + i);
    if (!toggle || !toggle.checked) return;
    var nameInput = document.getElementById("wave-name-" + i);
    var name = (nameInput ? nameInput.value.trim() : "") || ("ola_" + String(i+1).padStart(3, "0"));
    wavesData.push({ name: name, photos: w });
  });

  if (!wavesData.length) { alert("No hay olas seleccionadas"); return; }

  var nameCounts = {};
  wavesData.forEach(function(w) { nameCounts[w.name] = (nameCounts[w.name] || 0) + 1; });

  var nameIdx = {};
  var finalWaves = wavesData.map(function(w) {
    var folder = w.name;
    if (nameCounts[w.name] > 1) {
      nameIdx[w.name] = (nameIdx[w.name] || 0) + 1;
      folder = w.name + "/ola_" + nameIdx[w.name];
    }
    return Object.assign({}, w, { folder: folder });
  });

  document.getElementById("btn-confirm").disabled = true;
  var panel = document.getElementById("step-progress");
  panel.classList.remove("hidden"); panel.classList.add("visible");
  document.getElementById("summary-box").classList.add("hidden");
  document.getElementById("log-box").innerHTML = "";
  panel.scrollIntoView({behavior: "smooth"});

  var jpg = document.getElementById("inp-jpg").value.trim();
  var raw = document.getElementById("inp-raw").value.trim();

  fetch("/api/copy", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({jpg_dir: jpg, raw_dir: raw, out_dir: out, waves: finalWaves, no_exif: noExifPhotos}) })
    .then(function(r) { return r.json(); })
    .then(function(d) { if (d.stream_id) listenProgress(d.stream_id); });
}

// ── Join & Split ───────────────────────────────────────────

function syncWaveNames() {
  currentWaves.forEach(function(wave, idx) {
    var inp = document.getElementById('wave-name-' + idx);
    if (inp) wave.waveName = inp.value.trim() || wave.waveName;
  });
}

var _dragSrcIdx = null;

function setupDragHandlers(block, idx) {
  var handle = block.querySelector('.drag-handle');

  handle.addEventListener('dragstart', function(e) {
    _dragSrcIdx = idx;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(idx));
    setTimeout(function() { block.classList.add('dragging'); }, 0);
  });

  handle.addEventListener('dragend', function() {
    block.classList.remove('dragging');
    _dragSrcIdx = null;
    document.querySelectorAll('.wave-block.drag-over').forEach(function(b) {
      b.classList.remove('drag-over');
    });
  });

  block.addEventListener('dragover', function(e) {
    if (_dragSrcIdx === null || _dragSrcIdx === idx) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    block.classList.add('drag-over');
  });

  block.addEventListener('dragleave', function(e) {
    if (!block.contains(e.relatedTarget)) {
      block.classList.remove('drag-over');
    }
  });

  block.addEventListener('drop', function(e) {
    e.preventDefault();
    block.classList.remove('drag-over');
    if (_dragSrcIdx === null || _dragSrcIdx === idx) return;
    mergeWaves(_dragSrcIdx, idx);
  });
}

function mergeWaves(srcIdx, dstIdx) {
  syncWaveNames();
  var merged = currentWaves[dstIdx].concat(currentWaves[srcIdx]);
  merged.sort(function(a, b) { return new Date(a.datetime_full) - new Date(b.datetime_full); });
  merged.waveName = currentWaves[dstIdx].waveName;
  var lo = Math.min(srcIdx, dstIdx), hi = Math.max(srcIdx, dstIdx);
  currentWaves.splice(hi, 1);
  currentWaves.splice(lo, 1);
  currentWaves.splice(lo, 0, merged);
  renderWaves();
}

function splitWave(waveIdx, photoIdx) {
  if (photoIdx <= 0 || photoIdx >= currentWaves[waveIdx].length) return;
  syncWaveNames();
  var wave = currentWaves[waveIdx];
  var part1 = wave.slice(0, photoIdx);
  var part2 = wave.slice(photoIdx);
  part1.waveName = wave.waveName;
  part2.waveName = wave.waveName + '_b';
  currentWaves.splice(waveIdx, 1, part1, part2);
  renderWaves();
}

// Split context menu
var _splitTarget = null;

function showSplitMenu(event, waveIdx, photoIdx) {
  event.preventDefault();
  if (photoIdx === 0) return;
  _splitTarget = { waveIdx: waveIdx, photoIdx: photoIdx };
  var menu = document.getElementById('split-menu');
  menu.style.display = 'block';
  menu.style.left = event.clientX + 'px';
  menu.style.top = event.clientY + 'px';
}

function hideSplitMenu() {
  document.getElementById('split-menu').style.display = 'none';
  _splitTarget = null;
}

function doSplit() {
  if (!_splitTarget) return;
  splitWave(_splitTarget.waveIdx, _splitTarget.photoIdx);
  hideSplitMenu();
}

document.addEventListener('click', function(e) {
  if (!e.target.closest('#split-menu')) hideSplitMenu();
});

// ── Formato RAW opcional ────────────────────────────────────

function onRawExtChange() {
  var val = document.getElementById('sel-raw-ext').value;
  document.getElementById('raw-dir-row').classList.toggle('disabled', val === 'none');
}

// ── Lightbox ────────────────────────────────────────────────

var _lbWave = null, _lbIdx = null;

function openLightbox(waveIdx, photoIdx) {
  _lbWave = waveIdx;
  _lbIdx = photoIdx;
  document.getElementById('lightbox').classList.add('open');
  loadLightboxPhoto();
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightbox-img').src = '';
}

function navLightbox(dir) {
  var wave = currentWaves[_lbWave];
  _lbIdx = (_lbIdx + dir + wave.length) % wave.length;
  loadLightboxPhoto();
}

async function loadLightboxPhoto() {
  var wave = currentWaves[_lbWave];
  var photo = wave[_lbIdx];
  var jpg = document.getElementById("inp-jpg").value.trim();
  var imgEl = document.getElementById('lightbox-img');
  var caption = document.getElementById('lightbox-caption');
  var splitBtn = document.getElementById('lightbox-split-btn');

  imgEl.style.opacity = '0.35';
  caption.textContent = photo.jpg + '  (' + (_lbIdx + 1) + ' / ' + wave.length + ')';
  splitBtn.style.display = _lbIdx > 0 ? '' : 'none';

  try {
    var r = await fetch("/api/photo-full", { method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({jpg_dir: jpg, filename: photo.jpg}) });
    var data = await r.json();
    if (data.data) {
      imgEl.src = "data:image/jpeg;base64," + data.data;
      imgEl.style.opacity = '1';
    }
  } catch(e) { console.error(e); }
}

function splitFromLightbox() {
  if (_lbWave === null || _lbIdx === null || _lbIdx === 0) return;
  var wi = _lbWave, pi = _lbIdx;
  closeLightbox();
  splitWave(wi, pi);
}

document.addEventListener('keydown', function(e) {
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navLightbox(-1);
  if (e.key === 'ArrowRight') navLightbox(1);
});

function listenProgress(streamId) {
  var es = new EventSource("/api/progress/" + streamId);
  var logBox = document.getElementById("log-box");
  var pbar = document.getElementById("pbar");

  es.onmessage = function(e) {
    var msg = JSON.parse(e.data);
    if (msg.type === "log") {
      var line = document.createElement("div");
      line.className = "log-line " + (msg.level || "info");
      line.textContent = msg.text;
      logBox.appendChild(line);
      logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "progress") {
      pbar.style.width = msg.pct + "%"; pbar.textContent = msg.pct + "%";
    } else if (msg.type === "done") {
      es.close();
      pbar.style.width = "100%"; pbar.textContent = "100%";
      document.getElementById("btn-confirm").disabled = false;
      var box = document.getElementById("summary-box");
      box.classList.remove("hidden");
      document.getElementById("summary-content").innerHTML =
        "<p>🌊 Secuencias: " + msg.waves + " · 📸 JPGs: " + msg.jpgs + " · 🎞️ ARWs: " + msg.arws +
        (msg.no_exif > 0 ? " · ⚠ Sin EXIF: " + msg.no_exif : "") + "</p>";
    }
  };
}
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, home_dir=str(Path.home()))


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


@app.route("/api/photo-full", methods=["POST"])
def api_photo_full():
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    filename = data["filename"]
    filepath = jpg_dir / filename
    if not filepath.exists():
        return jsonify(error="Archivo no encontrado")
    try:
        det = get_photo_detection(filepath)

        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((800, 800), Image.LANCZOS)

            # Paso 1: fondo + watermark a baja calidad
            img_wm = _apply_watermark(img.copy())
            bg_buf = io.BytesIO()
            img_wm.save(bg_buf, format="JPEG", quality=15)
            bg_buf.seek(0)
            base_img = Image.open(bg_buf)
            base_img.load()  # forzar lectura antes de cerrar el buffer

        # Paso 2: pegar el recorte de detección a ALTA calidad sobre el fondo ya degradado
        if det["crop"]:
            try:
                crop_img = Image.open(io.BytesIO(base64.b64decode(det["crop"]))).convert("RGB")
                crop_size = min(int(base_img.width * 0.20), 165)
                crop_img = crop_img.resize((crop_size, crop_size), Image.LANCZOS)
                bordered = Image.new("RGB", (crop_size + 6, crop_size + 6), (255, 255, 255))
                bordered.paste(crop_img, (3, 3))
                margin = 12
                base_img.paste(bordered, (margin, base_img.height - bordered.height - margin))
            except Exception:
                pass

        # Paso 3: guardar el resultado final a calidad alta (fondo ya degradado, crop nítido)
        final_buf = io.BytesIO()
        base_img.save(final_buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(final_buf.getvalue()).decode()
        return jsonify(data=img_b64, crop=None, bbox=det.get("bbox") if det else None)
    except Exception as e:
        return jsonify(error=str(e))


@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    raw_dir = Path(data["raw_dir"])
    jpg_ext = data.get("jpg_ext", ".jpg")
    raw_ext = data.get("raw_ext", ".arw")

    if not jpg_dir.is_dir():
        return jsonify(error=f"Carpeta JPG no existe: {jpg_dir}")
    if raw_ext != "none" and not raw_dir.is_dir():
        return jsonify(error=f"Carpeta RAW no existe: {raw_dir}")

    all_items = list(jpg_dir.iterdir())
    total_files = sum(1 for f in all_items if f.is_file())
    subdirs = sum(1 for f in all_items if f.is_dir())

    photos, no_exif = scan_photos(jpg_dir, raw_dir, jpg_ext=jpg_ext, raw_ext=raw_ext)

    for p in photos:
        del p["datetime"]
    for p in no_exif:
        del p["datetime"]

    return jsonify(photos=photos, no_exif=no_exif, total_files=total_files, subdirs=subdirs)


@app.route("/assets/watermark.png")
def serve_watermark():
    from flask import send_file
    if WATERMARK_PATH.exists():
        return send_file(WATERMARK_PATH, mimetype="image/png")
    return "", 404


@app.route("/api/thumbnail", methods=["POST"])
def api_thumbnail():
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    filename = data["filename"]
    filepath = jpg_dir / filename
    if not filepath.exists():
        return jsonify(thumbnail=None)

    thumb = detect_person_thumbnail(filepath)
    if not thumb:
        thumb = make_thumbnail_b64(filepath, 300)
    return jsonify(thumbnail=thumb)


@app.route("/api/wave-thumbnail", methods=["POST"])
def api_wave_thumbnail():
    """
    Devuelve el thumbnail de la foto más cercana al centro de la ola
    que tenga detección de persona. Si ninguna tiene, usa la central.
    """
    data = request.json
    jpg_dir = Path(data["jpg_dir"])
    photos = data["photos"]  # lista de nombres de archivo

    n = len(photos)
    mid = n // 2

    # Orden de búsqueda: centro, luego alternando izq/der
    order = [mid]
    for offset in range(1, n):
        if mid - offset >= 0:
            order.append(mid - offset)
        if mid + offset < n:
            order.append(mid + offset)

    # Buscar la más cercana al centro con detección de persona
    for idx in order:
        filepath = jpg_dir / photos[idx]
        if not filepath.exists():
            continue
        det = get_photo_detection(filepath)
        if det["crop"] is not None:
            return jsonify(thumbnail=det["crop"], photo_idx=idx)

    # Ninguna tiene detección → thumbnail normal de la foto central
    filepath = jpg_dir / photos[mid]
    thumb = make_thumbnail_b64(filepath, 300)
    return jsonify(thumbnail=thumb, photo_idx=mid)



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
    waves = data["waves"]
    no_exif = data.get("no_exif", [])

    total_files = sum(len(w["photos"]) for w in waves) + len(no_exif)
    done = 0
    stats = {"jpg": 0, "arw": 0, "waves": len(waves), "no_exif": 0}
    manifest = []

    for wave in waves:
        wave_name = wave.get("folder", wave["name"])
        dest_jpg = out_dir / wave_name / "jpg"
        dest_raw = out_dir / wave_name / "raw"
        dest_jpg.mkdir(parents=True, exist_ok=True)
        dest_raw.mkdir(parents=True, exist_ok=True)

        display_name = wave["name"]
        if wave_name != display_name:
            display_name = wave_name.replace("/", " → ")

        q.put({"type": "log", "level": "info", "text": f"── {display_name} ({len(wave['photos'])} fotos) ──"})

        # Previsualización con marca de agua: foto intermedia de la ola
        photos_list = wave["photos"]
        mid_idx = len(photos_list) // 2
        mid_photo = photos_list[mid_idx]["jpg"]
        mid_src = jpg_dir / mid_photo
        preview_path = out_dir / wave_name / "preview.jpg"
        if mid_src.exists():
            ok = make_watermarked_preview(mid_src, preview_path)
            if ok:
                q.put({"type": "log", "level": "ok", "text": f"  🖼 Preview guardada: preview.jpg"})

        for p in wave["photos"]:
            jpg_name = p["jpg"]
            raw_name = p.get("raw")

            src_jpg = jpg_dir / jpg_name
            if src_jpg.exists():
                shutil.copy2(src_jpg, dest_jpg / jpg_name)
                stats["jpg"] += 1
                manifest.append({"file": jpg_name, "dest": f"{wave_name}/jpg/{jpg_name}"})
                q.put({"type": "log", "level": "ok", "text": f"  ✔ {jpg_name}"})
            else:
                q.put({"type": "log", "level": "err", "text": f"  ✘ JPG no encontrado: {jpg_name}"})

            if raw_name:
                src_raw = raw_dir / raw_name
                if src_raw.exists():
                    shutil.copy2(src_raw, dest_raw / raw_name)
                    stats["arw"] += 1
                    manifest.append({"file": raw_name, "dest": f"{wave_name}/raw/{raw_name}"})
                else:
                    q.put({"type": "log", "level": "warn", "text": f"  ⚠ RAW no encontrado: {raw_name}"})

            done += 1
            q.put({"type": "progress", "pct": int(done / total_files * 100)})

    if no_exif:
        dest_unc_jpg = out_dir / "sin_clasificar" / "jpg"
        dest_unc_raw = out_dir / "sin_clasificar" / "raw"
        dest_unc_jpg.mkdir(parents=True, exist_ok=True)
        dest_unc_raw.mkdir(parents=True, exist_ok=True)

        q.put({"type": "log", "level": "warn", "text": f"── sin_clasificar ({len(no_exif)} fotos) ──"})

        for p in no_exif:
            jpg_name = p["jpg"]
            raw_name = p.get("raw")

            src_jpg = jpg_dir / jpg_name
            if src_jpg.exists():
                shutil.copy2(src_jpg, dest_unc_jpg / jpg_name)
                stats["jpg"] += 1
                stats["no_exif"] += 1
                manifest.append({"file": jpg_name, "dest": f"sin_clasificar/jpg/{jpg_name}"})
                q.put({"type": "log", "level": "ok", "text": f"  ✔ {jpg_name}"})

            if raw_name:
                src_raw = raw_dir / raw_name
                if src_raw.exists():
                    shutil.copy2(src_raw, dest_unc_raw / raw_name)
                    stats["arw"] += 1

            done += 1
            q.put({"type": "progress", "pct": int(done / total_files * 100)})

    manifest_path = out_dir / "manifiesto.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    q.put({"type": "log", "level": "info", "text": f"\n📄 Manifiesto guardado: {manifest_path}"})

    q.put({"type": "done", "waves": stats["waves"], "jpgs": stats["jpg"], "arws": stats["arw"], "no_exif": stats["no_exif"]})
    q.put(None)
    time.sleep(5)
    progress_queues.pop(stream_id, None)


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


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def open_browser_when_ready(port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                url = f"http://127.0.0.1:{port}"
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", url])
                elif platform.system() == "Windows":
                    os.startfile(url)
                else:
                    subprocess.Popen(["xdg-open", url])
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    print(f"⚠ Abre manualmente: http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5070
    print(f"\n🌊 Wave Splitter → http://127.0.0.1:{port}\n")
    threading.Thread(target=open_browser_when_ready, args=(port,), daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)