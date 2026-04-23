#!/usr/bin/env python3
"""
SolarGuard AI — Data Capture Interface
=======================================
Step 1 of the pipeline:
  - Shows live stream from phone (or webcam)
  - Press "Capture" button → saves ONE image to dataset/
  - Label each image: Normal or Anomaly
  - Build a labelled dataset for training

Run:
    python3 capture_server.py --source http://PHONE_IP:8080/video
    python3 capture_server.py --source 0   (webcam fallback)

Then open http://localhost:8090 in your browser.
"""
from __future__ import annotations
import os, cv2, time, json, threading, argparse, shutil
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template_string

# ── Directories ───────────────────────────────────────
DATASET_DIR  = "dataset"
NORMAL_DIR   = os.path.join(DATASET_DIR, "normal")
ANOMALY_DIR  = os.path.join(DATASET_DIR, "anomaly")
for d in [NORMAL_DIR, ANOMALY_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)

# ── Camera ────────────────────────────────────────────
cap = None
cap_lock = threading.Lock()
latest_frame = None
frame_lock   = threading.Lock()

def open_camera(source):
    global cap
    src = int(source) if str(source).isdigit() else source
    with cap_lock:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {src}")
            return False
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[OK] Source opened: {src}")
        return True

def capture_loop():
    """Continuously read frames into latest_frame buffer."""
    global latest_frame
    while True:
        if cap and cap.isOpened():
            with cap_lock:
                ret, frame = cap.read()
            if ret and frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()
        time.sleep(0.04)

threading.Thread(target=capture_loop, daemon=True).start()

# ── MJPEG stream ──────────────────────────────────────
def gen_stream():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            # Overlay timestamp
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, ts, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        time.sleep(0.04)

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Capture endpoint ──────────────────────────────────
@app.route("/capture", methods=["POST"])
def capture():
    data  = request.get_json(silent=True) or {}
    label = data.get("label", "normal").lower()  # 'normal' or 'anomaly'

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        return jsonify({"ok": False, "error": "No frame available"})

    folder = NORMAL_DIR if label == "normal" else ANOMALY_DIR
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    fname  = f"{label}_{ts}.jpg"
    fpath  = os.path.join(folder, fname)
    cv2.imwrite(fpath, frame)

    # Counts
    n_normal  = len(os.listdir(NORMAL_DIR))
    n_anomaly = len(os.listdir(ANOMALY_DIR))
    print(f"[CAPTURE] {label.upper()} → {fpath}  (total: normal={n_normal}, anomaly={n_anomaly})")

    # Return thumbnail as base64 for preview
    small = cv2.resize(frame, (160, 120))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    import base64
    thumb = base64.b64encode(buf.tobytes()).decode() if ok else ""

    return jsonify({
        "ok": True, "label": label, "file": fname,
        "n_normal": n_normal, "n_anomaly": n_anomaly,
        "thumb": thumb
    })

@app.route("/counts")
def counts():
    return jsonify({
        "normal":  len(os.listdir(NORMAL_DIR)),
        "anomaly": len(os.listdir(ANOMALY_DIR)),
    })

@app.route("/clear", methods=["POST"])
def clear_dataset():
    data = request.get_json(silent=True) or {}
    label = data.get("label", "all")
    if label in ("normal", "all"):
        shutil.rmtree(NORMAL_DIR, ignore_errors=True)
        os.makedirs(NORMAL_DIR, exist_ok=True)
    if label in ("anomaly", "all"):
        shutil.rmtree(ANOMALY_DIR, ignore_errors=True)
        os.makedirs(ANOMALY_DIR, exist_ok=True)
    return jsonify({"ok": True})

@app.route("/")
def index():
    return render_template_string(CAPTURE_HTML)

# ── HTML ──────────────────────────────────────────────
CAPTURE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SolarGuard — Data Capture</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#060b14;--bg2:#0c1423;--border:rgba(99,179,237,.13);--accent:#38bdf8;--green:#22c55e;--red:#ef4444;--orange:#f97316;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;--card:rgba(13,22,40,.9)}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.bg-glow{position:fixed;border-radius:50%;filter:blur(120px);opacity:.12;pointer-events:none;z-index:0}
.g1{width:500px;height:500px;top:-150px;right:-100px;background:radial-gradient(circle,#0ea5e9,transparent 70%)}
.g2{width:400px;height:400px;bottom:-100px;left:-80px;background:radial-gradient(circle,#6366f1,transparent 70%)}
header{position:sticky;top:0;z-index:100;background:rgba(6,11,20,.95);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:0 24px}
.hdr{max-width:1300px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:58px}
.brand{display:flex;align-items:center;gap:10px;font-size:1rem;font-weight:700}
.brand span{color:var(--accent)}
.step-badge{background:rgba(56,189,248,.12);border:1px solid rgba(56,189,248,.3);color:var(--accent);font-size:.72rem;padding:4px 12px;border-radius:20px;font-weight:600}
main{position:relative;z-index:1;max-width:1300px;margin:0 auto;padding:20px 24px 48px;display:grid;grid-template-columns:1fr 380px;gap:20px}
@media(max-width:900px){main{grid-template-columns:1fr}}

.camera-panel{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.panel-head{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.panel-title{font-size:.8rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.06em}
.live-dot{display:flex;align-items:center;gap:5px;font-size:.68rem;color:var(--red);font-weight:700}
.rdot{width:6px;height:6px;border-radius:50%;background:var(--red);animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.feed-wrap{position:relative;background:#000}
#feed{display:block;width:100%;height:auto}
.feed-overlay{position:absolute;inset:0;pointer-events:none}
.corner{position:absolute;width:20px;height:20px;border-color:var(--accent);border-style:solid;opacity:.7}
.tl{top:10px;left:10px;border-width:2px 0 0 2px}
.tr{top:10px;right:10px;border-width:2px 2px 0 0}
.bl{bottom:10px;left:10px;border-width:0 0 2px 2px}
.br{bottom:10px;right:10px;border-width:0 2px 2px 0}
.flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .05s}
.flash.show{opacity:.6}

.controls{display:flex;flex-direction:column;gap:16px}

.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px}
.card-title{font-size:.78rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}

/* Label toggle */
.label-toggle{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.lbtn{padding:12px;border-radius:10px;font-size:.85rem;font-weight:600;cursor:pointer;border:2px solid transparent;transition:all .2s;text-align:center;user-select:none}
.lbtn.normal{background:rgba(34,197,94,.1);color:var(--green);border-color:rgba(34,197,94,.25)}
.lbtn.anomaly{background:rgba(239,68,68,.1);color:var(--red);border-color:rgba(239,68,68,.25)}
.lbtn.active.normal{background:rgba(34,197,94,.25);border-color:var(--green);box-shadow:0 0 16px rgba(34,197,94,.2)}
.lbtn.active.anomaly{background:rgba(239,68,68,.25);border-color:var(--red);box-shadow:0 0 16px rgba(239,68,68,.2)}

/* Capture button */
.capture-btn{width:100%;padding:18px;border-radius:12px;font-size:1.1rem;font-weight:800;cursor:pointer;border:none;background:linear-gradient(135deg,#0ea5e9,#6366f1);color:#fff;letter-spacing:.02em;transition:transform .1s,box-shadow .2s;box-shadow:0 4px 24px rgba(14,165,233,.3)}
.capture-btn:active{transform:scale(.97)}
.capture-btn:hover{box-shadow:0 6px 32px rgba(14,165,233,.45)}
.capture-btn:disabled{opacity:.5;cursor:not-allowed}
.shortcut{text-align:center;font-size:.7rem;color:var(--text3);margin-top:6px}

/* Counters */
.counters{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.counter{text-align:center;padding:14px;border-radius:10px}
.counter.n{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2)}
.counter.a{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2)}
.cnum{font-size:2rem;font-weight:800;display:block}
.counter.n .cnum{color:var(--green)}
.counter.a .cnum{color:var(--red)}
.clabel{font-size:.68rem;color:var(--text3);text-transform:uppercase;letter-spacing:.07em}

/* Gallery */
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;max-height:260px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.gthumb{position:relative;border-radius:6px;overflow:hidden;aspect-ratio:4/3}
.gthumb img{width:100%;height:100%;object-fit:cover}
.gthumb .glabel{position:absolute;bottom:0;left:0;right:0;font-size:.6rem;font-weight:600;text-align:center;padding:2px}
.gthumb.gn .glabel{background:rgba(34,197,94,.8);color:#fff}
.gthumb.ga .glabel{background:rgba(239,68,68,.8);color:#fff}

/* Action buttons */
.action-row{display:flex;gap:8px}
.abtn{flex:1;padding:10px;border-radius:8px;font-size:.75rem;font-weight:600;cursor:pointer;border:1px solid var(--border);background:rgba(255,255,255,.05);color:var(--text2);font-family:'Inter',sans-serif;transition:all .2s}
.abtn:hover{border-color:var(--accent);color:var(--accent);background:rgba(56,189,248,.08)}
.abtn.go{background:rgba(34,197,94,.15);border-color:var(--green);color:var(--green)}
.abtn.go:hover{background:rgba(34,197,94,.25)}

/* Toast */
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:rgba(13,22,40,.95);border:1px solid var(--border);border-radius:10px;padding:10px 20px;font-size:.82rem;backdrop-filter:blur(12px);opacity:0;transition:opacity .3s;pointer-events:none;z-index:200;white-space:nowrap}
.toast.show{opacity:1}
.toast.ok{border-color:rgba(34,197,94,.4);color:var(--green)}
.toast.err{border-color:rgba(239,68,68,.4);color:var(--red)}
</style>
</head>
<body>
<div class="bg-glow g1"></div><div class="bg-glow g2"></div>
<header>
  <div class="hdr">
    <div class="brand">☀️ SolarGuard <span>AI</span> &nbsp;—&nbsp; Data Capture</div>
    <div class="step-badge">Step 1 of 3 — Capture</div>
  </div>
</header>
<main>
  <!-- Camera feed -->
  <div class="camera-panel">
    <div class="panel-head">
      <span class="panel-title">📷 Live Phone / Camera Feed</span>
      <span class="live-dot"><span class="rdot"></span>LIVE</span>
    </div>
    <div class="feed-wrap">
      <img id="feed" src="/video_feed" alt="feed"/>
      <div class="feed-overlay">
        <div class="corner tl"></div><div class="corner tr"></div>
        <div class="corner bl"></div><div class="corner br"></div>
      </div>
      <div class="flash" id="flash"></div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">

    <!-- Label selector -->
    <div class="card">
      <div class="card-title">1. Choose Label</div>
      <div class="label-toggle">
        <div class="lbtn normal active" id="btnNormal" onclick="setLabel('normal')">✅ Normal</div>
        <div class="lbtn anomaly" id="btnAnomaly" onclick="setLabel('anomaly')">⚠️ Anomaly</div>
      </div>
    </div>

    <!-- Capture button -->
    <div class="card">
      <div class="card-title">2. Capture Image</div>
      <button class="capture-btn" id="captureBtn" onclick="captureImage()">📸 Capture</button>
      <div class="shortcut">Press <kbd style="background:rgba(255,255,255,.1);padding:1px 6px;border-radius:4px;font-family:monospace">Space</kbd> to capture</div>
    </div>

    <!-- Counters -->
    <div class="card">
      <div class="card-title">3. Dataset Progress</div>
      <div class="counters">
        <div class="counter n"><span class="cnum" id="cntNormal">0</span><span class="clabel">Normal</span></div>
        <div class="counter a"><span class="cnum" id="cntAnomaly">0</span><span class="clabel">Anomaly</span></div>
      </div>
      <div style="margin-top:12px;font-size:.72rem;color:var(--text3);text-align:center">
        Recommended: ≥ 40 normal + ≥ 20 anomaly images
      </div>
    </div>

    <!-- Recent captures -->
    <div class="card">
      <div class="card-title">Recent Captures</div>
      <div class="gallery" id="gallery"></div>
    </div>

    <!-- Actions -->
    <div class="action-row">
      <button class="abtn" onclick="clearDataset('normal')">🗑 Clear Normal</button>
      <button class="abtn" onclick="clearDataset('anomaly')">🗑 Clear Anomaly</button>
      <button class="abtn go" onclick="goTrain()">🧠 Train Model →</button>
    </div>

  </div>
</main>
<div class="toast" id="toast"></div>

<script>
let currentLabel = 'normal';
let capturing = false;

function setLabel(l) {
  currentLabel = l;
  document.getElementById('btnNormal').classList.toggle('active', l==='normal');
  document.getElementById('btnAnomaly').classList.toggle('active', l==='anomaly');
}

async function captureImage() {
  if (capturing) return;
  capturing = true;
  const btn = document.getElementById('captureBtn');
  btn.disabled = true;

  // Flash effect
  const flash = document.getElementById('flash');
  flash.classList.add('show');
  setTimeout(() => flash.classList.remove('show'), 150);

  try {
    const res = await fetch('/capture', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({label: currentLabel})
    });
    const d = await res.json();
    if (d.ok) {
      document.getElementById('cntNormal').textContent  = d.n_normal;
      document.getElementById('cntAnomaly').textContent = d.n_anomaly;
      addThumb(d.thumb, d.label, d.file);
      showToast(`✅ ${d.label.toUpperCase()} captured (${d.label==='normal'?d.n_normal:d.n_anomaly} total)`, 'ok');
    } else {
      showToast('❌ ' + (d.error || 'Capture failed'), 'err');
    }
  } catch(e) {
    showToast('❌ Network error', 'err');
  }

  setTimeout(() => { btn.disabled = false; capturing = false; }, 400);
}

function addThumb(base64, label, fname) {
  const gallery = document.getElementById('gallery');
  const div = document.createElement('div');
  div.className = 'gthumb ' + (label==='normal'?'gn':'ga');
  div.innerHTML = `<img src="data:image/jpeg;base64,${base64}"/><div class="glabel">${label}</div>`;
  gallery.prepend(div);
  // Keep max 30 thumbs
  while(gallery.children.length > 30) gallery.removeChild(gallery.lastChild);
}

function clearDataset(label) {
  if (!confirm(`Clear all ${label} images?`)) return;
  fetch('/clear', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({label})})
    .then(()=>loadCounts())
    .then(()=>showToast(`🗑 ${label} images cleared`, 'ok'));
}

function loadCounts() {
  return fetch('/counts').then(r=>r.json()).then(d=>{
    document.getElementById('cntNormal').textContent  = d.normal;
    document.getElementById('cntAnomaly').textContent = d.anomaly;
  });
}

function goTrain() {
  const n = parseInt(document.getElementById('cntNormal').textContent);
  if (n < 10) { showToast('⚠️ Capture at least 10 normal images first', 'err'); return; }
  window.location.href = 'http://localhost:8091';
}

let toastTimer;
function showToast(msg, type='ok') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 3000);
}

// Spacebar shortcut
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    captureImage();
  }
});

loadCounts();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SolarGuard AI — Data Capture")
    parser.add_argument("--source", default="0",
                        help="Camera: '0' for webcam, or http://PHONE_IP:8080/video")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    open_camera(args.source)
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  SolarGuard AI — Data Capture UI    ║")
    print(f"  ║  Open: http://localhost:{args.port}        ║")
    print(f"  ║  Press SPACE or click Capture        ║")
    print(f"  ╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
