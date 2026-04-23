#!/usr/bin/env python3
from __future__ import annotations
"""
SolarGuard AI — Real-Time Anomaly Detection Server
===================================================
Opens real webcam, computes anomaly scores by comparing each frame
against a learned "clean" reference baseline, streams video to the
browser dashboard via MJPEG, and pushes scores via Socket.IO.

Sends relay commands to ESP32 over WiFi when anomaly is detected.

Run:
    python server.py
    python server.py --camera 1 --esp-ip 192.168.1.100 --threshold 5.0

Then open http://localhost:5000 in your browser.
"""

import os
import cv2
import time
import math
import json
import logging
import argparse
import threading
import numpy as np
from datetime import datetime
from collections import deque

# Flask + SocketIO
from flask import Flask, Response, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# HTTP for ESP32 relay
try:
    import requests as http_requests
    HTTP_OK = True
except ImportError:
    HTTP_OK = False

# ── Logging ───────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("SolarGuard")

# ── Flask App ─────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "solarguard-ai-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ─────────────────────────────────────────────────────
#  Anomaly Detection Engine
#  Uses reference-frame deviation — same concept as FOMO-AD:
#  learn what "normal" looks like, flag deviations
# ─────────────────────────────────────────────────────
class AnomalyEngine:
    def __init__(self, threshold: float = 5.0, ref_frames: int = 40):
        self.threshold = threshold
        self.ref_frames_needed = ref_frames

        # Reference model
        self._ref_mean = None   # mean of clean frames (np.ndarray)
        self._ref_std  = None   # std of clean frames (np.ndarray)
        self._calibration_frames: list = []
        self.calibrated = False
        self.calibration_count = 0

        # Warmup: ignore first N frames after calibration to let score settle
        self.warmup_frames_needed = 20
        self.warmup_count = 0
        self.warmed_up = False

        # Score smoothing (larger window = more stable)
        self._score_window = deque(maxlen=8)

        # Stats
        self.total_frames = 0
        self.dirty_frames = 0

    def calibrate_frame(self, frame_gray: np.ndarray) -> bool:
        """
        Feed clean frames during calibration phase.
        Returns True when calibration is complete.
        """
        self._calibration_frames.append(frame_gray.astype(np.float32))
        self.calibration_count += 1
        if self.calibration_count >= self.ref_frames_needed:
            stack = np.stack(self._calibration_frames, axis=0)
            self._ref_mean = np.mean(stack, axis=0)
            self._ref_std  = np.std(stack, axis=0) + 1e-6   # avoid div/0
            self._calibration_frames = []
            self.calibrated = True
            self.warmup_count = 0
            self.warmed_up = False
            log.info(f"✅ Calibration complete ({self.ref_frames_needed} frames) — warming up ({self.warmup_frames_needed} frames)...")
            return True
        return False

    def reset_calibration(self):
        self._ref_mean = None
        self._ref_std  = None
        self._calibration_frames = []
        self.calibrated = False
        self.calibration_count = 0
        self.warmup_count = 0
        self.warmed_up = False
        self._score_window.clear()
        log.info("🔄 Calibration reset — recapturing reference")

    def score(self, frame_gray: np.ndarray):
        """
        Compute anomaly score (0–10) for a frame.
        Returns (score, heatmap_bgr, is_warmed_up).

        Method: pixel-wise normalised deviation from clean reference.
        Each pixel deviation = |pixel - mean| / std
        Score = 90th-percentile deviation, scaled to 0–10.
        During warmup phase, score is computed but NO relay decisions made.
        """
        self.total_frames += 1

        # Warmup: let rolling window fill before making decisions
        if not self.warmed_up:
            self.warmup_count += 1
            if self.warmup_count >= self.warmup_frames_needed:
                self.warmed_up = True
                log.info("✅ Warmup complete — anomaly detection ACTIVE")

        f = frame_gray.astype(np.float32)
        deviation = np.abs(f - self._ref_mean) / self._ref_std

        # 90th-percentile: more stable than 95th, less noise-sensitive
        raw = float(np.percentile(deviation, 90))

        # Scale: a well-calibrated still panel gives raw ~0.5–1.5 → score ~1–3
        # Significant change (dust/hand) gives raw ~3–5 → score ~6–10
        scaled = min(raw * 2.0, 10.0)

        # Smooth over rolling window
        self._score_window.append(scaled)
        smooth = float(np.mean(self._score_window))

        if self.warmed_up and smooth >= self.threshold:
            self.dirty_frames += 1

        # Build heatmap for overlay
        heatmap = self._make_heatmap(deviation, frame_gray.shape)

        return smooth, heatmap, self.warmed_up

    def _make_heatmap(self, deviation: np.ndarray, shape) -> np.ndarray:
        # Normalise deviation to 0-255
        d_norm = np.clip(deviation / (deviation.max() + 1e-6), 0, 1)
        d_u8 = (d_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        return heatmap


# ─────────────────────────────────────────────────────
#  ESP32 Relay Controller
# ─────────────────────────────────────────────────────
class RelayController:
    def __init__(self, esp32_ip: str, enabled: bool = True):
        self.base_url = f"http://{esp32_ip}"
        self.enabled = enabled and HTTP_OK
        self.state = False
        self._lock = threading.Lock()
        self.esp32_ip = esp32_ip

    def _send(self, path: str, label: str) -> bool:
        url = self.base_url + path
        if not self.enabled:
            # ESP32 disabled — log what WOULD have been sent
            log.info(f"[ESP32 DISABLED] Would send → GET {url}")
            return True
        # Real HTTP request to ESP32
        log.info(f"📡 Sending to ESP32 → GET {url}")
        try:
            r = http_requests.get(url, timeout=2)
            if r.status_code == 200:
                log.info(f"✅ ESP32 response OK → relay {label}")
                return True
            else:
                log.warning(f"⚠️  ESP32 returned HTTP {r.status_code}")
                return False
        except Exception as e:
            log.error(f"❌ ESP32 unreachable ({self.esp32_ip}): {e}")
            return False

    def on(self) -> bool:
        with self._lock:
            if not self.state:
                log.warning("🔴 RELAY ON  — panel dirty, triggering cleaning")
                self._send("/relay?state=ON", "ON")
                self.state = True
            return self.state

    def off(self) -> bool:
        with self._lock:
            if self.state:
                log.info("🟢 RELAY OFF — panel clean, stopping cleaning")
                self._send("/relay?state=OFF", "OFF")
                self.state = False
            return self.state


# ─────────────────────────────────────────────────────
#  Camera Thread
# ─────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, camera_index: int, engine: AnomalyEngine,
                 relay: RelayController, heatmap_alpha: float = 0.45):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.engine = engine
        self.relay = relay
        self.heatmap_alpha = heatmap_alpha

        self._lock = threading.Lock()
        self._frame_jpg = None   # latest MJPEG frame bytes (bytes)
        self._running = True

        # Shared state (read by Flask routes)
        self.score = 0.0
        self.is_dirty = False
        self.fps = 0.0
        self.frame_num = 0
        self.state = "calibrating"   # 'calibrating' | 'warmup' | 'monitoring'
        self.cap_open = False

        # Require N consecutive dirty frames before triggering relay
        # Prevents single-frame noise from firing the relay
        self._consecutive_dirty = 0
        self._consecutive_clean = 0
        self.DIRTY_CONFIRM_FRAMES = 5   # must be dirty for 5 frames in a row
        self.CLEAN_CONFIRM_FRAMES = 8   # must be clean for 8 frames to turn off

        # History for chart (last 120 readings)
        self.score_history = deque(maxlen=120)

        # Event log
        self.events = deque(maxlen=200)

        # Session counts
        self.clean_count = 0
        self.dirty_count = 0
        self.relay_triggers = 0

    def _log_event(self, level: str, msg: str):
        entry = {
            "t": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "msg": msg
        }
        self.events.appendleft(entry)
        socketio.emit("log", entry)
        log_fn = {"info": log.info, "warn": log.warning,
                   "error": log.error, "success": log.info}.get(level, log.info)
        log_fn(msg)

    def _emit_state(self):
        socketio.emit("state", {
            "score": round(self.score, 3),
            "is_dirty": self.is_dirty,
            "relay": self.relay.state,
            "frame": self.frame_num,
            "fps": round(self.fps, 1),
            "state": self.state,
            "calibration_progress": min(
                self.engine.calibration_count / self.engine.ref_frames_needed, 1.0
            ),
            "warmup_progress": min(
                self.engine.warmup_count / max(self.engine.warmup_frames_needed, 1), 1.0
            ),
            "threshold": self.engine.threshold,
            "clean_count": self.clean_count,
            "dirty_count": self.dirty_count,
            "relay_triggers": self.relay_triggers,
            "esp32_ip": self.relay.esp32_ip,
            "esp32_enabled": self.relay.enabled,
        })

    def get_jpeg(self):
        with self._lock:
            return self._frame_jpg

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self.camera_index}")
            self._log_event("error", f"Camera {self.camera_index} not found")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap_open = True
        self._log_event("success", f"Camera {self.camera_index} opened — calibrating ({self.engine.ref_frames_needed} frames needed)")

        fps_t0 = time.time()
        fps_count = 0
        last_emit = time.time()
        prev_dirty = False

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)   # mild denoise

            draw = frame.copy()

            if not self.engine.calibrated:
                # ── Calibration phase ──────────────────────────────────────
                self.state = "calibrating"
                done = self.engine.calibrate_frame(gray)
                progress = self.engine.calibration_count / self.engine.ref_frames_needed

                cv2.rectangle(draw, (0, 0), (640, 480), (30, 150, 30), 3)
                pct = int(progress * 100)
                bar_w = int(progress * 580)
                cv2.rectangle(draw, (30, 440), (610, 462), (40, 40, 40), -1)
                cv2.rectangle(draw, (30, 440), (30 + bar_w, 462), (50, 200, 80), -1)
                cv2.putText(draw, f"CALIBRATING — keep panel CLEAN: {pct}%",
                            (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
                cv2.putText(draw, "Step 1: Point camera at CLEAN solar panel",
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 1)

                if done:
                    self._log_event("success", "Reference baseline captured — warming up...")
                    self.state = "warmup"

            elif not self.engine.warmed_up:
                # ── Warmup phase (post-calibration settling) ───────────────
                self.state = "warmup"
                score, heatmap, _ = self.engine.score(gray)
                self.score = score
                wp = self.engine.warmup_count / self.engine.warmup_frames_needed

                cv2.rectangle(draw, (0, 0), (640, 480), (30, 100, 180), 3)
                cv2.putText(draw, f"WARMING UP — settling score: {score:.2f}",
                            (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 180, 255), 2)
                cv2.putText(draw, "Almost ready — do not move the camera",
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 1)

                if self.engine.warmed_up:
                    self._log_event("success", "Warmup complete — anomaly detection ACTIVE")
                    self.state = "monitoring"

            else:
                # ── Detection phase (real anomaly detection + relay control) ──
                self.state = "monitoring"
                score, heatmap, _ = self.engine.score(gray)
                self.score = score
                above_threshold = score >= self.engine.threshold

                # ── Consecutive-frame confirmation (debounce) ──────────────
                # Relay only fires after DIRTY_CONFIRM_FRAMES consecutive dirty frames
                # Relay turns off after CLEAN_CONFIRM_FRAMES consecutive clean frames
                if above_threshold:
                    self._consecutive_dirty += 1
                    self._consecutive_clean = 0
                else:
                    self._consecutive_clean += 1
                    self._consecutive_dirty = 0

                # Decide actual dirty state with debounce
                if self._consecutive_dirty >= self.DIRTY_CONFIRM_FRAMES:
                    is_dirty = True
                elif self._consecutive_clean >= self.CLEAN_CONFIRM_FRAMES:
                    is_dirty = False
                else:
                    is_dirty = self.is_dirty   # hold previous state

                self.is_dirty = is_dirty

                # Heatmap overlay
                heatmap_full = cv2.resize(heatmap, (640, 480))
                alpha = self.heatmap_alpha if is_dirty else 0.0
                if alpha > 0:
                    draw = cv2.addWeighted(draw, 1 - alpha, heatmap_full, alpha, 0)

                # ── Relay decision on state change ─────────────────────────
                if is_dirty and not prev_dirty:
                    # Panel just became dirty → send ON to ESP32
                    self.dirty_count += 1
                    self.relay_triggers += 1
                    self.relay.on()
                    msg = (f"🔴 ANOMALY DETECTED  score={score:.2f}  "
                           f"(>={self.engine.threshold})  "
                           f"→ HTTP GET /relay?state=ON → ESP32 {self.relay.esp32_ip}")
                    self._log_event("error", msg)
                    socketio.emit("history_row", {
                        "t": datetime.now().strftime("%H:%M:%S"),
                        "score": round(score, 2),
                        "status": "DIRTY",
                        "action": f"Relay ON → {self.relay.esp32_ip}"
                    })

                elif not is_dirty and prev_dirty:
                    # Panel just became clean → send OFF to ESP32
                    self.clean_count += 1
                    self.relay.off()
                    msg = (f"🟢 Panel CLEAN  score={score:.2f}  "
                           f"(<{self.engine.threshold})  "
                           f"→ HTTP GET /relay?state=OFF → ESP32 {self.relay.esp32_ip}")
                    self._log_event("success", msg)
                    socketio.emit("history_row", {
                        "t": datetime.now().strftime("%H:%M:%S"),
                        "score": round(score, 2),
                        "status": "CLEAN",
                        "action": f"Relay OFF → {self.relay.esp32_ip}"
                    })

                prev_dirty = is_dirty

                # Score history for chart
                self.score_history.append({
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "v": round(score, 3)
                })

                draw = self._draw_overlay(draw, score, is_dirty)

            # Encode as JPEG for MJPEG stream
            ok, buf = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._frame_jpg = buf.tobytes()

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                self.fps = fps_count / elapsed
                fps_count = 0
                fps_t0 = time.time()

            self.frame_num += 1

            # Emit state ~5×/sec
            if time.time() - last_emit >= 0.2:
                self._emit_state()
                last_emit = time.time()

        cap.release()
        log.info("Camera released")

    def _draw_overlay(self, frame: np.ndarray, score: float, is_dirty: bool) -> np.ndarray:
        H, W = frame.shape[:2]
        color = (40, 60, 230) if is_dirty else (50, 200, 80)

        # Top info bar
        cv2.rectangle(frame, (0, 0), (W, 36), (0, 0, 0), -1)
        status = "ANOMALY DETECTED" if is_dirty else "CLEAN"
        cv2.putText(frame, f"Score: {score:.3f}  |  {status}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Frame #{self.frame_num}  FPS:{self.fps:.1f}",
                    (W - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 150, 150), 1)

        # Score bar at bottom
        cv2.rectangle(frame, (0, H - 18), (W, H), (20, 20, 20), -1)
        bar_w = int((score / 10.0) * (W - 4))
        bar_color = (40, 60, 230) if is_dirty else (50, 200, 80)
        cv2.rectangle(frame, (2, H - 16), (2 + bar_w, H - 2), bar_color, -1)

        # Threshold marker
        tx = 2 + int((self.engine.threshold / 10.0) * (W - 4))
        cv2.line(frame, (tx, H - 20), (tx, H), (0, 140, 255), 2)

        # Border flash when dirty
        if is_dirty:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (40, 40, 230), 4)

        # Relay indicator
        r_color = (50, 220, 80) if self.relay.state else (80, 80, 80)
        cv2.circle(frame, (W - 20, H - 30), 8, r_color, -1)
        cv2.putText(frame, "RELAY", (W - 75, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, r_color, 1)

        return frame


# ─────────────────────────────────────────────────────
#  Global objects (set in main())
# ─────────────────────────────────────────────────────
camera_thread = None
relay = None
engine = None


# ─────────────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────────────
def generate_mjpeg():
    """Generator for MJPEG video stream."""
    while True:
        jpg = camera_thread.get_jpeg() if camera_thread else None
        if jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)   # ~25 fps max to browser


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    if not camera_thread:
        return jsonify({"error": "not started"})
    return jsonify({
        "score": camera_thread.score,
        "is_dirty": camera_thread.is_dirty,
        "relay": relay.state,
        "frame": camera_thread.frame_num,
        "fps": camera_thread.fps,
        "state": camera_thread.state,
        "threshold": engine.threshold,
        "calibrated": engine.calibrated,
    })


@app.route("/api/relay", methods=["POST"])
def api_relay():
    data = request.get_json(silent=True) or {}
    state = data.get("state", "").upper()
    if state == "ON":
        relay.on()
    elif state == "OFF":
        relay.off()
    return jsonify({"relay": relay.state})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    engine.reset_calibration()
    camera_thread._log_event("warn", "Manual recalibration triggered")
    return jsonify({"ok": True})


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    data = request.get_json(silent=True) or {}
    val = float(data.get("threshold", engine.threshold))
    engine.threshold = max(0.5, min(10.0, val))
    camera_thread._log_event("info", f"Threshold updated → {engine.threshold:.1f}")
    return jsonify({"threshold": engine.threshold})


@app.route("/api/history")
def api_history():
    if not camera_thread:
        return jsonify([])
    return jsonify(list(camera_thread.score_history))


@app.route("/api/events")
def api_events():
    if not camera_thread:
        return jsonify([])
    return jsonify(list(camera_thread.events))


@app.route("/")
def index():
    return render_template_string(open("templates/index.html").read())


# ─────────────────────────────────────────────────────
#  Socket.IO events
# ─────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    if camera_thread:
        emit("log", {"t": datetime.now().strftime("%H:%M:%S"),
                     "level": "info",
                     "msg": "Dashboard connected to SolarGuard server"})
        # Send existing event history
        emit("event_history", list(camera_thread.events))


@socketio.on("set_threshold")
def on_threshold(data):
    val = float(data.get("threshold", engine.threshold))
    engine.threshold = max(0.5, min(10.0, val))
    camera_thread._log_event("info", f"Threshold updated → {engine.threshold:.1f}")


@socketio.on("manual_relay")
def on_manual_relay(data):
    state = data.get("state", False)
    if state:
        relay.on()
        camera_thread._log_event("warn", "Manual relay ON")
    else:
        relay.off()
        camera_thread._log_event("info", "Manual relay OFF")


@socketio.on("recalibrate")
def on_recalibrate():
    engine.reset_calibration()
    camera_thread._log_event("warn", "Recalibration started — keep panel CLEAN")


# ─────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────
def main():
    global camera_thread, relay, engine

    parser = argparse.ArgumentParser(description="SolarGuard AI Server")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--esp-ip", default="192.168.1.100",
                        help="ESP32 IP address")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Anomaly score threshold (default: 5.0)")
    parser.add_argument("--ref-frames", type=int, default=40,
                        help="Number of frames for reference calibration (default: 40)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web server port (default: 5000)")
    parser.add_argument("--no-esp32", action="store_true",
                        help="Disable ESP32 relay (dashboard only)")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  SolarGuard AI — Real-Time Detection Server")
    log.info("=" * 55)
    log.info(f"  Camera      : {args.camera}")
    log.info(f"  ESP32 IP    : {args.esp_ip}  (disabled: {args.no_esp32})")
    log.info(f"  Threshold   : {args.threshold}")
    log.info(f"  Ref Frames  : {args.ref_frames}")
    log.info(f"  Dashboard   : http://localhost:{args.port}")
    log.info("=" * 55)

    engine = AnomalyEngine(
        threshold=args.threshold,
        ref_frames=args.ref_frames
    )
    relay = RelayController(
        esp32_ip=args.esp_ip,
        enabled=not args.no_esp32
    )
    camera_thread = CameraThread(
        camera_index=args.camera,
        engine=engine,
        relay=relay
    )
    camera_thread.start()

    socketio.run(app, host="0.0.0.0", port=args.port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
