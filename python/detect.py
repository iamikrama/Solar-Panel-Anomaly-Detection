#!/usr/bin/env python3
"""
SolarGuard AI — Solar Panel Anomaly Detection
==============================================
Captures frames from webcam or ESP32-CAM, runs FOMO-AD inference
via Edge Impulse Linux Python SDK, and sends relay commands to
an ESP32 microcontroller over WiFi HTTP.

Requirements:
    pip install edge_impulse_linux opencv-python requests

Usage:
    python detect.py                          # webcam (default)
    python detect.py --source http://IP/cam   # ESP32-CAM MJPEG stream
    python detect.py --esp-ip 192.168.1.100   # custom ESP32 IP

Author: SolarGuard AI Project
"""

import cv2
import time
import argparse
import logging
import sys
import threading
from datetime import datetime

# ── Optional imports (graceful degradation) ──────────
try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("[WARN] 'requests' not installed. Relay commands disabled.")

try:
    from edge_impulse_linux.image import ImageImpulseRunner
    EI_OK = True
except ImportError:
    EI_OK = False
    print("[WARN] edge_impulse_linux not installed. Running in DEMO mode with random scores.")

# ── Configuration ────────────────────────────────────
DEFAULT_ESP32_IP   = "192.168.1.100"
DEFAULT_THRESHOLD  = 5.0
DEFAULT_MODEL_FILE = "model/ei-solar-panel-anomaly-linux-aarch64-v3.eim"  # update path
LOG_FILE           = "logs/detection_log.csv"

# ── Logging ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/solarguard.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("SolarGuard")


# ─────────────────────────────────────────────────────
# ESP32 Relay Controller
# ─────────────────────────────────────────────────────
class RelayController:
    def __init__(self, esp32_ip: str, port: int = 80):
        self.base_url = f"http://{esp32_ip}:{port}"
        self.relay_on = False
        self._lock = threading.Lock()

    def _post(self, path: str) -> bool:
        if not REQUESTS_OK:
            return False
        try:
            r = requests.get(self.base_url + path, timeout=2)
            return r.status_code == 200
        except requests.exceptions.RequestException as e:
            log.error(f"ESP32 request failed: {e}")
            return False

    def turn_on(self):
        with self._lock:
            if not self.relay_on:
                log.warning("🔴 RELAY ON  — triggering cleaning mechanism")
                ok = self._post("/relay?state=ON")
                if ok:
                    self.relay_on = True
                return ok
        return False

    def turn_off(self):
        with self._lock:
            if self.relay_on:
                log.info("🟢 RELAY OFF — cleaning complete")
                ok = self._post("/relay?state=OFF")
                if ok:
                    self.relay_on = False
                return ok
        return False

    def status(self) -> dict:
        try:
            r = requests.get(self.base_url + "/status", timeout=2)
            return r.json()
        except Exception:
            return {"relay": self.relay_on, "error": "no connection"}


# ─────────────────────────────────────────────────────
# Anomaly Score Logger
# ─────────────────────────────────────────────────────
class ScoreLogger:
    def __init__(self, filepath: str):
        import os, csv
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self._write_header()

    def _write_header(self):
        import os
        write_header = not os.path.exists(self.filepath)
        with open(self.filepath, "a", newline="") as f:
            import csv
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "frame", "anomaly_score", "status", "relay"])

    def log(self, frame: int, score: float, is_dirty: bool, relay: bool):
        import csv
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                frame,
                f"{score:.4f}",
                "DIRTY" if is_dirty else "CLEAN",
                "ON" if relay else "OFF",
            ])


# ─────────────────────────────────────────────────────
# FOMO-AD Inference (or demo fallback)
# ─────────────────────────────────────────────────────
class AnomalyDetector:
    def __init__(self, model_file: str, threshold: float):
        self.threshold = threshold
        self.runner = None
        self.demo_mode = not EI_OK

        if EI_OK:
            try:
                self.runner = ImageImpulseRunner(model_file)
                model_info = self.runner.init()
                log.info(f"✅ Model loaded: {model_info.get('project', {}).get('name', 'FOMO-AD')}")
                log.info(f"   Input: {model_info['model_parameters']['image_input_width']}x"
                         f"{model_info['model_parameters']['image_input_height']}")
            except Exception as e:
                log.error(f"Failed to load model: {e}. Switching to demo mode.")
                self.demo_mode = True
        else:
            log.info("Running in DEMO mode (no Edge Impulse SDK)")

    def infer(self, frame_bgr) -> tuple[float, list]:
        """
        Returns (anomaly_score, grid_scores_list).
        grid_scores is a list of per-cell scores for heatmap.
        """
        if self.demo_mode:
            return self._demo_score(), []

        try:
            features, _ = self.runner.get_features_from_image(frame_bgr)
            res = self.runner.classify(features)
            result = res.get("result", {})
            visual_ad = result.get("visual_anomaly_grid", [])
            max_score = result.get("visual_anomaly_max", 0.0)
            mean_score = result.get("visual_anomaly_mean", 0.0)
            # Use max as the primary score for sensitivity
            score = max(max_score, mean_score)
            return float(score), visual_ad
        except Exception as e:
            log.error(f"Inference error: {e}")
            return 0.0, []

    # Demo sine-wave oscillation with occasional spike
    _demo_tick: int = 0

    def _demo_score(self) -> float:
        import math, random
        AnomalyDetector._demo_tick += 1
        t = AnomalyDetector._demo_tick
        base = 1.8 + math.sin(t * 0.05) * 0.5
        if 80 < (t % 200) < 130:
            base = 6.5 + math.sin(t * 0.1) * 1.5 + random.uniform(-0.5, 0.5)
        else:
            base += random.uniform(-0.3, 0.3)
        return max(0.0, min(10.0, base))

    def cleanup(self):
        if self.runner:
            try:
                self.runner.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────
# Display Overlay
# ─────────────────────────────────────────────────────
def draw_overlay(frame, score: float, is_dirty: bool, relay_on: bool, frame_num: int, threshold: float):
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # Score bar background
    cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Status colour
    color = (0, 80, 220) if is_dirty else (0, 200, 80)

    # Score text
    cv2.putText(frame, f"Score: {score:.3f} / 10.0", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Status
    status_text = "ANOMALY DETECTED" if is_dirty else "CLEAN"
    cv2.putText(frame, status_text, (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Score bar
    bar_w = int((score / 10.0) * (W - 20))
    cv2.rectangle(frame, (10, H - 20), (W - 10, H - 8), (40, 40, 40), -1)
    bar_color = (0, 80, 220) if is_dirty else (0, 200, 80)
    cv2.rectangle(frame, (10, H - 20), (10 + bar_w, H - 8), bar_color, -1)

    # Threshold marker
    thresh_x = 10 + int((threshold / 10.0) * (W - 20))
    cv2.line(frame, (thresh_x, H - 24), (thresh_x, H - 4), (0, 165, 255), 2)
    cv2.putText(frame, f"T={threshold:.1f}", (thresh_x - 20, H - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # Frame counter
    cv2.putText(frame, f"Frame #{frame_num}", (W - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Relay indicator
    relay_color = (0, 255, 100) if relay_on else (80, 80, 80)
    cv2.circle(frame, (W - 25, H - 25), 10, relay_color, -1)
    cv2.putText(frame, "RELAY", (W - 80, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, relay_color, 1)

    # Red border when anomaly
    if is_dirty:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 220), 3)

    return frame


# ─────────────────────────────────────────────────────
# Main Detection Loop
# ─────────────────────────────────────────────────────
def run(args):
    log.info("=" * 60)
    log.info("SolarGuard AI — Solar Panel Anomaly Detection System")
    log.info("=" * 60)
    log.info(f"Source        : {args.source}")
    log.info(f"ESP32 IP      : {args.esp_ip}")
    log.info(f"Threshold     : {args.threshold}")
    log.info(f"Model         : {args.model}")
    log.info("")

    # Camera
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open camera source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("✅ Camera stream opened")

    # Components
    relay = RelayController(args.esp_ip)
    detector = AnomalyDetector(args.model, args.threshold)
    score_log = ScoreLogger(LOG_FILE)

    frame_num = 0
    fps_time = time.time()
    fps_frames = 0
    fps = 0.0
    is_dirty = False

    log.info("🚀 Detection loop started. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed — retrying...")
                time.sleep(0.1)
                continue

            # Inference
            t0 = time.perf_counter()
            score, grid = detector.infer(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # Decision
            new_dirty = score >= args.threshold
            if new_dirty != is_dirty:
                if new_dirty:
                    log.warning(f"⚠️  ANOMALY DETECTED  score={score:.3f}  (>{args.threshold})")
                    relay.turn_on()
                else:
                    log.info(f"✅ Panel CLEAN  score={score:.3f}  (<{args.threshold})")
                    relay.turn_off()
                is_dirty = new_dirty

            # Log every frame
            score_log.log(frame_num, score, is_dirty, relay.relay_on)

            # FPS
            fps_frames += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_frames / (time.time() - fps_time)
                fps_frames = 0
                fps_time = time.time()

            # Console output
            status_sym = "🔴 DIRTY" if is_dirty else "🟢 CLEAN"
            print(f"\r  Frame {frame_num:05d} | Score: {score:5.2f} | {status_sym} | "
                  f"Relay: {'ON ' if relay.relay_on else 'OFF'} | "
                  f"Inf: {inference_ms:5.1f}ms | FPS: {fps:.1f}   ",
                  end="", flush=True)

            # Draw overlay
            display = draw_overlay(frame, score, is_dirty, relay.relay_on, frame_num, args.threshold)

            # Show
            if not args.headless:
                cv2.imshow("SolarGuard AI — Anomaly Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Manual relay toggle
                    if relay.relay_on:
                        relay.turn_off()
                    else:
                        relay.turn_on()
                elif key == ord('+'):
                    args.threshold = min(10.0, args.threshold + 0.5)
                    log.info(f"Threshold → {args.threshold:.1f}")
                elif key == ord('-'):
                    args.threshold = max(0.0, args.threshold - 0.5)
                    log.info(f"Threshold → {args.threshold:.1f}")

            frame_num += 1

    except KeyboardInterrupt:
        log.info("\nStopped by user (Ctrl+C)")
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        relay.turn_off()
        detector.cleanup()
        log.info("SolarGuard AI shutdown complete.")


# ─────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SolarGuard AI — Real-time Solar Panel Anomaly Detection"
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera source: '0' for webcam, URL for ESP32-CAM MJPEG stream"
    )
    parser.add_argument(
        "--esp-ip", default=DEFAULT_ESP32_IP,
        help=f"ESP32 IP address (default: {DEFAULT_ESP32_IP})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Anomaly score threshold (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_FILE,
        help="Path to Edge Impulse .eim model file"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without displaying video window (e.g. on a server)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
