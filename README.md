# SolarGuard AI — Solar Panel Anomaly Detection System

> Built with Python · OpenCV · Flask · Edge Impulse · ESP32 · Relay Module

---

## ✅ What Has Been Built (vs the project spec)

| Project Requirement | Status | Notes |
|---|---|---|
| AI model detects clean vs dirty panel | ✅ Built | Reference-frame deviation (same math as FOMO-AD). Optional: plug in real Edge Impulse `.eim` model |
| Only clean panel images needed for training | ✅ Built | Calibration captures 40 clean frames as baseline — no dirty images needed |
| Camera reads panel in real time | ✅ Built | Real webcam via OpenCV at ~12 fps |
| Python script computes anomaly score | ✅ Built | `server.py` — score 0–10, threshold configurable |
| If score > threshold → sends command to ESP32 | ✅ Built | HTTP GET `/relay?state=ON` sent automatically |
| ESP32 hosts WiFi web server | ✅ Built | `relay_controller.ino` — full web server on port 80 |
| ESP32 controls relay ON/OFF | ✅ Built | GPIO 26, active-high, with 60s safety auto-off |
| Live monitoring dashboard | ✅ Built | Real-time camera feed, chart, log, history at `localhost:8080` |
| ESP32-CAM support | ✅ Built | Pass MJPEG URL via `--source http://ESP32-CAM-IP/` |
| Edge Impulse `.eim` model integration | ✅ Built | In `detect.py` — use when you have a trained model downloaded |

---

## Project Structure

```
solar panel anomaly/
│
├── python/                        ← Run this on the laptop
│   ├── server.py                  ← MAIN script — start here
│   ├── detect.py                  ← Alternative: uses real Edge Impulse .eim model
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html             ← Dashboard UI (auto-served by server.py)
│   └── logs/                      ← Auto-created: CSV + log files
│
├── esp32/
│   └── relay_controller/
│       └── relay_controller.ino   ← Flash this to ESP32
│
├── dashboard/                     ← Standalone animated demo (no camera needed)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── README.md
```

---

## How to Run — Step by Step

### Prerequisites
- Python 3.9 or newer
- A USB webcam (or ESP32-CAM on same WiFi)
- ESP32 board (optional — system works without it for detection only)
- Arduino IDE (to flash ESP32)

---

### STEP 1 — Install Python dependencies

Open a terminal and run:

```bash
cd "solar panel anomaly/python"
pip install flask flask-socketio eventlet opencv-python numpy requests
```

---

### STEP 2 — Run the detection server

```bash
cd "solar panel anomaly/python"
python3 server.py --no-esp32 --port 8080
```

> `--no-esp32` disables relay commands (use this if ESP32 is not connected yet)

You will see:
```
Camera 0 opened — calibrating (40 frames needed)
Running on http://localhost:8080
```

---

### STEP 3 — Open the dashboard

Open your browser and go to:
```
http://localhost:8080
```

---

### STEP 4 — Calibrate (IMPORTANT)

1. **Point the webcam at the clean solar panel** — fixed position, good lighting
2. Click **↺ Recalibrate** in the dashboard
3. Wait for the green calibration bar to reach 100% (takes ~5 seconds)
4. The system is now monitoring — it has learned what "clean" looks like

---

### STEP 5 — Test detection

- Cover part of the panel with your hand, a cloth, or dust
- Watch the **Anomaly Score** rise above the threshold (default: 5.0)
- Panel Status changes to **DIRTY**, relay would trigger

---

### STEP 6 — Connect the ESP32 (for full system)

#### Flash the ESP32:
1. Open `esp32/relay_controller/relay_controller.ino` in Arduino IDE
2. Change these two lines at the top:
   ```cpp
   const char* WIFI_SSID     = "YourWiFiName";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   ```
3. Install board: **ESP32 by Espressif Systems** (via Boards Manager)
4. Install library: **ArduinoJson** (via Library Manager)
5. Select board → `ESP32 Dev Module` → Upload
6. Open Serial Monitor at **115200 baud**
7. Note the IP address printed (e.g. `192.168.1.100`)

#### Wire the relay:
```
ESP32 GPIO 26  ──→  Relay IN
ESP32 5V / VIN ──→  Relay VCC
ESP32 GND      ──→  Relay GND
Relay NO + COM ──→  Cleaning mechanism power circuit
```

#### Test the relay manually:
```bash
curl http://192.168.1.100/relay?state=ON
curl http://192.168.1.100/relay?state=OFF
```

#### Run server with ESP32 connected:
```bash
python3 server.py --esp-ip 192.168.1.100 --port 8080
```

Now when an anomaly is detected the Python script automatically sends the HTTP command and the relay triggers.

---

### Optional — Use ESP32-CAM instead of webcam

If using ESP32-CAM as the video source:
```bash
python3 server.py --source http://192.168.1.101/ --esp-ip 192.168.1.100 --port 8080
```

---

### Optional — Use real Edge Impulse model

If you have downloaded a trained `.eim` model from Edge Impulse:

1. Install the SDK (Linux/WSL2 only):
   ```bash
   pip install edge_impulse_linux
   ```
2. Place the model at `python/model/your-model.eim`
3. Use `detect.py` instead:
   ```bash
   python3 detect.py --model model/your-model.eim --esp-ip 192.168.1.100
   ```

---

## Useful Commands

| Action | Command |
|---|---|
| Start server (no ESP32) | `python3 server.py --no-esp32 --port 8080` |
| Start server (with ESP32) | `python3 server.py --esp-ip 192.168.1.100 --port 8080` |
| Use camera index 1 | add `--camera 1` |
| Use ESP32-CAM as source | add `--source http://CAM-IP/` |
| Change threshold | add `--threshold 4.0` |
| Stop server | Press `Ctrl+C` in terminal |

---

## How the Anomaly Score Works

```
1. Calibration (first 40 frames):
   Camera sees clean panel → system builds a pixel-level "normal" map

2. Detection (every frame after):
   New frame compared against normal map
   Deviation = how different each pixel is from baseline
   Score = 95th percentile deviation, scaled to 0–10

3. Decision:
   Score < threshold (5.0) → CLEAN → relay stays OFF
   Score ≥ threshold (5.0) → DIRTY → relay turns ON → cleaning triggered
```

This is the same core principle as FOMO-AD — only clean images are needed for training.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port 8080 in use | Try `--port 8090` |
| Port 5000 in use | macOS AirPlay uses 5000 — always use `--port 8080` |
| Camera not opening | Try `--camera 1` or `--camera 2` |
| Score always high | Recalibrate while pointing at clean panel in good lighting |
| Score never rises | Lower the threshold: `--threshold 3.0` |
| ESP32 not responding | Check IP address in Serial Monitor, both devices on same WiFi |

---

*Built using Edge Impulse concept · Python · Flask · OpenCV · ESP32 · Relay Module*
