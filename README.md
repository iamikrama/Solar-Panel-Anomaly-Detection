# SolarGuard AI — Solar Panel Anomaly Detection System

> **Phone Camera → Capture Dataset → Train Model → Live Detection → ESP32 Relay Trigger**

Built with Python · OpenCV · scikit-learn · Flask · ESP32

---

## What Is This Project?

An AI-powered system that automatically detects when a solar panel is dirty or covered, and triggers a physical cleaning mechanism — no human involvement needed.

Solar panels lose efficiency when covered with dust, bird droppings, or debris. This system automates the entire detection and cleaning process using a phone camera, machine learning, and an ESP32 microcontroller.

---

## System Architecture

```
📱 Phone Camera
      │
      │  HTTP stream (WiFi)
      ▼
🖥️  Python Server (laptop)
      │
      ├─ capture_server.py  →  capture labelled images (Step 1)
      ├─ train.py           →  train ML model on dataset (Step 2)
      └─ server.py          →  live detection + relay control (Step 3)
                                    │
                                    │  HTTP request (WiFi)
                                    ▼
                              📡 ESP32 Microcontroller
                                    │
                                    ▼
                              ⚡ Relay Module → Cleaning Mechanism
```

---

## Project Structure

```
solar panel anomaly/
│
├── python/
│   ├── capture_server.py      ← STEP 1: Capture images from phone
│   ├── train.py               ← STEP 2: Train the ML model
│   ├── server.py              ← STEP 3: Live detection + ESP32 control
│   ├── detect.py              ← Alternative: Edge Impulse .eim model
│   ├── requirements.txt
│   ├── dataset/               ← Auto-created: captured images
│   │   ├── normal/            ← Clean panel images
│   │   └── anomaly/           ← Dirty panel images
│   ├── model/                 ← Auto-created: saved trained model
│   │   └── solarguard_model.pkl
│   ├── templates/
│   │   └── index.html         ← Live detection dashboard
│   └── logs/
│
├── esp32/
│   └── relay_controller/
│       └── relay_controller.ino   ← Flash this to your ESP32
│
├── dashboard/                 ← Standalone demo (no server needed)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd "solar panel anomaly/python"
pip install -r requirements.txt
```

---

## STEP 1 — Capture Images from Phone

### Setup phone stream
**Android:** Install **"IP Webcam"** from Play Store → open app → tap **Start Server**
**iOS:** Install **"IP Camera Lite"** from App Store → open app

> Note the IP shown in the app (e.g. `192.168.1.5:8080`)
> Phone and laptop must be on the **same WiFi network**

### Run capture server
```bash
python3 capture_server.py --source http://192.168.1.5:8080/video
```

Open **http://localhost:8090** in your browser.

### How to capture
1. Select label: **✅ Normal** (clean panel) or **⚠️ Anomaly** (dirty/blocked)
2. Point phone at the panel
3. Press **📸 Capture** button — saves **one image per press**
4. You can also press **`Space`** as a shortcut
5. Repeat until you have enough images

**Recommended dataset size:**
- Normal images: **≥ 40** (more = better baseline)
- Anomaly images: **≥ 20** (optional — system works with only normal images)

Images are saved to `dataset/normal/` and `dataset/anomaly/` automatically.

---

## STEP 2 — Train the Model

```bash
python3 train.py
```

The script will:
- Load all images from `dataset/`
- Extract features (resize to 64×64 → grayscale → flatten)
- Apply PCA dimensionality reduction
- Auto-select model:
  - **IsolationForest** (if only normal images — unsupervised)
  - **One-Class SVM** (if anomaly images present — more accurate)
- Save model to `model/solarguard_model.pkl`
- Print accuracy report

```
  Normal correctly identified: 40/40 (100.0%)
  Anomaly correctly detected:  18/20 (90.0%)
  ✅ Model saved → model/solarguard_model.pkl
```

You can force a specific model type:
```bash
python3 train.py --model iso      # IsolationForest
python3 train.py --model svm_oc   # One-Class SVM
```

---

## STEP 3 — Live Detection

```bash
# Phone as camera, no ESP32 yet (testing)
python3 server.py --source http://192.168.1.5:8080/video --no-esp32 --port 8080

# With ESP32 connected
python3 server.py --source http://192.168.1.5:8080/video --esp-ip 192.168.1.100 --port 8080

# Webcam fallback (no phone)
python3 server.py --no-esp32 --port 8080
```

Open **http://localhost:8080** — the dashboard shows:
- Live camera feed with anomaly heatmap overlay
- Real-time anomaly score chart (0–10)
- Panel status: **CLEAN** / **DIRTY**
- Relay status and event log
- Adjustable threshold slider

### How detection works

```
1. If model/solarguard_model.pkl exists:
   → Loads trained model (from Step 2)
   → Each frame is classified by the model
   → Outputs anomaly score 0–10

2. If no model found:
   → Falls back to reference-frame detection
   → Captures 40 clean frames as baseline on startup
   → Scores deviation from baseline each frame

3. Score ≥ threshold (default 5.0):
   → Anomaly confirmed after 5 consecutive dirty frames
   → HTTP GET → http://ESP32-IP/relay?state=ON
   → ESP32 triggers relay → cleaning starts

4. Score < threshold for 8 consecutive frames:
   → HTTP GET → http://ESP32-IP/relay?state=OFF
   → Cleaning stops
```

---

## ESP32 Setup

### Flash the firmware
1. Open `esp32/relay_controller/relay_controller.ino` in Arduino IDE
2. Update WiFi credentials at the top:
   ```cpp
   const char* WIFI_SSID     = "YourWiFiName";
   const char* WIFI_PASSWORD = "YourPassword";
   ```
3. Board: **ESP32 Dev Module** (install via Boards Manager)
4. Library: **ArduinoJson** (install via Library Manager)
5. Upload → open Serial Monitor at **115200 baud**
6. Note the IP address shown (e.g. `192.168.1.100`)

### Wire the relay
```
ESP32 GPIO 26  →  Relay IN
ESP32 VIN/5V   →  Relay VCC
ESP32 GND      →  Relay GND
Relay NO + COM →  Cleaning mechanism circuit
```

### Test manually
```bash
curl http://192.168.1.100/relay?state=ON
curl http://192.168.1.100/relay?state=OFF
curl http://192.168.1.100/status
```

---

## Command Reference

| Action | Command |
|---|---|
| Capture images (phone) | `python3 capture_server.py --source http://PHONE_IP:8080/video` |
| Capture images (webcam) | `python3 capture_server.py --source 0` |
| Train model | `python3 train.py` |
| Train (force IsolationForest) | `python3 train.py --model iso` |
| Run detection (no ESP32) | `python3 server.py --no-esp32 --port 8080` |
| Run detection (with ESP32) | `python3 server.py --esp-ip 192.168.1.100 --port 8080` |
| Run with phone camera | `python3 server.py --source http://PHONE_IP:8080/video --port 8080` |
| Change threshold | add `--threshold 4.0` |
| Use second webcam | add `--source 1` |
| Stop any server | Press `Ctrl+C` in terminal |

---

## Technology Stack

| Component | Technology |
|---|---|
| Camera source | Phone (IP Webcam app) or webcam |
| Phone stream | HTTP MJPEG over WiFi |
| Image capture UI | Python Flask (capture_server.py) |
| Feature extraction | OpenCV + NumPy (64×64 grayscale flatten) |
| ML model | scikit-learn IsolationForest / One-Class SVM |
| Inference server | Python Flask + Socket.IO (server.py) |
| Live dashboard | HTML/CSS/JS (real-time via WebSocket) |
| ESP32 communication | HTTP GET over WiFi |
| Microcontroller | ESP32 |
| Physical output | Relay module (GPIO 26) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Phone stream not opening | Check IP in app, same WiFi network, use `/video` endpoint |
| Port 8080 in use | Try `--port 8090` |
| Port 5000 in use | macOS AirPlay uses 5000 — always use `--port 8080` |
| Score always high | Not enough training data — capture more normal images and retrain |
| Score never rises | Lower threshold: `--threshold 3.0` |
| Model loads but wrong | Recapture dataset with correct labels and retrain |
| ESP32 not responding | Check IP in Serial Monitor, both on same WiFi |
| Relay fires immediately | Warmup phase running — wait 20 frames after calibration |

---

## How the Anomaly Score Works

```
Training phase (train.py):
  40+ clean panel photos → feature extraction → IsolationForest learns "normal"

Inference phase (server.py), per frame:
  Frame → resize 64×64 → grayscale → flatten → PCA → model.predict()
  Model output → decision score → mapped to 0–10

  Score 0–4  →  CLEAN  →  relay stays OFF
  Score 5–10 →  DIRTY  →  relay triggers ON  →  ESP32 HTTP command sent

Debounce:
  Must see 5 consecutive dirty frames before relay fires
  Must see 8 consecutive clean frames before relay turns off
  (prevents false triggers from single noisy frames)
```

---

*Built with Python · OpenCV · scikit-learn · Flask · Socket.IO · ESP32 · Relay Module*
