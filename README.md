# SolarGuard AI — Solar Panel Anomaly Detection System

> **Phone Camera → Capture Dataset → Train Model → Live Detection → ESP32 Relay Trigger**

Built with Python · OpenCV · scikit-learn · Flask · ESP32

---

## System Overview

```
📱 Phone (IP Webcam app)
      │
      │  HTTP stream over WiFi
      ▼
🖥️  Laptop (Python scripts)
      │
      ├─ capture_server.py  →  press button to capture images
      ├─ train.py           →  train AI model on captured images
      └─ server.py          →  live detection + send command to ESP32
                                    │
                                    │  HTTP over WiFi
                                    ▼
                              📡 ESP32  →  Relay  →  Cleaning Mechanism
```

---

## Project Structure

```
solar panel anomaly/
│
├── python/
│   ├── capture_server.py      ← STEP 2: Capture images from phone
│   ├── train.py               ← STEP 3: Train the AI model
│   ├── server.py              ← STEP 4: Live detection + ESP32 control
│   ├── detect.py              ← Alternative: Edge Impulse .eim model
│   ├── requirements.txt       ← All Python dependencies
│   ├── dataset/               ← Auto-created when you capture images
│   │   ├── normal/
│   │   └── anomaly/
│   ├── model/                 ← Auto-created after training
│   │   └── solarguard_model.pkl
│   └── templates/
│       └── index.html         ← Live detection dashboard UI
│
├── esp32/
│   └── relay_controller/
│       └── relay_controller.ino
│
└── README.md
```

---

# 🚀 Full Setup Guide (Fresh PC — Start Here)

Follow every step in order. Do not skip any step.

---

## PRE-REQUISITE — Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.10 or newer**
3. During installation — **check the box that says "Add Python to PATH"**
4. Click Install Now
5. Verify by opening a terminal and running:
   ```
   python --version
   ```
   You should see something like `Python 3.11.x`

> **Windows users:** Use **Command Prompt** or **PowerShell** for all commands below.
> **Mac/Linux users:** Use **Terminal**.

---

## STEP 1 — Clone the Project

Open a terminal and run:

```bash
git clone https://github.com/iamikrama/Solar-Panel-Anomaly-Detection.git
```

Then enter the project folder:

```bash
cd Solar-Panel-Anomaly-Detection
```

Then enter the python subfolder:

```bash
cd python
```

> All commands from this point should be run from inside the `python/` folder.

---

## STEP 2 — Install Dependencies

Run this one command to install everything:

```bash
pip install -r requirements.txt
```

This installs: Flask, OpenCV, scikit-learn, Socket.IO, NumPy, and all other required packages.

It may take 1–3 minutes depending on your internet speed.

Verify it worked:
```bash
python -c "import cv2, flask, sklearn; print('All OK')"
```
You should see `All OK`.

---

## STEP 3 — Set Up Phone as Camera

1. Install **IP Webcam** app on your Android phone
   - Play Store → search "IP Webcam" by Pavel Khlebovich → Install

2. Open the app → scroll to the bottom → tap **"Start server"**

3. The app will show an IP address like:
   ```
   http://192.168.1.5:8080
   ```
   Note this down — you will use it in every command below.

> ⚠️ **Important:** Your phone and laptop must be connected to the **same WiFi network**.

4. Test it — open the IP address in your laptop browser. You should see the IP Webcam control panel with a live video.

---

## STEP 4 — Capture Images (Build Your Dataset)

Run the capture server:

```bash
python capture_server.py --source http://192.168.1.5:8080/video
```

> Replace `192.168.1.5` with your phone's actual IP from Step 3.

Open your browser and go to:
```
http://localhost:8090
```

You will see a live view from your phone camera.

**How to capture:**

1. Select label **"✅ Normal"** — point phone at the **clean solar panel**
2. Press the **📸 Capture** button (or press `Space` on keyboard)
3. Each press saves **one image**
4. Capture at least **40 normal images** from slightly different angles/lighting
5. Switch label to **"⚠️ Anomaly"** — cover part of the panel (hand, cloth, dust)
6. Capture at least **20 anomaly images**
7. Watch the counter in the dashboard

When done, press `Ctrl+C` in the terminal to stop the capture server.

---

## STEP 5 — Train the AI Model

Run:

```bash
python train.py
```

The script will:
- Load all your captured images
- Extract image features
- Train an AI model (IsolationForest or SVM automatically selected)
- Save the model to `model/solarguard_model.pkl`

You will see output like:
```
Loading dataset...
  Normal images  :  40
  Anomaly images :  20
Training model: iso
Normal correctly identified: 40/40 (100.0%)
Anomaly correctly detected:  18/20 (90.0%)
✅ Model saved → model/solarguard_model.pkl
```

---

## STEP 6 — Run Live Detection

### Without ESP32 (testing only)

```bash
python server.py --source http://192.168.1.5:8080/video --no-esp32 --port 8080
```

### With ESP32 connected

```bash
python server.py --source http://192.168.1.5:8080/video --esp-ip 192.168.1.100 --port 8080
```

> Replace `192.168.1.100` with your ESP32's IP (shown in Serial Monitor after flashing).

Open your browser and go to:
```
http://localhost:8080
```

You will see:
- Live camera feed from your phone
- Real-time anomaly score (0–10)
- Panel status: CLEAN or DIRTY
- Relay status and event log

**When score ≥ 5.0:** System sends HTTP command to ESP32 → relay turns ON → cleaning starts.

Press `Ctrl+C` in the terminal to stop the server.

---

## STEP 7 — Flash the ESP32 (Hardware Setup)

### Install Arduino IDE
1. Download from https://www.arduino.cc/en/software
2. Install and open it

### Add ESP32 board support
1. Open Arduino IDE → File → Preferences
2. In "Additional Board Manager URLs" paste:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Click OK
4. Tools → Board → Boards Manager → search "esp32" → Install **esp32 by Espressif Systems**

### Install required library
1. Sketch → Include Library → Manage Libraries
2. Search **ArduinoJson** → Install

### Flash the firmware
1. Open `esp32/relay_controller/relay_controller.ino`
2. Edit these two lines with your WiFi details:
   ```cpp
   const char* WIFI_SSID     = "YourWiFiName";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   ```
3. Connect ESP32 to laptop via USB
4. Tools → Board → select **ESP32 Dev Module**
5. Tools → Port → select the correct COM port
6. Click **Upload** (→ arrow button)
7. Open Tools → Serial Monitor → set baud rate to **115200**
8. You will see the ESP32's IP address printed (e.g. `192.168.1.100`)

### Wire the relay
```
ESP32 GPIO 26  →  Relay IN
ESP32 VIN/5V   →  Relay VCC
ESP32 GND      →  Relay GND
Relay NO + COM →  Your cleaning mechanism circuit
```

### Test the relay works
```bash
curl http://192.168.1.100/relay?state=ON
curl http://192.168.1.100/relay?state=OFF
```

---

## Summary — All Commands

```bash
# 1. Clone and enter project
git clone https://github.com/iamikrama/Solar-Panel-Anomaly-Detection.git
cd Solar-Panel-Anomaly-Detection/python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Capture images from phone camera
python capture_server.py --source http://PHONE_IP:8080/video
# → Open http://localhost:8090 and press Capture button

# 4. Train the model
python train.py

# 5a. Run detection (testing, no ESP32)
python server.py --source http://PHONE_IP:8080/video --no-esp32 --port 8080

# 5b. Run detection (with ESP32 relay)
python server.py --source http://PHONE_IP:8080/video --esp-ip ESP32_IP --port 8080
# → Open http://localhost:8080
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Use `python3` instead, or reinstall Python with "Add to PATH" checked |
| `pip` not found | Use `pip3` instead |
| Phone stream not connecting | Phone and laptop must be on same WiFi. Use IP shown in IP Webcam app |
| `http://localhost:8080` shows nothing | Wait 5 seconds after running server, then refresh |
| Port 8080 in use | Try `--port 8090` |
| Port 5000 in use (Mac) | macOS AirPlay uses port 5000 — always use `--port 8080` |
| Score always shows DIRTY | Retrain — capture more clean panel images in same lighting/angle |
| Score never goes DIRTY | Lower threshold: add `--threshold 3.0` to server command |
| ESP32 not responding | Check IP in Serial Monitor, same WiFi, try `curl http://IP/status` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |

---

## Technology Stack

| Component | Technology |
|---|---|
| Phone camera stream | Android IP Webcam app (MJPEG over HTTP) |
| Image capture interface | Python Flask (`capture_server.py`) |
| AI model | scikit-learn IsolationForest / One-Class SVM |
| Live detection server | Python Flask + Socket.IO (`server.py`) |
| Dashboard UI | HTML + CSS + JavaScript (WebSocket real-time) |
| ESP32 communication | HTTP GET request over WiFi |
| Microcontroller | ESP32 |
| Physical output | Relay module on GPIO 26 |

---

*Built with Python · OpenCV · scikit-learn · Flask · Socket.IO · ESP32*
