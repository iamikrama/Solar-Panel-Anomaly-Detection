/*
 * SolarGuard AI — ESP32 Relay Controller Firmware
 * ================================================
 * Connects to WiFi, hosts a simple HTTP web server on port 80.
 * Python script on laptop sends GET requests to control a relay
 * that triggers the panel cleaning mechanism.
 *
 * Endpoints:
 *   GET /relay?state=ON   → turn relay ON  (cleaner starts)
 *   GET /relay?state=OFF  → turn relay OFF (cleaner stops)
 *   GET /status           → JSON status of relay, uptime, WiFi
 *   GET /                 → simple HTML status page
 *
 * Hardware:
 *   - ESP32 (any variant)
 *   - Relay module connected to RELAY_PIN (GPIO 26)
 *   - Active HIGH relay (change RELAY_ACTIVE_HIGH if needed)
 *
 * Libraries:
 *   - WiFi.h (built-in with ESP32 Arduino core)
 *   - WebServer.h (built-in)
 *
 * Author: SolarGuard AI Project
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>  // Install via: Arduino Library Manager → ArduinoJson

// ── WiFi Credentials ─────────────────────────────────
const char* WIFI_SSID     = "SolarNet";          // ← change to your WiFi SSID
const char* WIFI_PASSWORD = "your_password";     // ← change to your WiFi password

// ── Hardware ──────────────────────────────────────────
const int RELAY_PIN       = 26;     // GPIO pin connected to relay IN
const bool RELAY_ACTIVE_HIGH = true; // true = relay ON when pin HIGH
                                     // false = relay ON when pin LOW (active-low modules)

// ── Web Server ────────────────────────────────────────
WebServer server(80);

// ── State ─────────────────────────────────────────────
bool relayState = false;
unsigned long startTime = 0;
unsigned long lastCommandTime = 0;
int commandCount = 0;
int relayOnCount = 0;

// ─────────────────────────────────────────────────────
// Relay Control
// ─────────────────────────────────────────────────────
void setRelay(bool on) {
  relayState = on;
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_HIGH ? (on ? HIGH : LOW)
                                             : (on ? LOW : HIGH));
  Serial.printf("[RELAY] %s\n", on ? "ON  → cleaning mechanism ACTIVE" : "OFF → cleaning mechanism STOPPED");
}

// ─────────────────────────────────────────────────────
// HTTP Route Handlers
// ─────────────────────────────────────────────────────
void handleRelay() {
  // GET /relay?state=ON  or  ?state=OFF
  if (!server.hasArg("state")) {
    server.send(400, "application/json", "{\"error\":\"Missing 'state' parameter\"}");
    return;
  }

  String stateArg = server.arg("state");
  stateArg.toUpperCase();

  bool newState;
  if (stateArg == "ON") {
    newState = true;
    relayOnCount++;
  } else if (stateArg == "OFF") {
    newState = false;
  } else {
    server.send(400, "application/json", "{\"error\":\"state must be ON or OFF\"}");
    return;
  }

  setRelay(newState);
  commandCount++;
  lastCommandTime = millis();

  // JSON response
  StaticJsonDocument<128> doc;
  doc["relay"]  = relayState ? "ON" : "OFF";
  doc["ok"]     = true;
  doc["uptime"] = (millis() - startTime) / 1000;

  String resp;
  serializeJson(doc, resp);
  server.send(200, "application/json", resp);
}

void handleStatus() {
  // GET /status → full JSON status
  StaticJsonDocument<256> doc;
  doc["relay"]         = relayState ? "ON" : "OFF";
  doc["uptime_sec"]    = (millis() - startTime) / 1000;
  doc["commands_recv"] = commandCount;
  doc["relay_on_count"]= relayOnCount;
  doc["wifi_rssi"]     = WiFi.RSSI();
  doc["ip"]            = WiFi.localIP().toString();
  doc["ssid"]          = WiFi.SSID();

  String resp;
  serializeJson(doc, resp);
  server.send(200, "application/json", resp);
}

void handleRoot() {
  // Simple human-readable status page
  String color = relayState ? "#ef4444" : "#22c55e";
  String status = relayState ? "ON — Cleaning ACTIVE" : "OFF — Panel OK";
  unsigned long upSec = (millis() - startTime) / 1000;
  unsigned long h = upSec / 3600, m = (upSec % 3600) / 60, s = upSec % 60;

  String html = R"rawliteral(
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="3">
<title>SolarGuard ESP32</title>
<style>
  body{font-family:monospace;background:#060b14;color:#e2e8f0;margin:0;padding:24px}
  h1{color:#38bdf8;font-size:1.4rem;margin-bottom:8px}
  .card{background:rgba(255,255,255,0.06);border:1px solid rgba(99,179,237,0.15);
        border-radius:12px;padding:20px;margin:12px 0;max-width:400px}
  .status{font-size:1.6rem;font-weight:bold;color:)rawliteral" + color + R"rawliteral(}
  .row{display:flex;justify-content:space-between;margin:6px 0;font-size:0.85rem}
  .label{color:#64748b} .val{color:#94a3b8}
</style></head><body>
<h1>⚡ SolarGuard AI — ESP32</h1>
<div class="card">
  <div class="status">Relay: )rawliteral" + status + R"rawliteral(</div>
  <div class="row"><span class="label">Uptime</span><span class="val">)rawliteral";

  html += String(h) + "h " + String(m) + "m " + String(s) + "s";
  html += R"rawliteral(</span></div>
  <div class="row"><span class="label">WiFi RSSI</span><span class="val">)rawliteral";
  html += String(WiFi.RSSI()) + " dBm";
  html += R"rawliteral(</span></div>
  <div class="row"><span class="label">Commands Received</span><span class="val">)rawliteral";
  html += String(commandCount);
  html += R"rawliteral(</span></div>
  <div class="row"><span class="label">Relay Triggers</span><span class="val">)rawliteral";
  html += String(relayOnCount);
  html += R"rawliteral(</span></div>
  <div class="row"><span class="label">IP Address</span><span class="val">)rawliteral";
  html += WiFi.localIP().toString();
  html += R"rawliteral(</span></div>
</div>
<p style="color:#475569;font-size:0.7rem">Auto-refreshes every 3 seconds</p>
</body></html>)rawliteral";

  server.send(200, "text/html", html);
}

void handleNotFound() {
  server.send(404, "application/json", "{\"error\":\"Not found\"}");
}

// ─────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n\n================================================");
  Serial.println("  SolarGuard AI — ESP32 Relay Controller");
  Serial.println("================================================");

  // Relay pin
  pinMode(RELAY_PIN, OUTPUT);
  setRelay(false);  // Start with relay OFF
  Serial.printf("Relay pin: GPIO %d (active %s)\n",
                RELAY_PIN, RELAY_ACTIVE_HIGH ? "HIGH" : "LOW");

  // WiFi
  Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[ERROR] WiFi connection failed! Restarting...");
    delay(3000);
    ESP.restart();
  }

  Serial.println("\n✅ WiFi connected!");
  Serial.printf("   IP Address : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("   RSSI       : %d dBm\n", WiFi.RSSI());

  // Routes
  server.on("/",          HTTP_GET, handleRoot);
  server.on("/relay",     HTTP_GET, handleRelay);
  server.on("/status",    HTTP_GET, handleStatus);
  server.onNotFound(handleNotFound);

  server.begin();
  startTime = millis();

  Serial.println("\n🚀 HTTP server started");
  Serial.printf("   Open: http://%s/\n", WiFi.localIP().toString().c_str());
  Serial.println("\nEndpoints:");
  Serial.printf("   GET /relay?state=ON   → Relay ON\n");
  Serial.printf("   GET /relay?state=OFF  → Relay OFF\n");
  Serial.printf("   GET /status           → JSON status\n");
  Serial.println("================================================\n");
}

// ─────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────
void loop() {
  server.handleClient();

  // WiFi watchdog — reconnect if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WARN] WiFi lost — reconnecting...");
    WiFi.reconnect();
    delay(5000);
  }

  // Safety: auto-turn off relay after 60 seconds if no command received
  // (prevents stuck relay if laptop script crashes)
  const unsigned long RELAY_TIMEOUT_MS = 60000;
  if (relayState && (millis() - lastCommandTime) > RELAY_TIMEOUT_MS) {
    Serial.println("[SAFETY] Relay auto-OFF (timeout — no command received for 60s)");
    setRelay(false);
  }

  delay(10);
}
