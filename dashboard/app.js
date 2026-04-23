/* ===================================================
   SolarGuard AI — app.js
   Simulates live anomaly detection, charts, camera feed,
   event log, history table and relay control.
   =================================================== */

// ── State ───────────────────────────────────────────
const state = {
  threshold:     5.0,
  currentScore:  0.0,
  isDirty:       false,
  relayOn:       false,
  frames:        0,
  cleanCount:    0,
  dirtyCount:    0,
  relayTriggers: 0,
  startTime:     Date.now(),
  scoreHistory:  [],
  maxHistory:    60,
  simPhase:      'clean',   // 'clean' | 'dirty' | 'transition'
  phaseTick:     0,
  dirtyDuration: 0,
};

// ── DOM Refs ─────────────────────────────────────────
const $ = id => document.getElementById(id);
const els = {
  anomalyScoreVal: $('anomalyScoreVal'),
  anomalyBar:      $('anomalyBar'),
  panelStatusVal:  $('panelStatusVal'),
  statusBadge:     $('statusBadge'),
  badgeText:       $('badgeText'),
  relayVal:        $('relayVal'),
  relayToggle:     $('relayToggle'),
  framesVal:       $('framesVal'),
  fpsVal:          $('fpsVal'),
  timeDisplay:     $('timeDisplay'),
  cameraCanvas:    $('cameraCanvas'),
  scoreChart:      $('scoreChart'),
  logContainer:    $('logContainer'),
  historyBody:     $('historyBody'),
  ssClean:         $('ssClean'),
  ssDirty:         $('ssDirty'),
  ssRelay:         $('ssRelay'),
  inferenceTime:   $('inferenceTime'),
  uptimeVal:       $('uptimeVal'),
  alertBanner:     $('alertBanner'),
  alertText:       $('alertText'),
  archAIStatus:    $('archAIStatus'),
  archRelayStatus: $('archRelayStatus'),
  archRelay:       $('archRelay'),
  arrow3:          $('arrow3'),
  arrow4:          $('arrow4'),
  camThreshold:    $('camThreshold'),
  heatmapOverlay:  $('heatmapOverlay'),
};

// ── Chart ────────────────────────────────────────────
const chartCtx = els.scoreChart.getContext('2d');
const CHART_W = 500, CHART_H = 220;
const PAD = { t: 16, r: 16, b: 32, l: 44 };
const INNER_W = CHART_W - PAD.l - PAD.r;
const INNER_H = CHART_H - PAD.t - PAD.b;

function drawChart() {
  const c = chartCtx;
  const dpr = window.devicePixelRatio || 1;
  const canvas = els.scoreChart;
  canvas.width  = CHART_W * dpr;
  canvas.height = CHART_H * dpr;
  c.scale(dpr, dpr);
  canvas.style.width  = CHART_W + 'px';
  canvas.style.height = CHART_H + 'px';

  c.clearRect(0, 0, CHART_W, CHART_H);

  // Background
  c.fillStyle = 'rgba(0,0,0,0)';
  c.fillRect(0, 0, CHART_W, CHART_H);

  const data = state.scoreHistory;
  const maxVal = 10;

  const x = i => PAD.l + (i / (state.maxHistory - 1)) * INNER_W;
  const y = v => PAD.t + INNER_H - (v / maxVal) * INNER_H;

  // Grid lines + Y labels
  c.strokeStyle = 'rgba(99,179,237,0.07)';
  c.lineWidth = 1;
  c.font = '10px JetBrains Mono, monospace';
  c.fillStyle = 'rgba(148,163,184,0.5)';
  c.textAlign = 'right';
  for (let v = 0; v <= 10; v += 2) {
    const yy = y(v);
    c.beginPath(); c.moveTo(PAD.l, yy); c.lineTo(PAD.l + INNER_W, yy); c.stroke();
    c.fillText(v.toFixed(0), PAD.l - 6, yy + 4);
  }

  // Threshold line
  const ty = y(state.threshold);
  c.strokeStyle = '#f97316';
  c.lineWidth = 1.5;
  c.setLineDash([6, 4]);
  c.beginPath(); c.moveTo(PAD.l, ty); c.lineTo(PAD.l + INNER_W, ty); c.stroke();
  c.setLineDash([]);

  if (data.length < 2) return;

  // Area fill — split clean / dirty
  for (let i = 1; i < data.length; i++) {
    const x0 = x(i-1), x1 = x(i);
    const y0 = y(data[i-1]), y1 = y(data[i]);
    const isDirtySegment = data[i] > state.threshold;
    const grad = c.createLinearGradient(0, PAD.t, 0, PAD.t + INNER_H);
    if (isDirtySegment) {
      grad.addColorStop(0, 'rgba(239,68,68,0.3)');
      grad.addColorStop(1, 'rgba(239,68,68,0.02)');
    } else {
      grad.addColorStop(0, 'rgba(34,197,94,0.2)');
      grad.addColorStop(1, 'rgba(34,197,94,0.02)');
    }
    c.fillStyle = grad;
    c.beginPath();
    c.moveTo(x0, PAD.t + INNER_H);
    c.lineTo(x0, y0);
    c.lineTo(x1, y1);
    c.lineTo(x1, PAD.t + INNER_H);
    c.closePath();
    c.fill();
  }

  // Line
  c.beginPath();
  c.moveTo(x(0), y(data[0]));
  for (let i = 1; i < data.length; i++) {
    c.lineTo(x(i), y(data[i]));
  }
  const lineGrad = c.createLinearGradient(PAD.l, 0, PAD.l + INNER_W, 0);
  lineGrad.addColorStop(0, '#38bdf8');
  lineGrad.addColorStop(1, state.isDirty ? '#ef4444' : '#22c55e');
  c.strokeStyle = lineGrad;
  c.lineWidth = 2.5;
  c.lineJoin = 'round';
  c.stroke();

  // Latest dot
  if (data.length > 0) {
    const li = data.length - 1;
    c.beginPath();
    c.arc(x(li), y(data[li]), 5, 0, Math.PI*2);
    c.fillStyle = state.isDirty ? '#ef4444' : '#22c55e';
    c.fill();
    c.strokeStyle = '#fff';
    c.lineWidth = 1.5;
    c.stroke();
  }
}

// ── Camera Simulation ────────────────────────────────
const camCtx = els.cameraCanvas.getContext('2d');
let camFrame = 0;

function drawCamera() {
  const c = camCtx;
  const W = 640, H = 360;
  c.clearRect(0, 0, W, H);

  // Sky gradient background
  const sky = c.createLinearGradient(0, 0, 0, H * 0.4);
  sky.addColorStop(0, '#0c1e35');
  sky.addColorStop(1, '#1a3a5c');
  c.fillStyle = sky;
  c.fillRect(0, 0, W, H * 0.4);

  // Ground
  c.fillStyle = '#1a2634';
  c.fillRect(0, H * 0.4, W, H * 0.6);

  // Solar panel grid
  const panelX = 80, panelY = 60, panelW = 480, panelH = 240;

  // Panel shadow
  c.shadowColor = 'rgba(0,0,0,0.5)';
  c.shadowBlur = 20;

  // Panel backing
  c.fillStyle = state.isDirty ? '#2a2010' : '#0d2040';
  c.fillRect(panelX, panelY, panelW, panelH);
  c.shadowBlur = 0;

  // Panel cells
  const cols = 6, rows = 3;
  const cw = panelW / cols, ch = panelH / rows;
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      const cx2 = panelX + col * cw + 2;
      const cy2 = panelY + r * ch + 2;
      const dirty = state.isDirty && Math.random() < 0.4;

      const cellGrad = c.createLinearGradient(cx2, cy2, cx2 + cw - 4, cy2 + ch - 4);
      if (dirty) {
        cellGrad.addColorStop(0, '#4a3820');
        cellGrad.addColorStop(1, '#3a2a10');
      } else {
        cellGrad.addColorStop(0, '#1a4a8a');
        cellGrad.addColorStop(1, '#0a2a5a');
      }
      c.fillStyle = cellGrad;
      c.fillRect(cx2, cy2, cw - 4, ch - 4);

      // Cell grid lines
      c.strokeStyle = dirty ? 'rgba(100,80,30,0.4)' : 'rgba(56,189,248,0.15)';
      c.lineWidth = 0.5;
      c.strokeRect(cx2, cy2, cw - 4, ch - 4);

      // Reflection shimmer
      if (!dirty) {
        c.fillStyle = `rgba(255,255,255,${0.03 + 0.02 * Math.sin(camFrame * 0.05 + r * 0.7 + col * 0.5)})`;
        c.fillRect(cx2 + 4, cy2 + 4, cw * 0.4, ch * 0.3);
      }
    }
  }

  // Dirt overlay when dirty
  if (state.isDirty) {
    const dirtGrad = c.createRadialGradient(panelX + panelW*0.5, panelY + panelH*0.5, 30,
                                             panelX + panelW*0.5, panelY + panelH*0.5, panelW*0.5);
    dirtGrad.addColorStop(0, `rgba(120,90,30,${0.3 + 0.1 * Math.sin(camFrame * 0.03)})`);
    dirtGrad.addColorStop(1, 'rgba(80,60,20,0.08)');
    c.fillStyle = dirtGrad;
    c.fillRect(panelX, panelY, panelW, panelH);

    // Dust particles
    c.fillStyle = 'rgba(160,130,60,0.25)';
    for (let d = 0; d < 30; d++) {
      const dx = panelX + (d * 47 % panelW);
      const dy = panelY + (d * 31 % panelH);
      c.beginPath();
      c.arc(dx, dy, 2 + (d % 3), 0, Math.PI * 2);
      c.fill();
    }
  }

  // Panel border / frame
  c.strokeStyle = state.isDirty ? '#5a4020' : '#2a5a8a';
  c.lineWidth = 3;
  c.strokeRect(panelX, panelY, panelW, panelH);

  // Status overlay text
  const label = state.isDirty ? 'ANOMALY DETECTED' : 'CLEAN';
  const labelColor = state.isDirty ? '#ef4444' : '#22c55e';
  c.font = 'bold 13px JetBrains Mono, monospace';
  c.fillStyle = state.isDirty ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.1)';
  c.fillRect(panelX + 8, panelY + 8, 180, 26);
  c.fillStyle = labelColor;
  c.fillText(label, panelX + 14, panelY + 26);

  // Score tag
  c.fillStyle = 'rgba(0,0,0,0.6)';
  c.fillRect(W - 140, 8, 132, 26);
  c.fillStyle = state.isDirty ? '#f87171' : '#86efac';
  c.font = 'bold 12px JetBrains Mono, monospace';
  c.fillText(`Score: ${state.currentScore.toFixed(2)}`, W - 134, 26);

  // Frame counter
  c.fillStyle = 'rgba(148,163,184,0.7)';
  c.font = '10px JetBrains Mono, monospace';
  c.fillText(`Frame #${state.frames}`, panelX + 8, H - 12);

  camFrame++;
}

// ── Heatmap Overlay ──────────────────────────────────
function updateHeatmap() {
  const el = els.heatmapOverlay;
  if (state.isDirty) {
    const intensity = Math.min((state.currentScore - state.threshold) / 5, 1);
    el.style.background = `radial-gradient(ellipse 60% 50% at 55% 48%, rgba(239,68,68,${intensity * 0.5}), transparent 80%)`;
    el.style.opacity = '1';
  } else {
    el.style.opacity = '0';
  }
}

// ── Score Simulation ─────────────────────────────────
function nextScore() {
  state.phaseTick++;

  if (state.simPhase === 'clean') {
    // Normal clean state — low jitter around 1.5–2.5
    const target = 1.8 + Math.sin(state.phaseTick * 0.08) * 0.5;
    state.currentScore += (target - state.currentScore) * 0.15 + (Math.random() - 0.5) * 0.3;
    state.currentScore = Math.max(0.1, Math.min(state.currentScore, 4.2));

    // Occasionally go dirty
    if (state.phaseTick > 40 && Math.random() < 0.015) {
      state.simPhase = 'dirty';
      state.dirtyDuration = 30 + Math.floor(Math.random() * 40);
      state.phaseTick = 0;
      addLog('warn', 'Anomaly pattern emerging — score rising');
    }
  } else if (state.simPhase === 'dirty') {
    // Dirty state — score climbs above threshold
    const target = 6.5 + Math.sin(state.phaseTick * 0.1) * 1.2;
    state.currentScore += (target - state.currentScore) * 0.12 + (Math.random() - 0.5) * 0.5;
    state.currentScore = Math.max(0, Math.min(state.currentScore, 9.8));

    if (state.phaseTick > state.dirtyDuration) {
      state.simPhase = 'clean';
      state.phaseTick = 0;
      addLog('success', 'Panel cleaning complete — score normalising');
      setRelay(false);
    }
  }

  return Math.max(0, state.currentScore);
}

// ── UI Updates ───────────────────────────────────────
function setRelay(on) {
  if (state.relayOn === on) return;
  state.relayOn = on;
  els.relayVal.textContent = on ? 'ON' : 'OFF';
  els.relayToggle.classList.toggle('on', on);
  els.archRelayStatus.textContent = on ? 'Active' : 'Standby';
  els.archRelay.classList.toggle('relay-on', on);
  els.arrow4.style.opacity = on ? '1' : '0.4';
  if (on) {
    state.relayTriggers++;
    els.ssRelay.textContent = state.relayTriggers;
    showAlert(`⚠️ Anomaly Detected (Score: ${state.currentScore.toFixed(2)}) — Relay triggered! Cleaning initiated.`);
    addLog('error', `Relay ON — HTTP POST → 192.168.1.100/relay?state=ON`);
  } else {
    addLog('info', `Relay OFF — HTTP POST → 192.168.1.100/relay?state=OFF`);
  }
}

function updateUI(score) {
  // Score card
  els.anomalyScoreVal.textContent = score.toFixed(2);
  const pct = Math.min(score / 10 * 100, 100);
  els.anomalyBar.style.width = pct + '%';

  const isDirty = score >= state.threshold;
  if (isDirty !== state.isDirty) {
    state.isDirty = isDirty;

    if (isDirty) {
      // Going dirty
      els.panelStatusVal.textContent = 'DIRTY';
      els.panelStatusVal.classList.add('dirty');
      els.statusBadge.className = 'status-badge dirty';
      els.badgeText.textContent = 'Anomaly Detected';
      document.getElementById('cardScore').style.setProperty('--accent', '#ef4444');
      setRelay(true);
      state.dirtyCount++;
      els.ssDirty.textContent = state.dirtyCount;
      addHistoryRow(score, 'DIRTY', 'Relay ON');
    } else {
      // Going clean
      els.panelStatusVal.textContent = 'CLEAN';
      els.panelStatusVal.classList.remove('dirty');
      els.statusBadge.className = 'status-badge clean';
      els.badgeText.textContent = 'Operational';
      setRelay(false);
      state.cleanCount++;
      els.ssClean.textContent = state.cleanCount;
      addHistoryRow(score, 'CLEAN', 'None');
    }
  }

  // Frames
  state.frames++;
  els.framesVal.textContent = state.frames.toLocaleString();
}

// ── Log ──────────────────────────────────────────────
function addLog(type, msg) {
  const now = new Date();
  const t = now.toTimeString().slice(0, 8);
  const el = document.createElement('div');
  el.className = `log-entry ${type}`;
  el.innerHTML = `<span class="log-time">${t}</span><span class="log-msg">${msg}</span>`;
  els.logContainer.prepend(el);
  // Keep last 50
  while (els.logContainer.children.length > 50) {
    els.logContainer.removeChild(els.logContainer.lastChild);
  }
}

function clearLog() {
  els.logContainer.innerHTML = '';
  addLog('info', 'Log cleared');
}

// ── History Table ─────────────────────────────────────
function addHistoryRow(score, status, action) {
  const now = new Date().toTimeString().slice(0, 8);
  const scoreClass = score >= state.threshold ? 'high' : score >= state.threshold * 0.7 ? 'med' : 'low';
  const statusClass = status === 'DIRTY' ? 'td-status-dirty' : 'td-status-clean';
  const actionClass = action.includes('Relay') ? 'td-action-relay' : 'td-action';
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${now}</td>
    <td class="td-score ${scoreClass}">${score.toFixed(2)}</td>
    <td class="${statusClass}">${status}</td>
    <td class="${actionClass}">${action}</td>
  `;
  els.historyBody.prepend(row);
  if (els.historyBody.children.length > 30) {
    els.historyBody.removeChild(els.historyBody.lastChild);
  }
}

// ── Alert ─────────────────────────────────────────────
let alertTimer = null;
function showAlert(msg) {
  els.alertText.textContent = msg;
  els.alertBanner.classList.add('show');
  clearTimeout(alertTimer);
  alertTimer = setTimeout(() => els.alertBanner.classList.remove('show'), 6000);
}
function dismissAlert() {
  els.alertBanner.classList.remove('show');
  clearTimeout(alertTimer);
}

// ── Threshold ─────────────────────────────────────────
function updateThreshold(val) {
  state.threshold = parseFloat(val) || 5.0;
  els.camThreshold.textContent = `Threshold: ${state.threshold.toFixed(1)}`;
  addLog('info', `Threshold updated → ${state.threshold.toFixed(1)}`);
}

// ── Clock + Uptime ────────────────────────────────────
function updateClock() {
  const now = new Date();
  els.timeDisplay.textContent = now.toTimeString().slice(0, 8);
  const upSec = Math.floor((Date.now() - state.startTime) / 1000);
  const h = String(Math.floor(upSec / 3600)).padStart(2, '0');
  const m = String(Math.floor((upSec % 3600) / 60)).padStart(2, '0');
  const s = String(upSec % 60).padStart(2, '0');
  els.uptimeVal.textContent = `${h}:${m}:${s}`;
}

// ── FPS Tracker ───────────────────────────────────────
let fpsFrames = 0, lastFpsTime = Date.now();
function trackFps() {
  fpsFrames++;
  const now = Date.now();
  if (now - lastFpsTime >= 1000) {
    els.fpsVal.textContent = fpsFrames;
    fpsFrames = 0;
    lastFpsTime = now;
    // Randomise inference time for realism
    const ms = 100 + Math.floor(Math.random() * 60);
    els.inferenceTime.textContent = `~${ms} ms`;
  }
}

// ── Main Loop ─────────────────────────────────────────
let tick = 0;
function loop() {
  const score = nextScore();

  // Push to history
  state.scoreHistory.push(score);
  if (state.scoreHistory.length > state.maxHistory) state.scoreHistory.shift();

  updateUI(score);
  drawCamera();
  drawChart();
  updateHeatmap();
  trackFps();
  updateClock();

  tick++;

  // Log every ~5 seconds
  if (tick % 25 === 0) {
    addLog('info', `Frame ${state.frames} — Score: ${score.toFixed(3)} — ${state.isDirty ? 'ANOMALY' : 'CLEAN'}`);
  }

  // Arch AI status pulse
  els.archAIStatus.textContent = tick % 4 < 2 ? 'Inferring' : 'Computing';
}

// ── Boot ──────────────────────────────────────────────
function boot() {
  // Pre-fill history with clean data
  for (let i = 0; i < state.maxHistory; i++) {
    state.scoreHistory.push(1.5 + Math.random() * 0.8);
  }

  addLog('success', 'SolarGuard AI system started');
  addLog('info', 'Loading FOMO-AD model from Edge Impulse SDK...');
  addLog('info', 'Camera stream initialised — 640×360 @ 10fps');
  addLog('info', 'ESP32 target: 192.168.1.100:80');
  addLog('info', `Anomaly threshold: ${state.threshold.toFixed(1)}`);

  setInterval(loop, 200);   // ~5 fps simulation ticks
  setInterval(updateClock, 1000);
}

window.addEventListener('DOMContentLoaded', boot);
