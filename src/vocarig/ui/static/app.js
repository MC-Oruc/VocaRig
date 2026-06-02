/* ============================================================
   VocaRig Lab — Premium UI Logic
   Chart.js integration, custom slider fills, scale toggles
   ============================================================ */

const state = {
  lipNames: [],
  arkitNames: [],
  lipValues: [],
  arkitValues: {},
  modelLipValues: [],
  targetLipValues: [],
  targetArkitValues: {},
  streamId: `vocarig-${Math.random().toString(16).slice(2)}`,
  model: { selected: "", options: [] },
  dataset: { selected: "", options: [] },
  device: { selected: "auto", effective: "unknown", cudaAvailable: false },
  playing: false,
  playTimer: null,
  poseSmoothing: {
    rafId: null,
    lastTime: 0,
  },
  audioPlayback: {
    frames: [],
    fps: 30,
    url: "",
    rafId: null,
    frameIndex: -1,
    active: false,
    inferMs: null,
    totalMs: null,
    nativePreparePending: false,
    preparedMode: "",
  },
  realtime: {
    mode: "idle",
    timer: null,
    pending: false,
    socket: null,
    socketReady: null,
    socketWaiters: new Map(),
    requestSeq: 0,
    skippedTicks: 0,
    droppedRequests: 0,
    warningTimer: null,
    intentionalSocketClose: false,
    resetNext: false,
    fileBuffer: null,
    fileSampleOffset: 0,
    audioContext: null,
    micStream: null,
    micSource: null,
    micProcessor: null,
    micGain: null,
    micQueue: [],
    liveWave: [],
  },
  waveform: {
    peaks: [],
    duration: 0,
    ready: false,
    playhead: 0,
  },
  trainSocket: null,
  syntheticSocket: null,
  metrics: null,
  meshUrl: "/mesh/ARKitMesh.glb",
  rig: null,
  previewMode: "mesh",
  rotation: { x: -0.03, y: 0 },
  zoom: 3.15,
  inferenceMode: "baked",
  lossChart: null,
  metricsChart: null,
  chartScale: { loss: "linear", metrics: "linear" },
  selectedMetricsPath: "",
  wasTraining: false,
  defaultAudioUrl: "/audio/voice-sample.wav",
  defaultAudioName: "voice-sample.wav",
};

const $ = (id) => document.getElementById(id);

const SMOOTH_EPSILON = 0.0015;
const SOFT_CONTROLS = new Set([
  "mouthPucker",
  "mouthFunnel",
  "mouthShrugUpper",
  "mouthShrugLower",
  "mouthRollUpper",
  "mouthRollLower",
]);
const SIDE_CONTROLS = new Set([
  "jawLeft",
  "jawRight",
  "mouthLeft",
  "mouthRight",
  "mouthLowerDownLeft",
  "mouthLowerDownRight",
  "mouthUpperUpLeft",
  "mouthUpperUpRight",
  "mouthPressLeft",
  "mouthPressRight",
  "mouthStretchLeft",
  "mouthStretchRight",
]);

const INFERENCE_MODES = {
  baked: {
    label: "Ön-hesaplı mod",
    ready: "Senkron kareleri hazır",
    prepare: "YÜZ + SESİ HAZIRLA",
  },
  "live-file": {
    label: "Canlı WAV modu",
    ready: "WAV akışı hazır",
    prepare: "CANLI WAV BAŞLAT",
  },
  mic: {
    label: "Mikrofon modu",
    ready: "Mikrofon akışı hazır",
    prepare: "MİKROFONU BAŞLAT",
  },
};

/* ── API helpers ── */
async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const response = await fetch(path, { headers, ...options });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function websocketUrl(path) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}${path}`;
}

function setLog(value) {
  $("pipelineLog").textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

/* ── Theme palette: CSS değişkenlerini okuyan tek renk kaynağı (canvas / Chart.js / 3D) ── */
function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (value && value.trim()) || fallback;
}

function palette() {
  return {
    s1: cssVar("--chart-1", "#c84f2b"),
    s1Fill: cssVar("--chart-1-fill", "rgba(200,79,43,0.08)"),
    s2: cssVar("--chart-2", "#1b1d21"),
    s2Fill: cssVar("--chart-2-fill", "rgba(27,29,33,0.05)"),
    s3: cssVar("--chart-3", "#c8761a"),
    s3Fill: cssVar("--chart-3-fill", "rgba(200,118,26,0.06)"),
    grid: cssVar("--chart-grid", "rgba(27,29,33,0.07)"),
    axis: cssVar("--chart-axis", "rgba(27,29,33,0.22)"),
    tick: cssVar("--chart-tick", "#5a5c62"),
    legend: cssVar("--chart-legend", "#5a5c62"),
    tipBg: cssVar("--chart-tip-bg", "#fbfaf7"),
    tipBorder: cssVar("--hairline-strong", "#b7b2a6"),
    tipTitle: cssVar("--ink", "#1b1d21"),
    tipBody: cssVar("--ink-2", "#5a5c62"),
    canvasBg: cssVar("--canvas-bg", "#ece9e2"),
    waveInk: cssVar("--wave-ink", "#84827c"),
    signal: cssVar("--accent", "#c84f2b"),
    wavePlayed: cssVar("--wave-played", "rgba(37,71,255,0.10)"),
    center: cssVar("--hairline", "#d8d4cb"),
    sliderTrack: cssVar("--slider-track", "rgba(27,29,33,0.10)"),
    sceneBg: cssVar("--viewport-scene", "#e7e4dc"),
    meshColor: cssVar("--mesh-color", "#c4c2bb"),
    meshRim: cssVar("--mesh-rim", "#edeff2"),
    hemiGround: cssVar("--mesh-hemi-ground", "#cfccc4"),
  };
}

/* ── Tema yönetimi (Light "Paper" / Dark "Graphite") ── */
function applyThemeAssets() {
  ["energyValue", "volumeThreshold", "styleAlpha", "styleAsymmetry"].forEach((id) => {
    const el = $(id);
    if (el) updateSliderFill(el);
  });
  recolorChart(state.lossChart);
  recolorChart(state.metricsChart);
  if (state.realtime.mode !== "mic") drawWaveformPlayhead(state.waveform.playhead || 0);
  recolorScene();
}

function recolorChart(chart) {
  if (!chart || typeof Chart === "undefined") return;
  const p = palette();
  const cols = [p.s1, p.s2, p.s3];
  const fills = [p.s1Fill, p.s2Fill, p.s3Fill];
  chart.data.datasets.forEach((dataset, index) => {
    if (cols[index]) dataset.borderColor = cols[index];
    if (dataset.fill && fills[index]) dataset.backgroundColor = fills[index];
  });
  const o = chart.options;
  if (o.plugins?.legend?.labels) o.plugins.legend.labels.color = p.legend;
  if (o.plugins?.tooltip) {
    o.plugins.tooltip.backgroundColor = p.tipBg;
    o.plugins.tooltip.borderColor = p.tipBorder;
    o.plugins.tooltip.titleColor = p.tipTitle;
    o.plugins.tooltip.bodyColor = p.tipBody;
  }
  ["x", "y"].forEach((axis) => {
    if (!o.scales?.[axis]) return;
    if (o.scales[axis].ticks) o.scales[axis].ticks.color = p.tick;
    if (o.scales[axis].grid) o.scales[axis].grid.color = p.grid;
    if (o.scales[axis].border) o.scales[axis].border.color = p.axis;
  });
  chart.update("none");
}

function recolorScene() {
  if (!state.rig) return;
  const { THREE, scene, meshRoot, hemi, rim } = state.rig;
  const p = palette();
  if (hemi) hemi.groundColor = new THREE.Color(p.hemiGround);
  if (rim) rim.color = new THREE.Color(p.meshRim);
  meshRoot.traverse((object) => {
    if (object.isMesh && object.material) object.material.color = new THREE.Color(p.meshColor);
  });
  renderMesh();
}

function setTheme(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("vocarig-theme", next); } catch {}
  const toggle = $("themeToggle");
  if (toggle) {
    toggle.setAttribute("aria-pressed", String(next === "dark"));
    const label = toggle.querySelector(".theme-toggle-label");
    if (label) label.textContent = next === "dark" ? "Koyu" : "Aydınlık";
  }
  applyThemeAssets();
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  setTheme(current === "dark" ? "light" : "dark");
}

/* ── Slider fill ── */
function updateSliderFill(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  const p = palette();
  input.style.background = `linear-gradient(to right, ${p.signal} ${pct}%, ${p.sliderTrack} ${pct}%)`;
}

function setupSliders() {
  const sliders = [
    { id: "energyValue", valId: "energyVal" },
    { id: "volumeThreshold", valId: "volumeThresholdVal", digits: 3 },
    { id: "styleAlpha", valId: "styleAlphaVal" },
    { id: "styleAsymmetry", valId: "styleAsymmetryVal" },
  ];
  sliders.forEach(({ id, valId, digits = 2 }) => {
    const input = $(id);
    const badge = $(valId);
    if (!input) return;
    updateSliderFill(input);
    input.addEventListener("input", () => {
      updateSliderFill(input);
      if (badge) badge.textContent = Number(input.value).toFixed(digits);
    });
  });
}

/* ── Scale toggles ── */
function setupScaleToggles() {
  document.querySelectorAll(".scale-toggle").forEach((toggle) => {
    const chartKey = toggle.dataset.chart;
    toggle.querySelectorAll(".scale-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const scale = btn.dataset.scale;
        state.chartScale[chartKey] = scale;
        toggle.querySelectorAll(".scale-btn").forEach((other) => {
          const isActive = other.dataset.scale === scale;
          other.classList.toggle("active", isActive);
          other.setAttribute("aria-pressed", String(isActive));
        });
        const chart = chartKey === "loss" ? state.lossChart : state.metricsChart;
        if (chart) {
          chart.options.scales.y.type = scale;
          chart.update("none");
        }
      });
    });
  });
}

/* ── Chart.js helpers ── */
function chartDefaults() {
  const p = palette();
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: true,
        position: "top",
        labels: {
          color: p.legend,
          boxWidth: 10,
          boxHeight: 3,
          padding: 14,
          font: { size: 11, family: "'Sora', sans-serif", weight: "600" },
          usePointStyle: true,
          pointStyle: "rectRounded",
        },
      },
      tooltip: {
        backgroundColor: p.tipBg,
        borderColor: p.tipBorder,
        borderWidth: 1,
        titleColor: p.tipTitle,
        bodyColor: p.tipBody,
        titleFont: { size: 12, weight: "700" },
        bodyFont: { size: 11, family: "'JetBrains Mono', monospace" },
        padding: 10,
        cornerRadius: 6,
        displayColors: true,
        boxPadding: 4,
      },
    },
    scales: {
      x: {
        offset: false,
        ticks: { color: p.tick, font: { size: 10, family: "'JetBrains Mono', monospace" }, maxTicksLimit: 12, maxRotation: 0 },
        grid: { color: p.grid, lineWidth: 1 },
        border: { color: p.axis },
      },
      y: {
        type: "logarithmic",
        ticks: {
          color: p.tick,
          font: { size: 10, family: "'JetBrains Mono', monospace" },
          callback: (value) => {
            const num = Number(value);
            if (num === 0) return "0";
            if (num >= 10) return num.toFixed(0);
            if (num >= 1) return num.toFixed(1);
            if (num >= 0.1) return num.toFixed(3);
            if (num >= 0.01) return num.toFixed(4);
            if (num >= 0.0001) return num.toFixed(5);
            return num.toExponential(2);
          },
        },
        grid: { color: p.grid, lineWidth: 1 },
        border: { color: p.axis },
      },
    },
  };
}

function createLossChart(rows) {
  const canvas = $("lossCanvas");
  if (!canvas) return;
  if (typeof Chart === "undefined") { drawFallbackChart(canvas, rows); return; }
  const ctx = canvas.getContext("2d");

  if (state.lossChart) {
    state.lossChart.destroy();
    state.lossChart = null;
  }

  const labels = rows.map((row) => row.epoch || "");
  const opts = chartDefaults();
  opts.scales.y.type = state.chartScale.loss;
  const p = palette();

  state.lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Eğitim Kaybı",
          data: rows.map((row) => row.train_loss),
          borderColor: p.s1,
          backgroundColor: p.s1Fill,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Doğrulama Kaybı",
          data: rows.map((row) => row.val_loss),
          borderColor: p.s2,
          backgroundColor: p.s2Fill,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Rollout Kaybı",
          data: rows.map((row) => row.train_rollout_loss),
          borderColor: p.s3,
          backgroundColor: p.s3Fill,
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.3,
          borderDash: [4, 3],
          fill: false,
        },
      ],
    },
    options: opts,
  });
}

function createMetricsChart(rows) {
  const canvas = $("metricsCanvas");
  if (!canvas) return;
  if (typeof Chart === "undefined") { drawFallbackChart(canvas, rows); return; }
  const ctx = canvas.getContext("2d");

  if (state.metricsChart) {
    state.metricsChart.destroy();
    state.metricsChart = null;
  }

  if (!rows || rows.length < 2) {
    drawEmptyChart(canvas, "Metrik grafiği için en az iki epoch gerekiyor.");
    return;
  }

  const labels = rows.map((row) => row.epoch || "");
  const opts = chartDefaults();
  opts.scales.y.type = state.chartScale.metrics;
  const p = palette();

  state.metricsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Eğitim Kaybı",
          data: rows.map((row) => row.train_loss),
          borderColor: p.s1,
          backgroundColor: p.s1Fill,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Doğrulama Kaybı",
          data: rows.map((row) => row.val_loss),
          borderColor: p.s2,
          backgroundColor: p.s2Fill,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Rollout Kaybı",
          data: rows.map((row) => row.train_rollout_loss),
          borderColor: p.s3,
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.3,
          borderDash: [4, 3],
          fill: false,
        },
      ],
    },
    options: opts,
  });
}

function drawFallbackChart(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width || canvas.clientWidth || 900;
  const h = canvas.height || canvas.clientHeight || 260;
  canvas.width = w; canvas.height = h;
  const p = palette();
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = p.canvasBg;
  ctx.fillRect(0, 0, w, h);
  if (!rows || rows.length < 2) {
    drawEmptyChart(canvas, "Grafik için yeterli metrik yok.");
    return;
  }
  const series = [["train_loss", p.s1], ["val_loss", p.s2], ["train_rollout_loss", p.s3]];
  const values = rows.flatMap((r) => series.map(([k]) => Number(r[k])).filter(Number.isFinite));
  const min = Math.min(...values); const max = Math.max(...values);
  ctx.strokeStyle = p.grid; ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) { const y = (h / 5) * i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  for (const [key, color] of series) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    rows.forEach((row, idx) => {
      const x = (idx / Math.max(1, rows.length - 1)) * w;
      const v = Number(row[key]); const norm = (v - min) / Math.max(1e-9, max - min);
      const y = h - 14 - norm * (h - 28);
      idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function drawEmptyChart(canvas, message) {
  const ctx = canvas.getContext("2d");
  const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width || 720));
  const height = Math.max(1, Math.floor(canvas.clientHeight || canvas.height || 320));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const p = palette();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = p.canvasBg;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    const y = (height / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.fillStyle = p.tick;
  ctx.font = "700 13px Sora, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
}

function updateLossChart(rows) {
  if (!state.lossChart) {
    createLossChart(rows);
    return;
  }
  const labels = rows.map((row) => row.epoch || "");
  state.lossChart.data.labels = labels;
  state.lossChart.data.datasets[0].data = rows.map((row) => row.train_loss);
  state.lossChart.data.datasets[1].data = rows.map((row) => row.val_loss);
  state.lossChart.data.datasets[2].data = rows.map((row) => row.train_rollout_loss);
  state.lossChart.update("none");
}

/* ── Data refresh ── */
async function refreshAll() {
  const metricsParams = state.selectedMetricsPath ? `?path=${encodeURIComponent(state.selectedMetricsPath)}` : "";
  const datasetParam = state.dataset.selected ? `?path=${encodeURIComponent(state.dataset.selected)}` : "";
  const [status, data, datasets, metrics, checkpoints, diagnosis, metricsOptions] = await Promise.all([
    api("/api/status"),
    api(`/api/data${datasetParam}`),
    api("/api/datasets"),
    api(`/api/metrics${metricsParams}`),
    api("/api/train/checkpoints"),
    api(`/api/diagnosis${metricsParams}`),
    api("/api/metrics/options"),
  ]);
  state.lipNames = status.lip_names;
  state.arkitNames = status.arkit_names;
  state.model = status.model;
  const selectedDataset = state.dataset.selected || datasets.selected;
  state.dataset = datasets;
  state.dataset.selected = selectedDataset;
  state.device = status.device;
  if (!state.lipValues.length) state.lipValues = state.lipNames.map(() => 0);
  if (!state.modelLipValues.length) state.modelLipValues = state.lipNames.map(() => 0);
  if (!state.targetLipValues.length) state.targetLipValues = state.lipNames.map(() => 0);
  if (!Object.keys(state.arkitValues).length) state.arkitValues = zeroArkitValues();
  if (!Object.keys(state.targetArkitValues).length) state.targetArkitValues = zeroArkitValues();
  renderDevice();
  renderModelSelect();
  renderDatasetSelect();
  renderDataset(data);
  renderMetrics(metrics);
  renderCheckpoints(checkpoints.entries || []);
  renderDiagnosis(diagnosis);
  renderMetricsOptions(metricsOptions);
  renderBars();
  renderArkitTable();
  updatePreview();
  $("modelStatus").textContent = status.files.checkpoint ? status.model.selected_name : "checkpoint yok";
  $("dataStatus").textContent = data.data_exists ? "veri seti hazır" : "veri seti yok";
  $("onnxStatus").textContent = status.files.onnx ? "onnx hazır" : "onnx yok";
}

function renderMetricsOptions(data) {
  const select = $("metricsSelect");
  if (!select) return;
  const options = data.options || [];
  if (!options.length) {
    select.innerHTML = `<option value="">Metrik dosyası yok</option>`;
    return;
  }
  if (!state.selectedMetricsPath) {
    state.selectedMetricsPath = data.selected;
  }
  select.innerHTML = options.map((item) => {
    const bestValLoss = item.best_val_loss;
    const valLoss = bestValLoss !== null && bestValLoss !== undefined && Number.isFinite(Number(bestValLoss))
      ? ` | Doğr: ${Number(bestValLoss).toFixed(4)}`
      : "";
    const label = `${item.name}${valLoss} (${item.completed_epochs || 0} ep)`;
    return `<option value="${escapeHtml(item.path)}"${item.path === state.selectedMetricsPath ? " selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
}

/* ── Render functions ── */
function renderDevice() {
  const badge = $("deviceBadge");
  badge.textContent = `CİHAZ: ${state.device.selected.toUpperCase()} → ${String(state.device.effective).toUpperCase()}`;
  badge.classList.toggle("active", String(state.device.effective).includes("cuda"));
  document.querySelectorAll("#deviceMenu [data-device]").forEach((button) => {
    const mode = button.dataset.device;
    button.classList.toggle("active", mode === state.device.selected);
    button.setAttribute("aria-checked", String(mode === state.device.selected));
    button.disabled = mode === "cuda" && !state.device.cuda_available;
  });
}

function renderModelSelect() {
  const select = $("activeModel");
  const options = state.model.options || [];
  if (!options.length) {
    select.innerHTML = `<option value="">.pt model yok</option>`;
    return;
  }
  select.innerHTML = options.map((item) => {
    const group = item.group === "checkpoint" ? "KONTROL" : item.group === "legacy" ? "ESKİ" : "MODEL";
    const size = compactFileSize(item.size_mb);
    const label = `${group} | ${compactFileName(item.name || item.label || item.path)}${size ? ` · ${size}` : ""}`;
    return `<option value="${escapeHtml(item.path)}"${item.path === state.model.selected ? " selected" : ""} title="${escapeHtml(item.path || label)}">${escapeHtml(label)}</option>`;
  }).join("");
}

function renderDatasetSelect() {
  const select = $("trainingDataset");
  if (!select) return;
  const options = state.dataset.options || [];
  if (!options.length) {
    select.innerHTML = `<option value="">Veri seti yok</option>`;
    return;
  }
  select.innerHTML = options.map((item) => {
    const minutes = compactDuration(item.duration_seconds);
    const size = compactFileSize(item.size_mb);
    const kind = String(item.kind || "data").toLowerCase();
    const kindLabel = { synthetic: "SENTETİK", real: "GERÇEK", data: "VERİ" }[kind] || kind.toUpperCase();
    const meta = [minutes, size].filter(Boolean).join(" · ");
    const label = `${kindLabel} | ${compactFileName(item.name || item.path)}${meta ? ` · ${meta}` : ""}`;
    return `<option value="${escapeHtml(item.path)}"${item.path === state.dataset.selected ? " selected" : ""} title="${escapeHtml(item.path || label)}">${escapeHtml(label)}</option>`;
  }).join("");
}

function renderDataset(data) {
  $("datasetDetails").innerHTML = [
    detailRow("Veri", data.data_exists ? "hazır" : "yok"),
    detailRow("Seçili", data.data_path || "—"),
    detailRow("Üst Veri", data.metadata_exists ? "hazır" : "yok"),
    detailRow("Ses Pencereleri", data.audio_windows_shape ? data.audio_windows_shape.join(" × ") : "—"),
    detailRow("Hedefler", data.y_shape ? data.y_shape.join(" × ") : "—"),
  ].join("");
}

function renderCheckpoints(entries) {
  const select = $("resumeCheckpoint");
  select.innerHTML = `<option value="">Yeni çalışma</option>${entries.map((entry) => (
    `<option value="${escapeHtml(entry.path)}">${escapeHtml(entry.path.split(/[\\/]/).slice(-3).join("/"))}</option>`
  )).join("")}`;
}

function renderMetrics(metrics) {
  state.metrics = metrics.exists ? metrics : null;
  $("metricsDetails").textContent = JSON.stringify(metrics, null, 2);
  createMetricsChart(metrics.history || []);
}

function renderDiagnosis(data) {
  const badge = $("diagnosisStatus");
  const statusMap = { healthy: "finished", insufficient_data: "insufficient_data", no_data: "" };
  const labelMap = { healthy: "SAĞLIKLI", insufficient_data: "YETERSİZ VERİ", no_data: "VERİ YOK" };
  badge.textContent = labelMap[data.overall_status] || String(data.overall_status || "no_data").toUpperCase().replace(/_/g, " ");
  badge.className = `status-pill ${statusMap[data.overall_status] || data.overall_status || ""}`;
  $("diagnosisDetails").innerHTML = [
    detailRow("Özet", data.summary || "—"),
    detailRow("Epoch", data.metrics?.epoch_count ?? "—"),
    detailRow("En İyi Doğrulama", formatNumber(data.metrics?.best_val_loss)),
    detailRow("Hassasiyet", data.metrics?.precision || "—"),
    detailRow("Cihaz", data.metrics?.device || "—"),
  ].join("");
}

function renderBars() {
  const rows = state.lipNames.map((name, index) => ({
    name,
    value: Number(state.lipValues[index] || 0),
  }));
  renderActiveBlendshapes(rows);
  $("lipBars").innerHTML = rows.map(({ name, value }) => {
    const pct = Math.round(value * 100);
    return `
      <div class="bar-row">
        <span>${escapeHtml(name)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span class="bar-value">${value.toFixed(3)}</span>
      </div>
    `;
  }).join("");
}

function renderActiveBlendshapes(rows) {
  const container = $("activeBlendshapes");
  const count = $("activeBlendshapeCount");
  if (!container || !count) return;
  const active = rows
    .filter((row) => row.value >= 0.006)
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);
  count.textContent = `${active.length} aktif`;
  if (!active.length) {
    container.innerHTML = `<div class="active-empty">Hareket bekleniyor</div>`;
    return;
  }
  container.innerHTML = active.map(({ name, value }) => {
    const pct = Math.max(2, Math.round(value * 100));
    return `
      <div class="active-blendshape">
        <span>${escapeHtml(name)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <strong>${value.toFixed(3)}</strong>
      </div>
    `;
  }).join("");
}

function renderArkitTable() {
  const values = state.arkitValues || {};
  $("arkitTable").querySelector("tbody").innerHTML = state.arkitNames.map((name) => {
    const value = Number(values[name] || 0);
    return `<tr><td>${escapeHtml(name)}</td><td>${value.toFixed(4)}</td></tr>`;
  }).join("");
}

function renderTelemetry(result = {}) {
  const tel = result.telemetry || {};
  const isGpu = String(tel.device || state.device.effective || "").includes("cuda");
  const latency = result.latency_ms ?? tel.infer_ms;
  const inferValue = Number.isFinite(Number(latency)) ? `${Number(latency).toFixed(2)} ms` : "-";
  $("telemetryCards").innerHTML = [
    telemetryCell("Kare", state.frame || 0),
    telemetryCell("Çıkarım", inferValue, "hot"),
    telemetryCell("Cihaz", tel.device || state.device.effective || "—", isGpu ? "gpu" : ""),
    telemetryCell("FPS", $("inferenceFps").value),
    telemetryCell("Kontrol", state.lipNames.length),
    telemetryCell("Akış", state.streamId.slice(-6)),
  ].join("");
}

/* ── Inference ── */
async function inferStep(reset = false) {
  const payload = {
    stream_id: state.streamId,
    previous_lip: reset ? zeroLipValues() : previousLipForModel(),
    delta_time: numberOr($("deltaTime").value, 1 / 30),
    time_since_audio_update: 0,
    energy: numberOr($("energyValue").value, 0),
    style_values: [numberOr($("styleAlpha").value, 0.5), numberOr($("styleAsymmetry").value, 0.5)],
    reset_state: reset,
  };
  const result = await api("/api/infer", { method: "POST", body: JSON.stringify(payload) });
  applyInference(result);
}

function applyInference(result) {
  state.frame = (state.frame || 0) + 1;
  state.modelLipValues = Array.isArray(result.lip_values) ? result.lip_values : state.modelLipValues;
  setPoseTarget(result.lip_values, result.arkit_values);
  renderTelemetry(result);
}

function newStreamId(prefix) {
  state.streamId = `${prefix}-${Math.random().toString(16).slice(2)}`;
  state.frame = 0;
}

async function runBakedAudioFile() {
  const file = await selectedAudioFile();
  if (!file) throw new Error("Ses dosyası bulunamadı");
  newStreamId("baked");
  stopLivePlayback();
  stopRealtimeInput();
  stopAudioPlayback();
  $("bakedBtn").disabled = true;
  $("modelStatus").textContent = "ön-hesaplı: tüm ses işleniyor";
  refreshAudioAction();
  const audioBuffer = await file.arrayBuffer();
  await drawWaveBytes(audioBuffer);
  const form = new FormData();
  form.append("file", file);
  form.append("style_values", JSON.stringify([numberOr($("styleAlpha").value, 0.5), numberOr($("styleAsymmetry").value, 0.5)]));
  form.append("volume_threshold", String(volumeThreshold()));
  try {
    const result = await api("/api/infer/audio", { method: "POST", body: form });
    setupAudioPlayback(file, result);
    $("modelStatus").textContent = `ön-hesaplı: ${result.frame_count} kare @ ${result.fps} fps`;
    setLog({ mode: "baked", frame_count: result.frame_count, infer_ms: result.infer_ms, first_frame: result.frames?.[0] });
    await startAudioPlayback();
  } finally {
    $("bakedBtn").disabled = false;
    refreshAudioAction();
  }
}

async function runNoBakedAudioFile() {
  const file = await selectedAudioFile();
  if (!file) throw new Error("Ses dosyası bulunamadı");
  newStreamId("no-baked");
  stopLivePlayback();
  stopRealtimeInput();
  stopAudioPlayback();
  $("noBakedBtn").disabled = true;
  $("modelStatus").textContent = "canlı WAV: dosya akışı başlatılıyor";
  refreshAudioAction();
  try {
    const bytes = await file.arrayBuffer();
    await drawWaveBytes(bytes.slice(0));
    const decoded = await decodeAudioBuffer(bytes.slice(0));
    if (state.audioPlayback.url) URL.revokeObjectURL(state.audioPlayback.url);
    const url = URL.createObjectURL(file);
    const audio = $("audioPreview");
    audio.src = url;
    audio.currentTime = 0;
    audio.onplay = () => {
      revealViewportForPlayback();
      startRealtimeFileTimer();
    };
    audio.onpause = () => {
      drawWaveformPlayhead(audio.currentTime || 0);
      if (!audio.ended) stopRealtimeTimer();
    };
    audio.onended = () => {
      stopRealtimeInput({ keepAudio: true });
      drawWaveformPlayhead(audio.duration || state.waveform.duration || 0);
    };
    audio.onseeked = () => {
      state.realtime.fileSampleOffset = Math.floor((audio.currentTime || 0) * decoded.sampleRate);
      drawWaveformPlayhead(audio.currentTime || 0);
    };
    state.audioPlayback.frames = [];
    state.audioPlayback.url = url;
    state.audioPlayback.frameIndex = -1;
    state.audioPlayback.active = false;
    state.audioPlayback.inferMs = null;
    state.audioPlayback.totalMs = null;
    state.audioPlayback.preparedMode = "live-file";
    state.realtime.mode = "file";
    state.realtime.fileBuffer = decoded;
    state.realtime.fileSampleOffset = 0;
    state.realtime.resetNext = true;
    resetPoseState();
    $("modelStatus").textContent = "canlı WAV: parçalar akıyor";
    revealViewportForPlayback();
    await startAudioPlayback("canlı WAV hazır - YÜZ + SESİ BAŞLAT düğmesine basın");
  } finally {
    $("noBakedBtn").disabled = false;
    refreshAudioAction();
  }
}

async function runRealMicTest() {
  newStreamId("real-test");
  stopLivePlayback();
  stopRealtimeInput();
  stopAudioPlayback();
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Mikrofon API kullanılamıyor");
  $("realTestBtn").disabled = true;
  $("modelStatus").textContent = "mikrofon: mikrofon bekleniyor";
  refreshAudioAction();
  try {
    const micConstraints = microphoneAudioConstraints();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("AudioContext kullanılamıyor");
    const context = new AudioContextClass();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(2048, 1, 1);
    const gain = context.createGain();
    gain.gain.value = 0;
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer;
      const chunk = mixAudioBuffer(input, 0, input.length);
      state.realtime.micQueue.push(chunk);
      drawLiveWaveform(chunk, input.sampleRate || context.sampleRate);
    };
    source.connect(processor);
    processor.connect(gain);
    gain.connect(context.destination);
    state.realtime.mode = "mic";
    state.realtime.audioContext = context;
    state.realtime.micStream = stream;
    state.realtime.micSource = source;
    state.realtime.micProcessor = processor;
    state.realtime.micGain = gain;
    state.realtime.micQueue = [];
    state.realtime.liveWave = [];
    state.realtime.resetNext = true;
    state.audioPlayback.preparedMode = "mic";
    resetPoseState();
    revealViewportForPlayback();
    startRealtimeMicTimer();
    const settings = stream.getAudioTracks()[0]?.getSettings?.() || {};
    $("modelStatus").textContent = micConstraints.noiseSuppression
      ? "mikrofon: canlı + gürültü bastırma"
      : "mikrofon: canlı";
    setLog({ mode: "real-test", mic_constraints: micConstraints, mic_settings: settings });
  } finally {
    $("realTestBtn").disabled = false;
    refreshAudioAction();
  }
}

function microphoneAudioConstraints() {
  const enabled = $("micNoiseSuppression")?.checked ?? true;
  const supported = navigator.mediaDevices?.getSupportedConstraints?.() || {};
  const audio = {
    channelCount: 1,
  };
  if (supported.noiseSuppression) audio.noiseSuppression = enabled;
  if (supported.echoCancellation) audio.echoCancellation = enabled;
  if (supported.autoGainControl) audio.autoGainControl = false;
  return audio;
}

async function selectedAudioFile() {
  const selected = $("audioFile").files[0];
  if (selected) return selected;
  const response = await fetch(state.defaultAudioUrl);
  if (!response.ok) throw new Error(`Varsayılan ses bulunamadı: ${state.defaultAudioUrl}`);
  const blob = await response.blob();
  return new File([blob], state.defaultAudioName, { type: blob.type || "audio/wav" });
}

async function loadDefaultAudioPreview() {
  try {
    const response = await fetch(state.defaultAudioUrl);
    if (!response.ok) return;
    const blob = await response.blob();
    const buffer = await blob.arrayBuffer();
    await drawWaveBytes(buffer);
    const audio = $("audioPreview");
    audio.src = state.defaultAudioUrl;
    refreshAudioAction();
    setAudioFileName(state.defaultAudioName);
    $("modelStatus").textContent = state.defaultAudioName;
  } catch (error) {
    console.warn("default audio unavailable", error);
  }
}

function setAudioFileName(name) {
  const label = $("audioFileName");
  if (label) label.textContent = name || state.defaultAudioName;
}

function revealViewportForPlayback() {
  if (!window.matchMedia("(max-width: 1320px)").matches) return;
  $("viewport")?.scrollIntoView({
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    block: "start",
    inline: "nearest",
  });
}

function audioHasSource(audio = $("audioPreview")) {
  return Boolean(audio?.currentSrc || audio?.src);
}

function hasAudioInput() {
  return Boolean($("audioFile")?.files?.[0]) || audioHasSource() || Boolean(state.defaultAudioUrl);
}

function currentMode() {
  return INFERENCE_MODES[state.inferenceMode] || INFERENCE_MODES.baked;
}

function setInferenceMode(mode) {
  if (!INFERENCE_MODES[mode]) return;
  state.inferenceMode = mode;
  document.querySelectorAll("[data-inference-mode]").forEach((button) => {
    const active = button.dataset.inferenceMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.setAttribute("aria-checked", String(active));
  });
  refreshAudioAction();
}

async function handleInferenceModeClick(mode) {
  setInferenceMode(mode);
  if (isAudioPreparationBusy()) return;
  await runSelectedInferenceMode();
}

function preparedInferenceMode() {
  if (state.realtime.mode === "file") return "live-file";
  if (state.realtime.mode === "mic") return "mic";
  if (state.audioPlayback.frames.length) return state.audioPlayback.preparedMode || "baked";
  return "";
}

function isCurrentModePrepared() {
  return preparedInferenceMode() === state.inferenceMode;
}

function playbackStateLabel({ ready, playing, prepared }) {
  if (state.realtime.mode === "mic") return "Mikrofon canlı akıyor";
  if (state.realtime.mode === "file" && playing) return "Canlı WAV akıyor";
  if (playing) return "Yüz + ses oynuyor";
  if (prepared) return currentMode().ready;
  if (ready) return state.inferenceMode === "baked" ? "Hazırlık bekliyor" : "Başlatmaya hazır";
  return "Ses bekleniyor";
}

function refreshAudioAction() {
  const button = $("audioPlayBtn");
  const audio = $("audioPreview");
  if (!button || !audio) return;
  const ready = hasAudioInput();
  const micActive = state.realtime.mode === "mic";
  const playing = micActive || (ready && !audio.paused && !audio.ended);
  const prepared = isCurrentModePrepared();
  const needsSyncPrep = ready && audio.paused && !prepared;
  const busy = isAudioPreparationBusy();
  button.disabled = !ready;
  button.classList.toggle("is-playing", playing);
  button.classList.toggle("is-busy", Boolean(busy));
  button.setAttribute("aria-pressed", String(playing));
  button.textContent = busy
    ? "HAZIRLANIYOR..."
    : (playing ? (micActive ? "MİKROFONU DURDUR" : "DURAKLAT") : (needsSyncPrep ? currentMode().prepare : "YÜZ + SESİ BAŞLAT"));
  const modeLabel = $("syncModeLabel");
  const stateLabel = $("syncStateLabel");
  if (modeLabel) modeLabel.textContent = currentMode().label;
  if (stateLabel) stateLabel.textContent = busy ? "Hazırlanıyor..." : playbackStateLabel({ ready, playing, prepared });
}

function isAudioGestureError(error) {
  return error?.name === "NotAllowedError";
}

async function startMutedPlaybackFallback(audio) {
  const wasMuted = audio.muted;
  audio.muted = true;
  try {
    await audio.play();
  } catch (error) {
    audio.muted = wasMuted;
    throw error;
  }
  window.setTimeout(() => {
    audio.muted = wasMuted;
    refreshAudioAction();
  }, 90);
}

function markPlaybackStarted() {
  const status = $("modelStatus")?.textContent || "";
  if (status.includes("bekliyor") || status.includes("hazır -")) {
    $("modelStatus").textContent = "yüz + ses oynatılıyor";
  }
}

function needsSyncedPreparation() {
  return !isCurrentModePrepared();
}

function isAudioPreparationBusy() {
  return Boolean(
    state.audioPlayback.nativePreparePending ||
    $("bakedBtn")?.disabled ||
    $("noBakedBtn")?.disabled ||
    $("realTestBtn")?.disabled
  );
}

async function runSelectedInferenceMode() {
  if (state.inferenceMode === "live-file") return runNoBakedAudioFile();
  if (state.inferenceMode === "mic") return runRealMicTest();
  return runBakedAudioFile();
}

function selectedAudioName() {
  const selected = $("audioFile").files[0];
  return selected ? selected.name : state.defaultAudioName;
}

function volumeThreshold() {
  return Math.max(0, Math.min(1, numberOr($("volumeThreshold")?.value, 0.015)));
}

function setupAudioPlayback(file, result) {
  const frames = Array.isArray(result.frames) ? result.frames : [];
  if (!frames.length) throw new Error("Model ses karesi döndürmedi");
  if (state.audioPlayback.url) URL.revokeObjectURL(state.audioPlayback.url);
  const url = URL.createObjectURL(file);
  const audio = $("audioPreview");
  audio.src = url;
  audio.currentTime = 0;
  audio.onplay = () => {
    state.audioPlayback.active = true;
    revealViewportForPlayback();
    scheduleAudioFrameSync();
  };
  audio.onpause = () => {
    state.audioPlayback.active = false;
    cancelAudioFrameSync();
    drawWaveformPlayhead(audio.currentTime || 0);
  };
  audio.onended = () => {
    state.audioPlayback.active = false;
    cancelAudioFrameSync();
    applyAudioFrame(frames.length - 1, true);
    drawWaveformPlayhead(audio.duration || state.waveform.duration || 0);
  };
  audio.onseeked = () => {
    syncAudioFrame(true);
    drawWaveformPlayhead(audio.currentTime || 0);
  };
  state.audioPlayback.frames = frames;
  state.audioPlayback.fps = Number(result.fps) || 30;
  state.audioPlayback.url = url;
  state.audioPlayback.inferMs = Number.isFinite(Number(result.infer_ms)) ? Number(result.infer_ms) : null;
  state.audioPlayback.totalMs = Number.isFinite(Number(result.latency_ms)) ? Number(result.latency_ms) : null;
  state.audioPlayback.preparedMode = "baked";
  state.audioPlayback.frameIndex = -1;
  state.audioPlayback.active = false;
  applyAudioFrame(0, true);
  refreshAudioAction();
}

async function startAudioPlayback(blockedMessage = "ses hazır - YÜZ + SESİ BAŞLAT düğmesine basın") {
  const audio = $("audioPreview");
  if (!audioHasSource(audio)) return false;
  try {
    revealViewportForPlayback();
    await audio.play();
    refreshAudioAction();
    markPlaybackStarted();
    return true;
  } catch (error) {
    refreshAudioAction();
    if (isAudioGestureError(error)) {
      try {
        await startMutedPlaybackFallback(audio);
        refreshAudioAction();
        markPlaybackStarted();
        return true;
      } catch {
        $("modelStatus").textContent = blockedMessage;
        return false;
      }
    }
    throw error;
  }
}

async function toggleAudioPlayback() {
  const audio = $("audioPreview");
  if (!hasAudioInput()) return;
  if (state.realtime.mode === "mic") {
    stopCurrentPlayback();
    return;
  }
  if (audio.paused) {
    if (needsSyncedPreparation()) {
      if (isAudioPreparationBusy()) return;
      await runSelectedInferenceMode();
      refreshAudioAction();
      return;
    }
    if (audio.ended) audio.currentTime = 0;
    await startAudioPlayback("oynatma bekliyor - YÜZ + SESİ BAŞLAT düğmesine basın");
  } else {
    audio.pause();
  }
  refreshAudioAction();
}

function guardUnsyncedNativePlayback() {
  const audio = $("audioPreview");
  if (!audioHasSource(audio) || !needsSyncedPreparation()) return false;
  audio.pause();
  audio.currentTime = 0;
  if (isAudioPreparationBusy()) {
    $("modelStatus").textContent = "senkron hazırlığı sürüyor";
    refreshAudioAction();
    return true;
  }
  state.audioPlayback.nativePreparePending = true;
  $("modelStatus").textContent = state.inferenceMode === "mic"
    ? "mikrofon hazırlanıyor"
    : (state.inferenceMode === "live-file" ? "canlı WAV hazırlanıyor" : "yüz ve ses hazırlanıyor");
  refreshAudioAction();
  runSelectedInferenceMode()
    .catch(showError)
    .finally(() => {
      state.audioPlayback.nativePreparePending = false;
      refreshAudioAction();
    });
  return true;
}

function stopLivePlayback() {
  state.playing = false;
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
}

function stopAudioPlayback() {
  stopRealtimeInput();
  const audio = $("audioPreview");
  cancelAudioFrameSync();
  stopPoseSmoothing();
  if (!audio.paused) audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (state.audioPlayback.url) URL.revokeObjectURL(state.audioPlayback.url);
  state.audioPlayback.frames = [];
  state.audioPlayback.url = "";
  state.audioPlayback.inferMs = null;
  state.audioPlayback.totalMs = null;
  state.audioPlayback.preparedMode = "";
  state.audioPlayback.frameIndex = -1;
  state.audioPlayback.active = false;
  clearWaveform();
  refreshAudioAction();
}

function resetAudioPlayback() {
  stopRealtimeInput({ keepAudio: true });
  const audio = $("audioPreview");
  cancelAudioFrameSync();
  stopPoseSmoothing();
  if (!audio.paused) audio.pause();
  if (audio.src) {
    audio.currentTime = 0;
    drawWaveformPlayhead(0);
  }
  state.audioPlayback.frameIndex = -1;
  state.audioPlayback.active = false;
  refreshAudioAction();
}

function stopCurrentPlayback() {
  stopLivePlayback();
  stopRealtimeInput({ keepAudio: true });
  cancelAudioFrameSync();
  const audio = $("audioPreview");
  if (!audio.paused) audio.pause();
  state.audioPlayback.active = false;
  $("modelStatus").textContent = "durduruldu";
  refreshAudioAction();
}

function stopRealtimeTimer() {
  if (state.realtime.timer) window.clearInterval(state.realtime.timer);
  state.realtime.timer = null;
}

function stopRealtimeInput(options = {}) {
  const keepAudio = Boolean(options.keepAudio);
  stopRealtimeTimer();
  closeRealtimeSocket(true);
  state.realtime.pending = false;
  state.realtime.resetNext = false;
  if (state.realtime.micProcessor) {
    state.realtime.micProcessor.onaudioprocess = null;
    state.realtime.micProcessor.disconnect();
  }
  if (state.realtime.micSource) state.realtime.micSource.disconnect();
  if (state.realtime.micGain) state.realtime.micGain.disconnect();
  if (state.realtime.micStream) {
    state.realtime.micStream.getTracks().forEach((track) => track.stop());
  }
  if (state.realtime.audioContext) {
    state.realtime.audioContext.close().catch(() => {});
  }
  if (!keepAudio && state.realtime.mode === "file") {
    const audio = $("audioPreview");
    if (!audio.paused) audio.pause();
  }
  state.realtime.mode = "idle";
  state.realtime.fileBuffer = null;
  state.realtime.fileSampleOffset = 0;
  state.realtime.audioContext = null;
  state.realtime.micStream = null;
  state.realtime.micSource = null;
  state.realtime.micProcessor = null;
  state.realtime.micGain = null;
  state.realtime.micQueue = [];
  state.realtime.skippedTicks = 0;
  state.realtime.droppedRequests = 0;
  clearRealtimeWarning();
}

function closeRealtimeSocket(intentional = false) {
  state.realtime.intentionalSocketClose = intentional;
  if (state.realtime.socketWaiters?.size) {
    const error = new Error("Canlı WebSocket bağlantısı kapandı");
    state.realtime.socketWaiters.forEach((waiter) => {
      window.clearTimeout(waiter.timeout);
      waiter.reject(error);
    });
    state.realtime.socketWaiters.clear();
  }
  const socket = state.realtime.socket;
  state.realtime.socket = null;
  state.realtime.socketReady = null;
  if (socket && socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
    socket.close(1000, "stop");
  }
}

function startRealtimeFileTimer() {
  if (state.realtime.mode !== "file") return;
  stopRealtimeTimer();
  const fps = Math.max(1, Math.min(60, numberOr($("inferenceFps").value, 30)));
  state.realtime.timer = window.setInterval(() => tickRealtimeFile().catch(showError), 1000 / fps);
}

function startRealtimeMicTimer() {
  stopRealtimeTimer();
  const fps = Math.max(1, Math.min(60, numberOr($("inferenceFps").value, 30)));
  state.realtime.timer = window.setInterval(() => tickRealtimeMic().catch(showError), 1000 / fps);
}

async function tickRealtimeFile() {
  if (state.realtime.mode !== "file" || !state.realtime.fileBuffer) return;
  if (state.realtime.pending) {
    noteRealtimeTickSkipped("file");
    return;
  }
  const audio = $("audioPreview");
  if (audio.paused || audio.ended) return;
  const buffer = state.realtime.fileBuffer;
  const end = Math.min(buffer.length, Math.floor((audio.currentTime || 0) * buffer.sampleRate));
  const start = Math.max(0, Math.min(state.realtime.fileSampleOffset, end));
  if (end <= start && !state.realtime.resetNext) return;
  const samples = mixAudioBuffer(buffer, start, end);
  state.realtime.fileSampleOffset = end;
  await sendRealtimeChunk(samples, buffer.sampleRate);
  drawWaveformPlayhead(audio.currentTime || 0);
}

async function tickRealtimeMic() {
  if (state.realtime.mode !== "mic" || !state.realtime.audioContext) return;
  if (state.realtime.pending) {
    noteRealtimeTickSkipped("mic");
    return;
  }
  const chunks = state.realtime.micQueue.splice(0);
  const samples = flattenChunks(chunks);
  if (!samples.length && !state.realtime.resetNext) return;
  await sendRealtimeChunk(samples, state.realtime.audioContext.sampleRate);
}

async function sendRealtimeChunk(samples, sampleRate) {
  state.realtime.pending = true;
  try {
    const result = await sendRealtimeSocketPayload({
      stream_id: state.streamId,
      samples,
      sample_rate: sampleRate,
      previous_lip: state.realtime.resetNext ? zeroLipValues() : previousLipForModel(),
      delta_time: numberOr($("deltaTime").value, 1 / 30),
      style_values: [numberOr($("styleAlpha").value, 0.5), numberOr($("styleAsymmetry").value, 0.5)],
      volume_threshold: volumeThreshold(),
      reset_state: state.realtime.resetNext,
    });
    state.realtime.resetNext = false;
    applyInference(result);
    if (!state.realtime.skippedTicks && !state.realtime.droppedRequests) clearRealtimeWarning();
  } catch (error) {
    if (!state.realtime.intentionalSocketClose) {
      state.realtime.droppedRequests += 1;
      showRealtimeWarning(`UYARI: WS istek düştü (${state.realtime.droppedRequests})`);
    }
    throw error;
  } finally {
    state.realtime.pending = false;
  }
}

function ensureRealtimeSocket() {
  const current = state.realtime.socket;
  if (current?.readyState === WebSocket.OPEN) return Promise.resolve(current);
  if (state.realtime.socketReady) return state.realtime.socketReady;

  closeRealtimeSocket(true);
  state.realtime.intentionalSocketClose = false;
  const socket = new WebSocket(websocketUrl("/ws/infer"));
  state.realtime.socket = socket;
  state.realtime.socketReady = new Promise((resolve, reject) => {
    let settled = false;
    const timeout = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      state.realtime.socketReady = null;
      showRealtimeWarning("UYARI: WS bağlantı gecikti");
      reject(new Error("Canlı WebSocket bağlantı zaman aşımı"));
      closeRealtimeSocket(true);
    }, 3000);

    socket.onopen = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      state.realtime.socketReady = null;
      resolve(socket);
    };
    socket.onmessage = handleRealtimeSocketMessage;
    socket.onerror = () => {
      showRealtimeWarning("UYARI: WS hata");
    };
    socket.onclose = () => {
      window.clearTimeout(timeout);
      if (!settled) {
        settled = true;
        state.realtime.socketReady = null;
        reject(new Error("Canlı WebSocket açılmadan kapandı"));
      }
      if (!state.realtime.intentionalSocketClose && state.realtime.mode !== "idle") {
        showRealtimeWarning("UYARI: WS bağlantı koptu");
      }
      state.realtime.socket = null;
      state.realtime.socketReady = null;
      if (state.realtime.socketWaiters.size) {
        const error = new Error("Canlı WebSocket bağlantısı kapandı");
        state.realtime.socketWaiters.forEach((waiter) => {
          window.clearTimeout(waiter.timeout);
          waiter.reject(error);
        });
        state.realtime.socketWaiters.clear();
      }
    };
  });
  return state.realtime.socketReady;
}

async function sendRealtimeSocketPayload(payload) {
  const socket = await ensureRealtimeSocket();
  if (socket.readyState !== WebSocket.OPEN) throw new Error("Canlı WebSocket açık değil");
  const requestId = `${state.streamId}-${++state.realtime.requestSeq}`;
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      state.realtime.socketWaiters.delete(requestId);
      reject(new Error("Canlı WebSocket yanıt zaman aşımı"));
    }, 2000);
    state.realtime.socketWaiters.set(requestId, { resolve, reject, timeout });
    try {
      socket.send(JSON.stringify({ ...payload, request_id: requestId }));
    } catch (error) {
      window.clearTimeout(timeout);
      state.realtime.socketWaiters.delete(requestId);
      reject(error);
    }
  });
}

function handleRealtimeSocketMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    state.realtime.droppedRequests += 1;
    showRealtimeWarning(`UYARI: WS bozuk cevap (${state.realtime.droppedRequests})`);
    return;
  }
  const requestId = payload.request_id;
  const waiter = requestId
    ? state.realtime.socketWaiters.get(requestId)
    : state.realtime.socketWaiters.values().next().value;
  if (!waiter) {
    state.realtime.droppedRequests += 1;
    showRealtimeWarning(`UYARI: WS sahipsiz cevap (${state.realtime.droppedRequests})`);
    return;
  }
  window.clearTimeout(waiter.timeout);
  if (requestId) state.realtime.socketWaiters.delete(requestId);
  else {
    const firstKey = state.realtime.socketWaiters.keys().next().value;
    if (firstKey) state.realtime.socketWaiters.delete(firstKey);
  }
  if (!payload.ok) {
    waiter.reject(new Error(payload.error || "Canlı WebSocket çıkarımı başarısız"));
    return;
  }
  delete payload.ok;
  delete payload.request_id;
  waiter.resolve(payload);
}

function noteRealtimeTickSkipped(mode) {
  state.realtime.skippedTicks += 1;
  if (state.realtime.skippedTicks === 1 || state.realtime.skippedTicks % 10 === 0) {
    showRealtimeWarning(`UYARI: ${mode} tick gecikti (${state.realtime.skippedTicks})`);
  }
}

function showRealtimeWarning(message) {
  const warning = $("realtimeWarning");
  if (!warning) return;
  warning.textContent = message;
  warning.hidden = false;
  if (state.realtime.warningTimer) window.clearTimeout(state.realtime.warningTimer);
  state.realtime.warningTimer = window.setTimeout(clearRealtimeWarning, 6000);
}

function clearRealtimeWarning() {
  const warning = $("realtimeWarning");
  if (!warning) return;
  warning.hidden = true;
  warning.textContent = "";
  if (state.realtime.warningTimer) window.clearTimeout(state.realtime.warningTimer);
  state.realtime.warningTimer = null;
}

function mixAudioBuffer(buffer, start, end) {
  const first = Math.max(0, Math.floor(start));
  const last = Math.max(first, Math.floor(end));
  const length = Math.max(0, last - first);
  const mixed = new Array(length).fill(0);
  const channels = Math.max(1, buffer.numberOfChannels || 1);
  for (let channel = 0; channel < channels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < length; index++) {
      mixed[index] += Number(data[first + index] || 0) / channels;
    }
  }
  return mixed;
}

function flattenChunks(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const samples = new Array(total);
  let offset = 0;
  chunks.forEach((chunk) => {
    for (let index = 0; index < chunk.length; index++) samples[offset + index] = chunk[index];
    offset += chunk.length;
  });
  return samples;
}

function scheduleAudioFrameSync() {
  cancelAudioFrameSync();
  const tick = () => {
    syncAudioFrame(false);
    drawWaveformPlayhead($("audioPreview").currentTime || 0);
    if (state.audioPlayback.active) {
      state.audioPlayback.rafId = window.requestAnimationFrame(tick);
    }
  };
  state.audioPlayback.rafId = window.requestAnimationFrame(tick);
}

function cancelAudioFrameSync() {
  if (state.audioPlayback.rafId) window.cancelAnimationFrame(state.audioPlayback.rafId);
  state.audioPlayback.rafId = null;
}

function syncAudioFrame(force) {
  const audio = $("audioPreview");
  const frames = state.audioPlayback.frames;
  if (!frames.length || !audio.src) return;
  const index = Math.max(0, Math.min(frames.length - 1, Math.floor(audio.currentTime * state.audioPlayback.fps)));
  applyAudioFrame(index, force);
}

function applyAudioFrame(index, force = false) {
  const frames = state.audioPlayback.frames;
  if (!frames.length) return;
  const clamped = Math.max(0, Math.min(frames.length - 1, index));
  if (!force && clamped === state.audioPlayback.frameIndex) return;
  const frame = frames[clamped];
  state.audioPlayback.frameIndex = clamped;
  state.frame = clamped + 1;
  state.modelLipValues = Array.isArray(frame.lip_values) ? frame.lip_values : state.modelLipValues;
  setPoseTarget(frame.lip_values, frame.arkit_values, { snap: force });
  renderTelemetry({ latency_ms: state.audioPlayback.inferMs, telemetry: { device: state.device.effective } });
}

function previousLipForModel() {
  return state.modelLipValues.length ? state.modelLipValues : state.lipValues;
}

function zeroLipValues() {
  return state.lipNames.map(() => 0);
}

function zeroArkitValues() {
  return Object.fromEntries(state.arkitNames.map((name) => [name, 0]));
}

function resetPoseState() {
  stopPoseSmoothing();
  const lipZeros = zeroLipValues();
  const arkitZeros = zeroArkitValues();
  state.modelLipValues = lipZeros.slice();
  state.targetLipValues = lipZeros.slice();
  state.lipValues = lipZeros.slice();
  state.targetArkitValues = { ...arkitZeros };
  state.arkitValues = { ...arkitZeros };
  renderBars();
  renderArkitTable();
  updatePreview();
}

function setPoseTarget(lipValues, arkitValues, options = {}) {
  const lipTarget = Array.isArray(lipValues)
    ? lipValues.map((value) => clamp01(value))
    : state.targetLipValues;
  const arkitTarget = arkitValues ? normalizeArkitValues(arkitValues) : state.targetArkitValues;

  state.targetLipValues = lipTarget;
  state.targetArkitValues = arkitTarget;

  if (options.snap || !state.lipValues.length) {
    stopPoseSmoothing();
    state.lipValues = lipTarget.slice();
    state.arkitValues = { ...arkitTarget };
    renderBars();
    renderArkitTable();
    updatePreview();
    return;
  }

  startPoseSmoothing();
}

function normalizeArkitValues(values) {
  const normalized = {};
  const names = state.arkitNames.length ? state.arkitNames : Object.keys(values || {});
  names.forEach((name) => {
    normalized[name] = clamp01(values?.[name] ?? 0);
  });
  return normalized;
}

function startPoseSmoothing() {
  if (state.poseSmoothing.rafId) return;
  state.poseSmoothing.lastTime = 0;
  state.poseSmoothing.rafId = window.requestAnimationFrame(stepPoseSmoothing);
}

function stopPoseSmoothing() {
  if (state.poseSmoothing.rafId) window.cancelAnimationFrame(state.poseSmoothing.rafId);
  state.poseSmoothing.rafId = null;
  state.poseSmoothing.lastTime = 0;
}

function stepPoseSmoothing(timestamp) {
  const last = state.poseSmoothing.lastTime || timestamp;
  const dt = Math.max(1 / 120, Math.min(0.05, (timestamp - last) / 1000 || 1 / 60));
  state.poseSmoothing.lastTime = timestamp;

  const lipResult = smoothLipValues(dt);
  const arkitResult = smoothArkitValues(dt);
  state.lipValues = lipResult.values;
  state.arkitValues = arkitResult.values;

  renderBars();
  renderArkitTable();
  updatePreview();

  if (lipResult.settled && arkitResult.settled) {
    stopPoseSmoothing();
    return;
  }
  state.poseSmoothing.rafId = window.requestAnimationFrame(stepPoseSmoothing);
}

function smoothLipValues(dt) {
  const target = state.targetLipValues.length ? state.targetLipValues : zeroLipValues();
  const current = state.lipValues.length ? state.lipValues : zeroLipValues();
  const size = Math.max(target.length, current.length, state.lipNames.length);
  const values = new Array(size);
  let settled = true;
  for (let index = 0; index < size; index++) {
    const name = state.lipNames[index] || "";
    const next = smoothBlendValue(name, current[index] || 0, target[index] || 0, dt);
    values[index] = next;
    if (Math.abs(next - (target[index] || 0)) > SMOOTH_EPSILON) settled = false;
  }
  return { values, settled };
}

function smoothArkitValues(dt) {
  const target = state.targetArkitValues || {};
  const current = state.arkitValues || {};
  const keys = state.arkitNames.length
    ? state.arkitNames
    : Array.from(new Set([...Object.keys(target), ...Object.keys(current)]));
  const values = {};
  let settled = true;
  keys.forEach((name) => {
    const targetValue = target[name] || 0;
    const next = smoothBlendValue(name, current[name] || 0, targetValue, dt);
    values[name] = next;
    if (Math.abs(next - targetValue) > SMOOTH_EPSILON) settled = false;
  });
  return { values, settled };
}

function smoothBlendValue(name, current, target, dt) {
  const diff = clamp01(target) - clamp01(current);
  if (Math.abs(diff) <= SMOOTH_EPSILON) return clamp01(target);
  const params = smoothingParams(name, diff);
  const alpha = 1 - Math.exp(-params.response * dt);
  const wantedStep = diff * alpha;
  const maxStep = params.maxPerSecond * dt;
  return clamp01(current + Math.max(-maxStep, Math.min(maxStep, wantedStep)));
}

function smoothingParams(name, diff) {
  const rising = diff > 0;
  let response = rising ? 18 : 13;
  let maxPerSecond = rising ? 4.8 : 3.2;

  if (name === "jawOpen") {
    response = rising ? 24 : 12;
    maxPerSecond = rising ? 6.2 : 2.8;
  } else if (name === "mouthClose") {
    response = rising ? 24 : 14;
    maxPerSecond = rising ? 6.0 : 3.0;
  } else if (SOFT_CONTROLS.has(name)) {
    response = rising ? 13 : 10;
    maxPerSecond = rising ? 3.0 : 2.4;
  } else if (SIDE_CONTROLS.has(name)) {
    response = rising ? 14 : 11;
    maxPerSecond = rising ? 3.0 : 2.4;
  }

  return { response, maxPerSecond };
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

async function drawWaveBytes(buffer) {
  try {
    const decoded = await decodeAudioBuffer(buffer);
    state.waveform.peaks = buildWaveformPeaks(decoded);
    state.waveform.duration = decoded.duration || 0;
    state.waveform.ready = true;
    state.waveform.playhead = 0;
    drawWaveformPlayhead(0);
  } catch (error) {
    console.warn("waveform decode failed", error);
    clearWaveform();
  }
}

async function decodeAudioBuffer(buffer) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error("AudioContext unavailable");
  const audioContext = new AudioContextClass();
  try {
    return await audioContext.decodeAudioData(buffer.slice(0));
  } finally {
    if (audioContext.close) audioContext.close().catch(() => {});
  }
}

function buildWaveformPeaks(audioBuffer) {
  const { width } = waveCanvasContext();
  const bucketCount = Math.max(1, Math.floor(width * 2));
  const channels = [];
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
    channels.push(audioBuffer.getChannelData(channel));
  }
  const sampleCount = audioBuffer.length;
  const samplesPerBucket = Math.max(1, Math.ceil(sampleCount / bucketCount));
  const peaks = [];
  for (let bucket = 0; bucket < bucketCount; bucket++) {
    const start = bucket * samplesPerBucket;
    const end = Math.min(sampleCount, start + samplesPerBucket);
    let min = 1;
    let max = -1;
    for (let sampleIndex = start; sampleIndex < end; sampleIndex++) {
      let sample = 0;
      for (const channel of channels) sample += channel[sampleIndex] || 0;
      sample /= Math.max(1, channels.length);
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    peaks.push({ min, max });
  }
  return peaks;
}

function drawWaveformPlayhead(time = state.waveform.playhead || 0) {
  const { ctx, width, height } = waveCanvasContext();
  state.waveform.playhead = time;
  drawWaveformBase(ctx, width, height);
  if (!state.waveform.ready || !state.waveform.peaks.length) return;

  const p = palette();
  const center = height * 0.5;
  const amplitude = Math.max(12, (height - 30) * 0.5);
  ctx.strokeStyle = p.waveInk;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const peak = state.waveform.peaks[Math.floor((x / Math.max(1, width - 1)) * (state.waveform.peaks.length - 1))];
    const y1 = center + peak.min * amplitude;
    const y2 = center + peak.max * amplitude;
    ctx.moveTo(x + 0.5, y1);
    ctx.lineTo(x + 0.5, y2);
  }
  ctx.stroke();

  const duration = state.waveform.duration || $("audioPreview").duration || 0;
  const progress = duration > 0 ? Math.max(0, Math.min(1, time / duration)) : 0;
  const playheadX = progress * width;
  ctx.fillStyle = p.wavePlayed;
  ctx.fillRect(0, 0, playheadX, height);
  ctx.strokeStyle = p.signal;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(playheadX, 8);
  ctx.lineTo(playheadX, height - 8);
  ctx.stroke();
}

function clearWaveform() {
  state.waveform.peaks = [];
  state.waveform.duration = 0;
  state.waveform.ready = false;
  state.waveform.playhead = 0;
  state.realtime.liveWave = [];
  const { ctx, width, height } = waveCanvasContext();
  drawWaveformBase(ctx, width, height);
}

function drawLiveWaveform(samples, sampleRate) {
  const keep = Math.max(1, Math.floor((sampleRate || 16000) * 2));
  state.realtime.liveWave = state.realtime.liveWave.concat(samples).slice(-keep);
  const { ctx, width, height } = waveCanvasContext();
  drawWaveformBase(ctx, width, height);
  const wave = state.realtime.liveWave;
  if (!wave.length) return;
  const center = height * 0.5;
  const amplitude = Math.max(12, (height - 30) * 0.5);
  const step = Math.max(1, Math.floor(wave.length / width));
  ctx.strokeStyle = palette().signal;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const start = x * step;
    const end = Math.min(wave.length, start + step);
    let min = 1;
    let max = -1;
    for (let index = start; index < end; index++) {
      const value = Number(wave[index] || 0);
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    ctx.moveTo(x + 0.5, center + min * amplitude);
    ctx.lineTo(x + 0.5, center + max * amplitude);
  }
  ctx.stroke();
}

function drawWaveformBase(ctx, width, height) {
  const p = palette();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = p.canvasBg;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = p.center;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function waveCanvasContext() {
  const canvas = $("waveCanvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width || 760));
  const height = Math.max(1, Math.floor(canvas.clientHeight || canvas.height || 132));
  const pixelWidth = Math.floor(width * dpr);
  const pixelHeight = Math.floor(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { canvas, ctx, width, height };
}

/* ── Pipeline actions ── */
async function generateSynthetic() {
  const result = await api("/api/synthetic/generate", {
    method: "POST",
    body: JSON.stringify({
      utterances: numberInput("syntheticUtterances"),
      seed: numberInput("syntheticSeed"),
      min_phonemes: numberInput("syntheticMinPhonemes"),
      max_phonemes: numberInput("syntheticMaxPhonemes"),
      silence_probability: numberInput("syntheticSilence"),
      tr_probability: numberInput("syntheticTrProbability"),
    }),
  });
  if (result.ok) {
    setLog({ message: "Sentetik veri üretimi kuyruğa alındı." });
  } else {
    setLog(result);
  }
}

async function startTraining() {
  const result = await api("/api/train/start", {
    method: "POST",
    body: JSON.stringify({
      data_path: $("trainingDataset")?.value || state.dataset.selected || null,
      epochs: numberInput("trainEpochs"),
      batch_size: numberInput("trainBatch"),
      learning_rate: numberInput("trainLr"),
      weight_decay: numberInput("trainWeightDecay"),
      validation_split: numberInput("trainValidationSplit"),
      sequence_window: numberInput("trainSequenceWindow"),
      sequence_stride: numberInput("trainSequenceStride"),
      checkpoint_interval: numberInput("trainCheckpointInterval"),
      metric_eval_interval: numberInput("trainMetricEvalInterval"),
      precision: $("trainPrecision").value,
      resume_checkpoint: $("resumeCheckpoint").value || null,
      final_teacher_forcing_ratio: numberInput("trainFinalTeacherForcing"),
      teacher_decay_start_epoch: numberInput("trainTeacherDecayStart"),
      teacher_decay_epochs: numberInput("trainTeacherDecayEpochs"),
      warmup_loss_steps: numberInput("trainWarmupLossSteps"),
      early_stopping_patience: numberInput("trainEarlyPatience"),
      early_stopping_min_delta: numberInput("trainEarlyMinDelta"),
      early_stopping_min_epochs: numberInput("trainEarlyMinEpochs"),
      target_val_loss: numberInput("trainTargetValLoss"),
      target_train_loss: numberInput("trainTargetTrainLoss"),
      divergence_loss: numberInput("trainDivergenceLoss"),
      overfit_gap_ratio: numberInput("trainOverfitGap"),
      stop_on_target_val_loss: checkedInput("stopTargetVal"),
      stop_on_target_train_loss: checkedInput("stopTargetTrain"),
      stop_on_divergence_loss: checkedInput("stopDivergence"),
      stop_on_plateau: checkedInput("stopPlateau"),
      stop_on_overfit_gap: checkedInput("stopOverfitGap"),
      pose_loss_weight: numberInput("lossPose"),
      delta_loss_weight: numberInput("lossDelta"),
      velocity_loss_weight: numberInput("lossVelocity"),
      jerk_loss_weight: numberInput("lossJerk"),
      silence_loss_weight: numberInput("lossSilence"),
      range_loss_weight: numberInput("lossRange"),
    }),
  });
  setLog(result);
  connectTrainSocket();
}

async function stopTraining() {
  setLog(await api("/api/train/stop", { method: "POST", body: "{}" }));
}

async function exportOnnx() {
  setLog(await api("/api/export", { method: "POST", body: JSON.stringify({ opset_version: 18 }) }));
  await refreshAll();
}

async function benchmark() {
  setLog(await api("/api/benchmark", { method: "POST", body: JSON.stringify({ iterations: 250, warmup: 10 }) }));
}

/* ── Device & Model ── */
async function setDevice(mode) {
  const result = await api("/api/device", { method: "POST", body: JSON.stringify({ device: mode }) });
  state.device = result;
  renderDevice();
}

async function selectModel(path) {
  if (!path) return;
  const result = await api("/api/models/select", { method: "POST", body: JSON.stringify({ path }) });
  state.model = result;
  state.selectedMetricsPath = "";
  renderModelSelect();
  await refreshAll();
}

/* ── WebSocket ── */
function connectTrainSocket() {
  if (state.trainSocket && state.trainSocket.readyState === WebSocket.OPEN) return;
  const socket = new WebSocket(websocketUrl("/ws/train"));
  state.trainSocket = socket;
  socket.addEventListener("message", (event) => renderTraining(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    if (state.trainSocket === socket) state.trainSocket = null;
  });
}

function connectSyntheticSocket() {
  if (state.syntheticSocket && state.syntheticSocket.readyState === WebSocket.OPEN) return;
  const socket = new WebSocket(websocketUrl("/ws/synthetic"));
  state.syntheticSocket = socket;
  socket.addEventListener("message", (event) => renderSyntheticProgress(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    if (state.syntheticSocket === socket) state.syntheticSocket = null;
    setTimeout(connectSyntheticSocket, 3000);
  });
}

function renderSyntheticProgress(data) {
  const statusText = $("dataStatus");
  const progEl = $("syntheticProgress");
  const btnEl = $("generateBtn");

  const synthLabels = { starting: "BAŞLIYOR", running: "ÇALIŞIYOR", saving: "KAYDEDİLİYOR", completed: "TAMAMLANDI", error: "HATA" };
  if (data.running) {
    statusText.textContent = `${synthLabels[data.status] || String(data.status || "").toUpperCase()} (${data.phase})`;
    progEl.hidden = false;
    btnEl.disabled = true;
    
    if (data.total > 0) {
      progEl.max = data.total;
      progEl.value = data.progress;
    } else {
      progEl.removeAttribute("value");
    }
  } else {
    progEl.hidden = true;
    btnEl.disabled = false;
    if (data.status === "error") {
      statusText.textContent = `HATA: ${data.error}`;
      statusText.style.color = "var(--red)";
    } else if (data.status === "completed") {
      statusText.textContent = `TAMAMLANDI (${data.frames} kare)`;
      statusText.style.color = "var(--green)";
      
      // Tamamlandıysa sadece bir kere listeyi yenile
      if (document.lastSyntheticStatus !== "completed") {
        refreshAll();
      }
    }
  }
  document.lastSyntheticStatus = data.status;
}

function renderTraining(data) {
  const status = data.status || "idle";
  const trainLabels = { idle: "BOŞTA", running: "ÇALIŞIYOR", stopping: "DURDURULUYOR", stopped: "DURDURULDU", finished: "TAMAMLANDI", failed: "BAŞARISIZ", error: "HATA" };
  $("trainStatus").textContent = trainLabels[status] || status.toUpperCase();
  $("trainStatus").className = `status-pill ${status}`;
  const rows = (data.events || []).filter((event) => event.event === "epoch");
  const last = rows[rows.length - 1] || {};
  const total = data.config?.epochs || last.epoch || 0;
  $("trainProgress").max = Math.max(1, total);
  $("trainProgress").value = last.epoch || 0;
  $("trainMetrics").innerHTML = [
    metricCard("Epoch", `${last.epoch || 0} / ${total}`),
    metricCard("Eğitim", formatNumber(last.train_loss)),
    metricCard("Rollout", formatNumber(last.train_rollout_loss)),
    metricCard("Doğrulama", formatNumber(last.val_loss)),
    metricCard("En İyi Doğrulama", formatNumber(last.best_val_loss)),
    metricCard("Öğretmen Oranı", formatPercent(last.teacher_forcing_ratio)),
    metricCard("Hassasiyet", data.config?.precision || "—"),
  ].join("");
  updateLossChart(rows);
  setLog(data.error || { status, last });
  
  // Only refresh when active training changes state to inactive to avoid infinite re-render loop
  if (state.wasTraining && !data.running) {
    state.wasTraining = false;
    refreshAll().catch(() => {});
  } else if (data.running) {
    state.wasTraining = true;
  }
}

/* ── 3D Mesh ── */
async function setupMesh() {
  try {
    const THREE = await import("three");
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const root = $("viewport");
    const p = palette();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, root.clientWidth / root.clientHeight, 0.01, 100);
    camera.position.set(0, 0.1, state.zoom);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    renderer.setSize(root.clientWidth, root.clientHeight);
    root.appendChild(renderer.domElement);
    const hemi = new THREE.HemisphereLight(0xffffff, new THREE.Color(p.hemiGround), 1.3);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.8);
    key.position.set(2.5, 2.2, 3.0);
    scene.add(key);
    const rim = new THREE.DirectionalLight(new THREE.Color(p.meshRim), 2.8);
    rim.position.set(-2.5, 0.6, -2.0);
    scene.add(rim);
    const gltf = await new GLTFLoader().loadAsync(state.meshUrl);
    const meshRoot = gltf.scene;
    scene.add(meshRoot);
    normalizeMesh(meshRoot, THREE);
    const morphMeshes = [];
    meshRoot.traverse((object) => {
      if (object.isMesh && object.morphTargetDictionary && object.morphTargetInfluences) morphMeshes.push(object);
      if (object.isMesh) {
        object.material = new THREE.MeshStandardMaterial({ color: new THREE.Color(p.meshColor), roughness: 0.52, metalness: 0.16 });
        object.geometry.computeVertexNormals();
      }
    });
    state.rig = { THREE, scene, camera, renderer, meshRoot, morphMeshes, root, hemi, rim };
    $("linePreview").classList.add("hidden");
    bindViewportControls();
    renderMesh();
    new ResizeObserver(() => resizeMesh()).observe(root);
  } catch (error) {
    $("linePreview").classList.remove("hidden");
    console.warn("mesh preview unavailable", error);
  }
}

function normalizeMesh(meshRoot, THREE) {
  const box = new THREE.Box3().setFromObject(meshRoot);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const scale = 2.42 / (Math.max(size.x, size.y, size.z) || 1);
  meshRoot.scale.setScalar(scale);
  meshRoot.position.set(-center.x * scale, -center.y * scale - 0.08, -center.z * scale);
}

function bindViewportControls() {
  const root = state.rig.root;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  root.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    root.setPointerCapture(event.pointerId);
  });
  root.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    state.rotation.y += (event.clientX - lastX) * 0.006;
    state.rotation.x += (event.clientY - lastY) * 0.004;
    state.rotation.x = Math.max(-0.55, Math.min(0.42, state.rotation.x));
    lastX = event.clientX;
    lastY = event.clientY;
    renderMesh();
  });
  root.addEventListener("pointerup", () => { dragging = false; });
  root.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = Math.max(2.2, Math.min(4.4, state.zoom + event.deltaY * 0.002));
    state.rig.camera.position.z = state.zoom;
    renderMesh();
  }, { passive: false });
}

function resizeMesh() {
  if (!state.rig) return;
  const { root, camera, renderer } = state.rig;
  camera.aspect = root.clientWidth / Math.max(1, root.clientHeight);
  camera.updateProjectionMatrix();
  renderer.setSize(root.clientWidth, root.clientHeight);
  renderMesh();
}

function renderMesh() {
  if (!state.rig) return;
  state.rig.meshRoot.rotation.set(state.rotation.x, state.rotation.y, 0);
  state.rig.renderer.render(state.rig.scene, state.rig.camera);
}

function updatePreview() {
  const jaw = value("jawOpen");
  const pucker = value("mouthPucker");
  const stretch = (value("mouthStretchLeft") + value("mouthStretchRight")) * 0.5;
  const close = value("mouthClose");
  const mouth = $("linePreview").querySelector(".mouth");
  const jawEl = $("linePreview").querySelector(".jaw");
  mouth.style.width = `${130 + stretch * 90 - pucker * 48}px`;
  mouth.style.height = `${10 + jaw * 58 - close * 8}px`;
  jawEl.style.transform = `translateY(${48 + jaw * 58}px)`;
  applyMeshBlendshapes();
}

function applyMeshBlendshapes() {
  if (!state.rig || state.previewMode !== "mesh") return;
  state.rig.morphMeshes.forEach((mesh) => {
    Object.entries(mesh.morphTargetDictionary).forEach(([name, index]) => {
      const mapped = state.arkitValues[name] ?? value(name);
      if (Number.isFinite(Number(mapped))) mesh.morphTargetInfluences[index] = Number(mapped);
    });
  });
  renderMesh();
}

function setPreviewMode(mode) {
  state.previewMode = mode;
  const meshActive = mode === "mesh";
  $("meshModeBtn").classList.toggle("active", meshActive);
  $("meshModeBtn").setAttribute("aria-pressed", String(meshActive));
  $("rigModeBtn").classList.toggle("active", !meshActive);
  $("rigModeBtn").setAttribute("aria-pressed", String(!meshActive));
  $("linePreview").classList.toggle("hidden", mode === "mesh" && Boolean(state.rig));
  if (state.rig) state.rig.renderer.domElement.style.display = mode === "mesh" ? "block" : "none";
}

/* ── Utility functions ── */
function value(name) {
  const index = state.lipNames.indexOf(name);
  return Math.max(0, Math.min(1, Number(state.lipValues[index] || 0)));
}

function telemetryCell(label, value, extraClass = "") {
  return `<div class="telemetry-cell ${extraClass}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function metricCard(label, value) {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function detailRow(label, value) {
  return `<div class="detail-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function compactFileName(value, max = 34) {
  const raw = String(value || "").replace(/\\/g, "/");
  const file = raw.split("/").filter(Boolean).pop() || raw || "—";
  if (file.length <= max) return file;
  const head = Math.max(12, Math.floor(max * 0.58));
  const tail = Math.max(8, max - head - 3);
  return `${file.slice(0, head)}...${file.slice(-tail)}`;
}

function compactFileSize(value) {
  const mb = Number(value);
  if (!Number.isFinite(mb) || mb <= 0) return "";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 10) return `${Math.round(mb)} MB`;
  return `${mb.toFixed(1)} MB`;
}

function compactDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const minutes = seconds / 60;
  if (minutes >= 90) return `${(minutes / 60).toFixed(1)} sa`;
  return `${Math.round(minutes)} dk`;
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(6) : "—";
}

function numberOr(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function numberInput(id) {
  const input = $(id);
  const number = Number(input?.value);
  return Number.isFinite(number) ? number : null;
}

function checkedInput(id) {
  return Boolean($(id)?.checked);
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
}

function showError(error) {
  setLog(error.message || String(error));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function getAvailableTabIds() {
  return new Set(Array.from(document.querySelectorAll(".tab[data-tab]")).map((button) => button.dataset.tab));
}

function normalizeTab(tabId, fallback = "inference") {
  const candidates = getAvailableTabIds();
  return candidates.has(tabId) ? tabId : fallback;
}

function readStoredTab(fallback = "inference") {
  const candidates = getAvailableTabIds();
  try {
    const saved = localStorage.getItem("vocarig-active-tab");
    return candidates.has(saved) ? saved : fallback;
  } catch {
    return fallback;
  }
}

function writeStoredTab(tabId) {
  try {
    localStorage.setItem("vocarig-active-tab", tabId);
  } catch {
    // localStorage unavailable or denied; ignore gracefully.
  }
}

function syncTabHash(tabId) {
  const nextHash = `#${tabId}`;
  const currentHash = window.location.hash;
  if (currentHash === nextHash) return;
  history.replaceState(
    null,
    "",
    `${window.location.pathname}${window.location.search}#${tabId}`,
  );
}

/* ── Event bindings ── */
function switchTab(tabId) {
  const validTab = normalizeTab(tabId);
  const button = document.querySelector(`.tab[data-tab="${validTab}"]`);
  if (!button) return;
  document.querySelectorAll(".tab").forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === validTab));
  resizeMesh();
  if (state.lossChart) state.lossChart.resize();
  if (state.metricsChart) state.metricsChart.resize();
}

function closeDeviceMenu({ restoreFocus = false } = {}) {
  const menu = $("deviceMenu");
  const badge = $("deviceBadge");
  if (!menu || !badge) return;
  menu.hidden = true;
  badge.setAttribute("aria-expanded", "false");
  if (restoreFocus) badge.focus();
}

function bind() {
  // Tab switching
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      const tabId = button.dataset.tab;
      switchTab(tabId);
      writeStoredTab(tabId);
      syncTabHash(tabId);
    });
  });

  // Device picker
  $("deviceBadge").addEventListener("click", () => {
    const menu = $("deviceMenu");
    menu.hidden = !menu.hidden;
    $("deviceBadge").setAttribute("aria-expanded", String(!menu.hidden));
  });
  document.querySelectorAll("#deviceMenu [data-device]").forEach((button) => {
    button.addEventListener("click", () => {
      setDevice(button.dataset.device)
        .then(() => {
          closeDeviceMenu();
        })
        .catch(showError);
    });
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".device-picker")) closeDeviceMenu();
  });

  // Keyboard accessibility
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeDeviceMenu({ restoreFocus: true });
    }
  });

  // Theme toggle (Light "Paper" / Dark "Graphite")
  $("themeToggle")?.addEventListener("click", toggleTheme);

  // Model, controls, actions
  $("activeModel").addEventListener("change", (event) => selectModel(event.target.value).catch(showError));
  $("trainingDataset")?.addEventListener("change", async (event) => {
    state.dataset.selected = event.target.value;
    const data = await api(`/api/data?path=${encodeURIComponent(state.dataset.selected)}`);
    renderDataset(data);
  });
  $("audioFile").addEventListener("change", () => {
    stopLivePlayback();
    stopRealtimeInput();
    stopAudioPlayback();
    const file = $("audioFile").files[0];
    setAudioFileName(selectedAudioName());
    if (!file) {
      loadDefaultAudioPreview().catch(showError);
      return;
    }
    file.arrayBuffer()
      .then(drawWaveBytes)
      .then(refreshAudioAction)
      .catch(showError);
  });
  $("resetBtn").addEventListener("click", () => {
    stopLivePlayback();
    resetAudioPlayback();
    state.frame = 0;
    resetPoseState();
    if (!state.audioPlayback.frames.length && state.realtime.mode === "idle") {
      inferStep(true).catch(showError);
    }
  });
  $("stopBtn").addEventListener("click", stopCurrentPlayback);
  document.querySelectorAll("[data-inference-mode]").forEach((button) => {
    button.addEventListener("click", () => handleInferenceModeClick(button.dataset.inferenceMode).catch(showError));
  });
  $("audioPlayBtn")?.addEventListener("click", () => toggleAudioPlayback().catch(showError));
  const audio = $("audioPreview");
  audio.addEventListener("play", () => {
    if (guardUnsyncedNativePlayback()) return;
    revealViewportForPlayback();
    refreshAudioAction();
  });
  ["pause", "ended", "loadedmetadata", "emptied"].forEach((eventName) => {
    audio.addEventListener(eventName, refreshAudioAction);
  });
  $("generateBtn").addEventListener("click", () => generateSynthetic().catch(showError));
  $("refreshBtn").addEventListener("click", () => refreshAll().catch(showError));
  $("trainBtn").addEventListener("click", () => startTraining().catch(showError));
  $("stopTrainBtn").addEventListener("click", () => stopTraining().catch(showError));
  $("exportBtn").addEventListener("click", () => exportOnnx().catch(showError));
  $("benchmarkBtn").addEventListener("click", () => benchmark().catch(showError));
  $("meshModeBtn").addEventListener("click", () => setPreviewMode("mesh"));
  $("rigModeBtn").addEventListener("click", () => setPreviewMode("rig"));
  $("metricsSelect").addEventListener("change", async (event) => {
    state.selectedMetricsPath = event.target.value;
    try {
      const metricsParams = `?path=${encodeURIComponent(state.selectedMetricsPath)}`;
      const [metrics, diagnosis] = await Promise.all([
        api(`/api/metrics${metricsParams}`),
        api(`/api/diagnosis${metricsParams}`),
      ]);
      renderMetrics(metrics);
      renderDiagnosis(diagnosis);
    } catch (err) {
      showError(err);
    }
  });
}

/* ── Init ── */
function init() {
  bind();
  const savedTheme = (() => { try { return localStorage.getItem("vocarig-theme"); } catch { return null; } })();
  setTheme(savedTheme === "dark" ? "dark" : "light");
  setupSliders();
  setupScaleToggles();
  setInferenceMode(state.inferenceMode);
  clearWaveform();
  setupMesh();
  connectTrainSocket();
  connectSyntheticSocket();
  refreshAll().catch(showError);
  loadDefaultAudioPreview().catch(showError);
  
  // Handle initial hash link
  const hash = window.location.hash.replace("#", "");
  const initialTab = hash ? normalizeTab(hash, null) : readStoredTab("inference");
  if (initialTab) switchTab(initialTab);
  if (initialTab) syncTabHash(initialTab);
  if (initialTab && hash !== initialTab) writeStoredTab(initialTab);
  
  // Listen for hash changes
  window.addEventListener("hashchange", () => {
    const newHash = window.location.hash.replace("#", "");
    const nextTab = normalizeTab(newHash, readStoredTab("inference"));
    if (nextTab) {
      switchTab(nextTab);
      writeStoredTab(nextTab);
      syncTabHash(nextTab);
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
