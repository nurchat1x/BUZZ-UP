import "./style.css";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { handleDrowsinessAlert, setSoundEnabled } from "./alert";
import { createFaceLandmarker } from "./detector";
import { makeAwakeResult, makeDrowsyResult } from "./demo";
import {
  DEFAULT_CONSECUTIVE_FRAMES,
  DEFAULT_EAR_THRESHOLD,
  EarTracker,
  type DrowsinessResult,
} from "./ear";
import {
  episodeChartRows,
  eventsForDriver,
  loadFleetEvents,
  summarizeDrivers,
  type ChartPoint,
  type DriverSummary,
} from "./fleet";
import {
  clearLogs,
  ensureSessionId,
  eventsToCsv,
  formatEventDisplay,
  formatEventLabel,
  formatEventTime,
  formatModeLabel,
  formatSessionLabel,
  handleFatigueLog,
  readEvents,
  setLogCooldown,
  setLogEnabled,
  type FatigueEvent,
} from "./logger";
import {
  findNearestStop,
  formatRating,
  loadBusStops,
  type BusStopsData,
  type NearestStop,
} from "./route";

const video = document.querySelector<HTMLVideoElement>("#video")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const ctx = canvas.getContext("2d")!;
const placeholder = document.querySelector<HTMLElement>("#placeholder")!;
const btnStart = document.querySelector<HTMLButtonElement>("#btn-start")!;
const btnStop = document.querySelector<HTMLButtonElement>("#btn-stop")!;
const statusEl = document.querySelector<HTMLElement>("#status")!;
const earEl = document.querySelector<HTMLElement>("#ear")!;
const closedEl = document.querySelector<HTMLElement>("#closed")!;
const confidenceEl = document.querySelector<HTMLElement>("#confidence")!;
const errorEl = document.querySelector<HTMLElement>("#error")!;
const thresholdInput = document.querySelector<HTMLInputElement>("#threshold")!;
const framesInput = document.querySelector<HTMLInputElement>("#frames")!;
const thresholdVal = document.querySelector<HTMLElement>("#threshold-val")!;
const framesVal = document.querySelector<HTMLElement>("#frames-val")!;
const soundInput = document.querySelector<HTMLInputElement>("#sound")!;

const btnDemoDrowsy = document.querySelector<HTMLButtonElement>("#btn-demo-drowsy")!;
const btnDemoAwake = document.querySelector<HTMLButtonElement>("#btn-demo-awake")!;
const btnDemoClear = document.querySelector<HTMLButtonElement>("#btn-demo-clear")!;
const demoHint = document.querySelector<HTMLElement>("#demo-hint")!;

const routeSelect = document.querySelector<HTMLSelectElement>("#route-select")!;
const latInput = document.querySelector<HTMLInputElement>("#lat")!;
const lngInput = document.querySelector<HTMLInputElement>("#lng")!;
const btnGeo = document.querySelector<HTMLButtonElement>("#btn-geo")!;
const btnZoomMe = document.querySelector<HTMLButtonElement>("#btn-zoom-me")!;
const btnFindStop = document.querySelector<HTMLButtonElement>("#btn-find-stop")!;
const stopDetails = document.querySelector<HTMLElement>("#stop-details")!;
const stopAdvice = document.querySelector<HTMLElement>("#stop-advice")!;
const mapEl = document.querySelector<HTMLElement>("#map")!;

const logEnabledInput = document.querySelector<HTMLInputElement>("#log-enabled")!;
const logCooldownInput = document.querySelector<HTMLInputElement>("#log-cooldown")!;
const logCooldownVal = document.querySelector<HTMLElement>("#log-cooldown-val")!;
const sessionLabel = document.querySelector<HTMLElement>("#session-label")!;
const logList = document.querySelector<HTMLElement>("#log-list")!;
const btnExportCsv = document.querySelector<HTMLButtonElement>("#btn-export-csv")!;
const btnClearLogs = document.querySelector<HTMLButtonElement>("#btn-clear-logs")!;

const fleetDemoInput = document.querySelector<HTMLInputElement>("#fleet-demo")!;
const fleetMetrics = document.querySelector<HTMLElement>("#fleet-metrics")!;
const fleetTableBody = document.querySelector<HTMLTableSectionElement>("#fleet-table tbody")!;
const fleetEmpty = document.querySelector<HTMLElement>("#fleet-empty")!;
const fleetDriverSelect = document.querySelector<HTMLSelectElement>("#fleet-driver-select")!;
const fleetEventsBody = document.querySelector<HTMLTableSectionElement>("#fleet-events-table tbody")!;
const earChart = document.querySelector<HTMLCanvasElement>("#ear-chart")!;
const earChartEmpty = document.querySelector<HTMLElement>("#ear-chart-empty")!;
const earChartDetail = document.querySelector<HTMLElement>("#ear-chart-detail")!;
const liveStatus = document.querySelector<HTMLElement>("#live-status")!;
const liveEar = document.querySelector<HTMLElement>("#live-ear")!;
const liveClosed = document.querySelector<HTMLElement>("#live-closed")!;

const tracker = new EarTracker(DEFAULT_EAR_THRESHOLD, DEFAULT_CONSECUTIVE_FRAMES);

let landmarker: FaceLandmarker | null = null;
let stream: MediaStream | null = null;
let rafId = 0;
let lastVideoTime = -1;
let running = false;
let demoActive = false;
let busData: BusStopsData | null = null;
let nearestStop: NearestStop | null = null;
let map: LeafletMap | null = null;
let markersLayer: LeafletLayerGroup | null = null;
let lastLiveResult: DrowsinessResult | null = null;
let fleetEventsCache: FatigueEvent[] = [];
let selectedDriverId: string | null = null;
let earChartInstance: ChartInstance | null = null;
let chartPointsCache: ChartPoint[] = [];
let userMarker: LeafletMarker | null = null;

function showError(message: string | null): void {
  if (!message) {
    errorEl.textContent = "";
    errorEl.classList.remove("visible");
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.add("visible");
}

function applyResult(result: DrowsinessResult, mode: string): void {
  lastLiveResult = result;
  renderResult(result);
  renderLiveDriver(result);
  updateStopAdvice();
  void handleDrowsinessAlert(result.status, mode === "demo" ? 0 : 5);
  handleFatigueLog(result, mode, nearestStop, mode === "demo" ? 0 : undefined);
  refreshLogList();
  if (document.querySelector(".tab.active")?.getAttribute("data-tab") === "fleet") {
    renderFleet(false);
  }
}

function renderResult(result: DrowsinessResult): void {
  statusEl.textContent = result.status;
  statusEl.classList.remove("status-awake", "status-asleep", "status-missing");
  if (result.status === "Спит") statusEl.classList.add("status-asleep");
  else if (result.status === "Не Спит") statusEl.classList.add("status-awake");
  else statusEl.classList.add("status-missing");

  earEl.textContent = result.faceDetected ? result.ear.toFixed(3) : "—";
  closedEl.textContent = String(result.closedFrames);
  confidenceEl.textContent = result.faceDetected
    ? `${Math.round(result.confidence * 100)}%`
    : "—";
}

function syncCanvasSize(): void {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

async function ensureLandmarker(): Promise<FaceLandmarker> {
  if (landmarker) return landmarker;
  btnStart.disabled = true;
  btnStart.textContent = "Загрузка модели…";
  try {
    landmarker = await createFaceLandmarker();
    return landmarker;
  } finally {
    btnStart.textContent = "▶ Включить камеру";
  }
}

function loop(): void {
  if (!running || !landmarker || demoActive) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    syncCanvasSize();
    const ts = performance.now();
    const detection = landmarker.detectForVideo(video, ts);
    const face = detection.faceLandmarks[0] ?? null;
    const w = canvas.width || video.videoWidth || 640;
    const h = canvas.height || video.videoHeight || 480;
    const result = tracker.update(face, w, h);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    applyResult(result, "web_live");
  }

  rafId = requestAnimationFrame(loop);
}

async function startCamera(): Promise<void> {
  showError(null);
  try {
    await ensureLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });
    video.srcObject = stream;
    await video.play();
    placeholder.classList.add("hidden");
    tracker.reset();
    clearDemoSimulation(false);
    running = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    lastVideoTime = -1;
    rafId = requestAnimationFrame(loop);
  } catch (err) {
    const msg =
      err instanceof Error
        ? err.message
        : "Не удалось открыть камеру. Нужен HTTPS (или localhost) и разрешение в браузере.";
    showError(msg);
    await stopCamera();
  }
}

async function stopCamera(): Promise<void> {
  running = false;
  cancelAnimationFrame(rafId);
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.classList.remove("hidden");
  tracker.reset();
  if (!demoActive) {
    renderResult({
      status: "Лицо не найдено",
      ear: 0,
      leftEar: 0,
      rightEar: 0,
      faceDetected: false,
      closedFrames: 0,
      confidence: 0,
    });
    statusEl.textContent = "—";
  }
  btnStart.disabled = false;
  btnStop.disabled = true;
}

function setDemoActive(active: boolean): void {
  demoActive = active;
  demoHint.classList.toggle("hidden", !active);
  btnDemoClear.disabled = !active;
}

function clearDemoSimulation(resetStatus = true): void {
  setDemoActive(false);
  if (resetStatus && !running) {
    statusEl.textContent = "—";
    statusEl.classList.remove("status-awake", "status-asleep");
    statusEl.classList.add("status-missing");
    earEl.textContent = "—";
    closedEl.textContent = "0";
    confidenceEl.textContent = "—";
  }
  if (running) {
    lastVideoTime = -1;
    rafId = requestAnimationFrame(loop);
  }
}

function activateDemo(result: DrowsinessResult): void {
  setDemoActive(true);
  cancelAnimationFrame(rafId);
  applyResult(result, "demo");
}

function refreshLogList(): void {
  const events = readEvents(80).slice().reverse();
  logList.innerHTML = events.length
    ? events.map((e) => `<li>${formatEventDisplay(e)}</li>`).join("")
    : "<li>Пока нет событий. Включите камеру или Demo mode.</li>";
  sessionLabel.textContent = `Текущая смена: ${formatSessionLabel(ensureSessionId())}`;
}

function ensureMap(): LeafletMap {
  if (map) return map;
  map = L.map(mapEl);
  map.setView([48.0, 67.0], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
    maxZoom: 18,
  }).addTo(map);
  markersLayer = L.layerGroup();
  markersLayer.addTo(map);
  setTimeout(() => map?.invalidateSize(), 80);
  return map;
}

function renderMap(
  routeId: string,
  userLat: number,
  userLon: number,
  stop: NearestStop | null,
  focusUser = false,
): void {
  if (!busData) return;
  const m = ensureMap();
  markersLayer?.clearLayers();
  userMarker = null;
  const route = busData.routes[routeId];
  const points: Array<[number, number]> = [[userLat, userLon]];

  userMarker = L.circleMarker([userLat, userLon], {
    radius: 8,
    color: "#3ecfb0",
    fillColor: "#3ecfb0",
    fillOpacity: 0.9,
  })
    .bindPopup("Вы здесь")
    .addTo(markersLayer!);

  for (const s of route?.stops ?? []) {
    const isNearest = stop && s.id === stop.id;
    L.circleMarker([s.lat, s.lng], {
      radius: isNearest ? 9 : 6,
      color: isNearest ? "#e8a54b" : "#1c9279",
      fillColor: isNearest ? "#e8a54b" : "#1c9279",
      fillOpacity: 0.85,
    })
      .bindPopup(`<b>${s.name}</b><br>${s.address ?? ""}`)
      .addTo(markersLayer!);
    points.push([s.lat, s.lng]);
  }

  if (focusUser) {
    m.flyTo([userLat, userLon], 15);
    setTimeout(() => userMarker?.openPopup(), 450);
  } else if (points.length > 1) {
    m.fitBounds(L.latLngBounds(points), { padding: [28, 28] });
  } else {
    m.setView([userLat, userLon], 10);
  }
}

function updateStopAdvice(): void {
  stopAdvice.classList.remove("warn", "danger", "ok");
  if (!nearestStop) {
    stopAdvice.textContent = "🔍 Найдите остановку справа";
    return;
  }
  const km = nearestStop.distance_km;
  const drowsy = lastLiveResult?.status === "Спит";
  if (drowsy && km <= 20) {
    stopAdvice.textContent = "⚠️ Рекомендуется немедленный отдых! Близкая остановка найдена.";
    stopAdvice.classList.add("warn");
  } else if (drowsy) {
    stopAdvice.textContent = "🚨 Критично! Нужен отдых, но ближайшая остановка далеко.";
    stopAdvice.classList.add("danger");
  } else if (km <= 30) {
    stopAdvice.textContent = "💡 Близкая остановка — можно планировать отдых";
    stopAdvice.classList.add("ok");
  } else {
    stopAdvice.textContent = "ℹ️ Дальняя остановка — продолжайте движение";
  }
}

function renderStopDetails(stop: NearestStop | null): void {
  if (!stop) {
    stopDetails.innerHTML = "<p class='hint'>Остановка не найдена.</p>";
    updateStopAdvice();
    return;
  }
  const services = (stop.services ?? []).slice(0, 4).join(", ") || "—";
  stopDetails.innerHTML = `
    <p><strong>📍 ${stop.name}</strong></p>
    <p>${stop.distance_km} км · ${formatRating(stop)}</p>
    ${stop.address ? `<p>${stop.address}</p>` : ""}
    <p>${services}</p>
  `;
  updateStopAdvice();
}

async function initRoutes(): Promise<void> {
  busData = await loadBusStops();
  routeSelect.innerHTML = Object.entries(busData.routes)
    .map(([id, r]) => `<option value="${id}">${r.name}</option>`)
    .join("");
  ensureMap();
}

function findAndShowStop(focusUser = false): void {
  if (!busData) return;
  const lat = Number(latInput.value);
  const lng = Number(lngInput.value);
  const routeId = routeSelect.value;
  nearestStop = findNearestStop(lat, lng, routeId, busData);
  renderStopDetails(nearestStop);
  renderMap(routeId, lat, lng, nearestStop, focusUser);
}

function zoomToMyLocation(): void {
  if (!navigator.geolocation) {
    showError("Геолокация недоступна в этом браузере.");
    return;
  }
  showError(null);
  btnZoomMe.disabled = true;
  btnZoomMe.textContent = "…";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      latInput.value = String(Math.round(pos.coords.latitude * 10000) / 10000);
      lngInput.value = String(Math.round(pos.coords.longitude * 10000) / 10000);
      findAndShowStop(true);
      btnZoomMe.disabled = false;
      btnZoomMe.textContent = "🎯 На меня";
    },
    () => {
      showError("Не удалось получить геолокацию. Разрешите доступ в браузере.");
      btnZoomMe.disabled = false;
      btnZoomMe.textContent = "🎯 На меня";
    },
    { enableHighAccuracy: true, timeout: 12000 },
  );
}

function renderLiveDriver(result: DrowsinessResult | null): void {
  if (!result) {
    liveStatus.textContent = "—";
    liveStatus.className = "value status-missing";
    liveEar.textContent = "—";
    liveClosed.textContent = "0";
    return;
  }
  liveStatus.textContent =
    result.status === "Спит"
      ? `😴 ${result.status}`
      : result.status === "Не Спит"
        ? `👁️ ${result.status}`
        : `❓ ${result.status}`;
  liveStatus.classList.remove("status-awake", "status-asleep", "status-missing");
  if (result.status === "Спит") liveStatus.classList.add("status-asleep");
  else if (result.status === "Не Спит") liveStatus.classList.add("status-awake");
  else liveStatus.classList.add("status-missing");
  liveEar.textContent = result.faceDetected ? result.ear.toFixed(3) : "—";
  liveClosed.textContent = String(result.closedFrames);
}

function showChartDetail(point: ChartPoint | null): void {
  if (!point) {
    earChartDetail.textContent = "Выберите точку на графике";
    return;
  }
  earChartDetail.textContent = `${point.time} · EAR ${point.ear.toFixed(3)} · ${formatEventLabel(point.event)}`;
}

function drawEarChart(points: ChartPoint[]): void {
  chartPointsCache = points;
  if (earChartInstance) {
    earChartInstance.destroy();
    earChartInstance = null;
  }

  if (points.length === 0) {
    earChart.classList.add("hidden");
    earChartEmpty.classList.remove("hidden");
    showChartDetail(null);
    return;
  }

  earChart.classList.remove("hidden");
  earChartEmpty.classList.add("hidden");
  showChartDetail(null);

  earChartInstance = new Chart(earChart, {
    type: "line",
    data: {
      labels: points.map((p) => p.time),
      datasets: [
        {
          label: "EAR",
          data: points.map((p) => p.ear),
          borderColor: "#3ecfb0",
          backgroundColor: "rgba(62, 207, 176, 0.12)",
          pointBackgroundColor: "#e8a54b",
          pointBorderColor: "#fff",
          pointRadius: 5,
          pointHoverRadius: 9,
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: true },
      onClick: (_evt: unknown, elements: Array<{ index: number }>) => {
        if (!elements.length) return;
        showChartDetail(chartPointsCache[elements[0].index] ?? null);
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10, 32, 28, 0.95)",
          titleColor: "#fff",
          bodyColor: "#dcf0eb",
          borderColor: "rgba(180, 220, 210, 0.3)",
          borderWidth: 1,
          callbacks: {
            label: (ctx: { raw: unknown; dataIndex: number }) => {
              const ear = Number(ctx.raw);
              const point = chartPointsCache[ctx.dataIndex];
              const event = point ? formatEventLabel(point.event) : "";
              return [`EAR: ${ear.toFixed(3)}`, event].filter(Boolean);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "rgba(220, 240, 235, 0.65)",
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
          },
          grid: { color: "rgba(180, 220, 210, 0.08)" },
        },
        y: {
          min: 0,
          suggestedMax: 0.35,
          ticks: { color: "rgba(220, 240, 235, 0.65)" },
          grid: { color: "rgba(180, 220, 210, 0.12)" },
          title: {
            display: true,
            text: "EAR",
            color: "rgba(220, 240, 235, 0.7)",
          },
        },
      },
    },
  });
}

function renderDriverDetails(summaries: DriverSummary[]): void {
  if (summaries.length === 0) {
    fleetDriverSelect.innerHTML = "";
    fleetEventsBody.innerHTML = "";
    drawEarChart([]);
    return;
  }

  const prev = selectedDriverId;
  fleetDriverSelect.innerHTML = summaries
    .map((s) => `<option value="${s.driver_id}">${s.name}</option>`)
    .join("");

  const preferred =
    summaries.find((s) => s.driver_id === prev) ||
    summaries.find((s) => s.name === "Вы (live)") ||
    summaries[0];
  selectedDriverId = preferred.driver_id;
  fleetDriverSelect.value = selectedDriverId;

  const driverEvents = eventsForDriver(fleetEventsCache, selectedDriverId);
  const chartRows = episodeChartRows(driverEvents);
  drawEarChart(chartRows);

  const rows = driverEvents.slice(-40).reverse();
  fleetEventsBody.innerHTML = rows
    .map(
      (e) => `
      <tr>
        <td>${formatEventTime(e.logged_at)}</td>
        <td>${formatEventLabel(e.event)}</td>
        <td>${typeof e.ear === "number" ? e.ear.toFixed(3) : "—"}</td>
        <td>${e.status ?? "—"}</td>
        <td>${e.nearest_stop_name || "—"}</td>
        <td>${e.nearest_stop_km ?? "—"}</td>
        <td>${formatModeLabel(e.mode)}</td>
      </tr>`,
    )
    .join("");
}

function renderFleet(resetDriver = true): void {
  const includeDemo = fleetDemoInput.checked;
  fleetEventsCache = loadFleetEvents(includeDemo);
  const summaries = summarizeDrivers(fleetEventsCache);
  const totalEpisodes = summaries.reduce((s, x) => s + x.episodes, 0);
  const active = summaries.filter((s) => s.status.includes("Сонливость")).length;
  const live = summaries.filter((s) => s.name === "Вы (live)").length;

  fleetMetrics.innerHTML = `
    <div class="metric"><label>Водителей</label><div class="value">${summaries.length}</div></div>
    <div class="metric"><label>Эпизодов сонливости</label><div class="value">${totalEpisodes}</div></div>
    <div class="metric"><label>Сейчас «Спит»</label><div class="value">${active}</div></div>
    <div class="metric"><label>Live-сессий</label><div class="value">${live}</div></div>
  `;

  fleetEmpty.classList.toggle("hidden", summaries.length > 0);
  fleetTableBody.innerHTML = summaries
    .map(
      (s) => `
      <tr>
        <td>${s.name}</td>
        <td>${s.vehicle}</td>
        <td>${s.route}</td>
        <td>${s.status}</td>
        <td>${s.episodes}</td>
        <td>${s.min_ear != null ? s.min_ear.toFixed(3) : "—"}</td>
        <td>${formatEventTime(s.last_event)}</td>
      </tr>`,
    )
    .join("");

  if (resetDriver) selectedDriverId = null;
  renderDriverDetails(summaries);
  renderLiveDriver(lastLiveResult);
}

function setupTabs(): void {
  const tabs = document.querySelectorAll<HTMLButtonElement>(".tab");
  const monitor = document.querySelector<HTMLElement>("#panel-monitor")!;
  const fleet = document.querySelector<HTMLElement>("#panel-fleet")!;
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const id = tab.dataset.tab;
      monitor.classList.toggle("hidden", id !== "monitor");
      fleet.classList.toggle("hidden", id !== "fleet");
      if (id === "fleet") {
        renderFleet(false);
        setTimeout(() => earChartInstance?.update(), 120);
      }
      if (id === "monitor") setTimeout(() => map?.invalidateSize(), 80);
    });
  });
}

thresholdInput.addEventListener("input", () => {
  const v = Number(thresholdInput.value);
  tracker.earThreshold = v;
  thresholdVal.textContent = v.toFixed(2);
});

framesInput.addEventListener("input", () => {
  const v = Number(framesInput.value);
  tracker.consecutiveFrames = v;
  framesVal.textContent = String(v);
});

soundInput.addEventListener("change", () => setSoundEnabled(soundInput.checked));

btnStart.addEventListener("click", () => void startCamera());
btnStop.addEventListener("click", () => void stopCamera());

btnDemoDrowsy.addEventListener("click", () => activateDemo(makeDrowsyResult()));
btnDemoAwake.addEventListener("click", () => activateDemo(makeAwakeResult()));
btnDemoClear.addEventListener("click", () => clearDemoSimulation(true));

logEnabledInput.addEventListener("change", () => setLogEnabled(logEnabledInput.checked));
logCooldownInput.addEventListener("input", () => {
  const v = Number(logCooldownInput.value);
  setLogCooldown(v);
  logCooldownVal.textContent = String(v);
});

btnExportCsv.addEventListener("click", () => {
  const csv = eventsToCsv(readEvents(2000));
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `buzzup_fatigue_${ensureSessionId()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

btnClearLogs.addEventListener("click", () => {
  clearLogs();
  refreshLogList();
  renderFleet(true);
});

fleetDriverSelect.addEventListener("change", () => {
  selectedDriverId = fleetDriverSelect.value;
  const summaries = summarizeDrivers(fleetEventsCache);
  renderDriverDetails(summaries);
});

btnGeo.addEventListener("click", () => {
  if (!navigator.geolocation) {
    showError("Геолокация недоступна в этом браузере.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      latInput.value = String(Math.round(pos.coords.latitude * 10000) / 10000);
      lngInput.value = String(Math.round(pos.coords.longitude * 10000) / 10000);
      findAndShowStop(false);
    },
    () => showError("Не удалось получить геолокацию. Разрешите доступ или введите координаты вручную."),
  );
});

btnZoomMe.addEventListener("click", zoomToMyLocation);
btnFindStop.addEventListener("click", () => findAndShowStop(false));
fleetDemoInput.addEventListener("change", () => renderFleet(true));

setupTabs();
refreshLogList();
void initRoutes()
  .then(() => findAndShowStop())
  .catch((err) => {
    showError(err instanceof Error ? err.message : "Ошибка загрузки остановок");
  });

renderResult({
  status: "Лицо не найдено",
  ear: 0,
  leftEar: 0,
  rightEar: 0,
  faceDetected: false,
  closedFrames: 0,
  confidence: 0,
});
statusEl.textContent = "—";
