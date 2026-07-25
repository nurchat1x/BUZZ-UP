import type { DrowsinessResult } from "./ear";
import type { NearestStop } from "./route";

const STORAGE_KEY = "buzzup_fatigue_log_v1";
const SESSION_KEY = "buzzup_session_id";
const DEFAULT_COOLDOWN_SEC = 10;

export const EVENT_LABELS_RU: Record<string, string> = {
  drowsy_start: "Начало сонливости",
  drowsy_continue: "Сонливость продолжается",
  drowsy_end: "Водитель проснулся",
};

export const MODE_LABELS_RU: Record<string, string> = {
  opencv_live: "камера OpenCV",
  local: "WebRTC",
  cloud: "облако",
  fleet_demo: "демо",
  demo: "демо (скринкаст)",
  web_live: "браузер live",
};

export function formatModeLabel(mode?: string): string {
  if (!mode) return "—";
  return MODE_LABELS_RU[mode] ?? mode;
}

export function formatEventLabel(eventType: string): string {
  return EVENT_LABELS_RU[eventType] ?? eventType;
}

export interface FatigueEvent {
  logged_at: string;
  session_id: string;
  event: string;
  mode: string;
  status: string;
  ear: number;
  left_ear: number;
  right_ear: number;
  confidence: number;
  closed_frames: number;
  nearest_stop_name?: string;
  nearest_stop_km?: number;
  driver_id?: string;
  driver_name?: string;
  vehicle?: string;
  route?: string;
}

let logEnabled = true;
let cooldownSec = DEFAULT_COOLDOWN_SEC;
let episodeActive = false;
let lastLogTs = 0;

export function setLogEnabled(v: boolean): void {
  logEnabled = v;
}

export function setLogCooldown(sec: number): void {
  cooldownSec = sec;
}

export function ensureSessionId(): string {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    const stamp = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const day =
      `${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}` +
      `-${pad(stamp.getHours())}${pad(stamp.getMinutes())}${pad(stamp.getSeconds())}`;
    id = `${day}-${Math.random().toString(16).slice(2, 8)}`;
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function formatSessionLabel(sessionId: string): string {
  const parts = sessionId.split("-");
  if (parts.length >= 2 && parts[0].length === 8 && /^\d+$/.test(parts[0]) && /^\d+$/.test(parts[1])) {
    const day = parts[0].slice(6, 8);
    const month = parts[0].slice(4, 6);
    const hour = parts[1].slice(0, 2);
    const minute = parts[1].slice(2, 4);
    return `Смена ${day}.${month} · ${hour}:${minute}`;
  }
  return `Смена ${sessionId.slice(0, 10)}`;
}

export function readEvents(limit = 500): FatigueEvent[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const all = JSON.parse(raw) as FatigueEvent[];
    return all.slice(-limit);
  } catch {
    return [];
  }
}

function writeEvents(events: FatigueEvent[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(events.slice(-2000)));
}

function appendEvent(event: FatigueEvent): void {
  const events = readEvents(2000);
  events.push(event);
  writeEvents(events);
}

function buildEvent(
  result: DrowsinessResult,
  mode: string,
  eventType: string,
  nearestStop: NearestStop | null,
): FatigueEvent {
  const event: FatigueEvent = {
    logged_at: new Date().toISOString(),
    session_id: ensureSessionId(),
    event: eventType,
    mode,
    status: result.status,
    ear: result.ear,
    left_ear: result.leftEar,
    right_ear: result.rightEar,
    confidence: result.confidence,
    closed_frames: result.closedFrames,
    driver_id: `live-${ensureSessionId()}`,
    driver_name: "Вы (live)",
    vehicle: "Текущая сессия",
    route: "—",
  };
  if (nearestStop) {
    event.nearest_stop_name = nearestStop.name;
    event.nearest_stop_km = nearestStop.distance_km;
  }
  return event;
}

export function handleFatigueLog(
  result: DrowsinessResult,
  mode: string,
  nearestStop: NearestStop | null = null,
  forceCooldown = cooldownSec,
): void {
  if (!logEnabled) return;

  if (result.status !== "Спит") {
    if (episodeActive) {
      appendEvent(buildEvent(result, mode, "drowsy_end", nearestStop));
      episodeActive = false;
    }
    return;
  }

  const now = performance.now() / 1000;
  if (!episodeActive) {
    appendEvent(buildEvent(result, mode, "drowsy_start", nearestStop));
    episodeActive = true;
    lastLogTs = now;
    return;
  }

  if (now - lastLogTs < forceCooldown) return;
  lastLogTs = now;
  appendEvent(buildEvent(result, mode, "drowsy_continue", nearestStop));
}

export function clearLogs(): void {
  localStorage.removeItem(STORAGE_KEY);
  episodeActive = false;
  lastLogTs = 0;
}

export function eventsToCsv(events: FatigueEvent[]): string {
  const fields = [
    "logged_at",
    "session_id",
    "event",
    "mode",
    "status",
    "ear",
    "left_ear",
    "right_ear",
    "confidence",
    "closed_frames",
    "nearest_stop_name",
    "nearest_stop_km",
  ];
  const escape = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [fields.join(",")];
  for (const e of events) {
    lines.push(fields.map((f) => escape((e as unknown as Record<string, unknown>)[f])).join(","));
  }
  return lines.join("\n");
}

export function formatEventTime(loggedAt?: string): string {
  if (!loggedAt) return "—";
  try {
    const dt = new Date(loggedAt);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${pad(dt.getDate())}.${pad(dt.getMonth() + 1)} ${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
  } catch {
    return loggedAt.slice(0, 16);
  }
}

export function formatEventDisplay(event: FatigueEvent): string {
  const when = formatEventTime(event.logged_at);
  const what = EVENT_LABELS_RU[event.event] ?? event.event;
  const ear = typeof event.ear === "number" ? `EAR ${event.ear.toFixed(3)}` : "EAR —";
  const stop =
    event.nearest_stop_name && event.nearest_stop_km != null
      ? ` · до «${event.nearest_stop_name}» ${event.nearest_stop_km} км`
      : event.nearest_stop_name
        ? ` · ${event.nearest_stop_name}`
        : "";
  return `${when} · ${what} · ${ear}${stop}`;
}
