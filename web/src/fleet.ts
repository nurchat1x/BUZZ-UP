import type { FatigueEvent } from "./logger";
import { readEvents } from "./logger";

export const DEMO_DRIVERS = [
  {
    driver_id: "demo-asylbek",
    name: "Асылбек Н.",
    vehicle: "KZ 123 AB",
    route: "Алматы — Астана",
  },
  {
    driver_id: "demo-maria",
    name: "Мария К.",
    vehicle: "KZ 456 CD",
    route: "Алматы — Шымкент",
  },
  {
    driver_id: "demo-dauran",
    name: "Даурен Т.",
    vehicle: "KZ 789 EF",
    route: "Астана — Караганда",
  },
] as const;

export const STATUS_OK = "🟢 Бодрствует";
export const STATUS_DROWSY = "🔴 Сонливость";
export const STATUS_UNKNOWN = "⚪ Нет данных";

export interface DriverSummary {
  driver_id: string;
  name: string;
  vehicle: string;
  route: string;
  status: string;
  episodes: number;
  events_count: number;
  last_event?: string;
  min_ear: number | null;
  mode: string;
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

export function generateDemoEvents(now = new Date()): FatigueEvent[] {
  const rng = seededRandom(42);
  const events: FatigueEvent[] = [];
  const profiles: Array<[(typeof DEMO_DRIVERS)[number], number, number]> = [
    [DEMO_DRIVERS[0], 3, 0],
    [DEMO_DRIVERS[1], 1, 4],
    [DEMO_DRIVERS[2], 2, 2],
  ];

  for (const [driver, episodeCount, hourOffset] of profiles) {
    for (let ep = 0; ep < episodeCount; ep++) {
      const startAt = new Date(
        now.getTime() - (hourOffset + ep * 2 + rng() * 0.6 + 0.2) * 3600_000,
      );
      const earStart = Math.round((0.018 + rng() * 0.027) * 1000) / 1000;
      const base: FatigueEvent = {
        logged_at: startAt.toISOString(),
        session_id: driver.driver_id,
        driver_id: driver.driver_id,
        driver_name: driver.name,
        vehicle: driver.vehicle,
        route: driver.route,
        event: "drowsy_start",
        mode: "fleet_demo",
        status: "Спит",
        ear: earStart,
        left_ear: earStart,
        right_ear: Math.round(earStart * (0.8 + rng() * 0.3) * 1000) / 1000,
        confidence: 1,
        closed_frames: 28 + Math.floor(rng() * 18),
        nearest_stop_name: "Астана — ТРК «Хан Шатир»",
        nearest_stop_km: Math.round((3 + rng() * 15) * 10) / 10,
      };
      events.push(base);

      if (rng() > 0.4) {
        events.push({
          ...base,
          logged_at: new Date(startAt.getTime() + (12 + Math.floor(rng() * 24)) * 1000).toISOString(),
          event: "drowsy_continue",
        });
      }

      events.push({
        ...base,
        logged_at: new Date(startAt.getTime() + (4 + Math.floor(rng() * 11)) * 1000).toISOString(),
        event: "drowsy_end",
        status: "Не Спит",
        ear: Math.round((0.19 + rng() * 0.09) * 1000) / 1000,
        confidence: Math.round((0.45 + rng() * 0.17) * 100) / 100,
        closed_frames: 0,
      });
    }
  }

  events.sort((a, b) => a.logged_at.localeCompare(b.logged_at));
  return events;
}

export function loadFleetEvents(includeDemo: boolean): FatigueEvent[] {
  const real = readEvents(2000);
  return includeDemo ? [...generateDemoEvents(), ...real] : real;
}

export function eventsForDriver(events: FatigueEvent[], driverId: string): FatigueEvent[] {
  return events.filter((e) => (e.driver_id || e.session_id) === driverId);
}

export interface ChartPoint {
  time: string;
  ear: number;
  event: string;
}

export function episodeChartRows(events: FatigueEvent[]): ChartPoint[] {
  const rows: ChartPoint[] = [];
  for (const event of events) {
    if (!["drowsy_start", "drowsy_continue", "drowsy_end"].includes(event.event)) continue;
    const dt = new Date(event.logged_at);
    if (Number.isNaN(dt.getTime())) continue;
    const pad = (n: number) => String(n).padStart(2, "0");
    rows.push({
      time: `${pad(dt.getHours())}:${pad(dt.getMinutes())}:${pad(dt.getSeconds())}`,
      ear: Number(event.ear) || 0,
      event: event.event,
    });
  }
  return rows;
}

export function summarizeDrivers(events: FatigueEvent[]): DriverSummary[] {
  const byDriver = new Map<string, FatigueEvent[]>();
  const meta = new Map<string, { name: string; vehicle: string; route: string }>();

  for (const event of events) {
    const did = event.driver_id || event.session_id || "unknown";
    if (!byDriver.has(did)) byDriver.set(did, []);
    byDriver.get(did)!.push(event);
    if (!meta.has(did)) {
      meta.set(did, {
        name: event.driver_name || did,
        vehicle: event.vehicle || "—",
        route: event.route || "—",
      });
    }
  }

  const summaries: DriverSummary[] = [];
  for (const [did, driverEvents] of byDriver) {
    driverEvents.sort((a, b) => a.logged_at.localeCompare(b.logged_at));
    const episodes = driverEvents.filter((e) => e.event === "drowsy_start").length;
    const drowsyRows = driverEvents.filter(
      (e) => e.event === "drowsy_start" || e.event === "drowsy_continue",
    );
    const minEar =
      drowsyRows.length > 0 ? Math.min(...drowsyRows.map((e) => e.ear ?? 1)) : null;
    const last = driverEvents[driverEvents.length - 1];
    const active = last?.event === "drowsy_start" || last?.event === "drowsy_continue";
    const m = meta.get(did)!;
    summaries.push({
      driver_id: did,
      name: m.name,
      vehicle: m.vehicle,
      route: m.route,
      status: active ? STATUS_DROWSY : episodes ? STATUS_OK : STATUS_UNKNOWN,
      episodes,
      events_count: driverEvents.length,
      last_event: last?.logged_at,
      min_ear: minEar,
      mode: last?.mode || "—",
    });
  }

  summaries.sort((a, b) => {
    if (a.name === "Вы (live)") return -1;
    if (b.name === "Вы (live)") return 1;
    return a.name.localeCompare(b.name, "ru");
  });
  return summaries;
}
