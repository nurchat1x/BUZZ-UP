export interface BusStop {
  id: number;
  name: string;
  lat: number;
  lng: number;
  address?: string;
  services?: string[];
  amenities?: string[];
  rating?: number;
  reviews_count?: number;
  rating_source?: string;
  description?: string;
}

export interface NearestStop extends BusStop {
  distance_km: number;
}

export interface BusRoute {
  name: string;
  distance_km?: number;
  stops: BusStop[];
}

export interface BusStopsData {
  routes: Record<string, BusRoute>;
}

export function calculateDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dlat = toRad(lat2 - lat1);
  const dlon = toRad(lon2 - lon1);
  const a =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dlon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function findNearestStop(
  userLat: number,
  userLon: number,
  routeId: string,
  data: BusStopsData,
): NearestStop | null {
  const route = data.routes[routeId];
  if (!route) return null;
  let nearest: NearestStop | null = null;
  let minDist = Infinity;
  for (const stop of route.stops) {
    const d = calculateDistance(userLat, userLon, stop.lat, stop.lng);
    if (d < minDist) {
      minDist = d;
      nearest = { ...stop, distance_km: Math.round(d * 100) / 100 };
    }
  }
  return nearest;
}

export function formatRating(stop: BusStop): string {
  if (stop.rating == null) return "—";
  let base = `⭐ ${stop.rating}/5`;
  if (stop.reviews_count) {
    base += ` (${stop.reviews_count} отзывов`;
    if (stop.rating_source) base += `, ${stop.rating_source}`;
    base += ")";
  } else if (stop.rating_source) {
    base += ` (${stop.rating_source})`;
  }
  return base;
}

export async function loadBusStops(): Promise<BusStopsData> {
  const res = await fetch("./bus_stops.json");
  if (!res.ok) throw new Error("Не удалось загрузить bus_stops.json");
  return res.json() as Promise<BusStopsData>;
}
