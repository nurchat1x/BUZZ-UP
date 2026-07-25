interface LeafletMap {
  setView(latlng: [number, number], zoom: number): LeafletMap;
  fitBounds(bounds: unknown, opts?: { padding?: [number, number] }): LeafletMap;
  invalidateSize(): void;
}

interface LeafletLayerGroup {
  addTo(map: LeafletMap): LeafletLayerGroup;
  clearLayers(): void;
}

interface LeafletMarker {
  bindPopup(html: string): LeafletMarker;
  addTo(layer: LeafletLayerGroup): LeafletMarker;
}

declare const L: {
  map(el: HTMLElement): LeafletMap;
  tileLayer(url: string, opts?: Record<string, unknown>): { addTo(map: LeafletMap): unknown };
  layerGroup(): LeafletLayerGroup;
  circleMarker(latlng: [number, number], opts?: Record<string, unknown>): LeafletMarker;
  latLngBounds(points: Array<[number, number]>): unknown;
};
