/* Chart.js from CDN */
interface ChartDataset {
  label?: string;
  data: number[];
  borderColor?: string;
  backgroundColor?: string | string[];
  pointBackgroundColor?: string;
  pointBorderColor?: string;
  pointRadius?: number;
  pointHoverRadius?: number;
  tension?: number;
  fill?: boolean;
}

interface ChartConfiguration {
  type: string;
  data: {
    labels: string[];
    datasets: ChartDataset[];
  };
  options?: Record<string, unknown>;
}

interface ChartInstance {
  destroy(): void;
  update(): void;
}

interface ChartConstructor {
  new (ctx: HTMLCanvasElement | CanvasRenderingContext2D, config: ChartConfiguration): ChartInstance;
}

declare const Chart: ChartConstructor;
