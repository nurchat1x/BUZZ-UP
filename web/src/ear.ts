/** Eye landmark indices and EAR — mirror of drowsiness_detector.py */

export const LEFT_EYE = [33, 160, 158, 133, 153, 144] as const;
export const RIGHT_EYE = [362, 385, 387, 263, 373, 380] as const;

export const DEFAULT_EAR_THRESHOLD = 0.21;
export const DEFAULT_CONSECUTIVE_FRAMES = 30;

export type Landmark = { x: number; y: number; z?: number };

function dist(a: Landmark, b: Landmark, width: number, height: number): number {
  const dx = (a.x - b.x) * width;
  const dy = (a.y - b.y) * height;
  return Math.hypot(dx, dy);
}

export function computeEar(
  landmarks: Landmark[],
  eyeIndices: readonly number[],
  width: number,
  height: number,
): number {
  const [i1, i2, i3, i4, i5, i6] = eyeIndices;
  const p1 = landmarks[i1];
  const p2 = landmarks[i2];
  const p3 = landmarks[i3];
  const p4 = landmarks[i4];
  const p5 = landmarks[i5];
  const p6 = landmarks[i6];
  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0;

  const vertical = dist(p2, p6, width, height) + dist(p3, p5, width, height);
  const horizontal = dist(p1, p4, width, height);
  if (horizontal < 1e-6) return 0;
  return vertical / (2 * horizontal);
}

export type DrowsinessStatus = "Не Спит" | "Спит" | "Лицо не найдено";

export interface DrowsinessResult {
  status: DrowsinessStatus;
  ear: number;
  leftEar: number;
  rightEar: number;
  faceDetected: boolean;
  closedFrames: number;
  confidence: number;
}

export class EarTracker {
  earThreshold: number;
  consecutiveFrames: number;
  private closedCounter = 0;

  constructor(
    earThreshold = DEFAULT_EAR_THRESHOLD,
    consecutiveFrames = DEFAULT_CONSECUTIVE_FRAMES,
  ) {
    this.earThreshold = earThreshold;
    this.consecutiveFrames = consecutiveFrames;
  }

  reset(): void {
    this.closedCounter = 0;
  }

  update(
    landmarks: Landmark[] | null,
    width: number,
    height: number,
  ): DrowsinessResult {
    if (!landmarks || landmarks.length === 0) {
      this.closedCounter = 0;
      return {
        status: "Лицо не найдено",
        ear: 0,
        leftEar: 0,
        rightEar: 0,
        faceDetected: false,
        closedFrames: 0,
        confidence: 0,
      };
    }

    const leftEar = computeEar(landmarks, LEFT_EYE, width, height);
    const rightEar = computeEar(landmarks, RIGHT_EYE, width, height);
    const ear = (leftEar + rightEar) / 2;

    if (ear < this.earThreshold) {
      this.closedCounter += 1;
    } else {
      this.closedCounter = 0;
    }

    const drowsy = this.closedCounter >= this.consecutiveFrames;
    const status: DrowsinessStatus = drowsy ? "Спит" : "Не Спит";

    let confidence: number;
    if (drowsy) {
      confidence = Math.min(1, this.closedCounter / Math.max(this.consecutiveFrames, 1));
    } else {
      const margin = Math.max(ear - this.earThreshold, 0);
      confidence = Math.min(1, 0.5 + margin * 2);
    }

    return {
      status,
      ear: round3(ear),
      leftEar: round3(leftEar),
      rightEar: round3(rightEar),
      faceDetected: true,
      closedFrames: this.closedCounter,
      confidence: round2(confidence),
    };
  }
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
