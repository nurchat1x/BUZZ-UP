import type { DrowsinessResult } from "./ear";

export function makeDrowsyResult(): DrowsinessResult {
  return {
    status: "Спит",
    ear: 0.028,
    leftEar: 0.034,
    rightEar: 0.022,
    faceDetected: true,
    closedFrames: 32,
    confidence: 1.0,
  };
}

export function makeAwakeResult(): DrowsinessResult {
  return {
    status: "Не Спит",
    ear: 0.218,
    leftEar: 0.21,
    rightEar: 0.226,
    faceDetected: true,
    closedFrames: 0,
    confidence: 0.52,
  };
}
