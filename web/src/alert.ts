/** Web Audio siren — same pattern as alert_sound.py */

const DEFAULT_COOLDOWN_SEC = 5;

let audioCtx: AudioContext | null = null;
let lastAlertTs = 0;
let soundEnabled = true;

export function setSoundEnabled(enabled: boolean): void {
  soundEnabled = enabled;
}

export function isSoundEnabled(): boolean {
  return soundEnabled;
}

async function ensureContext(): Promise<AudioContext | null> {
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) return null;
  if (!audioCtx) audioCtx = new Ctx();
  if (audioCtx.state === "suspended") {
    try {
      await audioCtx.resume();
    } catch {
      return null;
    }
  }
  return audioCtx;
}

function tone(
  ctx: AudioContext,
  freq: number,
  start: number,
  dur: number,
  volume: number,
): void {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "square";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(volume, start + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(start);
  osc.stop(start + dur + 0.05);
}

export async function playDrowsinessAlert(): Promise<void> {
  const ctx = await ensureContext();
  if (!ctx) return;
  const t = ctx.currentTime;
  tone(ctx, 880, t, 0.22, 0.22);
  tone(ctx, 660, t + 0.28, 0.22, 0.22);
  tone(ctx, 880, t + 0.56, 0.32, 0.26);
}

export async function handleDrowsinessAlert(
  status: string,
  cooldownSec = DEFAULT_COOLDOWN_SEC,
): Promise<void> {
  if (!soundEnabled || status !== "Спит") return;
  const now = performance.now() / 1000;
  if (now - lastAlertTs < cooldownSec) return;
  lastAlertTs = now;
  await playDrowsinessAlert();
}
