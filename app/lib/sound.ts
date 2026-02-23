export type BeepKind = "rep" | "goal";

let audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) {
    throw new Error("Web Audio API not supported");
  }
  if (!audioCtx) audioCtx = new Ctx();
  return audioCtx;
}

function playTone(opts: { frequencyHz: number; durationMs: number; volume: number }) {
  const ctx = getAudioContext();

  if (ctx.state === "suspended") {
    // Best effort: resume only after user gesture; if it fails, we just won't beep.
    ctx.resume().catch(() => {
      // ignore
    });
  }

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = "sine";
  osc.frequency.value = opts.frequencyHz;

  const now = ctx.currentTime;
  const attack = 0.004;
  const release = 0.06;

  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, opts.volume), now + attack);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + opts.durationMs / 1000 + release);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(now);
  osc.stop(now + opts.durationMs / 1000 + release + 0.01);
}

export function playBeep(kind: BeepKind) {
  try {
    if (kind === "rep") {
      playTone({ frequencyHz: 880, durationMs: 55, volume: 0.12 });
      return;
    }

    // goal
    playTone({ frequencyHz: 660, durationMs: 80, volume: 0.14 });
    setTimeout(() => {
      try {
        playTone({ frequencyHz: 990, durationMs: 120, volume: 0.14 });
      } catch {
        // ignore
      }
    }, 90);
  } catch {
    // ignore
  }
}
