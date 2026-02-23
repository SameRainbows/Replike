export type WorkoutSession = {
  id: string;
  startedAt: number;
  endedAt: number;
  durationSec: number;
  mode: "free" | "plan" | "custom";
  planId?: string;
  planName?: string;
  customWorkout?: {
    id: string;
    name: string;
    rounds: number;
    steps: Array<
      | { kind: "work_reps"; exercise: string; targetReps: number; label: string }
      | { kind: "work_time"; exercise: string; workSec: number; label: string }
      | { kind: "rest"; restSec: number; label: string }
    >;
  };
  goal?: {
    exercise: string;
    targetReps: number;
    reached: boolean;
  };
  quality?: {
    clean: number;
    ok: number;
    sloppy: number;
    avgRomPct?: number;
    byExercise: Record<
      string,
      {
        clean: number;
        ok: number;
        sloppy: number;
        avgRomPct?: number;
      }
    >;
  };
  totalReps: number;
  totalRejects: number;
  repsByExercise: Record<string, number>;
};

const STORAGE_KEY = "repdetect:sessions:v1";

export function loadSessions(): WorkoutSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as WorkoutSession[];
  } catch {
    return [];
  }
}

export function saveSessions(sessions: WorkoutSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

export function appendSession(next: WorkoutSession, maxSessions = 50): WorkoutSession[] {
  const sessions = loadSessions();
  const merged = [next, ...sessions];
  const trimmed = merged.slice(0, maxSessions);
  saveSessions(trimmed);
  return trimmed;
}

export function clearSessions() {
  localStorage.removeItem(STORAGE_KEY);
}
