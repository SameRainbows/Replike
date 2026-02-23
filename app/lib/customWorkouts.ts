export type CustomWorkoutStep =
  | {
      kind: "work_reps";
      exercise: string;
      targetReps: number;
      label: string;
    }
  | {
      kind: "work_time";
      exercise: string;
      workSec: number;
      label: string;
    }
  | {
      kind: "rest";
      restSec: number;
      label: string;
    };

export type CustomWorkout = {
  id: string;
  name: string;
  rounds: number;
  steps: CustomWorkoutStep[];
  updatedAt: number;
  createdAt: number;
};

const STORAGE_KEY = "repdetect:customWorkouts:v1";

export function loadCustomWorkouts(): CustomWorkout[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as CustomWorkout[];
  } catch {
    return [];
  }
}

export function saveCustomWorkouts(workouts: CustomWorkout[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(workouts));
  try {
    window.dispatchEvent(new Event("repdetect:customWorkouts"));
  } catch {
    // ignore
  }
}

export function upsertCustomWorkout(next: CustomWorkout, maxWorkouts = 50): CustomWorkout[] {
  const all = loadCustomWorkouts();
  const idx = all.findIndex((w) => w.id === next.id);
  const merged = idx === -1 ? [next, ...all] : all.map((w) => (w.id === next.id ? next : w));
  const trimmed = merged.slice(0, maxWorkouts);
  saveCustomWorkouts(trimmed);
  return trimmed;
}

export function deleteCustomWorkout(id: string): CustomWorkout[] {
  const all = loadCustomWorkouts();
  const next = all.filter((w) => w.id !== id);
  saveCustomWorkouts(next);
  return next;
}
