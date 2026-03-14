export type AppSettings = {
  calibrationEnabled: boolean;
  soundOnRep: boolean;
  soundOnGoal: boolean;
  theme: "dark" | "light" | "system";
};

const STORAGE_KEY = "repdetect:settings:v1";

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { calibrationEnabled: false, soundOnRep: true, soundOnGoal: true, theme: "system" };
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      calibrationEnabled: Boolean(parsed.calibrationEnabled),
      soundOnRep: parsed.soundOnRep !== undefined ? Boolean(parsed.soundOnRep) : true,
      soundOnGoal: parsed.soundOnGoal !== undefined ? Boolean(parsed.soundOnGoal) : true,
      theme: parsed.theme === "dark" || parsed.theme === "light" ? parsed.theme : "system",
    };
  } catch {
    return { calibrationEnabled: false, soundOnRep: true, soundOnGoal: true, theme: "system" };
  }
}

export function saveSettings(next: AppSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  try {
    window.dispatchEvent(new Event("repdetect:settings"));
  } catch {
    // ignore
  }
}
