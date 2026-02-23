export type AppSettings = {
  calibrationEnabled: boolean;
};

const STORAGE_KEY = "repdetect:settings:v1";

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { calibrationEnabled: false };
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      calibrationEnabled: Boolean(parsed.calibrationEnabled),
    };
  } catch {
    return { calibrationEnabled: false };
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
