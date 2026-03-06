export type FitnessGoal = "build_muscle" | "lose_weight" | "stay_active";
export type FitnessLevel = "beginner" | "intermediate" | "advanced";

export type DailyMetrics = {
    dateStr: string; // YYYY-MM-DD
    sleepHours: number;
    waterGlasses: number;
};

export type UserProfile = {
    hasCompletedOnboarding: boolean;
    goal: FitnessGoal;
    level: FitnessLevel;
    metrics: DailyMetrics[];
};

const STORAGE_KEY = "repdetect:profile:v1";

export function getProfile(): UserProfile {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return { hasCompletedOnboarding: false, goal: "stay_active", level: "beginner", metrics: [] };
        }
        const p = JSON.parse(raw) as UserProfile;
        if (!p.metrics) p.metrics = [];
        return p;
    } catch {
        return { hasCompletedOnboarding: false, goal: "stay_active", level: "beginner", metrics: [] };
    }
}

export function saveProfile(profile: Partial<UserProfile>) {
    const current = getProfile();
    const next = { ...current, ...profile };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));

    try {
        window.dispatchEvent(new Event("repdetect:profile"));
    } catch { }
}

export function logDailyMetrics(sleep: number, water: number) {
    const current = getProfile();
    const today = new Date().toLocaleDateString();

    const existingIndex = current.metrics.findIndex(m => m.dateStr === today);
    if (existingIndex >= 0) {
        current.metrics[existingIndex] = { dateStr: today, sleepHours: sleep, waterGlasses: water };
    } else {
        current.metrics.push({ dateStr: today, sleepHours: sleep, waterGlasses: water });
    }

    // Keep last 30 days
    if (current.metrics.length > 30) current.metrics.shift();

    saveProfile({ metrics: current.metrics });
}
