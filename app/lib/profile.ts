export type FitnessGoal = "build_muscle" | "lose_weight" | "stay_active";
export type FitnessLevel = "beginner" | "intermediate" | "advanced";

export type UserProfile = {
    hasCompletedOnboarding: boolean;
    goal: FitnessGoal;
    level: FitnessLevel;
};

const STORAGE_KEY = "repdetect:profile:v1";

export function getProfile(): UserProfile {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return { hasCompletedOnboarding: false, goal: "stay_active", level: "beginner" };
        }
        return JSON.parse(raw) as UserProfile;
    } catch {
        return { hasCompletedOnboarding: false, goal: "stay_active", level: "beginner" };
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
