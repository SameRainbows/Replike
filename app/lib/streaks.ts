import type { WorkoutSession } from "./workoutHistory";

export function calculateStreak(sessions: WorkoutSession[]): { current: number, best: number, isCurrentlyActive: boolean } {
    if (!sessions || sessions.length === 0) return { current: 0, best: 0, isCurrentlyActive: false };

    // Get unique days where a workout happened
    const uniqueDays = new Set<string>();
    for (const s of sessions) {
        const d = new Date(s.startedAt);
        // Format YYYY-MM-DD
        uniqueDays.add(`${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`);
    }

    const sortedDays = Array.from(uniqueDays)
        .map(d => new Date(d).getTime())
        .sort((a, b) => b - a); // Newest first

    if (sortedDays.length === 0) return { current: 0, best: 0, isCurrentlyActive: false };

    let currentStreak = 0;
    let bestStreak = 0;

    // Check if the streak is active today or yesterday
    const todayStr = new Date().toLocaleDateString();
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toLocaleDateString();

    const latestSessionDate = new Date(sortedDays[0]).toLocaleDateString();
    const isCurrentlyActive = latestSessionDate === todayStr || latestSessionDate === yesterdayStr;

    // Calculate current streak
    if (isCurrentlyActive) {
        currentStreak = 1;
        for (let i = 0; i < sortedDays.length - 1; i++) {
            const currentMs = sortedDays[i];
            const nextMs = sortedDays[i + 1]; // "next" chronologically backward
            const diffDays = Math.round((currentMs - nextMs) / (1000 * 60 * 60 * 24));

            if (diffDays === 1) {
                currentStreak++;
            } else {
                break;
            }
        }
    }

    // Calculate best streak globally
    let tempStreak = 1;
    bestStreak = 1;
    for (let i = 0; i < sortedDays.length - 1; i++) {
        const currentMs = sortedDays[i];
        const nextMs = sortedDays[i + 1];
        const diffDays = Math.round((currentMs - nextMs) / (1000 * 60 * 60 * 24));

        if (diffDays === 1) {
            tempStreak++;
            if (tempStreak > bestStreak) bestStreak = tempStreak;
        } else {
            tempStreak = 1;
        }
    }

    return {
        current: currentStreak,
        best: Math.max(bestStreak, currentStreak),
        isCurrentlyActive
    };
}
