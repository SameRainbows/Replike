import type { WorkoutSession } from "./workoutHistory";

export function generateCoachInsight(sessions: WorkoutSession[]): string {
    if (sessions.length === 0) {
        return "Welcome to RepDetect! Complete your first workout to get personalized coaching insights.";
    }

    const now = Date.now();
    const oneDayMs = 24 * 60 * 60 * 1000;

    // Sort sessions newest first
    const sorted = [...sessions].sort((a, b) => b.startedAt - a.startedAt);

    const lastSession = sorted[0];
    const daysSinceLast = (now - lastSession.startedAt) / oneDayMs;

    // Check for inactivity
    if (daysSinceLast > 3) {
        return `It's been ${Math.floor(daysSinceLast)} days since your last workout. Ease back into it with a quick 5-minute Free Mode session!`;
    }

    // Check last 7 days volume distribution
    const lastWeek = sorted.filter(s => (now - s.startedAt) < 7 * oneDayMs);
    let legVolume = 0;
    let upperVolume = 0;
    let coreVolume = 0;
    let cardioVolume = 0;

    for (const s of lastWeek) {
        if (!s.repsByExercise) continue;
        for (const [ex, count] of Object.entries(s.repsByExercise)) {
            if (ex === "squats" || ex === "lunges" || ex === "jump_squats") legVolume += count;
            if (ex === "push_ups") upperVolume += count;
            if (ex === "sit_ups" || ex === "plank") coreVolume += count;
            if (ex === "jumping_jacks" || ex === "high_knees" || ex === "burpees") cardioVolume += count;
        }
    }

    if (lastWeek.length > 0) {
        if (legVolume > 0 && upperVolume === 0) {
            return "Great leg volume this week! Your upper body is resting. Consider adding Push-ups to your next routine to balance out.";
        }
        if (upperVolume > 0 && legVolume === 0) {
            return "Solid upper body focus lately! Don't let your legs fall behind. Try the 'Leg Day Burner' guided plan next.";
        }
        if (coreVolume === 0 && (legVolume > 0 || upperVolume > 0 || cardioVolume > 0)) {
            return "You're consistently active, but missing core activation. Throwing in 30 seconds of Planks at the end of your set works wonders.";
        }
    }

    // Check Quality (sloppy ratio) of last session
    if (lastSession.quality) {
        const totalGraded = lastSession.quality.clean + lastSession.quality.ok + lastSession.quality.sloppy;
        if (totalGraded > 10) {
            const sloppyRatio = lastSession.quality.sloppy / totalGraded;
            if (sloppyRatio > 0.4) {
                return "Coach noticed some of your reps were graded as 'Sloppy' last time. Focus on full Range of Motion today, even if it means doing fewer reps.";
            } else if (sloppyRatio < 0.1) {
                return "Incredible form on your last session! Over 90% of your reps were clean. Keep that standard up.";
            }
        }
    }

    // Default encouragement
    return "You're doing great! Keep building your weekly volume step by step. Consistency is your greatest asset.";
}
