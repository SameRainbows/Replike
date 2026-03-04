"use client";

import { useEffect, useMemo, useState } from "react";
import { loadSessions, type WorkoutSession } from "@/app/lib/workoutHistory";
import { EXERCISE_LABELS } from "@/app/pose/exercises/types";

export default function TrendsPage() {
    const [sessions, setSessions] = useState<WorkoutSession[]>([]);

    useEffect(() => {
        setSessions(loadSessions());
    }, []);

    const stats = useMemo(() => {
        let totalReps = 0;
        let totalWorkouts = sessions.length;
        let totalTimeSec = 0;

        const repCounts: Record<string, number> = {};

        for (const s of sessions) {
            totalReps += s.totalReps || 0;
            totalTimeSec += s.durationSec || 0;

            if (s.repsByExercise) {
                for (const [ex, count] of Object.entries(s.repsByExercise)) {
                    repCounts[ex] = (repCounts[ex] || 0) + count;
                }
            }
        }

        const topExercises = Object.entries(repCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([ex, count]) => ({ ex, count }));

        return { totalReps, totalWorkouts, totalTimeSec, topExercises };
    }, [sessions]);

    return (
        <section className="stack">
            <header className="stack">
                <h1 className="h1">Trends</h1>
                <p className="lead">Your all-time stats and personal records.</p>
            </header>

            {sessions.length === 0 ? (
                <div className="card">
                    <div className="muted" style={{ fontSize: 14 }}>
                        No workout data available yet. Complete a session to see your trends!
                    </div>
                </div>
            ) : (
                <div className="grid">
                    <div className="card stack">
                        <div className="card__title">Total Workouts</div>
                        <div style={{ fontSize: 36, fontWeight: 800, color: "var(--accent)" }}>
                            {stats.totalWorkouts}
                        </div>
                    </div>
                    <div className="card stack">
                        <div className="card__title">Total Reps</div>
                        <div style={{ fontSize: 36, fontWeight: 800, color: "var(--accent)" }}>
                            {stats.totalReps.toLocaleString()}
                        </div>
                    </div>
                    <div className="card stack">
                        <div className="card__title">Time Active</div>
                        <div style={{ fontSize: 36, fontWeight: 800, color: "var(--accent)" }}>
                            {Math.floor(stats.totalTimeSec / 60)} <span style={{ fontSize: 16 }}>min</span>
                        </div>
                    </div>
                </div>
            )}

            {stats.topExercises.length > 0 && (
                <div className="card stack">
                    <div className="card__title" style={{ marginBottom: 8 }}>Top Exercises</div>
                    <div style={{ display: "grid", gap: 12 }}>
                        {stats.topExercises.map(({ ex, count }, i) => (
                            <div key={ex} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                                    <div style={{
                                        color: "rgba(255,255,255,0.4)",
                                        fontWeight: 800,
                                        width: 20
                                    }}>#{i + 1}</div>
                                    <div style={{ fontWeight: 600 }}>
                                        {EXERCISE_LABELS[ex as keyof typeof EXERCISE_LABELS] || ex.replace("_", " ")}
                                    </div>
                                </div>
                                <div style={{ fontWeight: 800, color: "rgba(230, 237, 246, 0.9)" }}>
                                    {count.toLocaleString()} <span style={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>reps</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </section>
    );
}
