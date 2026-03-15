"use client";

import { useEffect, useMemo, useState } from "react";
import { loadSessions, type WorkoutSession } from "@/app/lib/workoutHistory";
import { EXERCISE_LABELS } from "@/app/pose/exercises/types";
import { generateCoachInsight } from "@/app/lib/coach";
import { calculateStreak } from "@/app/lib/streaks";

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

    const coachMessage = useMemo(() => generateCoachInsight(sessions), [sessions]);

    const heatmapDays = useMemo(() => {
        const days = [];
        const now = new Date();
        now.setHours(0, 0, 0, 0);

        // Map sessions by midnight timestamp
        const activityMap = new Map<number, number>();
        for (const s of sessions) {
            const d = new Date(s.startedAt);
            d.setHours(0, 0, 0, 0);
            const ts = d.getTime();
            activityMap.set(ts, (activityMap.get(ts) || 0) + 1);
        }

        // Generate last 60 days
        for (let i = 59; i >= 0; i--) {
            const d = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
            const ts = d.getTime();
            const count = activityMap.get(ts) || 0;
            days.push({ date: d, count });
        }
        return days;
    }, [sessions]);

    const streaks = useMemo(() => calculateStreak(sessions), [sessions]);

    return (
        <section className="stack">
            <header className="stack" style={{ marginBottom: 12 }}>
                <h1 className="h1">Trends & Insights</h1>
                <p className="lead">Your all-time stats, personal records, and AI coaching.</p>
            </header>

            <div className="card" style={{
                background: "linear-gradient(145deg, rgba(232, 132, 94, 0.08) 0%, rgba(20, 20, 20, 0.4) 100%)",
                border: "1px solid rgba(232, 132, 94, 0.15)",
                display: "flex",
                gap: 16,
                alignItems: "flex-start",
                padding: 20
            }}>
                <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: 20,
                    background: "rgba(232, 132, 94, 0.15)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--accent)",
                    flexShrink: 0
                }}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                    </svg>
                </div>
                <div style={{ display: "grid", gap: 6 }}>
                    <div style={{ fontWeight: 800, fontSize: 14, color: "var(--accent)" }}>Smart Coach</div>
                    <div style={{ fontSize: 14, color: "var(--muted)", lineHeight: 1.5 }}>
                        {coachMessage}
                    </div>
                </div>
            </div>
            {sessions.length === 0 ? (
                <div className="card">
                    <div className="muted" style={{ fontSize: 14 }}>
                        No workout data available yet. Complete a session to see your trends!
                    </div>
                </div>
            ) : (
                <div className="stack" style={{ gap: 24 }}>
                    <div className="card stack">
                        <div className="card__title">Activity Map (Last 60 Days)</div>
                        <div style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(15, 1fr)",
                            gap: 6,
                            padding: 12,
                            background: "rgba(0,0,0,0.15)",
                            borderRadius: 12,
                            overflowX: "auto"
                        }}>
                            {heatmapDays.map((d, i) => (
                                <div
                                    key={i}
                                    title={`${d.date.toLocaleDateString()}: ${d.count} workouts`}
                                    style={{
                                        aspectRatio: "1/1",
                                        borderRadius: 4,
                                        background: d.count > 0
                                            ? `rgba(232, 132, 94, ${Math.min(0.2 + (d.count * 0.2), 1)})`
                                            : "rgba(255,255,255,0.04)",
                                        border: d.count > 0 ? "1px solid rgba(232, 132, 94, 0.3)" : "1px solid rgba(255,255,255,0.02)"
                                    }}
                                />
                            ))}
                        </div>
                    </div>

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

                    <div className="grid">
                        <div className="card stack">
                            <div className="card__title">Current Streak</div>
                            <div style={{ fontSize: 36, fontWeight: 800, color: "#ff8c6b" }}>
                                {streaks.current} <span style={{ fontSize: 16 }}>days</span>
                            </div>
                            {streaks.isCurrentlyActive ? (
                                <div style={{ fontSize: 13, color: "var(--accent)" }}>🔥 Active today</div>
                            ) : (
                                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.4)" }}>Complete a workout today to continue</div>
                            )}
                        </div>
                        <div className="card stack">
                            <div className="card__title">Best Streak</div>
                            <div style={{ fontSize: 36, fontWeight: 800, color: "var(--accent)" }}>
                                {streaks.best} <span style={{ fontSize: 16 }}>days</span>
                            </div>
                        </div>
                        <div className="card stack">
                            <div className="card__title">Trophy Cabinet</div>
                            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 8 }}>
                                {stats.totalWorkouts >= 1 && <div title="First Workout" style={{ fontSize: 24 }}>🌱</div>}
                                {stats.totalWorkouts >= 10 && <div title="10 Workouts" style={{ fontSize: 24 }}>🥉</div>}
                                {stats.totalWorkouts >= 50 && <div title="50 Workouts" style={{ fontSize: 24 }}>🥈</div>}
                                {stats.totalWorkouts >= 100 && <div title="Century Club" style={{ fontSize: 24 }}>🥇</div>}
                                {streaks.best >= 3 && <div title="3 Day Streak" style={{ fontSize: 24 }}>🔥</div>}
                                {streaks.best >= 7 && <div title="7 Day Streak" style={{ fontSize: 24 }}>🌋</div>}
                                {stats.totalWorkouts === 0 && <div style={{ fontSize: 13, color: "rgba(255,255,255,0.4)" }}>Earn your first trophy by working out!</div>}
                            </div>
                        </div>
                    </div>

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
                </div>
            )}
        </section>
    );
}
