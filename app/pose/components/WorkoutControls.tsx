import React, { useState } from "react";
import type { ExerciseId } from "../exercises/types";
import { EXERCISE_LABELS, ALL_EXERCISE_IDS } from "../exercises/types";
import { type CustomWorkout, encodeCustomWorkout } from "../../lib/customWorkouts";
import { ExerciseDemoModal } from "./ExerciseDemoModal";

type PlanMode = "free" | "plan" | "custom";
type Status = "init" | "loading" | "running" | "error";

interface WorkoutControlsProps {
    status: Status;
    planMode: PlanMode;
    exercise: ExerciseId;
    sessionRunning: boolean;
    activePlan: any;
    activeCustomWorkout: CustomWorkout | null;
    customWorkouts: CustomWorkout[];
    planStateActive: boolean;
    customStateActive: boolean;
    repCount: number;
    displayPhase: string;
    onExerciseChange: (ex: ExerciseId) => void;
    onPlanModeChange: (mode: PlanMode) => void;
    onSessionToggle: () => void;
    onSetupClick: () => void;
    onPlanStartStop: () => void;
    onCustomStartStop: () => void;
    onCustomWorkoutSelect: (id: string) => void;
    onReset: () => void;
    onSaveSession: () => void;
}

export function WorkoutControls({
    status,
    planMode,
    exercise,
    sessionRunning,
    activePlan,
    activeCustomWorkout,
    customWorkouts,
    planStateActive,
    customStateActive,
    repCount,
    displayPhase,
    onExerciseChange,
    onPlanModeChange,
    onSessionToggle,
    onSetupClick,
    onPlanStartStop,
    onCustomStartStop,
    onCustomWorkoutSelect,
    onReset,
    onSaveSession,
}: WorkoutControlsProps) {
    const [demoOpen, setDemoOpen] = useState(false);

    return (
        <div className="workout-controls-grid">
            {demoOpen && (
                <ExerciseDemoModal
                    exercise={exercise}
                    onClose={() => setDemoOpen(false)}
                />
            )}
            <div style={{ display: "grid", gap: 16 }}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
                    <div style={{ display: "grid", gap: 4 }}>
                        <label style={{ fontSize: 12, color: "#a7b4c7" }}>Mode</label>
                        <select
                            value={planMode}
                            onChange={(e) => onPlanModeChange(e.target.value as PlanMode)}
                            style={{
                                background: "rgba(255,255,255,0.06)",
                                color: "#e6edf6",
                                border: "1px solid rgba(255,255,255,0.12)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 16,
                                cursor: "pointer",
                                outline: "none",
                                fontWeight: 600,
                                minWidth: 140,
                            }}
                        >
                            <option value="free" style={{ background: "#0e111a", color: "#e6edf6" }}>
                                Free workout
                            </option>
                            <option value="plan" style={{ background: "#0e111a", color: "#e6edf6" }}>
                                Guided plan
                            </option>
                            <option value="custom" style={{ background: "#0e111a", color: "#e6edf6" }}>
                                Custom build
                            </option>
                        </select>
                    </div>

                    {planMode === "free" && (
                        <div style={{ display: "grid", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <label style={{ fontSize: 12, color: "#a7b4c7" }}>Exercise</label>
                                <button
                                    type="button"
                                    onClick={() => setDemoOpen(true)}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "var(--accent)",
                                        fontSize: 12,
                                        cursor: "pointer",
                                        padding: 0,
                                        textDecoration: "underline",
                                    }}
                                >
                                    Demo
                                </button>
                            </div>
                            <select
                                value={exercise}
                                onChange={(e) => onExerciseChange(e.target.value as ExerciseId)}
                                style={{
                                    background: "rgba(255,255,255,0.06)",
                                    color: "#e6edf6",
                                    border: "1px solid rgba(255,255,255,0.12)",
                                    borderRadius: 10,
                                    padding: "10px 12px",
                                    fontSize: 16,
                                    cursor: "pointer",
                                    outline: "none",
                                    fontWeight: 600,
                                    minWidth: 160,
                                }}
                            >
                                {ALL_EXERCISE_IDS.map((id) => (
                                    <option key={id} value={id} style={{ background: "#0e111a", color: "#e6edf6" }}>
                                        {EXERCISE_LABELS[id]}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}

                    {planMode === "custom" && (
                        <div style={{ display: "grid", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <label style={{ fontSize: 12, color: "#a7b4c7" }}>Select workout</label>
                                {activeCustomWorkout && (
                                    <button
                                        type="button"
                                        onClick={() => {
                                            const code = encodeCustomWorkout(activeCustomWorkout);
                                            if (code) {
                                                const url = `${window.location.origin}/builder?import=${code}`;
                                                navigator.clipboard.writeText(url).then(() => alert("Share Link copied to clipboard!"));
                                            }
                                        }}
                                        style={{
                                            background: "rgba(60, 242, 176, 0.1)",
                                            border: "1px solid rgba(60, 242, 176, 0.2)",
                                            color: "#3cf2b0",
                                            fontSize: 11,
                                            cursor: "pointer",
                                            padding: "2px 8px",
                                            borderRadius: 999,
                                            fontWeight: 600,
                                        }}
                                    >
                                        Share
                                    </button>
                                )}
                            </div>
                            <select
                                value={activeCustomWorkout?.id || ""}
                                onChange={(e) => onCustomWorkoutSelect(e.target.value)}
                                style={{
                                    background: "rgba(255,255,255,0.06)",
                                    color: "#e6edf6",
                                    border: "1px solid rgba(255,255,255,0.12)",
                                    borderRadius: 10,
                                    padding: "10px 12px",
                                    fontSize: 16,
                                    cursor: "pointer",
                                    outline: "none",
                                    fontWeight: 600,
                                    minWidth: 160,
                                }}
                            >
                                {customWorkouts.length === 0 ? (
                                    <option value="" disabled style={{ background: "#0e111a", color: "#e6edf6" }}>
                                        No workouts (Go to Builder)
                                    </option>
                                ) : (
                                    <>
                                        <option value="" disabled style={{ background: "#0e111a", color: "#e6edf6" }}>
                                            Choose...
                                        </option>
                                        {customWorkouts.map((w) => (
                                            <option key={w.id} value={w.id} style={{ background: "#0e111a", color: "#e6edf6" }}>
                                                {w.name}
                                            </option>
                                        ))}
                                    </>
                                )}
                            </select>
                        </div>
                    )}

                    {status !== "init" && status !== "error" && (
                        <button
                            type="button"
                            onClick={onSessionToggle}
                            style={{
                                background: sessionRunning ? "rgba(255, 180, 80, 0.12)" : "rgba(60, 242, 176, 0.14)",
                                color: sessionRunning ? "#ffd8a8" : "#e6edf6",
                                border: sessionRunning
                                    ? "1px solid rgba(255, 180, 80, 0.28)"
                                    : "1px solid rgba(60, 242, 176, 0.35)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                alignSelf: "end",
                            }}
                        >
                            {sessionRunning ? "Pause" : "Resume"}
                        </button>
                    )}

                    <button
                        type="button"
                        onClick={onSetupClick}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 10,
                            padding: "10px 12px",
                            fontSize: 14,
                            cursor: "pointer",
                            alignSelf: "end",
                        }}
                    >
                        Setup
                    </button>

                    {planMode === "plan" && (
                        <button
                            type="button"
                            onClick={onPlanStartStop}
                            style={{
                                background: planStateActive ? "rgba(255, 80, 80, 0.12)" : "rgba(60, 242, 176, 0.14)",
                                color: "#e6edf6",
                                border: planStateActive
                                    ? "1px solid rgba(255, 80, 80, 0.28)"
                                    : "1px solid rgba(60, 242, 176, 0.35)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                alignSelf: "end",
                                fontWeight: 800,
                            }}
                        >
                            {planStateActive ? "Stop plan" : "Start plan"}
                        </button>
                    )}

                    {planMode === "custom" && (
                        <button
                            type="button"
                            onClick={onCustomStartStop}
                            style={{
                                background: customStateActive ? "rgba(255, 80, 80, 0.12)" : "rgba(60, 242, 176, 0.14)",
                                color: "#e6edf6",
                                border: customStateActive
                                    ? "1px solid rgba(255, 80, 80, 0.28)"
                                    : "1px solid rgba(60, 242, 176, 0.35)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                alignSelf: "end",
                                fontWeight: 800,
                            }}
                        >
                            {customStateActive ? "Stop workout" : "Start workout"}
                        </button>
                    )}

                    <button
                        type="button"
                        onClick={onReset}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 10,
                            padding: "10px 12px",
                            fontSize: 14,
                            cursor: "pointer",
                            alignSelf: "end",
                        }}
                    >
                        Reset
                    </button>

                    {planMode === "free" && (
                        <button
                            type="button"
                            onClick={onSaveSession}
                            style={{
                                background: "rgba(60, 242, 176, 0.14)",
                                color: "#e6edf6",
                                border: "1px solid rgba(60, 242, 176, 0.35)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                alignSelf: "end",
                                fontWeight: 800,
                            }}
                        >
                            Save session
                        </button>
                    )}
                </div>
            </div>

            <div style={{ display: "grid", gap: 8, justifyItems: "end" }}>
                <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: 18 }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Reps</div>
                        <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: -0.2 }}>
                            {repCount}
                        </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Phase</div>
                        <div style={{ fontSize: 16, fontWeight: 700 }}>{displayPhase}</div>
                    </div>
                </div>

                <div style={{ fontSize: 12, color: "#a7b4c7" }}>Status: {status}</div>
            </div>
        </div>
    );
}
