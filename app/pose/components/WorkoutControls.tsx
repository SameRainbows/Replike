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
    hasStarted: boolean;
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
    hasStarted,
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
                        <label style={{ fontSize: 12, color: "var(--muted)" }}>Mode</label>
                        <select
                            value={planMode}
                            onChange={(e) => onPlanModeChange(e.target.value as PlanMode)}
                            style={{
                                background: "var(--surface)",
                                color: "var(--text)",
                                border: "1px solid var(--border)",
                                borderRadius: 10,
                                padding: "10px 12px",
                                fontSize: 16,
                                cursor: "pointer",
                                outline: "none",
                                fontWeight: 600,
                                minWidth: 140,
                            }}
                        >
                            <option value="free" style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                Free workout
                            </option>
                            <option value="plan" style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                Guided plan
                            </option>
                            <option value="custom" style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                Custom build
                            </option>
                        </select>
                    </div>

                    {planMode === "free" && (
                        <div style={{ display: "grid", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <label style={{ fontSize: 12, color: "var(--muted)" }}>Exercise</label>
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
                                    background: "var(--surface)",
                                    color: "var(--text)",
                                    border: "1px solid var(--border)",
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
                                    <option key={id} value={id} style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                        {EXERCISE_LABELS[id]}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}

                    {planMode === "custom" && (
                        <div style={{ display: "grid", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <label style={{ fontSize: 12, color: "var(--muted)" }}>Select workout</label>
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
                                            background: "rgba(232, 132, 94, 0.1)",
                                            border: "1px solid rgba(232, 132, 94, 0.2)",
                                            color: "var(--accent)",
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
                                    background: "var(--surface)",
                                    color: "var(--text)",
                                    border: "1px solid var(--border)",
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
                                    <option value="" disabled style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                        No workouts (Go to Builder)
                                    </option>
                                ) : (
                                    <>
                                        <option value="" disabled style={{ background: "var(--bg0)", color: "var(--text)" }}>
                                            Choose...
                                        </option>
                                        {customWorkouts.map((w) => (
                                            <option key={w.id} value={w.id} style={{ background: "var(--bg0)", color: "var(--text)" }}>
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
                            className={`btn ${!sessionRunning ? "btn--primary" : ""}`}
                            onClick={onSessionToggle}
                            style={{
                                alignSelf: "end",
                                ...(sessionRunning ? {
                                    background: "rgba(255, 180, 80, 0.12)",
                                    borderColor: "rgba(255, 180, 80, 0.28)",
                                    color: "var(--accent-2)"
                                } : {})
                            }}
                        >
                            {sessionRunning ? "Pause" : hasStarted ? "Resume" : "Start Workout"}
                        </button>
                    )}

                    <button
                        type="button"
                        className="btn"
                        onClick={onSetupClick}
                        style={{ alignSelf: "end" }}
                    >
                        Setup
                    </button>

                    {planMode === "plan" && (
                        <button
                            type="button"
                            className={`btn ${!planStateActive ? "btn--primary" : ""}`}
                            onClick={onPlanStartStop}
                            style={{
                                alignSelf: "end",
                                ...(planStateActive ? {
                                    background: "rgba(255, 80, 80, 0.12)",
                                    borderColor: "rgba(255, 80, 80, 0.28)",
                                    color: "#f56565"
                                } : {})
                            }}
                        >
                            {planStateActive ? "Stop plan" : "Start plan"}
                        </button>
                    )}

                    {planMode === "custom" && (
                        <button
                            type="button"
                            className={`btn ${!customStateActive ? "btn--primary" : ""}`}
                            onClick={onCustomStartStop}
                            style={{
                                alignSelf: "end",
                                ...(customStateActive ? {
                                    background: "rgba(255, 80, 80, 0.12)",
                                    borderColor: "rgba(255, 80, 80, 0.28)",
                                    color: "#f56565"
                                } : {})
                            }}
                        >
                            {customStateActive ? "Stop workout" : "Start workout"}
                        </button>
                    )}

                    <button
                        type="button"
                        className="btn"
                        onClick={onReset}
                        style={{ alignSelf: "end" }}
                    >
                        Reset
                    </button>

                    {planMode === "free" && (
                        <button
                            type="button"
                            className="btn btn--primary"
                            onClick={onSaveSession}
                            style={{ alignSelf: "end" }}
                        >
                            Save session
                        </button>
                    )}
                </div>
            </div>

            <div style={{ display: "grid", gap: 8, justifyItems: "end" }}>
                <div style={{ fontSize: 12, color: "var(--muted)" }}>Status: {status}</div>
            </div>
        </div>
    );
}
