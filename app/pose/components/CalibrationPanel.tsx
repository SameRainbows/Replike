import React from "react";
import type { ExerciseId } from "../exercises/types";

interface CalibrationPanelProps {
    exercise: ExerciseId;
    calibrationEnabled: boolean;
    isCalibrated: boolean;
    autoCalibActive: boolean;
    autoCalibStableMs: number;
    autoCalibHint: string;
    onManualCalibClick: () => void;
    onClearCalibClick: () => void;
}

export function CalibrationPanel({
    exercise,
    calibrationEnabled,
    isCalibrated,
    autoCalibActive,
    autoCalibStableMs,
    autoCalibHint,
    onManualCalibClick,
    onClearCalibClick,
}: CalibrationPanelProps) {
    if (!calibrationEnabled) return null;

    return (
        <div
            style={{
                display: "grid",
                gap: 10,
                gridTemplateColumns: "1fr",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12,
                padding: 12,
                background: "rgba(255,255,255,0.02)",
            }}
        >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>Calibration</div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <button
                        type="button"
                        onClick={onManualCalibClick}
                        disabled={exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank"}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 10,
                            padding: "8px 10px",
                            fontSize: 13,
                            cursor: "pointer",
                            opacity: (exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank") ? 0.6 : 1,
                        }}
                    >
                        Manual calibrate
                    </button>
                    <button
                        type="button"
                        onClick={onClearCalibClick}
                        disabled={exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank"}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 10,
                            padding: "8px 10px",
                            fontSize: 13,
                            cursor: "pointer",
                            opacity: (exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank") ? 0.6 : 1,
                        }}
                    >
                        Clear
                    </button>
                </div>
            </div>
            {(exercise === "burpees" || exercise === "push_ups" || exercise === "sit_ups" || exercise === "plank") ? (
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>{exercise.replace("_", " ")} don’t require calibration.</div>
            ) : autoCalibActive ? (
                <div style={{ display: "grid", gap: 8 }}>
                    <div style={{ fontSize: 12, color: "#ededec" }}>Auto-calibrating…</div>
                    <div style={{ fontSize: 12, color: "#a7b4c7" }}>{autoCalibHint}</div>
                    <div
                        style={{
                            height: 8,
                            borderRadius: 999,
                            overflow: "hidden",
                            border: "1px solid rgba(255,255,255,0.10)",
                            background: "rgba(0,0,0,0.18)",
                        }}
                    >
                        <div
                            style={{
                                height: "100%",
                                width: `${Math.round((Math.min(900, autoCalibStableMs) / 900) * 100)}%`,
                                background: "rgba(232, 132, 94, 0.55)",
                            }}
                        />
                    </div>
                    <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                        Hold still for ~1 second. This runs automatically when calibration is missing.
                    </div>
                </div>
            ) : (
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                    {isCalibrated ? "Calibrated. Use Manual calibrate if you want to refine it." : "Ready for calibration."}
                </div>
            )}
        </div>
    );
}
