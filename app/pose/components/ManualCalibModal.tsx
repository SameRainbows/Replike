import React from "react";
import type { ExerciseId } from "../exercises/types";

interface ManualCalibModalProps {
    exercise: ExerciseId;
    manualCalibStep: number;
    onSetStep: (step: number) => void;
    onCapture: (step: number) => void;
    onClose: () => void;
}

const CALIB_STEPS = [
    {
        title: "Standing position",
        hint: "Stand up straight and neutral.",
        actionLabel: "Capture step 1",
    },
    {
        title: "Bottom position",
        hint: "Hold the bottom/open position of the movement.",
        actionLabel: "Capture step 2 and finish",
    },
];

export function ManualCalibModal({
    exercise,
    manualCalibStep,
    onSetStep,
    onCapture,
    onClose,
}: ManualCalibModalProps) {
    const stepInfo = CALIB_STEPS[manualCalibStep];

    return (
        <div
            role="dialog"
            aria-modal="true"
            className="modalBackdrop"
            onClick={onClose}
        >
            <div
                className="modalCard"
                onClick={(e) => e.stopPropagation()}
            >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                    <div style={{ display: "grid", gap: 4 }}>
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Manual calibration</div>
                        <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.2 }}>
                            {exercise.replace("_", " ")}
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 10,
                            padding: "8px 10px",
                            fontSize: 13,
                            cursor: "pointer",
                        }}
                    >
                        Close
                    </button>
                </div>

                <div style={{ border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, padding: 12 }}>
                    <div style={{ fontWeight: 800 }}>{`Step ${manualCalibStep + 1}/2: ${stepInfo.title}`}</div>
                    <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
                        {stepInfo.hint}
                    </div>
                </div>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <button
                        type="button"
                        onClick={() => onCapture(manualCalibStep)}
                        style={{
                            background: "rgba(60, 242, 176, 0.14)",
                            color: "#e6edf6",
                            border: "1px solid rgba(60, 242, 176, 0.35)",
                            borderRadius: 12,
                            padding: "10px 12px",
                            fontSize: 14,
                            cursor: "pointer",
                            fontWeight: 800,
                        }}
                    >
                        {stepInfo.actionLabel}
                    </button>

                    <button
                        type="button"
                        onClick={() => onSetStep(manualCalibStep === 0 ? 1 : 0)}
                        style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#e6edf6",
                            border: "1px solid rgba(255,255,255,0.12)",
                            borderRadius: 12,
                            padding: "10px 12px",
                            fontSize: 14,
                            cursor: "pointer",
                        }}
                    >
                        {manualCalibStep === 0 ? "Skip to step 2" : "Back to step 1"}
                    </button>
                </div>

                <div className="muted" style={{ fontSize: 12 }}>
                    Tip: Keep your full body in frame and hold the position still for 1 second before capturing.
                </div>
            </div>
        </div>
    );
}
