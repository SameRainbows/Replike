import React from "react";
import type { ExerciseId } from "../exercises/types";
import { EXERCISE_LABELS } from "../exercises/types";
import { ExerciseAnimation } from "./ExerciseAnimation";

interface ExerciseDemoModalProps {
    exercise: ExerciseId;
    onClose: () => void;
}

const DEMO_HINTS: Record<ExerciseId, { title: string; hint: string }> = {
    jumping_jacks: {
        title: "Jumping Jacks",
        hint: "Start with feet together and hands at your sides. Jump feet out and bring arms overhead. Jump back to start.",
    },
    squats: {
        title: "Bodyweight Squats",
        hint: "Keep your chest up and back straight. Lower your hips until thighs are parallel to the floor, then stand back up.",
    },
    lunges: {
        title: "Alternating Lunges",
        hint: "Step forward with one leg and lower your hips until both knees are bent at a 90-degree angle. Push back to start and alternate.",
    },
    high_knees: {
        title: "High Knees",
        hint: "Run in place bringing your knees up as high as your waist. Keep a fast pace and pump your arms.",
    },
    jump_squats: {
        title: "Jump Squats",
        hint: "Perform a standard squat, but explode upwards into a jump from the bottom position. Land softly and go right into the next squat.",
    },
    burpees: {
        title: "Burpees",
        hint: "Drop into a squat, kick your feet back to a plank, return to the squat, and stand up (or jump).",
    },
    push_ups: {
        title: "Push-ups",
        hint: "Start in a plank position. Lower your body until your chest is just above the floor, then push back up. Keep your core tight.",
    },
    sit_ups: {
        title: "Sit-ups",
        hint: "Lie on your back with knees bent. Contract your abs to lift your torso to your knees, then slowly lower back down.",
    },
    plank: {
        title: "Plank Hold",
        hint: "Hold a push-up position (on hands or forearms) keeping your body in a straight line from head to heels.",
    },
};

export function ExerciseDemoModal({ exercise, onClose }: ExerciseDemoModalProps) {
    const info = DEMO_HINTS[exercise] || { title: EXERCISE_LABELS[exercise], hint: "Follow standard form for this exercise." };

    return (
        <div
            role="dialog"
            aria-modal="true"
            className="modalBackdrop"
            onClick={onClose}
        >
            <div className="modalCard" onClick={(e) => e.stopPropagation()}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                    <div style={{ display: "grid", gap: 4 }}>
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Exercise Demo</div>
                        <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.2 }}>{info.title}</div>
                    </div>
                    <button type="button" className="btn" onClick={onClose}>
                        Close
                    </button>
                </div>

                <div className="surface--inset" style={{ padding: 16, display: "grid", gap: 12 }}>

                    <ExerciseAnimation exercise={exercise} />

                    <div style={{ fontWeight: 800, fontSize: 14 }}>How to do it:</div>
                    <div className="muted" style={{ fontSize: 14, lineHeight: 1.6 }}>
                        {info.hint}
                    </div>

                    <div style={{
                        marginTop: 10,
                        padding: 12,
                        background: "rgba(60, 242, 176, 0.08)",
                        border: "1px solid rgba(60, 242, 176, 0.2)",
                        borderRadius: 12,
                        color: "#d6ffe9",
                        fontSize: 13,
                        display: "flex",
                        gap: 10,
                        alignItems: "center"
                    }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="16" x2="12" y2="12" />
                            <line x1="12" y1="8" x2="12.01" y2="8" />
                        </svg>
                        Tracker Tip: Keep your full body in frame and use slow, controlled movements.
                    </div>
                </div>
            </div>
        </div>
    );
}
