import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export type ExerciseId =
    | "jumping_jacks"
    | "squats"
    | "lunges"
    | "high_knees"
    | "jump_squats"
    | "burpees"
    | "push_ups"
    | "sit_ups"
    | "plank";

export type DecisionKind = "none" | "rep" | "reject";

export type RepState = {
    exercise: ExerciseId;
    repCount: number;
    phase: "unknown" | "closed" | "open" | "up" | "down";
    lastPhaseChangeMs: number;
    lastRepMs: number;
    reachedTarget: boolean;
    lastSide: "left" | "right" | "none";
    feedback: string;
    decisionId: number;
    decisionKind: DecisionKind;
    decisionMessage: string;
    /** For plank: accumulated hold time in ms */
    holdMs?: number;
};

export type RepEvent = {
    id: string;
    ts: number;
    exercise: ExerciseId;
    kind: Exclude<DecisionKind, "none">;
    message: string;
    reps: number;
};

export type TrackingHealth = {
    level: "good" | "partial" | "lost";
    fps: number;
    hint: string;
    missing: Array<"upper" | "lower" | "arms" | "head">;
};

export type RepQualityLabel = "clean" | "ok" | "sloppy";

export type QualityAgg = {
    clean: number;
    ok: number;
    sloppy: number;
    romSum: number;
    romCount: number;
    byExercise: Record<
        string,
        {
            clean: number;
            ok: number;
            sloppy: number;
            romSum: number;
            romCount: number;
        }
    >;
};

export type JumpingJackCalibration = {
    openAnkleRatio: number;
    closedAnkleRatio: number;
    openArmsLift: number;
    closedArmsLift: number;
};

export type SquatCalibration = {
    topKneeAngle: number;
    bottomKneeAngle: number;
};

export type LungeCalibration = {
    topKneeAngle: number;
    bottomKneeAngle: number;
};

export type HighKneesCalibration = {
    upLift: number;
    downLift: number;
};

export type JumpSquatCalibration = {
    topKneeAngle: number;
    bottomKneeAngle: number;
};

export type Calibration = {
    jumping_jacks?: JumpingJackCalibration;
    squats?: SquatCalibration;
    lunges?: LungeCalibration;
    high_knees?: HighKneesCalibration;
    jump_squats?: JumpSquatCalibration;
};

export type WorkoutStep =
    | {
        kind: "work_reps";
        exercise: ExerciseId;
        targetReps: number;
        label: string;
    }
    | {
        kind: "work_time";
        exercise: ExerciseId;
        workSec: number;
        label: string;
    }
    | {
        kind: "rest";
        restSec: number;
        label: string;
    };

export type WorkoutPlan = {
    id: string;
    name: string;
    steps: WorkoutStep[];
};

export type CustomRunStep =
    | { kind: "work_reps"; exercise: ExerciseId; targetReps: number; label: string }
    | { kind: "work_time"; exercise: ExerciseId; workSec: number; label: string }
    | { kind: "rest"; restSec: number; label: string };

export const EXERCISE_LABELS: Record<ExerciseId, string> = {
    jumping_jacks: "Jumping jacks",
    squats: "Squats",
    lunges: "Lunges",
    high_knees: "High knees",
    jump_squats: "Jump squats",
    burpees: "Burpees",
    push_ups: "Push-ups",
    sit_ups: "Sit-ups",
    plank: "Plank",
};

export const ALL_EXERCISE_IDS: ExerciseId[] = [
    "jumping_jacks",
    "squats",
    "lunges",
    "high_knees",
    "jump_squats",
    "burpees",
    "push_ups",
    "sit_ups",
    "plank",
];
