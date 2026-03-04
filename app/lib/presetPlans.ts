import type { WorkoutPlan } from "../pose/exercises/types";

export const PRESET_PLANS: WorkoutPlan[] = [
    {
        id: "p1",
        name: "Classic full-body sprint",
        steps: [
            { kind: "work_time", exercise: "jumping_jacks", workSec: 30, label: "Jumping Jacks" },
            { kind: "rest", restSec: 15, label: "Rest" },
            { kind: "work_reps", exercise: "squats", targetReps: 15, label: "Squats" },
            { kind: "rest", restSec: 15, label: "Rest" },
            { kind: "work_time", exercise: "burpees", workSec: 30, label: "Burpees" },
        ],
    },
    {
        id: "p2",
        name: "Leg day burner",
        steps: [
            { kind: "work_reps", exercise: "squats", targetReps: 20, label: "Wait for it..." },
            { kind: "rest", restSec: 10, label: "Rest" },
            { kind: "work_reps", exercise: "lunges", targetReps: 16, label: "Alt Lunges" },
            { kind: "rest", restSec: 15, label: "Rest" },
            { kind: "work_reps", exercise: "jump_squats", targetReps: 10, label: "Jump Squats" },
        ],
    },
    {
        id: "p3",
        name: "Core Crusher",
        steps: [
            { kind: "work_reps", exercise: "sit_ups", targetReps: 15, label: "Sit-ups" },
            { kind: "rest", restSec: 10, label: "Quick rest" },
            { kind: "work_time", exercise: "plank", workSec: 45, label: "Plank Hold" },
            { kind: "rest", restSec: 15, label: "Rest" },
            { kind: "work_reps", exercise: "high_knees", targetReps: 20, label: "High Knees" },
        ],
    },
    {
        id: "p4",
        name: "Upper Body Pump",
        steps: [
            { kind: "work_reps", exercise: "push_ups", targetReps: 10, label: "Push-ups" },
            { kind: "rest", restSec: 15, label: "Shake it out" },
            { kind: "work_time", exercise: "jumping_jacks", workSec: 30, label: "Jumping Jacks" },
            { kind: "rest", restSec: 15, label: "Rest" },
            { kind: "work_reps", exercise: "push_ups", targetReps: 10, label: "Push-ups" },
        ],
    },
    {
        id: "p5",
        name: "The Daily Dozen (Cardio)",
        steps: [
            { kind: "work_time", exercise: "high_knees", workSec: 20, label: "High Knees" },
            { kind: "rest", restSec: 10, label: "Rest" },
            { kind: "work_reps", exercise: "jumping_jacks", targetReps: 20, label: "Jumping Jacks" },
            { kind: "rest", restSec: 10, label: "Rest" },
            { kind: "work_time", exercise: "burpees", workSec: 20, label: "Burpees" },
        ],
    },
];
