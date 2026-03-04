import { useState, useCallback } from "react";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { ExerciseId, RepState, RepEvent, Calibration } from "../exercises/types";
import { randomId } from "../exercises/utils";
import {
    updateJumpingJackState,
    updateSquatState,
    updateLungeState,
    updateHighKneesState,
    updateJumpSquatState,
    updateBurpeeState,
    updatePushUpState,
    updateSitUpState,
    updatePlankState,
    classifyRep,
    romPctFromFeedback,
} from "../exercises";

const INITIAL_REP_STATE: RepState = {
    exercise: "squats",
    repCount: 0,
    phase: "unknown",
    lastPhaseChangeMs: 0,
    lastRepMs: 0,
    reachedTarget: false,
    lastSide: "none",
    feedback: "",
    decisionId: 0,
    decisionKind: "none",
    decisionMessage: "",
    holdMs: 0,
};

export function useRepCounter() {
    const [repState, setRepState] = useState<RepState>(INITIAL_REP_STATE);
    const [events, setEvents] = useState<RepEvent[]>([]);

    const resetFreeSession = useCallback(() => {
        setRepState((s) => ({ ...s, repCount: 0, holdMs: 0 }));
        setEvents([]);
    }, []);

    const changeExercise = useCallback((ex: ExerciseId) => {
        setRepState((s) => ({
            ...s,
            exercise: ex,
            repCount: 0,
            phase: "unknown",
            lastPhaseChangeMs: 0,
            lastRepMs: 0,
            reachedTarget: false,
            lastSide: "none",
            feedback: "",
            holdMs: 0,
        }));
    }, []);

    const processFrame = useCallback(
        (landmarks: NormalizedLandmark[], nowMs: number, exercise: ExerciseId, calibration: Calibration) => {
            setRepState((prev) => {
                let next: RepState;

                switch (exercise) {
                    case "jumping_jacks":
                        next = updateJumpingJackState(prev, landmarks, nowMs, calibration.jumping_jacks);
                        break;
                    case "squats":
                        next = updateSquatState(prev, landmarks, nowMs, calibration.squats);
                        break;
                    case "lunges":
                        next = updateLungeState(prev, landmarks, nowMs, calibration.lunges);
                        break;
                    case "high_knees":
                        next = updateHighKneesState(prev, landmarks, nowMs, calibration.high_knees);
                        break;
                    case "jump_squats":
                        next = updateJumpSquatState(prev, landmarks, nowMs, calibration.jump_squats);
                        break;
                    case "burpees":
                        next = updateBurpeeState(prev, landmarks, nowMs);
                        break;
                    case "push_ups":
                        next = updatePushUpState(prev, landmarks, nowMs);
                        break;
                    case "sit_ups":
                        next = updateSitUpState(prev, landmarks, nowMs);
                        break;
                    case "plank":
                        next = updatePlankState(prev, landmarks, nowMs);
                        break;
                    default:
                        return prev;
                }

                if (next.decisionId !== prev.decisionId && next.decisionKind !== "none") {
                    setEvents((evs) => {
                        const newEv: RepEvent = {
                            id: randomId("ev"),
                            ts: nowMs,
                            exercise,
                            kind: next.decisionKind as "rep" | "reject",
                            message: next.decisionMessage,
                            reps: next.repCount,
                        };
                        return [newEv, ...evs].slice(0, 50);
                    });
                }

                return next;
            });
        },
        []
    );

    return { repState, setRepState, events, setEvents, resetFreeSession, changeExercise, processFrame };
}
