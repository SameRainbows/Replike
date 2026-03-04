import { useState, useRef } from "react";
import type { WorkoutPlan, CustomRunStep } from "../exercises/types";

export function useWorkoutRunner() {
    const [planState, setPlanState] = useState<{
        active: boolean;
        planId: string;
        stepIndex: number;
        stepStartedAt: number;
        stepStartReps: number;
    }>({
        active: false,
        planId: "",
        stepIndex: 0,
        stepStartedAt: 0,
        stepStartReps: 0,
    });

    const [customState, setCustomState] = useState<{
        active: boolean;
        workoutId: string;
        roundIndex: number;
        stepIndex: number;
        stepStartedAt: number;
        stepStartReps: number;
    }>({
        active: false,
        workoutId: "",
        roundIndex: 0,
        stepIndex: 0,
        stepStartedAt: 0,
        stepStartReps: 0,
    });

    const planRunRef = useRef<{ startedAt: number } | null>(null);
    const customRunRef = useRef<{ startedAt: number } | null>(null);

    const getActivePlanStep = (plan: WorkoutPlan | undefined) => {
        if (!plan || !planState.active) return null;
        return plan.steps[planState.stepIndex] ?? null;
    };

    const getActiveCustomStep = (customSteps: CustomRunStep[] | undefined) => {
        if (!customSteps || !customState.active) return null;
        return customSteps[customState.stepIndex] ?? null;
    };

    return {
        planState,
        setPlanState,
        customState,
        setCustomState,
        planRunRef,
        customRunRef,
        getActivePlanStep,
        getActiveCustomStep,
    };
}
