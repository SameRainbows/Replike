import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind, ExerciseId, RepQualityLabel, QualityAgg } from "./types";
import { clamp01, parsePctFromText, parseAllPcts } from "./utils";

export function romPctFromFeedback(exercise: ExerciseId, feedback: string) {
    if (!feedback) return null;

    if (exercise === "jumping_jacks") {
        const pcts = parseAllPcts(feedback);
        if (pcts.length >= 2) return (pcts[0] + pcts[1]) / 2;
        return pcts[0] ?? null;
    }

    if (exercise === "squats" || exercise === "jump_squats" || exercise === "lunges" || exercise === "high_knees") {
        return parsePctFromText(feedback);
    }

    return null;
}

export function classifyRep(exercise: ExerciseId, romPct: number | null, tempoMs: number): RepQualityLabel {
    const minTempo =
        exercise === "jumping_jacks"
            ? 380
            : exercise === "high_knees"
                ? 300
                : exercise === "jump_squats"
                    ? 520
                    : exercise === "lunges"
                        ? 750
                        : exercise === "squats"
                            ? 700
                            : exercise === "push_ups"
                                ? 600
                                : exercise === "sit_ups"
                                    ? 650
                                    : 650;

    if (tempoMs > 0 && tempoMs < minTempo) return "sloppy";

    if (romPct === null) return "ok";
    if (romPct >= 70) return "clean";
    if (romPct >= 50) return "ok";
    return "sloppy";
}

export function emptyAgg(): QualityAgg {
    return { clean: 0, ok: 0, sloppy: 0, romSum: 0, romCount: 0, byExercise: {} };
}

export function getCoachCue(exercise: ExerciseId, repState: RepState) {
    if (!repState.feedback) return "";

    if (exercise === "squats") {
        const pct = parsePctFromText(repState.feedback);
        if (pct === null) return "";
        if (pct < 35) return "Coach: Go a bit deeper (controlled).";
        if (pct < 55) return "Coach: Nice. Try a little more depth.";
        if (repState.phase === "down") return "Coach: Drive up — stand tall at the top.";
        return "Coach: Smooth tempo — stay balanced.";
    }

    if (exercise === "jumping_jacks") {
        const nums = repState.feedback.match(/([0-9]{1,3})%/g)?.map((t) => Number(t.replace("%", ""))) ?? [];
        const legs = Number.isFinite(nums[0]) ? nums[0] : null;
        const arms = Number.isFinite(nums[1]) ? nums[1] : null;
        if (legs !== null && legs < 55) return "Coach: Wider feet.";
        if (arms !== null && arms < 55) return "Coach: Reach higher with your arms.";
        return "Coach: Great rhythm — keep it steady.";
    }

    if (exercise === "push_ups") {
        const pct = parsePctFromText(repState.feedback);
        if (pct === null) return "";
        if (pct < 40) return "Coach: Go lower — chest near the ground.";
        if (repState.phase === "down") return "Coach: Push up strong — full arm extension.";
        return "Coach: Keep your core tight.";
    }

    return "";
}
