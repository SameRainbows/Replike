import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind, HighKneesCalibration } from "./types";
import { avg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

export function updateHighKneesState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number,
    calib?: HighKneesCalibration
): RepState {
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lKnee = getLandmark(landmarks, 25);
    const rKnee = getLandmark(landmarks, 26);

    const minVisible =
        isLandmarkConfident(lHip, 0.35) &&
        isLandmarkConfident(rHip, 0.35) &&
        isLandmarkConfident(lKnee, 0.35) &&
        isLandmarkConfident(rKnee, 0.35);

    if (!minVisible) return { ...prev, phase: "unknown", feedback: "Make sure hips and knees are visible." };

    const hipY = avg(lHip!.y, rHip!.y);
    const leftLift = hipY - lKnee!.y;
    const rightLift = hipY - rKnee!.y;
    const upThreshold = calib ? (calib.upLift + calib.downLift) / 2 : 0.05;

    const leftUp = leftLift > upThreshold;
    const rightUp = rightLift > upThreshold;

    const sideUp: "left" | "right" | "none" = leftUp && !rightUp ? "left" : rightUp && !leftUp ? "right" : "none";
    const minRepMs = 300;

    let repCount = prev.repCount;
    let lastRepMs = prev.lastRepMs;
    let lastSide = prev.lastSide;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    const lift = Math.max(leftLift, rightLift);
    const liftPct = calib
        ? clamp01((lift - calib.downLift) / Math.max(calib.upLift - calib.downLift, 1e-6))
        : clamp01((lift - 0.02) / 0.12);

    if (sideUp === "none") feedback = `Lift: ${(liftPct * 100).toFixed(0)}%. Drive one knee higher.`;
    else feedback = `Lift: ${(liftPct * 100).toFixed(0)}%. Knee up (${sideUp}). Alternate.`;

    if (sideUp !== "none") {
        const canCount = nowMs - lastRepMs > minRepMs;
        const alternated = lastSide === "none" || lastSide !== sideUp;
        if (canCount && alternated) {
            repCount += 1;
            lastRepMs = nowMs;
            lastSide = sideUp;
            decisionId += 1;
            decisionKind = "rep";
            decisionMessage = "Rep counted.";
        } else if (canCount && !alternated) {
            decisionId += 1;
            decisionKind = "reject";
            decisionMessage = "Alternate legs for clean reps.";
        }
    }

    return {
        ...prev,
        repCount,
        phase: sideUp === "none" ? "down" : "up",
        lastPhaseChangeMs: prev.lastPhaseChangeMs,
        lastRepMs,
        reachedTarget: true,
        lastSide,
        feedback,
        decisionId,
        decisionKind,
        decisionMessage,
    };
}
