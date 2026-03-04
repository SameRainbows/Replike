import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind, JumpSquatCalibration } from "./types";
import { angleDeg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

export function updateJumpSquatState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number,
    calib?: JumpSquatCalibration
): RepState {
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lKnee = getLandmark(landmarks, 25);
    const rKnee = getLandmark(landmarks, 26);
    const lAnkle = getLandmark(landmarks, 27);
    const rAnkle = getLandmark(landmarks, 28);

    const minVisible =
        isLandmarkConfident(lHip, 0.4) &&
        isLandmarkConfident(rHip, 0.4) &&
        isLandmarkConfident(lKnee, 0.4) &&
        isLandmarkConfident(rKnee, 0.4) &&
        isLandmarkConfident(lAnkle, 0.4) &&
        isLandmarkConfident(rAnkle, 0.4);

    if (!minVisible) return { ...prev, phase: "unknown" };

    const lKneeAngle = angleDeg(lHip!, lKnee!, lAnkle!);
    const rKneeAngle = angleDeg(rHip!, rKnee!, rAnkle!);
    const kneeAngle = Math.min(lKneeAngle, rKneeAngle);

    const top = calib ? calib.topKneeAngle : 175;
    const bottom = calib ? calib.bottomKneeAngle : 110;
    const downEnter = calib ? bottom + 12 : 118;
    const downExit = calib ? bottom + 28 : 140;
    const upEnter = calib ? top - 10 : 165;
    const upExit = calib ? top - 20 : 155;
    const minRepMs = 520;

    const isDown = prev.phase === "down" ? kneeAngle < downExit : kneeAngle < downEnter;
    const isUp = prev.phase === "up" ? kneeAngle > upExit : kneeAngle > upEnter;

    const minPhaseMs = 200;
    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") nextPhase = isDown ? "down" : "up";
    else if (canSwitch) {
        if (prev.phase === "up" && isDown) nextPhase = "down";
        if (prev.phase === "down" && isUp) nextPhase = "up";
    }

    let repCount = prev.repCount;
    let reachedTarget = prev.reachedTarget;
    let lastRepMs = prev.lastRepMs;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    if (nextPhase === "down") reachedTarget = true;

    const depthPct = clamp01((top - kneeAngle) / Math.max(top - bottom, 1e-6));
    if (kneeAngle < downEnter) feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (explode up).`;
    else if (depthPct > 0.45) feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (go lower).`;
    else feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (start).`;

    if (prev.phase === "down" && nextPhase === "up") {
        if (reachedTarget && nowMs - lastRepMs > minRepMs) {
            repCount += 1;
            lastRepMs = nowMs;
            reachedTarget = false;
            decisionId += 1;
            decisionKind = "rep";
            decisionMessage = "Rep counted.";
        } else {
            decisionId += 1;
            decisionKind = "reject";
            decisionMessage = reachedTarget ? "Too fast. Control the landing." : "Not deep enough.";
        }
    }

    return {
        ...prev,
        repCount,
        phase: nextPhase,
        lastPhaseChangeMs: nextPhase !== prev.phase ? nowMs : prev.lastPhaseChangeMs,
        lastRepMs,
        reachedTarget,
        feedback,
        decisionId,
        decisionKind,
        decisionMessage,
    };
}
