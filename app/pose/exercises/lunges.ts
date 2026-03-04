import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind, LungeCalibration } from "./types";
import { angleDeg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

export function updateLungeState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number,
    calib?: LungeCalibration
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

    if (!minVisible) return { ...prev, phase: "unknown", feedback: "Step back so hips/knees/ankles are visible." };

    const leftAngle = angleDeg(lHip!, lKnee!, lAnkle!);
    const rightAngle = angleDeg(rHip!, rKnee!, rAnkle!);

    const activeSide: "left" | "right" = leftAngle < rightAngle ? "left" : "right";
    const activeAngle = Math.min(leftAngle, rightAngle);

    const top = calib ? calib.topKneeAngle : 175;
    const bottom = calib ? calib.bottomKneeAngle : 115;
    const downEnter = calib ? bottom + 10 : 120;
    const downExit = calib ? bottom + 25 : 140;
    const upEnter = calib ? top - 8 : 170;
    const upExit = calib ? top - 18 : 160;
    const minPhaseMs = 220;
    const minRepMs = 750;
    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;

    const isDown = prev.phase === "down" ? activeAngle < downExit : activeAngle < downEnter;
    const isUp = prev.phase === "up" ? activeAngle > upExit : activeAngle > upEnter;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") nextPhase = isDown ? "down" : "up";
    else if (canSwitch) {
        if (prev.phase === "up" && isDown) nextPhase = "down";
        if (prev.phase === "down" && isUp) nextPhase = "up";
    }

    let repCount = prev.repCount;
    let reachedTarget = prev.reachedTarget;
    let lastRepMs = prev.lastRepMs;
    let lastSide = prev.lastSide;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    let feedback = prev.feedback;
    const depthPct = clamp01((top - activeAngle) / Math.max(top - bottom, 1e-6));
    if (activeAngle < downEnter) feedback = `Lunge (${activeSide}) depth: ${(depthPct * 100).toFixed(0)}%. Push up.`;
    else if (depthPct > 0.45) feedback = `Lunge (${activeSide}) depth: ${(depthPct * 100).toFixed(0)}%. Go lower.`;
    else feedback = `Lunge (${activeSide}) depth: ${(depthPct * 100).toFixed(0)}%.`;

    if (nextPhase === "down") {
        reachedTarget = true;
        lastSide = activeSide;
    }

    if (prev.phase === "down" && nextPhase === "up") {
        if (reachedTarget && nowMs - lastRepMs > minRepMs) {
            repCount += 1;
            lastRepMs = nowMs;
            reachedTarget = false;
        }
    }

    return {
        ...prev,
        repCount,
        phase: nextPhase,
        lastPhaseChangeMs: nextPhase !== prev.phase ? nowMs : prev.lastPhaseChangeMs,
        lastRepMs,
        reachedTarget,
        lastSide,
        feedback,
        decisionId,
        decisionKind,
        decisionMessage,
    };
}
