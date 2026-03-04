import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind } from "./types";
import { angleDeg, avg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

/**
 * Push-up detection:
 * Uses elbow angle (shoulder→elbow→wrist) to detect down/up phases.
 * Down phase = arms bent (elbow angle < threshold).
 * Up phase = arms extended (elbow angle > threshold).
 * Rep counted on transition from down → up.
 */
export function updatePushUpState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number
): RepState {
    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lElbow = getLandmark(landmarks, 13);
    const rElbow = getLandmark(landmarks, 14);
    const lWrist = getLandmark(landmarks, 15);
    const rWrist = getLandmark(landmarks, 16);
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);

    const minVisible =
        isLandmarkConfident(lShoulder, 0.3) &&
        isLandmarkConfident(rShoulder, 0.3) &&
        isLandmarkConfident(lElbow, 0.3) &&
        isLandmarkConfident(rElbow, 0.3) &&
        isLandmarkConfident(lWrist, 0.25) &&
        isLandmarkConfident(rWrist, 0.25) &&
        isLandmarkConfident(lHip, 0.3) &&
        isLandmarkConfident(rHip, 0.3);

    if (!minVisible) {
        return { ...prev, phase: "unknown", feedback: "Keep shoulders, elbows, wrists, and hips visible." };
    }

    // Elbow angle: shoulder → elbow → wrist
    const lElbowAngle = angleDeg(lShoulder!, lElbow!, lWrist!);
    const rElbowAngle = angleDeg(rShoulder!, rElbow!, rWrist!);
    const elbowAngle = Math.min(lElbowAngle, rElbowAngle);

    // Verify plank-like position: shoulders and hips roughly horizontal
    const shoulderY = avg(lShoulder!.y, rShoulder!.y);
    const hipY = avg(lHip!.y, rHip!.y);
    const isHorizontal = Math.abs(shoulderY - hipY) < 0.25;

    if (!isHorizontal) {
        return { ...prev, phase: "unknown", feedback: "Get into push-up position (plank)." };
    }

    const topAngle = 160;
    const bottomAngle = 100;
    const downEnter = 110;
    const downExit = 130;
    const upEnter = 155;
    const upExit = 145;
    const minRepMs = 600;
    const minPhaseMs = 200;

    const isDown = prev.phase === "down" ? elbowAngle < downExit : elbowAngle < downEnter;
    const isUp = prev.phase === "up" ? elbowAngle > upExit : elbowAngle > upEnter;

    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") {
        nextPhase = isDown ? "down" : "up";
    } else if (canSwitch) {
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

    const depthPct = clamp01((topAngle - elbowAngle) / Math.max(topAngle - bottomAngle, 1e-6));
    if (elbowAngle < downEnter) feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (bottom). Push up.`;
    else if (depthPct > 0.4) feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (go lower).`;
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
            decisionMessage = reachedTarget ? "Too fast. Controlled reps." : "Not deep enough.";
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
