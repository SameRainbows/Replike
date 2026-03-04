import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind } from "./types";
import { angleDeg, avg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

/**
 * Sit-up detection:
 * Uses the angle between hip, shoulder, and knee to detect the torso angle.
 * Down phase = lying back (torso angle large / body flat).
 * Up phase = sitting up (torso angle small / body curled).
 * Rep counted on transition from up → down (one full sit-up cycle).
 */
export function updateSitUpState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number
): RepState {
    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lKnee = getLandmark(landmarks, 25);
    const rKnee = getLandmark(landmarks, 26);

    const minVisible =
        isLandmarkConfident(lShoulder, 0.3) &&
        isLandmarkConfident(rShoulder, 0.3) &&
        isLandmarkConfident(lHip, 0.3) &&
        isLandmarkConfident(rHip, 0.3) &&
        isLandmarkConfident(lKnee, 0.3) &&
        isLandmarkConfident(rKnee, 0.3);

    if (!minVisible) {
        return { ...prev, phase: "unknown", feedback: "Keep shoulders, hips, and knees visible." };
    }

    // Torso angle: knee → hip → shoulder
    const lTorsoAngle = angleDeg(lKnee!, lHip!, lShoulder!);
    const rTorsoAngle = angleDeg(rKnee!, rHip!, rShoulder!);
    const torsoAngle = avg(lTorsoAngle, rTorsoAngle);

    // Down (lying) = large angle (~150-180°)
    // Up (sitting) = small angle (~60-110°)
    const downAngle = 145;
    const upAngle = 100;
    const downEnter = 140;
    const downExit = 130;
    const upEnter = 105;
    const upExit = 115;
    const minRepMs = 650;
    const minPhaseMs = 200;

    const isDown = prev.phase === "down" ? torsoAngle > downExit : torsoAngle > downEnter;
    const isUp = prev.phase === "up" ? torsoAngle < upExit : torsoAngle < upEnter;

    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") {
        nextPhase = isUp ? "up" : "down";
    } else if (canSwitch) {
        if (prev.phase === "down" && isUp) nextPhase = "up";
        if (prev.phase === "up" && isDown) nextPhase = "down";
    }

    let repCount = prev.repCount;
    let reachedTarget = prev.reachedTarget;
    let lastRepMs = prev.lastRepMs;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    if (nextPhase === "up") reachedTarget = true;

    const crunchPct = clamp01((torsoAngle - upAngle) / Math.max(downAngle - upAngle, 1e-6));
    const risePct = 1 - crunchPct;
    if (torsoAngle < upEnter) feedback = `Rise: ${(risePct * 100).toFixed(0)}% (top). Lower back down.`;
    else if (risePct > 0.4) feedback = `Rise: ${(risePct * 100).toFixed(0)}% (sit up more).`;
    else feedback = `Rise: ${(risePct * 100).toFixed(0)}% (start).`;

    if (prev.phase === "up" && nextPhase === "down") {
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
            decisionMessage = reachedTarget ? "Too fast. Control the descent." : "Sit up higher.";
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
