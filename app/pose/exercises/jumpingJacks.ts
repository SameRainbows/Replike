import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind, JumpingJackCalibration } from "./types";
import { dist, avg, getLandmark, isLandmarkConfident, clamp01 } from "./utils";

export function updateJumpingJackState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number,
    calib?: JumpingJackCalibration
): RepState {
    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lWrist = getLandmark(landmarks, 15);
    const rWrist = getLandmark(landmarks, 16);
    const lAnkle = getLandmark(landmarks, 27);
    const rAnkle = getLandmark(landmarks, 28);
    const nose = getLandmark(landmarks, 0);

    const minVisible =
        isLandmarkConfident(lShoulder, 0.35) &&
        isLandmarkConfident(rShoulder, 0.35) &&
        isLandmarkConfident(lWrist, 0.25) &&
        isLandmarkConfident(rWrist, 0.25) &&
        isLandmarkConfident(lAnkle, 0.25) &&
        isLandmarkConfident(rAnkle, 0.25) &&
        isLandmarkConfident(nose, 0.25);

    if (!minVisible) {
        return { ...prev, phase: "unknown" };
    }

    const shoulderWidth = dist(lShoulder!, rShoulder!);
    const ankleWidth = dist(lAnkle!, rAnkle!);
    const ankleToShoulderRatio = ankleWidth / Math.max(shoulderWidth, 1e-6);
    const wristsY = avg(lWrist!.y, rWrist!.y);
    const headY = nose!.y;
    const armsLift = headY - wristsY;

    const defaultArmsUp = armsLift > 0.02;

    const openEnter = calib ? calib.openAnkleRatio * 0.96 : 1.35;
    const openExit = calib
        ? Math.min(openEnter * 0.88, Math.max((calib.closedAnkleRatio ?? 1.1) * 1.04, 1.12))
        : 1.18;

    const legsOpen =
        prev.phase === "open"
            ? ankleToShoulderRatio > openExit
            : ankleToShoulderRatio > openEnter;

    const armsUp = calib
        ? prev.phase === "open"
            ? armsLift > calib.openArmsLift * 0.6
            : armsLift > calib.openArmsLift * 0.75
        : defaultArmsUp;

    const open = armsUp && legsOpen;
    const closed = !open;

    const minPhaseMs = 180;
    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;
    const minRepMs = 400;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") {
        nextPhase = open ? "open" : "closed";
    } else if (canSwitch) {
        if (prev.phase === "closed" && open) nextPhase = "open";
        if (prev.phase === "open" && closed) nextPhase = "closed";
    }

    let repCount = prev.repCount;
    let reachedTarget = prev.reachedTarget;
    let lastRepMs = prev.lastRepMs;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    if (nextPhase === "open") reachedTarget = true;

    const openness = calib
        ? clamp01(
            (ankleToShoulderRatio - calib.closedAnkleRatio) /
            Math.max(calib.openAnkleRatio - calib.closedAnkleRatio, 1e-6)
        )
        : clamp01((ankleToShoulderRatio - 1.15) / 0.5);

    const liftPct = calib
        ? clamp01((armsLift - calib.closedArmsLift) / Math.max(calib.openArmsLift - calib.closedArmsLift, 1e-6))
        : clamp01((armsLift - 0.02) / 0.08);

    if (open) {
        feedback = `Open: ${(openness * 100).toFixed(0)}% legs, ${(liftPct * 100).toFixed(0)}% arms.`;
    } else {
        feedback = `Aim for: ${(openness * 100).toFixed(0)}% legs, ${(liftPct * 100).toFixed(0)}% arms.`;
    }

    if (prev.phase === "open" && nextPhase === "closed") {
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
            decisionMessage = reachedTarget ? "Too fast. Slow down." : "Didn't reach full open position.";
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
