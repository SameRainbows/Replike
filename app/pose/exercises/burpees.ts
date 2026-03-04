import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind } from "./types";
import { avg, angleDeg, getLandmark, isLandmarkConfident } from "./utils";

export function updateBurpeeState(prev: RepState, landmarks: NormalizedLandmark[], nowMs: number): RepState {
    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lKnee = getLandmark(landmarks, 25);
    const rKnee = getLandmark(landmarks, 26);
    const lAnkle = getLandmark(landmarks, 27);
    const rAnkle = getLandmark(landmarks, 28);

    const minVisible =
        isLandmarkConfident(lShoulder, 0.35) &&
        isLandmarkConfident(rShoulder, 0.35) &&
        isLandmarkConfident(lHip, 0.35) &&
        isLandmarkConfident(rHip, 0.35) &&
        isLandmarkConfident(lKnee, 0.35) &&
        isLandmarkConfident(rKnee, 0.35) &&
        isLandmarkConfident(lAnkle, 0.35) &&
        isLandmarkConfident(rAnkle, 0.35);

    if (!minVisible) return { ...prev, phase: "unknown", feedback: "Keep your full body in frame." };

    const shoulderY = avg(lShoulder!.y, rShoulder!.y);
    const hipY = avg(lHip!.y, rHip!.y);

    const lKneeAngle = angleDeg(lHip!, lKnee!, lAnkle!);
    const rKneeAngle = angleDeg(rHip!, rKnee!, rAnkle!);
    const kneeAngle = Math.min(lKneeAngle, rKneeAngle);

    const standLike = kneeAngle > 165 && hipY < 0.58;
    const crouchLike = kneeAngle < 145 || hipY > 0.62;
    const plankLike = shoulderY > 0.52 && hipY > 0.56 && Math.abs(hipY - shoulderY) < 0.16;

    const minPhaseMs = 220;
    const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;
    const minRepMs = 950;

    let nextPhase = prev.phase;
    if (prev.phase === "unknown") nextPhase = standLike ? "up" : plankLike ? "open" : "down";
    else if (canSwitch) {
        if (plankLike) nextPhase = "open";
        else if (crouchLike) nextPhase = "down";
        else if (standLike) nextPhase = "up";
    }

    let repCount = prev.repCount;
    let reachedTarget = prev.reachedTarget;
    let lastRepMs = prev.lastRepMs;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    if (nextPhase === "open") reachedTarget = true;

    if (nextPhase === "up") feedback = "Stand tall, then drop down.";
    else if (nextPhase === "down") feedback = "Hands down, kick back to plank.";
    else if (nextPhase === "open") feedback = "Plank. Drive feet in, then stand.";

    if (prev.phase === "open" && nextPhase === "up") {
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
            decisionMessage = reachedTarget ? "Too fast. Control the rep." : "Hit a solid plank position.";
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
