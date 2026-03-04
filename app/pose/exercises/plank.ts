import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RepState, DecisionKind } from "./types";
import { avg, getLandmark, isLandmarkConfident } from "./utils";

/**
 * Plank timer:
 * Detects and measures how long the user holds a valid plank position.
 * Instead of counting traditional reps, this tracks hold duration.
 * Every 10 seconds of plank hold counts as 1 "rep" for the counter.
 * The feedback shows the current hold time in seconds.
 */
export function updatePlankState(
    prev: RepState,
    landmarks: NormalizedLandmark[],
    nowMs: number
): RepState {
    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lAnkle = getLandmark(landmarks, 27);
    const rAnkle = getLandmark(landmarks, 28);
    const lElbow = getLandmark(landmarks, 13);
    const rElbow = getLandmark(landmarks, 14);

    const minVisible =
        isLandmarkConfident(lShoulder, 0.3) &&
        isLandmarkConfident(rShoulder, 0.3) &&
        isLandmarkConfident(lHip, 0.3) &&
        isLandmarkConfident(rHip, 0.3) &&
        isLandmarkConfident(lAnkle, 0.25) &&
        isLandmarkConfident(rAnkle, 0.25);

    if (!minVisible) {
        return {
            ...prev,
            phase: "unknown",
            feedback: "Keep shoulders, hips, and ankles visible.",
            holdMs: 0,
        };
    }

    const shoulderY = avg(lShoulder!.y, rShoulder!.y);
    const hipY = avg(lHip!.y, rHip!.y);
    const ankleY = avg(lAnkle!.y, rAnkle!.y);

    // Plank: body roughly horizontal — shoulders, hips, ankles should have
    // similar Y values (high Y = lower on screen). shoulders < hips < ankles
    // with small differences
    const bodyRange = Math.abs(ankleY - shoulderY);
    const hipDrop = hipY - Math.min(shoulderY, ankleY);
    const hipSag = Math.max(shoulderY, ankleY) - hipY;

    // Body must be roughly horizontal (not standing)
    const isHorizontalish = shoulderY > 0.35 && hipY > 0.35;
    // Shoulders, hips, ankles should be roughly aligned
    const isAligned = bodyRange < 0.3 && hipDrop < 0.15 && hipSag < 0.15;

    const inPlank = isHorizontalish && isAligned;

    let holdMs = prev.holdMs ?? 0;
    let repCount = prev.repCount;
    let lastRepMs = prev.lastRepMs;
    let feedback = prev.feedback;
    let decisionId = prev.decisionId;
    let decisionKind: DecisionKind = "none";
    let decisionMessage = "";

    if (inPlank) {
        if (prev.phase === "open") {
            // Continue holding
            const dt = nowMs - prev.lastPhaseChangeMs > 0 ? nowMs - (prev.lastRepMs || nowMs) : 0;
            holdMs = (prev.holdMs ?? 0) + (nowMs - (prev.lastRepMs || nowMs));
            // Use lastRepMs as the "last tick" marker
            lastRepMs = nowMs;
        } else {
            // Just entered plank
            holdMs = 0;
            lastRepMs = nowMs;
        }

        const holdSec = Math.floor(holdMs / 1000);
        feedback = `Plank: ${holdSec}s held. Keep your core tight!`;

        // Count a "rep" every 10 seconds
        const prevReps = Math.floor((prev.holdMs ?? 0) / 10000);
        const currentReps = Math.floor(holdMs / 10000);
        if (currentReps > prevReps) {
            repCount = currentReps;
            decisionId += 1;
            decisionKind = "rep";
            decisionMessage = `${currentReps * 10}s held!`;
        }
    } else {
        if (prev.phase === "open" && (prev.holdMs ?? 0) > 2000) {
            // Just broke plank after holding
            const heldSec = Math.floor((prev.holdMs ?? 0) / 1000);
            feedback = `Plank broken after ${heldSec}s. Get back in position.`;
        } else {
            feedback = "Get into plank position (forearms or hands on ground, body straight).";
        }
        holdMs = 0;
    }

    return {
        ...prev,
        repCount,
        phase: inPlank ? "open" : "down",
        lastPhaseChangeMs: (inPlank ? "open" : "down") !== prev.phase ? nowMs : prev.lastPhaseChangeMs,
        lastRepMs,
        reachedTarget: inPlank,
        feedback,
        decisionId,
        decisionKind,
        decisionMessage,
        holdMs,
    };
}
