"use client";

import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useEffect, useMemo, useRef, useState } from "react";

type ExerciseId =
  | "jumping_jacks"
  | "squats"
  | "lunges"
  | "high_knees"
  | "jump_squats"
  | "burpees";

type DecisionKind = "none" | "rep" | "reject";

type RepState = {
  exercise: ExerciseId;
  repCount: number;
  phase: "unknown" | "closed" | "open" | "up" | "down";
  lastPhaseChangeMs: number;
  lastRepMs: number;
  reachedTarget: boolean;
  lastSide: "left" | "right" | "none";
  feedback: string;
  decisionId: number;
  decisionKind: DecisionKind;
  decisionMessage: string;
};

type RepEvent = {
  id: string;
  ts: number;
  exercise: ExerciseId;
  kind: Exclude<DecisionKind, "none">;
  message: string;
  reps: number;
};

type JumpingJackCalibration = {
  openAnkleRatio: number;
  closedAnkleRatio: number;
  openArmsLift: number;
  closedArmsLift: number;
};

type SquatCalibration = {
  topKneeAngle: number;
  bottomKneeAngle: number;
};

type LungeCalibration = {
  topKneeAngle: number;
  bottomKneeAngle: number;
};

type HighKneesCalibration = {
  upLift: number;
  downLift: number;
};

type JumpSquatCalibration = {
  topKneeAngle: number;
  bottomKneeAngle: number;
};

type Calibration = {
  jumping_jacks?: JumpingJackCalibration;
  squats?: SquatCalibration;
  lunges?: LungeCalibration;
  high_knees?: HighKneesCalibration;
  jump_squats?: JumpSquatCalibration;
};

type WorkoutStep =
  | {
      kind: "work_reps";
      exercise: ExerciseId;
      targetReps: number;
      label: string;
    }
  | {
      kind: "rest";
      restSec: number;
      label: string;
    };

type WorkoutPlan = {
  id: string;
  name: string;
  steps: WorkoutStep[];
};

function formatUnknownError(e: unknown) {
  if (e instanceof Error) {
    return `${e.name}: ${e.message}`;
  }

  // DOMException is not always instanceof Error across environments
  if (typeof e === "object" && e !== null) {
    const maybeDomEx = e as { name?: unknown; message?: unknown; code?: unknown };
    if (typeof maybeDomEx.name === "string" || typeof maybeDomEx.message === "string") {
      return `${String(maybeDomEx.name ?? "Error")}: ${String(maybeDomEx.message ?? "")}`.trim();
    }
  }

  if (e instanceof Event) {
    const anyEvent = e as any;
    const target = anyEvent?.target;
    const targetTag = target?.tagName ? String(target.tagName).toLowerCase() : "unknown";
    const targetError = target?.error;
    const targetSrc = target?.src || target?.currentSrc;

    const parts = [
      `Event: ${e.type}`,
      `target: ${targetTag}`,
      targetSrc ? `src: ${String(targetSrc)}` : null,
      targetError ? `target.error: ${String(targetError?.message ?? targetError)}` : null,
    ].filter(Boolean);

    return parts.join(" | ");
  }

  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

function clamp01(x: number) {
  return Math.min(1, Math.max(0, x));
}

function newId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function dist(a: NormalizedLandmark, b: NormalizedLandmark) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function angleDeg(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) {
  // Angle ABC (at point b)
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;

  const dot = abx * cbx + aby * cby;
  const ab = Math.hypot(abx, aby);
  const cb = Math.hypot(cbx, cby);
  const denom = Math.max(ab * cb, 1e-6);
  const cos = Math.max(-1, Math.min(1, dot / denom));
  return (Math.acos(cos) * 180) / Math.PI;
}

function avg(a: number, b: number) {
  return (a + b) / 2;
}

function getLandmark(landmarks: NormalizedLandmark[], idx: number) {
  const lm = landmarks[idx];
  if (!lm) return null;
  return lm;
}

function isLandmarkConfident(lm: NormalizedLandmark | null, minVis = 0.5) {
  if (!lm) return false;
  if (typeof lm.visibility !== "number") return true;
  return lm.visibility >= minVis;
}

function smoothLandmarks(
  prev: NormalizedLandmark[] | null,
  next: NormalizedLandmark[],
  alpha: number
): NormalizedLandmark[] {
  if (!prev || prev.length !== next.length) return next.map((l) => ({ ...l }));
  return next.map((n, i) => {
    const p = prev[i] ?? n;
    return {
      ...n,
      x: p.x + (n.x - p.x) * alpha,
      y: p.y + (n.y - p.y) * alpha,
      z: typeof n.z === "number" && typeof p.z === "number" ? p.z + (n.z - p.z) * alpha : n.z,
    };
  });
}

function updateJumpingJackState(
  prev: RepState,
  landmarks: NormalizedLandmark[],
  nowMs: number,
  calib?: JumpingJackCalibration
): RepState {
  // MediaPipe Pose landmark indices (BlazePose)
  // 11: left shoulder, 12: right shoulder
  // 15: left wrist, 16: right wrist
  // 27: left ankle, 28: right ankle
  // 0: nose
  const lShoulder = getLandmark(landmarks, 11);
  const rShoulder = getLandmark(landmarks, 12);
  const lWrist = getLandmark(landmarks, 15);
  const rWrist = getLandmark(landmarks, 16);
  const lAnkle = getLandmark(landmarks, 27);
  const rAnkle = getLandmark(landmarks, 28);
  const nose = getLandmark(landmarks, 0);

  const minVisible =
    isLandmarkConfident(lShoulder) &&
    isLandmarkConfident(rShoulder) &&
    isLandmarkConfident(lWrist, 0.4) &&
    isLandmarkConfident(rWrist, 0.4) &&
    isLandmarkConfident(lAnkle, 0.4) &&
    isLandmarkConfident(rAnkle, 0.4) &&
    isLandmarkConfident(nose, 0.4);

  if (!minVisible) {
    return {
      ...prev,
      phase: "unknown",
    };
  }

  const shoulderWidth = dist(lShoulder!, rShoulder!);
  const ankleWidth = dist(lAnkle!, rAnkle!);

  // Normalize widths by shoulder width for body-size invariance
  const ankleToShoulderRatio = ankleWidth / Math.max(shoulderWidth, 1e-6);

  // Arms "up" if wrists are above head/nose region (y smaller = higher)
  const wristsY = avg(lWrist!.y, rWrist!.y);
  const headY = nose!.y;
  const armsLift = headY - wristsY;

  const defaultArmsUp = armsLift > 0.03;

  // Legs "open" if feet are spread wide relative to shoulders
  // Use hysteresis thresholds
  const openEnter = calib ? calib.openAnkleRatio : 1.45;
  const openExit = calib ? Math.min(calib.openAnkleRatio, Math.max(calib.closedAnkleRatio, 1.1)) : 1.25;
  const legsOpenEnter = openEnter;
  const legsOpenExit = openExit;

  const legsOpen =
    prev.phase === "open"
      ? ankleToShoulderRatio > legsOpenExit
      : ankleToShoulderRatio > legsOpenEnter;

  const armsUp = calib
    ? prev.phase === "open"
      ? armsLift > calib.openArmsLift * 0.7
      : armsLift > calib.openArmsLift * 0.85
    : defaultArmsUp;

  // Combine into overall "open" phase
  const open = armsUp && legsOpen;
  const closed = !open;

  // Timing: ignore ultra-fast toggles from jitter
  const minPhaseMs = 180;
  const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;
  const minRepMs = 500;

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
      decisionMessage = reachedTarget ? "Too fast. Slow down." : "Didn’t reach full open position.";
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

function updateJumpSquatState(
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

function updateBurpeeState(prev: RepState, landmarks: NormalizedLandmark[], nowMs: number): RepState {
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

function updateSquatState(
  prev: RepState,
  landmarks: NormalizedLandmark[],
  nowMs: number,
  calib?: SquatCalibration
): RepState {
  // 23/24: hips, 25/26: knees, 27/28: ankles
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

  if (!minVisible) {
    return { ...prev, phase: "unknown" };
  }

  const lKneeAngle = angleDeg(lHip!, lKnee!, lAnkle!);
  const rKneeAngle = angleDeg(rHip!, rKnee!, rAnkle!);
  const kneeAngle = Math.min(lKneeAngle, rKneeAngle);

  const top = calib ? calib.topKneeAngle : 175;
  const bottom = calib ? calib.bottomKneeAngle : 110;
  const downEnter = calib ? bottom + 10 : 115;
  const downExit = calib ? bottom + 25 : 135;
  const upEnter = calib ? top - 8 : 165;
  const upExit = calib ? top - 18 : 155;
  const minRepMs = 700;

  const isDown =
    prev.phase === "down"
      ? kneeAngle < downExit
      : kneeAngle < downEnter;

  const isUp =
    prev.phase === "up"
      ? kneeAngle > upExit
      : kneeAngle > upEnter;

  const minPhaseMs = 220;
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

  const depthPct = clamp01((top - kneeAngle) / Math.max(top - bottom, 1e-6));
  if (kneeAngle < downEnter) feedback = `Depth: ${(depthPct * 100).toFixed(0)}% (bottom). Drive up.`;
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
      decisionMessage = reachedTarget ? "Too fast. Slow down." : "Not deep enough.";
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

function updateLungeState(
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

function updateHighKneesState(
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

export default function PoseRepCounter() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const smoothedRef = useRef<NormalizedLandmark[] | null>(null);
  const lastLandmarksRef = useRef<NormalizedLandmark[] | null>(null);

  const [exercise, setExercise] = useState<ExerciseId>("jumping_jacks");
  const [sessionRunning, setSessionRunning] = useState<boolean>(true);
  const [calibration, setCalibration] = useState<Calibration>({});
  const [events, setEvents] = useState<RepEvent[]>([]);
  const [manualCalibOpen, setManualCalibOpen] = useState(false);
  const [manualCalibStep, setManualCalibStep] = useState<0 | 1>(0);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimeoutRef = useRef<number | null>(null);

  const exerciseRef = useRef<ExerciseId>(exercise);
  const sessionRunningRef = useRef<boolean>(sessionRunning);
  const calibrationRef = useRef<Calibration>(calibration);

  const [autoCalib, setAutoCalib] = useState<{
    active: boolean;
    step: 0 | 1;
    stableMs: number;
    lastOkMs: number;
    lastCaptureMs: number;
  }>(() => ({ active: false, step: 0, stableMs: 0, lastOkMs: 0, lastCaptureMs: 0 }));
  const [status, setStatus] = useState<
    "idle" | "loading_model" | "requesting_camera" | "running" | "error"
  >("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [repState, setRepState] = useState<RepState>(() => ({
    exercise: "jumping_jacks",
    repCount: 0,
    phase: "unknown",
    lastPhaseChangeMs: 0,
    lastRepMs: 0,
    reachedTarget: false,
    lastSide: "none",
    feedback: "",
    decisionId: 0,
    decisionKind: "none",
    decisionMessage: "",
  }));

  const displayPhase = useMemo(() => {
    if (repState.phase === "unknown") return "No pose";
    if (repState.phase === "open") return "Open";
    if (repState.phase === "closed") return "Closed";
    if (repState.phase === "up") return "Up";
    return "Down";
  }, [repState.phase]);

  const workoutPlans = useMemo<WorkoutPlan[]>(() => {
    const plans: WorkoutPlan[] = [
      {
        id: "starter_cardio",
        name: "Starter cardio (8 min)",
        steps: [
          { kind: "work_reps", exercise: "jumping_jacks", targetReps: 30, label: "Jumping jacks" },
          { kind: "rest", restSec: 30, label: "Rest" },
          { kind: "work_reps", exercise: "high_knees", targetReps: 40, label: "High knees" },
          { kind: "rest", restSec: 40, label: "Rest" },
          { kind: "work_reps", exercise: "jump_squats", targetReps: 15, label: "Jump squats" },
          { kind: "rest", restSec: 45, label: "Rest" },
          { kind: "work_reps", exercise: "burpees", targetReps: 10, label: "Burpees" },
        ],
      },
      {
        id: "legs_builder",
        name: "Legs builder (10 min)",
        steps: [
          { kind: "work_reps", exercise: "squats", targetReps: 20, label: "Squats" },
          { kind: "rest", restSec: 45, label: "Rest" },
          { kind: "work_reps", exercise: "lunges", targetReps: 16, label: "Lunges" },
          { kind: "rest", restSec: 45, label: "Rest" },
          { kind: "work_reps", exercise: "jump_squats", targetReps: 12, label: "Jump squats" },
          { kind: "rest", restSec: 60, label: "Rest" },
          { kind: "work_reps", exercise: "high_knees", targetReps: 50, label: "High knees" },
        ],
      },
    ];
    return plans;
  }, []);

  const [planMode, setPlanMode] = useState<"free" | "plan">("free");
  const [selectedPlanId, setSelectedPlanId] = useState<string>("starter_cardio");
  const [planState, setPlanState] = useState<{
    active: boolean;
    planId: string;
    stepIndex: number;
    stepStartedAt: number;
    stepStartReps: number;
  }>(() => ({ active: false, planId: "starter_cardio", stepIndex: 0, stepStartedAt: 0, stepStartReps: 0 }));
  const [planNowMs, setPlanNowMs] = useState<number>(0);

  function showToast(message: string) {
    setToast(message);
    if (toastTimeoutRef.current) window.clearTimeout(toastTimeoutRef.current);
    toastTimeoutRef.current = window.setTimeout(() => {
      setToast(null);
      toastTimeoutRef.current = null;
    }, 1400);
  }

  const calibSteps = useMemo(() => {
    if (exercise === "jumping_jacks") {
      return [
        {
          title: "Closed position",
          hint: "Stand with feet together and arms down.",
          actionLabel: "Capture closed",
        },
        {
          title: "Open position",
          hint: "Do a full jumping jack: feet wide and arms overhead.",
          actionLabel: "Capture open",
        },
      ] as const;
    }

    if (exercise === "high_knees") {
      return [
        {
          title: "Rest position",
          hint: "Stand tall with both feet on the ground.",
          actionLabel: "Capture rest",
        },
        {
          title: "Knee-up position",
          hint: "Lift one knee as high as you can (above hip if possible).",
          actionLabel: "Capture knee-up",
        },
      ] as const;
    }

    return [
      {
        title: "Top position",
        hint: "Stand tall (top of rep).",
        actionLabel: "Capture top",
      },
      {
        title: "Bottom position",
        hint: "Go to your deepest position (bottom of rep).",
        actionLabel: "Capture bottom",
      },
    ] as const;
  }, [exercise]);

  const autoCalibHint = useMemo(() => {
    const step = autoCalib.step;
    const stepName = `Step ${step + 1}/2`;

    if (exercise === "jumping_jacks") {
      return step === 0
        ? `${stepName}: Stand closed (feet together, arms down).`
        : `${stepName}: Go open (feet wide, arms overhead).`;
    }

    if (exercise === "high_knees") {
      return step === 0
        ? `${stepName}: Stand at rest (both feet down).`
        : `${stepName}: Hold a knee-up position.`;
    }

    return step === 0
      ? `${stepName}: Stand tall (top position).`
      : `${stepName}: Hold your deepest position (bottom).`;
  }, [autoCalib.step, exercise]);

  useEffect(() => {
    const hasCalib = exercise === "burpees" ? true : Boolean(calibration[exercise]);
    setAutoCalib((s) => ({
      ...s,
      active: !hasCalib,
      step: 0,
      stableMs: 0,
      lastOkMs: 0,
      lastCaptureMs: 0,
    }));
  }, [exercise, calibration]);

  function captureCalibrationFrame(step: 0 | 1) {
    const lms = lastLandmarksRef.current;
    if (!lms) return;

    if (exercise === "squats") {
      const lHip = getLandmark(lms, 23);
      const rHip = getLandmark(lms, 24);
      const lKnee = getLandmark(lms, 25);
      const rKnee = getLandmark(lms, 26);
      const lAnkle = getLandmark(lms, 27);
      const rAnkle = getLandmark(lms, 28);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35)
      )
        return;

      const lAngle = angleDeg(lHip!, lKnee!, lAnkle!);
      const rAngle = angleDeg(rHip!, rKnee!, rAnkle!);
      const angle = Math.min(lAngle, rAngle);

      setCalibration((c) => ({
        ...c,
        squats: {
          topKneeAngle: step === 0 ? angle : c.squats?.topKneeAngle ?? 175,
          bottomKneeAngle: step === 1 ? angle : c.squats?.bottomKneeAngle ?? 110,
        },
      }));

      return;
    }

    if (exercise === "lunges") {
      const lHip = getLandmark(lms, 23);
      const rHip = getLandmark(lms, 24);
      const lKnee = getLandmark(lms, 25);
      const rKnee = getLandmark(lms, 26);
      const lAnkle = getLandmark(lms, 27);
      const rAnkle = getLandmark(lms, 28);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35)
      )
        return;

      const lAngle = angleDeg(lHip!, lKnee!, lAnkle!);
      const rAngle = angleDeg(rHip!, rKnee!, rAnkle!);
      const angle = Math.min(lAngle, rAngle);

      setCalibration((c) => ({
        ...c,
        lunges: {
          topKneeAngle: step === 0 ? angle : c.lunges?.topKneeAngle ?? 175,
          bottomKneeAngle: step === 1 ? angle : c.lunges?.bottomKneeAngle ?? 115,
        },
      }));

      return;
    }

    if (exercise === "jumping_jacks") {
      const lShoulder = getLandmark(lms, 11);
      const rShoulder = getLandmark(lms, 12);
      const lWrist = getLandmark(lms, 15);
      const rWrist = getLandmark(lms, 16);
      const lAnkle = getLandmark(lms, 27);
      const rAnkle = getLandmark(lms, 28);
      const nose = getLandmark(lms, 0);
      if (
        !isLandmarkConfident(lShoulder, 0.35) ||
        !isLandmarkConfident(rShoulder, 0.35) ||
        !isLandmarkConfident(lWrist, 0.35) ||
        !isLandmarkConfident(rWrist, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35) ||
        !isLandmarkConfident(nose, 0.35)
      )
        return;

      const shoulderWidth = dist(lShoulder!, rShoulder!);
      const ankleWidth = dist(lAnkle!, rAnkle!);
      const ratio = ankleWidth / Math.max(shoulderWidth, 1e-6);
      const headY = nose!.y;
      const wristsY = avg(lWrist!.y, rWrist!.y);
      const lift = headY - wristsY;

      setCalibration((c) => ({
        ...c,
        jumping_jacks: {
          openAnkleRatio: step === 1 ? ratio : c.jumping_jacks?.openAnkleRatio ?? 1.45,
          closedAnkleRatio: step === 0 ? ratio : c.jumping_jacks?.closedAnkleRatio ?? 1.15,
          openArmsLift: step === 1 ? lift : c.jumping_jacks?.openArmsLift ?? 0.08,
          closedArmsLift: step === 0 ? lift : c.jumping_jacks?.closedArmsLift ?? 0.02,
        },
      }));

      return;
    }

    if (exercise === "high_knees") {
      const lHip = getLandmark(lms, 23);
      const rHip = getLandmark(lms, 24);
      const lKnee = getLandmark(lms, 25);
      const rKnee = getLandmark(lms, 26);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35)
      )
        return;

      const hipY = avg(lHip!.y, rHip!.y);
      const lift = Math.max(hipY - lKnee!.y, hipY - rKnee!.y);

      setCalibration((c) => ({
        ...c,
        high_knees: {
          upLift: step === 1 ? lift : c.high_knees?.upLift ?? 0.1,
          downLift: step === 0 ? lift : c.high_knees?.downLift ?? 0.02,
        },
      }));

      return;
    }

    if (exercise === "jump_squats") {
      const lHip = getLandmark(lms, 23);
      const rHip = getLandmark(lms, 24);
      const lKnee = getLandmark(lms, 25);
      const rKnee = getLandmark(lms, 26);
      const lAnkle = getLandmark(lms, 27);
      const rAnkle = getLandmark(lms, 28);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35)
      )
        return;

      const lAngle = angleDeg(lHip!, lKnee!, lAnkle!);
      const rAngle = angleDeg(rHip!, rKnee!, rAnkle!);
      const angle = Math.min(lAngle, rAngle);

      setCalibration((c) => ({
        ...c,
        jump_squats: {
          topKneeAngle: step === 0 ? angle : c.jump_squats?.topKneeAngle ?? 175,
          bottomKneeAngle: step === 1 ? angle : c.jump_squats?.bottomKneeAngle ?? 110,
        },
      }));

      return;
    }
  }

  useEffect(() => {
    try {
      const raw = localStorage.getItem("repdetect:v1");
      if (!raw) return;
      const parsed = JSON.parse(raw) as {
        exercise?: ExerciseId;
        calibration?: Calibration;
        planMode?: "free" | "plan";
        selectedPlanId?: string;
      };

      if (parsed.exercise) setExercise(parsed.exercise);
      if (parsed.calibration) setCalibration(parsed.calibration);
      if (parsed.planMode) setPlanMode(parsed.planMode);
      if (parsed.selectedPlanId) setSelectedPlanId(parsed.selectedPlanId);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(
        "repdetect:v1",
        JSON.stringify({
          exercise,
          calibration,
          planMode,
          selectedPlanId,
        })
      );
    } catch {
      // ignore
    }
  }, [exercise, calibration, planMode, selectedPlanId]);

  useEffect(() => {
    exerciseRef.current = exercise;
  }, [exercise]);

  useEffect(() => {
    sessionRunningRef.current = sessionRunning;
  }, [sessionRunning]);

  useEffect(() => {
    calibrationRef.current = calibration;
  }, [calibration]);

  useEffect(() => {
    setRepState((s) => ({
      ...s,
      exercise,
      repCount: 0,
      phase: "unknown",
      lastPhaseChangeMs: 0,
      lastRepMs: 0,
      reachedTarget: false,
      lastSide: "none",
      feedback: "",
      decisionId: 0,
      decisionKind: "none",
      decisionMessage: "",
    }));
    setEvents([]);
  }, [exercise]);

  useEffect(() => {
    if (repState.decisionKind === "none") return;
    if (!repState.decisionMessage) return;

    const kind = repState.decisionKind === "rep" ? "rep" : "reject";

    if (kind === "rep") {
      // Avoid filling the screen with "Rep counted" lines. Use a toast for most reps.
      showToast(`+1 ${exercise.replace("_", " ")}`);
      const milestone = repState.repCount === 1 || repState.repCount % 5 === 0;
      if (!milestone) return;
    }

    setEvents((prev) => {
      const next: RepEvent = {
        id: newId(),
        ts: Date.now(),
        exercise,
        kind,
        message: kind === "rep" ? `Milestone: ${repState.repCount}` : repState.decisionMessage,
        reps: repState.repCount,
      };
      const merged = [next, ...prev];
      return merged.slice(0, 20);
    });
  }, [repState.decisionId]);

  const activePlan = useMemo(() => workoutPlans.find((p) => p.id === selectedPlanId) ?? workoutPlans[0], [selectedPlanId, workoutPlans]);
  const activePlanStep = useMemo(() => {
    if (!planState.active) return null;
    const plan = workoutPlans.find((p) => p.id === planState.planId);
    if (!plan) return null;
    return plan.steps[planState.stepIndex] ?? null;
  }, [planState, workoutPlans]);

  useEffect(() => {
    if (!planState.active) return;
    setPlanNowMs(Date.now());
    const id = window.setInterval(() => setPlanNowMs(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [planState.active]);

  useEffect(() => {
    if (!planState.active) return;
    if (!activePlanStep) return;

    if (activePlanStep.kind === "work_reps") {
      if (exercise !== activePlanStep.exercise) setExercise(activePlanStep.exercise);

      const repsDone = repState.repCount - planState.stepStartReps;
      if (repsDone >= activePlanStep.targetReps) {
        const nextIndex = planState.stepIndex + 1;
        if (nextIndex >= activePlan.steps.length) {
          showToast("Workout complete");
          setPlanState((s) => ({ ...s, active: false }));
          return;
        }
        setPlanState((s) => ({
          ...s,
          stepIndex: nextIndex,
          stepStartedAt: Date.now(),
          stepStartReps: repState.repCount,
        }));
      }
    }

    if (activePlanStep.kind === "rest") {
      const elapsedSec = (planNowMs - planState.stepStartedAt) / 1000;
      if (elapsedSec >= activePlanStep.restSec) {
        const nextIndex = planState.stepIndex + 1;
        if (nextIndex >= activePlan.steps.length) {
          showToast("Workout complete");
          setPlanState((s) => ({ ...s, active: false }));
          return;
        }
        setPlanState((s) => ({
          ...s,
          stepIndex: nextIndex,
          stepStartedAt: Date.now(),
          stepStartReps: repState.repCount,
        }));
      }
    }
  }, [planState.active, planState.stepIndex, planState.stepStartReps, planState.stepStartedAt, planNowMs, repState.repCount, activePlanStep, activePlan, exercise]);

  function isPoseStableForAutoCalibration(
    landmarks: NormalizedLandmark[],
    step: 0 | 1
  ): boolean {
    if (exercise === "jumping_jacks") {
      const lShoulder = getLandmark(landmarks, 11);
      const rShoulder = getLandmark(landmarks, 12);
      const lWrist = getLandmark(landmarks, 15);
      const rWrist = getLandmark(landmarks, 16);
      const lAnkle = getLandmark(landmarks, 27);
      const rAnkle = getLandmark(landmarks, 28);
      const nose = getLandmark(landmarks, 0);
      if (
        !isLandmarkConfident(lShoulder, 0.35) ||
        !isLandmarkConfident(rShoulder, 0.35) ||
        !isLandmarkConfident(lWrist, 0.35) ||
        !isLandmarkConfident(rWrist, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35) ||
        !isLandmarkConfident(nose, 0.35)
      )
        return false;

      const shoulderWidth = dist(lShoulder!, rShoulder!);
      const ankleWidth = dist(lAnkle!, rAnkle!);
      const ratio = ankleWidth / Math.max(shoulderWidth, 1e-6);
      const headY = nose!.y;
      const wristsY = avg(lWrist!.y, rWrist!.y);
      const armsLift = headY - wristsY;

      const isClosed = ratio < 1.25 && armsLift < 0.04;
      const isOpen = ratio > 1.4 && armsLift > 0.06;
      return step === 0 ? isClosed : isOpen;
    }

    if (exercise === "squats" || exercise === "lunges") {
      const lHip = getLandmark(landmarks, 23);
      const rHip = getLandmark(landmarks, 24);
      const lKnee = getLandmark(landmarks, 25);
      const rKnee = getLandmark(landmarks, 26);
      const lAnkle = getLandmark(landmarks, 27);
      const rAnkle = getLandmark(landmarks, 28);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35) ||
        !isLandmarkConfident(lAnkle, 0.35) ||
        !isLandmarkConfident(rAnkle, 0.35)
      )
        return false;

      const lAngle = angleDeg(lHip!, lKnee!, lAnkle!);
      const rAngle = angleDeg(rHip!, rKnee!, rAnkle!);
      const angle = Math.min(lAngle, rAngle);
      const isTop = angle > 165;
      const isBottom = angle < 130;
      return step === 0 ? isTop : isBottom;
    }

    if (exercise === "high_knees") {
      const lHip = getLandmark(landmarks, 23);
      const rHip = getLandmark(landmarks, 24);
      const lKnee = getLandmark(landmarks, 25);
      const rKnee = getLandmark(landmarks, 26);
      if (
        !isLandmarkConfident(lHip, 0.35) ||
        !isLandmarkConfident(rHip, 0.35) ||
        !isLandmarkConfident(lKnee, 0.35) ||
        !isLandmarkConfident(rKnee, 0.35)
      )
        return false;

      const hipY = avg(lHip!.y, rHip!.y);
      const lift = Math.max(hipY - lKnee!.y, hipY - rKnee!.y);
      const isRest = lift < 0.04;
      const isUp = lift > 0.09;
      return step === 0 ? isRest : isUp;
    }

    return false;
  }

  useEffect(() => {
    let landmarker: PoseLandmarker | null = null;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let lastVideoTime = -1;
    let cancelled = false;

    async function start() {
      try {
        setErrorMessage(null);
        setStatus("loading_model");

        // Load MediaPipe WASM assets from our own origin (copied into /public/mediapipe/wasm on install)
        const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");

        landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });

        setStatus("requesting_camera");

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        const video = videoRef.current;
        if (!video) throw new Error("Video element not found");

        const onVideoError = (ev: Event) => {
          if (cancelled) return;
          setErrorMessage(`Video error: ${formatUnknownError(ev)}`);
          setStatus("error");
        };
        video.addEventListener("error", onVideoError);

        video.srcObject = stream;

        // Some browsers throw a DOMException here if not triggered by a user gesture.
        // We still call play() because most browsers allow it for getUserMedia streams.
        await video.play();

        setStatus("running");

        const canvas = canvasRef.current;
        if (!canvas) throw new Error("Canvas element not found");

        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Canvas 2D context not available");

        const drawingUtils = new DrawingUtils(ctx);

        const render = () => {
          rafId = requestAnimationFrame(render);

          if (!landmarker) return;
          if (!videoRef.current) return;

          const videoEl = videoRef.current;
          if (videoEl.readyState < 2) return;

          // Resize canvas to match the *rendered* video size.
          // Using intrinsic videoWidth/videoHeight can desync the overlay when the video is
          // displayed responsively (width: 100%, height: auto), causing visible offsets.
          const rect = videoEl.getBoundingClientRect();
          const w = Math.round(rect.width);
          const h = Math.round(rect.height);
          if (!w || !h) return;

          if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
          }

          const nowMs = performance.now();

          if (videoEl.currentTime === lastVideoTime) return;
          lastVideoTime = videoEl.currentTime;

          const result = landmarker.detectForVideo(videoEl, nowMs);

          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (result.landmarks && result.landmarks[0]) {
            const raw = result.landmarks[0];
            const smoothed = smoothLandmarks(smoothedRef.current, raw, 0.25);
            smoothedRef.current = smoothed;
            lastLandmarksRef.current = smoothed;

            // Hands-free calibration: when missing calibration, wait for stable poses and capture automatically.
            if (autoCalib.active) {
              const ok = isPoseStableForAutoCalibration(smoothed, autoCalib.step);
              const stableNeededMs = 900;
              const now = nowMs;

              setAutoCalib((s) => {
                if (!s.active) return s;
                const okNow = isPoseStableForAutoCalibration(smoothed, s.step);
                const lastOkMs = okNow ? (s.lastOkMs || now) : 0;
                const stableMs = okNow ? now - lastOkMs : 0;
                const cooldownOk = now - s.lastCaptureMs > 700;

                if (okNow && stableMs >= stableNeededMs && cooldownOk) {
                  captureCalibrationFrame(s.step);
                  showToast(s.step === 0 ? "Calibrated step 1/2" : "Calibration complete");
                  return {
                    ...s,
                    step: s.step === 0 ? 1 : 0,
                    active: s.step === 0,
                    stableMs: 0,
                    lastOkMs: 0,
                    lastCaptureMs: now,
                  };
                }

                return {
                  ...s,
                  stableMs,
                  lastOkMs,
                };
              });
            }

            drawingUtils.drawLandmarks(smoothed, {
              radius: (data) => 2 + 3 * clamp01(data.from?.z ? 0 : 1),
              color: "#3cf2b0",
            });

            drawingUtils.drawConnectors(
              smoothed,
              PoseLandmarker.POSE_CONNECTIONS,
              {
                color: "rgba(60, 242, 176, 0.35)",
                lineWidth: 3,
              }
            );

            setRepState((prev) => {
              const ex = exerciseRef.current;
              const running = sessionRunningRef.current;
              const calib = calibrationRef.current;

              if (prev.exercise !== ex) {
                return {
                  exercise: ex,
                  repCount: 0,
                  phase: "unknown",
                  lastPhaseChangeMs: 0,
                  lastRepMs: 0,
                  reachedTarget: false,
                  lastSide: "none",
                  feedback: "",
                  decisionId: 0,
                  decisionKind: "none",
                  decisionMessage: "",
                };
              }

              if (!running) {
                return {
                  ...prev,
                  feedback: "Paused.",
                  decisionKind: "none",
                  decisionMessage: "",
                };
              }

              if (ex === "jumping_jacks") return updateJumpingJackState(prev, smoothed, nowMs, calib.jumping_jacks);
              if (ex === "squats") return updateSquatState(prev, smoothed, nowMs, calib.squats);
              if (ex === "lunges") return updateLungeState(prev, smoothed, nowMs, calib.lunges);
              if (ex === "high_knees") return updateHighKneesState(prev, smoothed, nowMs, calib.high_knees);
              if (ex === "jump_squats") return updateJumpSquatState(prev, smoothed, nowMs, calib.jump_squats);
              if (ex === "burpees") return updateBurpeeState(prev, smoothed, nowMs);
              return prev;
            });
          } else {
            setRepState((prev) => ({
              ...prev,
              phase: "unknown",
              feedback: "",
            }));
          }
        };

        render();
      } catch (e) {
        const message = formatUnknownError(e);
        setErrorMessage(message);
        setStatus("error");
      }
    }

    const onUnhandledRejection = (ev: PromiseRejectionEvent) => {
      if (cancelled) return;
      setErrorMessage(`Unhandled promise rejection: ${formatUnknownError(ev.reason)}`);
      setStatus("error");
    };

    const onWindowError = (ev: ErrorEvent) => {
      if (cancelled) return;
      const details = ev.error ? formatUnknownError(ev.error) : `${ev.message}`;
      setErrorMessage(`Window error: ${details}`);
      setStatus("error");
    };

    window.addEventListener("unhandledrejection", onUnhandledRejection);
    window.addEventListener("error", onWindowError);

    start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);

      window.removeEventListener("unhandledrejection", onUnhandledRejection);
      window.removeEventListener("error", onWindowError);

      if (stream) {
        for (const t of stream.getTracks()) t.stop();
      }

      if (landmarker) {
        landmarker.close();
      }
    };
  }, []);

  return (
    <section
      style={{
        display: "grid",
        gridTemplateColumns: "1fr",
        gap: 12,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(220px, 1fr) auto",
          gap: 12,
          alignItems: "center",
          justifyContent: "space-between",
          padding: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 12,
          background: "rgba(255,255,255,0.03)",
        }}
      >
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 220 }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Exercise</div>
              <select
                value={exercise}
                onChange={(e) => setExercise(e.target.value as ExerciseId)}
                disabled={planState.active}
                style={{
                  background: "rgba(255,255,255,0.06)",
                  color: "#e6edf6",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: "10px 12px",
                  fontSize: 14,
                  outline: "none",
                  opacity: planState.active ? 0.7 : 1,
                }}
              >
                <option value="jumping_jacks">Jumping jacks</option>
                <option value="squats">Squats</option>
                <option value="lunges">Lunges</option>
                <option value="high_knees">High knees</option>
                <option value="jump_squats">Jump squats</option>
                <option value="burpees">Burpees</option>
              </select>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 220 }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Mode</div>
              <select
                value={planMode}
                onChange={(e) => {
                  const v = e.target.value as "free" | "plan";
                  setPlanMode(v);
                  if (v === "free") setPlanState((s) => ({ ...s, active: false }));
                }}
                disabled={planState.active}
                style={{
                  background: "rgba(255,255,255,0.06)",
                  color: "#e6edf6",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: "10px 12px",
                  fontSize: 14,
                  outline: "none",
                  opacity: planState.active ? 0.7 : 1,
                }}
              >
                <option value="free">Free workout</option>
                <option value="plan">Guided plan</option>
              </select>
            </div>

            {planMode === "plan" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 260 }}>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>Plan</div>
                <select
                  value={selectedPlanId}
                  onChange={(e) => setSelectedPlanId(e.target.value)}
                  disabled={planState.active}
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#e6edf6",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 10,
                    padding: "10px 12px",
                    fontSize: 14,
                    outline: "none",
                    opacity: planState.active ? 0.7 : 1,
                  }}
                >
                  {workoutPlans.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              type="button"
              onClick={() => setSessionRunning((s) => !s)}
              style={{
                background: sessionRunning ? "rgba(255,255,255,0.06)" : "rgba(60, 242, 176, 0.14)",
                color: "#e6edf6",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: 14,
                cursor: "pointer",
                alignSelf: "end",
              }}
            >
              {sessionRunning ? "Pause" : "Resume"}
            </button>

            {planMode === "plan" && (
              <button
                type="button"
                onClick={() => {
                  if (planState.active) {
                    setPlanState((s) => ({ ...s, active: false }));
                    showToast("Plan stopped");
                    return;
                  }

                  setPlanState({
                    active: true,
                    planId: selectedPlanId,
                    stepIndex: 0,
                    stepStartedAt: Date.now(),
                    stepStartReps: repState.repCount,
                  });
                  showToast("Plan started");
                }}
                style={{
                  background: planState.active ? "rgba(255, 80, 80, 0.12)" : "rgba(60, 242, 176, 0.14)",
                  color: "#e6edf6",
                  border: planState.active
                    ? "1px solid rgba(255, 80, 80, 0.28)"
                    : "1px solid rgba(60, 242, 176, 0.35)",
                  borderRadius: 10,
                  padding: "10px 12px",
                  fontSize: 14,
                  cursor: "pointer",
                  alignSelf: "end",
                  fontWeight: 800,
                }}
              >
                {planState.active ? "Stop plan" : "Start plan"}
              </button>
            )}

            <button
              type="button"
              onClick={() =>
                setRepState((s) => ({
                  ...s,
                  repCount: 0,
                  phase: "unknown",
                  lastPhaseChangeMs: 0,
                  lastRepMs: 0,
                  reachedTarget: false,
                  lastSide: "none",
                  feedback: "",
                }))
              }
              style={{
                background: "rgba(255,255,255,0.06)",
                color: "#e6edf6",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: 14,
                cursor: "pointer",
                alignSelf: "end",
              }}
            >
              Reset
            </button>
          </div>

          <div
            style={{
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12,
              padding: "10px 12px",
              background: "rgba(0,0,0,0.18)",
              color: repState.feedback ? "#d6ffe9" : "#a7b4c7",
              fontSize: 13,
              minHeight: 42,
              display: "flex",
              alignItems: "center",
            }}
          >
            {repState.feedback || "Move into frame to begin."}
          </div>

          {planState.active && activePlanStep && (
            <div
              style={{
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12,
                padding: 12,
                background: "rgba(255,255,255,0.02)",
                display: "grid",
                gap: 8,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>Guided plan</div>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>{`${planState.stepIndex + 1}/${activePlan.steps.length}`}</div>
              </div>
              {activePlanStep.kind === "work_reps" ? (
                <div style={{ display: "grid", gap: 6 }}>
                  <div style={{ fontWeight: 800 }}>{`${activePlanStep.label}: ${activePlanStep.targetReps} reps`}</div>
                  <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                    {`Progress: ${Math.max(0, repState.repCount - planState.stepStartReps)}/${activePlanStep.targetReps}`}
                  </div>
                  {activePlanStep.exercise === "burpees" && (
                    <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                      Burpees track best with a stable camera and full body visible.
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ display: "grid", gap: 6 }}>
                  <div style={{ fontWeight: 800 }}>{activePlanStep.label}</div>
                  <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                    {`Time left: ${Math.max(0, Math.ceil(activePlanStep.restSec - (planNowMs - planState.stepStartedAt) / 1000))}s`}
                  </div>
                </div>
              )}
            </div>
          )}

          <div
            style={{
              display: "grid",
              gap: 10,
              gridTemplateColumns: "1fr",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12,
              padding: 12,
              background: "rgba(255,255,255,0.02)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Calibration</div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button
                  type="button"
                  onClick={() => {
                    setManualCalibStep(0);
                    setManualCalibOpen(true);
                  }}
                  disabled={exercise === "burpees"}
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#e6edf6",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 10,
                    padding: "8px 10px",
                    fontSize: 13,
                    cursor: "pointer",
                    opacity: exercise === "burpees" ? 0.6 : 1,
                  }}
                >
                  Manual calibrate
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setCalibration((c) => ({
                      ...c,
                      [exercise]: undefined,
                    }))
                  }
                  disabled={exercise === "burpees"}
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#e6edf6",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 10,
                    padding: "8px 10px",
                    fontSize: 13,
                    cursor: "pointer",
                    opacity: exercise === "burpees" ? 0.6 : 1,
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
            {exercise === "burpees" ? (
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Burpees don’t require calibration.</div>
            ) : autoCalib.active ? (
              <div style={{ display: "grid", gap: 8 }}>
                <div style={{ fontSize: 12, color: "#d6ffe9" }}>Auto-calibrating…</div>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>{autoCalibHint}</div>
                <div
                  style={{
                    height: 8,
                    borderRadius: 999,
                    overflow: "hidden",
                    border: "1px solid rgba(255,255,255,0.10)",
                    background: "rgba(0,0,0,0.18)",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.round((Math.min(900, autoCalib.stableMs) / 900) * 100)}%`,
                      background: "rgba(60, 242, 176, 0.55)",
                    }}
                  />
                </div>
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                  Hold still for ~1 second. This runs automatically when calibration is missing.
                </div>
              </div>
            ) : (
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                Calibrated. Use Manual calibrate if you want to refine it.
              </div>
            )}
          </div>

          {manualCalibOpen && (
            <div
              role="dialog"
              aria-modal="true"
              style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0,0,0,0.55)",
                display: "grid",
                placeItems: "center",
                padding: 16,
                zIndex: 50,
              }}
              onClick={() => setManualCalibOpen(false)}
            >
              <div
                className="card"
                style={{
                  width: "min(680px, 100%)",
                  padding: 16,
                  display: "grid",
                  gap: 12,
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                  <div style={{ display: "grid", gap: 4 }}>
                    <div style={{ fontSize: 12, color: "#a7b4c7" }}>Manual calibration</div>
                    <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.2 }}>
                      {exercise.replace("_", " ")}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setManualCalibOpen(false)}
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      color: "#e6edf6",
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: 10,
                      padding: "8px 10px",
                      fontSize: 13,
                      cursor: "pointer",
                    }}
                  >
                    Close
                  </button>
                </div>

                <div style={{ border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, padding: 12 }}>
                  <div style={{ fontWeight: 800 }}>{`Step ${manualCalibStep + 1}/2: ${calibSteps[manualCalibStep].title}`}</div>
                  <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
                    {calibSteps[manualCalibStep].hint}
                  </div>
                </div>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <button
                    type="button"
                    onClick={() => {
                      captureCalibrationFrame(manualCalibStep);
                      if (manualCalibStep === 0) setManualCalibStep(1);
                      else setManualCalibOpen(false);
                    }}
                    style={{
                      background: "rgba(60, 242, 176, 0.14)",
                      color: "#e6edf6",
                      border: "1px solid rgba(60, 242, 176, 0.35)",
                      borderRadius: 12,
                      padding: "10px 12px",
                      fontSize: 14,
                      cursor: "pointer",
                      fontWeight: 800,
                    }}
                  >
                    {calibSteps[manualCalibStep].actionLabel}
                  </button>

                  <button
                    type="button"
                    onClick={() => setManualCalibStep((s) => (s === 0 ? 1 : 0))}
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      color: "#e6edf6",
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: 12,
                      padding: "10px 12px",
                      fontSize: 14,
                      cursor: "pointer",
                    }}
                  >
                    {manualCalibStep === 0 ? "Skip to step 2" : "Back to step 1"}
                  </button>
                </div>

                <div className="muted" style={{ fontSize: 12 }}>
                  Tip: Keep your full body in frame and hold the position still for 1 second before capturing.
                </div>
              </div>
            </div>
          )}

          <div
            style={{
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12,
              padding: 12,
              background: "rgba(255,255,255,0.02)",
              display: "grid",
              gap: 10,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Rep log</div>
              <button
                type="button"
                onClick={() => setEvents([])}
                style={{
                  background: "rgba(255,255,255,0.06)",
                  color: "#e6edf6",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: "6px 10px",
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                Clear log
              </button>
            </div>

            <div style={{ display: "grid", gap: 8, maxHeight: 170, overflow: "auto" }}>
              {events.length === 0 ? (
                <div style={{ fontSize: 12, color: "#a7b4c7" }}>No events yet.</div>
              ) : (
                events.map((ev) => (
                  <div
                    key={ev.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 10,
                      fontSize: 12,
                      color: ev.kind === "rep" ? "#d6ffe9" : "#ffd0d0",
                      border: "1px solid rgba(255,255,255,0.08)",
                      background: "rgba(0,0,0,0.18)",
                      borderRadius: 10,
                      padding: "8px 10px",
                    }}
                  >
                    <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ev.kind === "rep" ? `Rep ${ev.reps}` : "Rejected"}: {ev.message}
                    </div>
                    <div style={{ color: "#a7b4c7", flex: "0 0 auto" }}>
                      {new Date(ev.ts).toLocaleTimeString()}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {toast && (
          <div
            style={{
              position: "fixed",
              right: 16,
              bottom: 16,
              zIndex: 60,
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(0,0,0,0.7)",
              color: "#e6edf6",
              padding: "10px 12px",
              fontSize: 13,
              maxWidth: 280,
            }}
          >
            {toast}
          </div>
        )}

        <div style={{ display: "grid", gap: 8, justifyItems: "end" }}>
          <div style={{ display: "grid", gridTemplateColumns: "auto auto", gap: 18 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Reps</div>
              <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: -0.2 }}>
                {repState.repCount}
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>Phase</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{displayPhase}</div>
            </div>
          </div>

          <div style={{ fontSize: 12, color: "#a7b4c7" }}>Status: {status}</div>
        </div>
      </div>

      {status === "error" && (
        <div
          style={{
            padding: 12,
            borderRadius: 12,
            border: "1px solid rgba(255, 80, 80, 0.35)",
            background: "rgba(255, 80, 80, 0.08)",
            color: "#ffd0d0",
          }}
        >
          {errorMessage ?? "Unknown error"}
        </div>
      )}

      <div
        style={{
          position: "relative",
          width: "100%",
          borderRadius: 16,
          overflow: "hidden",
          border: "1px solid rgba(255,255,255,0.08)",
          background: "#05070c",
        }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: "100%",
            height: "auto",
            transform: "scaleX(-1)",
            display: "block",
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            transform: "scaleX(-1)",
          }}
        />

        {autoCalib.active && (
          <div
            style={{
              position: "absolute",
              left: 12,
              top: 12,
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(0,0,0,0.55)",
              padding: "10px 12px",
              color: "#e6edf6",
              maxWidth: 340,
              display: "grid",
              gap: 8,
            }}
          >
            <div style={{ fontWeight: 800, fontSize: 13 }}>Auto-calibrating</div>
            <div style={{ fontSize: 12, color: "#a7b4c7", lineHeight: 1.4 }}>{autoCalibHint}</div>
            <div
              style={{
                height: 8,
                borderRadius: 999,
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.10)",
                background: "rgba(0,0,0,0.18)",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${Math.round((Math.min(900, autoCalib.stableMs) / 900) * 100)}%`,
                  background: "rgba(60, 242, 176, 0.55)",
                }}
              />
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          color: "#a7b4c7",
          fontSize: 13,
          lineHeight: 1.5,
          padding: "0 4px",
        }}
      >
        Tip: If reps don’t count, step back so your full body is visible and keep the camera stable.
      </div>
    </section>
  );
}
