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
  | "pull_ups"
  | "chin_ups";

type RepState = {
  exercise: ExerciseId;
  repCount: number;
  phase: "unknown" | "closed" | "open" | "up" | "down";
  lastPhaseChangeMs: number;
  lastRepMs: number;
  reachedTarget: boolean;
  lastSide: "left" | "right" | "none";
  feedback: string;
};

type BarLine = {
  // Normalized (0..1) screen-space coordinates relative to the video.
  // We treat the bar as horizontal and use y as the main signal.
  y: number;
  x1: number;
  x2: number;
};

type BarAutoState =
  | { status: "idle" }
  | { status: "sampling"; samples: number; startedMs: number }
  | { status: "done" };

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

function clamp(x: number, min: number, max: number) {
  return Math.min(max, Math.max(min, x));
}

function median(values: number[]) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid]!;
  return (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function clamp01(x: number) {
  return Math.min(1, Math.max(0, x));
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
  nowMs: number
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
  const armsUp = wristsY < headY - 0.03;

  // Legs "open" if feet are spread wide relative to shoulders
  // Use hysteresis thresholds
  const legsOpenEnter = 1.45;
  const legsOpenExit = 1.25;

  const legsOpen =
    prev.phase === "open"
      ? ankleToShoulderRatio > legsOpenExit
      : ankleToShoulderRatio > legsOpenEnter;

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

  if (nextPhase === "open") reachedTarget = true;

  if (open) {
    feedback = "Good. Fully extend arms and step wide.";
  } else {
    feedback = "Raise arms above head and step wider to count.";
  }

  if (prev.phase === "open" && nextPhase === "closed") {
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
    feedback,
  };
}

function updateSquatState(prev: RepState, landmarks: NormalizedLandmark[], nowMs: number): RepState {
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

  // Hysteresis thresholds for squat depth
  const downEnter = 115;
  const downExit = 135;
  const upEnter = 165;
  const upExit = 155;
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

  if (nextPhase === "down") reachedTarget = true;

  if (kneeAngle < downEnter) {
    feedback = "Good depth. Drive up.";
  } else if (kneeAngle < 140) {
    feedback = "Go a bit lower for a full rep.";
  } else {
    feedback = "Stand tall, then squat down.";
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
    feedback,
  };
}

function updateLungeState(prev: RepState, landmarks: NormalizedLandmark[], nowMs: number): RepState {
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

  const downEnter = 120;
  const downExit = 140;
  const upEnter = 170;
  const upExit = 160;
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

  let feedback = prev.feedback;
  if (activeAngle < downEnter) feedback = `Good lunge (${activeSide}). Push back up.`;
  else if (activeAngle < 150) feedback = `Go a bit lower (${activeSide}) for a full rep.`;
  else feedback = "Take a longer step and lunge down.";

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
  };
}

function updateHighKneesState(prev: RepState, landmarks: NormalizedLandmark[], nowMs: number): RepState {
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
  const leftUp = lKnee!.y < hipY - 0.05;
  const rightUp = rKnee!.y < hipY - 0.05;

  const sideUp: "left" | "right" | "none" = leftUp && !rightUp ? "left" : rightUp && !leftUp ? "right" : "none";
  const minRepMs = 300;

  let repCount = prev.repCount;
  let lastRepMs = prev.lastRepMs;
  let lastSide = prev.lastSide;
  let feedback = prev.feedback;

  if (sideUp === "none") {
    feedback = "Drive one knee up above hip height.";
  } else {
    feedback = `Knee up (${sideUp}). Keep alternating.`;
  }

  if (sideUp !== "none") {
    const canCount = nowMs - lastRepMs > minRepMs;
    const alternated = lastSide === "none" || lastSide !== sideUp;
    if (canCount && alternated) {
      repCount += 1;
      lastRepMs = nowMs;
      lastSide = sideUp;
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
  };
}

function updateBarExerciseState(
  prev: RepState,
  landmarks: NormalizedLandmark[],
  nowMs: number,
  bar: BarLine | null,
  label: "Pull-ups" | "Chin-ups"
): RepState {
  if (!bar) {
    return {
      ...prev,
      phase: "unknown",
      feedback: `Set the bar line to start counting ${label.toLowerCase()}.`,
    };
  }

  const mouthL = getLandmark(landmarks, 9);
  const mouthR = getLandmark(landmarks, 10);
  const lWrist = getLandmark(landmarks, 15);
  const rWrist = getLandmark(landmarks, 16);
  const lShoulder = getLandmark(landmarks, 11);
  const rShoulder = getLandmark(landmarks, 12);
  const lElbow = getLandmark(landmarks, 13);
  const rElbow = getLandmark(landmarks, 14);

  const minVisible =
    isLandmarkConfident(mouthL, 0.35) &&
    isLandmarkConfident(mouthR, 0.35) &&
    isLandmarkConfident(lWrist, 0.35) &&
    isLandmarkConfident(rWrist, 0.35) &&
    isLandmarkConfident(lShoulder, 0.35) &&
    isLandmarkConfident(rShoulder, 0.35) &&
    isLandmarkConfident(lElbow, 0.35) &&
    isLandmarkConfident(rElbow, 0.35);

  if (!minVisible) {
    return {
      ...prev,
      phase: "unknown",
      feedback: "Make sure your face and arms are visible.",
    };
  }

  const mouthY = avg(mouthL!.y, mouthR!.y);
  const wristsY = avg(lWrist!.y, rWrist!.y);
  const wristNearBar = Math.abs(wristsY - bar.y) < 0.12;

  const lElbowAngle = angleDeg(lShoulder!, lElbow!, lWrist!);
  const rElbowAngle = angleDeg(rShoulder!, rElbow!, rWrist!);
  const elbowAngle = Math.min(lElbowAngle, rElbowAngle);

  const topEnter = 0.015;
  const topExit = 0.005;
  const chinAboveBar = prev.phase === "up" ? mouthY < bar.y - topExit : mouthY < bar.y - topEnter;

  const bottomEnter = 165;
  const bottomExit = 155;
  const elbowsExtended = prev.phase === "down" ? elbowAngle > bottomExit : elbowAngle > bottomEnter;

  const minRepMs = 900;
  const minPhaseMs = 200;
  const canSwitch = nowMs - prev.lastPhaseChangeMs > minPhaseMs;

  let nextPhase = prev.phase;
  if (prev.phase === "unknown") {
    nextPhase = chinAboveBar ? "up" : "down";
  } else if (canSwitch) {
    if (prev.phase === "down" && chinAboveBar && wristNearBar) nextPhase = "up";
    if (prev.phase === "up" && elbowsExtended && wristNearBar) nextPhase = "down";
  }

  let repCount = prev.repCount;
  let reachedTarget = prev.reachedTarget;
  let lastRepMs = prev.lastRepMs;
  let feedback = prev.feedback;

  if (!wristNearBar) {
    feedback = "Hands must stay on the bar (wrists near the bar line).";
  } else if (chinAboveBar) {
    feedback = "Top reached. Lower with control to full extension.";
  } else if (!elbowsExtended) {
    feedback = "Go to full extension at the bottom to count.";
  } else {
    feedback = "Pull until your chin clears the bar.";
  }

  if (nextPhase === "up") reachedTarget = true;

  if (prev.phase === "up" && nextPhase === "down") {
    if (reachedTarget && elbowsExtended && wristNearBar && nowMs - lastRepMs > minRepMs) {
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
    feedback,
  };
}

export default function PoseRepCounter() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const smoothedRef = useRef<NormalizedLandmark[] | null>(null);
  const lastLandmarksRef = useRef<NormalizedLandmark[] | null>(null);

  const [barLine, setBarLine] = useState<BarLine | null>(null);
  const [barDraft, setBarDraft] = useState<{ x: number; y: number } | null>(null);
  const [barAuto, setBarAuto] = useState<BarAutoState>({ status: "idle" });
  const barAutoSamplesRef = useRef<number[]>([]);

  const [exercise, setExercise] = useState<ExerciseId>("jumping_jacks");
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
  }));

  const needsBar = exercise === "pull_ups" || exercise === "chin_ups";

  const displayPhase = useMemo(() => {
    if (repState.phase === "unknown") return "No pose";
    if (repState.phase === "open") return "Open";
    if (repState.phase === "closed") return "Closed";
    if (repState.phase === "up") return "Up";
    return "Down";
  }, [repState.phase]);

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
    }));
    setBarDraft(null);
    setBarAuto({ status: "idle" });
    barAutoSamplesRef.current = [];
  }, [exercise]);

  useEffect(() => {
    if (!needsBar) {
      setBarDraft(null);
      setBarAuto({ status: "idle" });
      barAutoSamplesRef.current = [];
    }
  }, [needsBar]);

  useEffect(() => {
    // Ensure the "set bar" message goes away immediately after bar is set.
    if (!needsBar) return;
    if (!barLine) return;
    setRepState((s) => ({
      ...s,
      feedback: "",
      phase: s.phase === "unknown" ? "down" : s.phase,
    }));
  }, [barLine, needsBar]);

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

          // Resize canvas to match the displayed video size
          const w = videoEl.videoWidth;
          const h = videoEl.videoHeight;
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

          // Draw mirrored overlay to match mirrored video
          ctx.save();
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);

          if (result.landmarks && result.landmarks[0]) {
            const raw = result.landmarks[0];
            const smoothed = smoothLandmarks(smoothedRef.current, raw, 0.25);
            smoothedRef.current = smoothed;
            lastLandmarksRef.current = smoothed;

            // Auto bar sampling: collect wrist y for a short window and use median.
            if (needsBar && barAuto.status === "sampling") {
              const lWrist = getLandmark(smoothed, 15);
              const rWrist = getLandmark(smoothed, 16);
              if (isLandmarkConfident(lWrist, 0.35) && isLandmarkConfident(rWrist, 0.35)) {
                const wristsY = avg(lWrist!.y, rWrist!.y);
                barAutoSamplesRef.current.push(wristsY);
              }

              const elapsed = nowMs - barAuto.startedMs;
              const sampleCount = barAutoSamplesRef.current.length;
              if (elapsed > 900) {
                if (sampleCount >= 10) {
                  const y = clamp(median(barAutoSamplesRef.current) - 0.02, 0.02, 0.98);
                  setBarLine({ y, x1: 0.05, x2: 0.95 });
                  setBarAuto({ status: "done" });
                } else {
                  setBarAuto({ status: "idle" });
                }
                barAutoSamplesRef.current = [];
              } else {
                setBarAuto({ status: "sampling", samples: sampleCount, startedMs: barAuto.startedMs });
              }
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
              if (prev.exercise !== exercise) {
                return {
                  exercise,
                  repCount: 0,
                  phase: "unknown",
                  lastPhaseChangeMs: 0,
                  lastRepMs: 0,
                  reachedTarget: false,
                  lastSide: "none",
                  feedback: "",
                };
              }

              if (exercise === "jumping_jacks") return updateJumpingJackState(prev, smoothed, nowMs);
              if (exercise === "squats") return updateSquatState(prev, smoothed, nowMs);
              if (exercise === "lunges") return updateLungeState(prev, smoothed, nowMs);
              if (exercise === "high_knees") return updateHighKneesState(prev, smoothed, nowMs);
              if (exercise === "pull_ups")
                return updateBarExerciseState(prev, smoothed, nowMs, barLine, "Pull-ups");
              if (exercise === "chin_ups")
                return updateBarExerciseState(prev, smoothed, nowMs, barLine, "Chin-ups");
              return prev;
            });
          } else {
            setRepState((prev) => ({
              ...prev,
              phase: "unknown",
              feedback: "",
            }));
          }

          // Draw bar overlay if set
          if (barLine) {
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.strokeStyle = "rgba(255, 210, 80, 0.95)";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(barLine.x1 * canvas.width, barLine.y * canvas.height);
            ctx.lineTo(barLine.x2 * canvas.width, barLine.y * canvas.height);
            ctx.stroke();
            ctx.restore();
          }

          // Draw draft point while setting the bar
          if (barDraft) {
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.fillStyle = "rgba(255, 210, 80, 0.95)";
            ctx.beginPath();
            ctx.arc(barDraft.x * canvas.width, barDraft.y * canvas.height, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }

          ctx.restore();
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
  }, [exercise]);

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
                style={{
                  background: "rgba(255,255,255,0.06)",
                  color: "#e6edf6",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: "10px 12px",
                  fontSize: 14,
                  outline: "none",
                }}
              >
                <option value="jumping_jacks">Jumping jacks</option>
                <option value="squats">Squats</option>
                <option value="lunges">Lunges</option>
                <option value="high_knees">High knees</option>
                <option value="pull_ups">Pull-ups</option>
                <option value="chin_ups">Chin-ups</option>
              </select>
            </div>

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

          {needsBar && (
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                Bar: {barLine ? `set (y=${barLine.y.toFixed(3)})` : "not set"}
              </div>
              <button
                type="button"
                onClick={() => {
                  setBarLine(null);
                  setBarDraft(null);
                  setBarAuto({ status: "idle" });
                  barAutoSamplesRef.current = [];
                }}
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
                Clear bar
              </button>

              <button
                type="button"
                onClick={() => {
                  setBarDraft(null);
                  barAutoSamplesRef.current = [];
                  setBarAuto({ status: "sampling", samples: 0, startedMs: performance.now() });
                }}
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
                {barAuto.status === "sampling" ? `Auto-setting… (${barAuto.samples})` : "Auto-set bar"}
              </button>

              <div style={{ fontSize: 12, color: "#a7b4c7" }}>
                {barDraft
                  ? "Now click again to finish the bar."
                  : "Click the video twice to set the bar."}
              </div>
            </div>
          )}
        </div>

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
          cursor: needsBar ? "crosshair" : "default",
        }}
        onClick={(ev) => {
          if (!needsBar) return;
          const target = ev.currentTarget as HTMLDivElement;
          const rect = target.getBoundingClientRect();
          const x = clamp((ev.clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
          const y = clamp((ev.clientY - rect.top) / Math.max(rect.height, 1), 0, 1);

          // Clicking while auto-sampling cancels auto mode.
          if (barAuto.status === "sampling") {
            setBarAuto({ status: "idle" });
            barAutoSamplesRef.current = [];
          }

          if (!barDraft) {
            setBarDraft({ x, y });
            return;
          }

          const x1 = clamp(barDraft.x, 0, 1);
          const x2 = clamp(x, 0, 1);
          const lineY = clamp((barDraft.y + y) / 2, 0.02, 0.98);
          setBarLine({ y: lineY, x1: Math.min(x1, x2), x2: Math.max(x1, x2) });
          setBarDraft(null);
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
            width: "100%",
            height: "100%",
          }}
        />
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
