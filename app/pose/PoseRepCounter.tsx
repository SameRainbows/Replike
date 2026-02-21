"use client";

import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useEffect, useMemo, useRef, useState } from "react";

type ExerciseId = "jumping_jacks" | "squats";

type RepState = {
  exercise: ExerciseId;
  repCount: number;
  phase: "unknown" | "closed" | "open" | "up" | "down";
  lastPhaseChangeMs: number;
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

  let nextPhase = prev.phase;
  if (prev.phase === "unknown") {
    nextPhase = open ? "open" : "closed";
  } else if (canSwitch) {
    if (prev.phase === "closed" && open) nextPhase = "open";
    if (prev.phase === "open" && closed) nextPhase = "closed";
  }

  let repCount = prev.repCount;
  // Count a rep when returning to closed after being open
  if (prev.phase === "open" && nextPhase === "closed") {
    repCount += 1;
  }

  return {
    ...prev,
    repCount,
    phase: nextPhase,
    lastPhaseChangeMs: nextPhase !== prev.phase ? nowMs : prev.lastPhaseChangeMs,
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
  // Count when returning to up from down
  if (prev.phase === "down" && nextPhase === "up") repCount += 1;

  return {
    ...prev,
    repCount,
    phase: nextPhase,
    lastPhaseChangeMs: nextPhase !== prev.phase ? nowMs : prev.lastPhaseChangeMs,
  };
}

export default function PoseRepCounter() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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
  }));

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
    }));
  }, [exercise]);

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
            const lms = result.landmarks[0];

            drawingUtils.drawLandmarks(lms, {
              radius: (data) => 2 + 3 * clamp01(data.from?.z ? 0 : 1),
              color: "#3cf2b0",
            });

            drawingUtils.drawConnectors(
              lms,
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
                };
              }

              if (exercise === "jumping_jacks") return updateJumpingJackState(prev, lms, nowMs);
              if (exercise === "squats") return updateSquatState(prev, lms, nowMs);
              return prev;
            });
          } else {
            setRepState((prev) => ({
              ...prev,
              phase: "unknown",
            }));
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
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          alignItems: "center",
          justifyContent: "space-between",
          padding: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 12,
          background: "rgba(255,255,255,0.03)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
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
          </select>
        </div>

        <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontSize: 12, color: "#a7b4c7" }}>Reps</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{repState.repCount}</div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontSize: 12, color: "#a7b4c7" }}>Phase</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{displayPhase}</div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontSize: 12, color: "#a7b4c7" }}>Status</div>
            <div style={{ fontSize: 12 }}>{status}</div>
          </div>

          <button
            type="button"
            onClick={() =>
              setRepState((s) => ({
                ...s,
                repCount: 0,
                phase: "unknown",
                lastPhaseChangeMs: 0,
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
            }}
          >
            Reset
          </button>
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
        Tip: If reps don’t count, step back so your full body (ankles and wrists) is visible.
      </div>
    </section>
  );
}
