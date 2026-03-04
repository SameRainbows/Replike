import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { ExerciseId, TrackingHealth } from "./types";

export function randomId(prefix: string) {
    const rnd = Math.random().toString(16).slice(2);
    return `${prefix}_${Date.now().toString(16)}_${rnd}`;
}

export function newId() {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function formatUnknownError(e: unknown) {
    if (e instanceof Error) {
        return `${e.name}: ${e.message}`;
    }

    if (typeof e === "object" && e !== null) {
        const maybeDomEx = e as { name?: unknown; message?: unknown; code?: unknown };
        if (typeof maybeDomEx.name === "string" || typeof maybeDomEx.message === "string") {
            return `${String(maybeDomEx.name ?? "Error")}: ${String(maybeDomEx.message ?? "")}`.trim();
        }
    }

    if (typeof Event !== "undefined" && e instanceof Event) {
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

export function clamp01(x: number) {
    return Math.min(1, Math.max(0, x));
}

export function parsePctFromText(text: string) {
    const m = /([0-9]{1,3})%/.exec(text);
    if (!m) return null;
    const n = Number(m[1]);
    if (!Number.isFinite(n)) return null;
    return Math.max(0, Math.min(100, n));
}

export function parseAllPcts(text: string) {
    const matches = text.match(/([0-9]{1,3})%/g) ?? [];
    return matches
        .map((t) => Number(t.replace("%", "")))
        .filter((n) => Number.isFinite(n))
        .map((n) => Math.max(0, Math.min(100, n)));
}

export function dist(a: NormalizedLandmark, b: NormalizedLandmark) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.hypot(dx, dy);
}

export function angleDeg(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark) {
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

export function avg(a: number, b: number) {
    return (a + b) / 2;
}

export function getLandmark(landmarks: NormalizedLandmark[], idx: number) {
    const lm = landmarks[idx];
    if (!lm) return null;
    return lm;
}

export function isLandmarkConfident(lm: NormalizedLandmark | null, minVis = 0.5) {
    if (!lm) return false;
    if (typeof lm.visibility !== "number") return true;
    return lm.visibility >= minVis;
}

export function smoothLandmarks(
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

export function computeTrackingHealth(exercise: ExerciseId, landmarks: NormalizedLandmark[] | null, fps: number): TrackingHealth {
    if (!landmarks || landmarks.length === 0) {
        return {
            level: "lost",
            fps,
            hint: "No pose detected. Step into frame.",
            missing: ["upper", "lower"],
        };
    }

    const lShoulder = getLandmark(landmarks, 11);
    const rShoulder = getLandmark(landmarks, 12);
    const lHip = getLandmark(landmarks, 23);
    const rHip = getLandmark(landmarks, 24);
    const lKnee = getLandmark(landmarks, 25);
    const rKnee = getLandmark(landmarks, 26);
    const lAnkle = getLandmark(landmarks, 27);
    const rAnkle = getLandmark(landmarks, 28);
    const lWrist = getLandmark(landmarks, 15);
    const rWrist = getLandmark(landmarks, 16);
    const nose = getLandmark(landmarks, 0);

    const upperOk = isLandmarkConfident(lShoulder, 0.35) && isLandmarkConfident(rShoulder, 0.35) && isLandmarkConfident(lHip, 0.35) && isLandmarkConfident(rHip, 0.35);
    const lowerOk = isLandmarkConfident(lKnee, 0.35) && isLandmarkConfident(rKnee, 0.35) && isLandmarkConfident(lAnkle, 0.25) && isLandmarkConfident(rAnkle, 0.25);

    const armsOk =
        exercise === "jumping_jacks" || exercise === "burpees" || exercise === "push_ups"
            ? isLandmarkConfident(lWrist, 0.25) && isLandmarkConfident(rWrist, 0.25)
            : true;

    const headOk =
        exercise === "jumping_jacks"
            ? isLandmarkConfident(nose, 0.25)
            : true;

    const missing: TrackingHealth["missing"] = [];
    if (!upperOk) missing.push("upper");
    if (!lowerOk) missing.push("lower");
    if (!armsOk) missing.push("arms");
    if (!headOk) missing.push("head");

    if (missing.length === 0) {
        return { level: "good", fps, hint: "Tracking looks good.", missing };
    }

    const hint =
        missing.includes("lower")
            ? "Show your full body (knees/ankles). Step back or tilt the camera down."
            : missing.includes("upper")
                ? "Show your full upper body (shoulders/hips). Step back or raise the camera."
                : missing.includes("arms")
                    ? "Arms are hard to see. Improve lighting and keep wrists in frame."
                    : "Move into frame and improve lighting.";

    return {
        level: missing.length >= 2 ? "lost" : "partial",
        fps,
        hint,
        missing,
    };
}
