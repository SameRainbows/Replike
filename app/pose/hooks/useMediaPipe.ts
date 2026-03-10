import { useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

export function useMediaPipe(
    videoRef: React.RefObject<HTMLVideoElement>,
    canvasRef: React.RefObject<HTMLCanvasElement>,
    onPoseResult: (result: any, nowMs: number) => void,
    onStatusChange: (status: "init" | "loading" | "running" | "error", err?: string) => void
) {
    const [landmarker, setLandmarker] = useState<PoseLandmarker | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const rAFRef = useRef<number>(0);
    const sessionRunningRef = useRef(false);

    const onPoseResultRef = useRef(onPoseResult);
    useEffect(() => {
        onPoseResultRef.current = onPoseResult;
    }, [onPoseResult]);

    // Expose a setter ref so we can toggle from parent without re-rendering everything
    const setSessionRunning = (val: boolean) => {
        sessionRunningRef.current = val;
    };

    useEffect(() => {
        let active = true;
        let pm: PoseLandmarker | null = null;
        let fallbackTimer: any;

        async function init() {
            try {
                onStatusChange("loading");

                fallbackTimer = setTimeout(() => {
                    if (active && !pm) onStatusChange("error", "Model load is taking a long time...");
                }, 10000);

                const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
                pm = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                        delegate: "GPU",
                    },
                    runningMode: "VIDEO",
                    numPoses: 1,
                    minPoseDetectionConfidence: 0.5,
                    minPosePresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                if (!active) {
                    pm?.close();
                    return;
                }

                clearTimeout(fallbackTimer);
                setLandmarker(pm);

                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false,
                });

                if (!active || !videoRef.current) {
                    stream.getTracks().forEach((t) => t.stop());
                    return;
                }

                streamRef.current = stream;
                const video = videoRef.current;
                video.srcObject = stream;

                await new Promise<void>((resolve, reject) => {
                    video.onloadedmetadata = () => resolve();
                    video.onerror = (e) => reject(e);
                });

                if (!active) return;
                video.play().catch(() => { });
                onStatusChange("running");

                // The render loop
                const loop = () => {
                    if (!active) return;
                    rAFRef.current = requestAnimationFrame(loop);

                    if (!video || video.readyState < 2 || !pm) return;

                    const nowMs = performance.now();
                    pm.detectForVideo(video, nowMs, (result) => {
                        if (!active) return;

                        // Draw to canvas
                        if (canvasRef.current) {
                            const canvas = canvasRef.current;
                            const ctx = canvas.getContext("2d");
                            if (ctx) {
                                if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                                    canvas.width = video.videoWidth;
                                    canvas.height = video.videoHeight;
                                }
                                ctx.clearRect(0, 0, canvas.width, canvas.height);

                                if (result.landmarks && result.landmarks.length > 0) {
                                    const drawingUtils = new DrawingUtils(ctx);
                                    for (const pose of result.landmarks) {
                                        drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {
                                            color: "rgba(60, 242, 176, 0.4)",
                                            lineWidth: 2,
                                        });
                                        drawingUtils.drawLandmarks(pose, {
                                            color: "rgba(60, 242, 176, 0.9)",
                                            lineWidth: 2,
                                            radius: 3,
                                        });
                                    }
                                }
                            }
                        }

                        // Only pass to state machines if running
                        if (sessionRunningRef.current) {
                            onPoseResultRef.current(result, Date.now());
                        }
                    });
                };

                rAFRef.current = requestAnimationFrame(loop);

            } catch (err: any) {
                if (!active) return;
                clearTimeout(fallbackTimer);
                onStatusChange("error", err?.message || "Failed to initialize camera or model.");
            }
        }

        init();

        return () => {
            active = false;
            clearTimeout(fallbackTimer);
            cancelAnimationFrame(rAFRef.current);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((t) => t.stop());
            }
            pm?.close();
        };
    }, []);

    return { setSessionRunning };
}
