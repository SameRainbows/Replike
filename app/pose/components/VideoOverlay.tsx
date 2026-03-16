import React from "react";

interface VideoOverlayProps {
    videoRef: React.RefObject<HTMLVideoElement>;
    canvasRef: React.RefObject<HTMLCanvasElement>;
    autoCalibActive: boolean;
    calibrationEnabled: boolean;
    autoCalibHint: string;
    autoCalibStableMs: number;
    repCount?: number;
    displayPhase?: string;
}

export function VideoOverlay({
    videoRef,
    canvasRef,
    autoCalibActive,
    calibrationEnabled,
    autoCalibHint,
    autoCalibStableMs,
    repCount,
    displayPhase,
}: VideoOverlayProps) {
    return (
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
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                    transform: "scaleX(-1)",
                }}
            />

            {autoCalibActive && calibrationEnabled && (
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
                                width: `${Math.round((Math.min(900, autoCalibStableMs) / 900) * 100)}%`,
                                background: "rgba(232, 132, 94, 0.55)",
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Rep Counter HUD overlay */}
            {typeof repCount === "number" && displayPhase && (
                <div
                    style={{
                        position: "absolute",
                        top: 16,
                        right: 16,
                        background: "rgba(10, 10, 15, 0.5)",
                        backdropFilter: "blur(12px)",
                        border: "1px solid rgba(255, 255, 255, 0.12)",
                        borderRadius: 16,
                        padding: "16px 24px",
                        display: "flex",
                        alignItems: "center",
                        gap: 24,
                        color: "var(--text)",
                        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
                    }}
                >
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                        <div style={{ fontSize: 13, color: "rgba(255,255,255,0.7)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Phase</div>
                        <div style={{ fontSize: 20, fontWeight: 700 }}>{displayPhase}</div>
                    </div>
                    
                    <div style={{ width: 1, height: 40, background: "rgba(255,255,255,0.15)" }} />

                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
                        <div style={{ fontSize: 13, color: "rgba(255,255,255,0.7)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Reps</div>
                        <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: -1, lineHeight: 1, color: "var(--accent)" }}>
                            {repCount}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
