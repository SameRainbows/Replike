import React from "react";

interface VideoOverlayProps {
    videoRef: React.RefObject<HTMLVideoElement>;
    canvasRef: React.RefObject<HTMLCanvasElement>;
    autoCalibActive: boolean;
    calibrationEnabled: boolean;
    autoCalibHint: string;
    autoCalibStableMs: number;
}

export function VideoOverlay({
    videoRef,
    canvasRef,
    autoCalibActive,
    calibrationEnabled,
    autoCalibHint,
    autoCalibStableMs,
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
                    inset: 0,
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
                                background: "rgba(60, 242, 176, 0.55)",
                            }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
