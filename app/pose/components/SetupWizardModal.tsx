import React from "react";
import type { TrackingHealth } from "../exercises/types";

interface SetupWizardModalProps {
    trackingHealth: TrackingHealth;
    onClose: () => void;
}

export function SetupWizardModal({ trackingHealth, onClose }: SetupWizardModalProps) {
    return (
        <div
            role="dialog"
            aria-modal="true"
            className="modalBackdrop"
            onClick={onClose}
        >
            <div className="modalCard" onClick={(e) => e.stopPropagation()}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                    <div style={{ display: "grid", gap: 4 }}>
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Setup wizard</div>
                        <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.2 }}>Make tracking rock solid</div>
                    </div>
                    <button type="button" className="btn" onClick={onClose}>
                        Close
                    </button>
                </div>

                <div className="surface--inset" style={{ padding: 12, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 800, fontSize: 13 }}>Live status</div>
                    <div className="muted" style={{ fontSize: 13 }}>
                        {trackingHealth.level === "good"
                            ? "Tracking looks good. You’re ready to go."
                            : trackingHealth.level === "partial"
                                ? "Tracking is partial — the model is losing some landmarks."
                                : "Tracking is lost — the model can’t see a stable pose."}
                    </div>
                    <div className="muted" style={{ fontSize: 12 }}>{`FPS: ${Math.round(trackingHealth.fps)}`}</div>
                </div>

                <div className="surface--inset" style={{ padding: 12, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 800, fontSize: 13 }}>Checklist</div>
                    <div className="muted" style={{ fontSize: 13 }}>Do these in order for the fastest fix:</div>
                    <div style={{ display: "grid", gap: 6, fontSize: 13 }}>
                        <div style={{ color: "rgba(230, 237, 246, 0.92)" }}>1) Step back until your full body is visible</div>
                        <div style={{ color: "rgba(230, 237, 246, 0.92)" }}>2) Improve lighting (face the light, avoid backlight)</div>
                        <div style={{ color: "rgba(230, 237, 246, 0.92)" }}>3) Keep the camera stable (no wobble)</div>
                    </div>
                </div>

                <div className="surface--inset" style={{ padding: 12, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 800, fontSize: 13 }}>What to fix right now</div>
                    <div className="muted" style={{ fontSize: 13 }}>{trackingHealth.hint}</div>
                </div>
            </div>
        </div>
    );
}
