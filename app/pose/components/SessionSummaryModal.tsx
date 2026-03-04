import React from "react";
import type { QualityAgg } from "../exercises/types";

interface SessionSummaryModalProps {
    lastSummary: QualityAgg | null;
    onClose: () => void;
}

export function SessionSummaryModal({ lastSummary, onClose }: SessionSummaryModalProps) {
    if (!lastSummary) return null;

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
                        <div style={{ fontSize: 12, color: "#a7b4c7" }}>Session summary</div>
                        <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: -0.2 }}>Rep quality</div>
                    </div>
                    <button type="button" className="btn" onClick={onClose}>
                        Close
                    </button>
                </div>

                <div className="surface--inset" style={{ padding: 12, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 800, fontSize: 13 }}>Totals</div>
                    <div className="muted" style={{ fontSize: 13 }}>
                        {`${lastSummary.clean} clean · ${lastSummary.ok} ok · ${lastSummary.sloppy} sloppy`}
                        {typeof lastSummary.romSum === "number" && lastSummary.romCount > 0
                            ? ` · Avg ROM ${Math.round(lastSummary.romSum / lastSummary.romCount)}%`
                            : ""}
                    </div>
                </div>

                <div className="surface--inset" style={{ padding: 12, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 800, fontSize: 13 }}>By exercise</div>
                    {Object.keys(lastSummary.byExercise || {}).length === 0 ? (
                        <div className="muted" style={{ fontSize: 13 }}>
                            No quality data recorded.
                        </div>
                    ) : (
                        <div style={{ display: "grid", gap: 8 }}>
                            {Object.entries(lastSummary.byExercise)
                                .sort((a, b) => (b[1].clean + b[1].ok + b[1].sloppy) - (a[1].clean + a[1].ok + a[1].sloppy))
                                .map(([ex, q]) => (
                                    <div
                                        key={ex}
                                        style={{
                                            display: "flex",
                                            justifyContent: "space-between",
                                            gap: 12,
                                            alignItems: "baseline",
                                        }}
                                    >
                                        <div style={{ fontWeight: 800, fontSize: 13 }}>{ex.replaceAll("_", " ")}</div>
                                        <div className="muted" style={{ fontSize: 13, textAlign: "right" }}>
                                            {`${q.clean} clean · ${q.ok} ok · ${q.sloppy} sloppy`}
                                            {typeof q.romSum === "number" && q.romCount > 0
                                                ? ` · Avg ROM ${Math.round(q.romSum / q.romCount)}%`
                                                : ""}
                                        </div>
                                    </div>
                                ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
