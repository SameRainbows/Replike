import React from "react";
import type { RepEvent } from "../exercises/types";

interface RepLogProps {
    events: RepEvent[];
    onClear: () => void;
}

export function RepLog({ events, onClear }: RepLogProps) {
    return (
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
                    onClick={onClear}
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
                                color: ev.kind === "rep" ? "#ededec" : "#ffd0d0",
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
    );
}
