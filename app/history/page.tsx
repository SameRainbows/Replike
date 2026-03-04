"use client";

import { useRef, useEffect, useMemo, useState } from "react";
import { clearSessions, loadSessions, saveSessions, type WorkoutSession } from "@/app/lib/workoutHistory";

function formatDuration(sec: number) {
  const s = Math.max(0, Math.floor(sec));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<WorkoutSession[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setSessions(loadSessions());
  }, []);

  const handleExport = () => {
    if (sessions.length === 0) return;
    const data = JSON.stringify(sessions, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `repdetect_history_${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (Array.isArray(parsed)) {
          // Simplistic merge: prefer existing based on ID, add new ones
          const existingIds = new Set(sessions.map(s => s.id));
          const toAdd = parsed.filter(s => s.id && typeof s.totalReps === "number" && !existingIds.has(s.id));

          if (toAdd.length > 0) {
            const merged = [...sessions, ...toAdd].sort((a, b) => b.endedAt - a.endedAt);
            saveSessions(merged);
            setSessions(merged);
            alert(`Imported ${toAdd.length} new sessions.`);
          } else {
            alert("No new sessions found in file.");
          }
        }
      } catch (err) {
        alert("Failed to parse history file.");
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
    };
    reader.readAsText(file);
  };

  const hasSessions = sessions.length > 0;

  const rows = useMemo(() => {
    return sessions.map((s) => {
      const entries = Object.entries(s.repsByExercise || {}).filter(([, v]) => v > 0);
      entries.sort((a, b) => b[1] - a[1]);
      return { s, entries };
    });
  }, [sessions]);

  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">History</h1>
        <p className="lead">Saved sessions are stored locally in your browser.</p>
      </header>

      <div className="card" style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <div className="muted" style={{ fontSize: 13 }}>
          {hasSessions ? `${sessions.length} session${sessions.length === 1 ? "" : "s"}` : "No sessions yet."}
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn"
            style={{ height: 38, padding: "0 12px" }}
            onClick={() => fileInputRef.current?.click()}
          >
            Import data
          </button>
          <input
            type="file"
            accept=".json"
            ref={fileInputRef}
            onChange={handleImport}
            style={{ display: "none" }}
          />

          <button
            type="button"
            className="btn"
            style={{ height: 38, padding: "0 12px" }}
            onClick={handleExport}
            disabled={!hasSessions}
          >
            Export data
          </button>

          <button
            type="button"
            onClick={() => {
              if (!hasSessions) return;
              const ok = window.confirm("Clear all saved sessions?");
              if (!ok) return;
              clearSessions();
              setSessions([]);
            }}
            className="btn"
            style={{ height: 38, padding: "0 12px", border: "1px solid rgba(255, 80, 80, 0.28)", color: "#ffd0d0" }}
            disabled={!hasSessions}
          >
            Clear history
          </button>
        </div>
      </div>

      {rows.map(({ s, entries }) => (
        <div key={s.id} className="card stack">
          <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
            <div style={{ display: "grid", gap: 4 }}>
              <div style={{ fontWeight: 800, letterSpacing: -0.2 }}>
                {s.mode === "plan"
                  ? s.planName ?? "Guided plan"
                  : s.mode === "custom"
                    ? s.customWorkout?.name ?? "Custom workout"
                    : "Free workout"}
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                {new Date(s.endedAt).toLocaleString()}
              </div>
            </div>
            <div className="muted" style={{ fontSize: 12 }}>
              {formatDuration(s.durationSec)}
            </div>
          </div>

          <div className="grid grid--2">
            <div className="card stack" style={{ padding: 12, background: "rgba(0,0,0,0.12)" }}>
              <div className="card__title">Totals</div>
              <div className="muted" style={{ fontSize: 13 }}>
                Reps: {s.totalReps}
              </div>
              {s.quality && (
                <div className="muted" style={{ fontSize: 13 }}>
                  Quality: {s.quality.clean} clean · {s.quality.ok} ok · {s.quality.sloppy} sloppy
                  {typeof s.quality.avgRomPct === "number" ? ` · Avg ROM ${Math.round(s.quality.avgRomPct)}%` : ""}
                </div>
              )}
              {s.goal && (
                <div className="muted" style={{ fontSize: 13 }}>
                  Goal: {s.goal.targetReps} {s.goal.exercise.replaceAll("_", " ")} — {s.goal.reached ? "Reached" : "Not reached"}
                </div>
              )}
              <div className="muted" style={{ fontSize: 13 }}>
                Rejected: {s.totalRejects}
              </div>
            </div>

            <div className="card stack" style={{ padding: 12, background: "rgba(0,0,0,0.12)" }}>
              <div className="card__title">By exercise</div>
              {entries.length === 0 ? (
                <div className="muted" style={{ fontSize: 13 }}>
                  No reps recorded.
                </div>
              ) : (
                <div style={{ display: "grid", gap: 6 }}>
                  {entries.map(([k, v]) => (
                    <div key={k} className="muted" style={{ fontSize: 13, display: "flex", justifyContent: "space-between", gap: 10 }}>
                      <span>{k.replaceAll("_", " ")}</span>
                      <span style={{ color: "rgba(230, 237, 246, 0.92)" }}>{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}

      {!hasSessions && (
        <div className="card stack">
          <h2 className="h2">How to get sessions here</h2>
          <p className="p">Complete a guided plan (auto-saved), or save a free workout from the Workout page.</p>
        </div>
      )}
    </section>
  );
}
