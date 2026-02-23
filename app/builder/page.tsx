"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  deleteCustomWorkout,
  loadCustomWorkouts,
  type CustomWorkout,
  type CustomWorkoutStep,
  upsertCustomWorkout,
} from "@/app/lib/customWorkouts";

function newId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function defaultWorkout(): CustomWorkout {
  const now = Date.now();
  return {
    id: `cw_${newId()}`,
    name: "My workout",
    rounds: 1,
    steps: [
      { kind: "work_reps", exercise: "squats", targetReps: 10, label: "Squats" },
      { kind: "rest", restSec: 30, label: "Rest" },
      { kind: "work_time", exercise: "jumping_jacks", workSec: 30, label: "Jumping jacks" },
    ],
    createdAt: now,
    updatedAt: now,
  };
}

function stepLabel(s: CustomWorkoutStep) {
  if (s.kind === "rest") return `${s.label} · ${s.restSec}s`;
  if (s.kind === "work_reps") return `${s.label} · ${s.targetReps} reps`;
  return `${s.label} · ${s.workSec}s`;
}

export default function BuilderPage() {
  const [workouts, setWorkouts] = useState<CustomWorkout[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    setWorkouts(loadCustomWorkouts());

    const onChange = () => setWorkouts(loadCustomWorkouts());
    window.addEventListener("storage", onChange);
    window.addEventListener("repdetect:customWorkouts", onChange);
    return () => {
      window.removeEventListener("storage", onChange);
      window.removeEventListener("repdetect:customWorkouts", onChange);
    };
  }, []);

  const active = useMemo(() => workouts.find((w) => w.id === activeId) ?? null, [workouts, activeId]);

  useEffect(() => {
    if (activeId) return;
    if (workouts.length > 0) setActiveId(workouts[0].id);
  }, [workouts, activeId]);

  function save(next: CustomWorkout) {
    const updated: CustomWorkout = { ...next, updatedAt: Date.now() };
    const all = upsertCustomWorkout(updated);
    setWorkouts(all);
    setActiveId(updated.id);
  }

  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">Workout Builder</h1>
        <p className="lead">Build custom workouts with reps, timed intervals, and rest. Stored locally in your browser.</p>
      </header>

      <div className="split">
        <div className="card stack">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
            <div className="card__title">Your workouts</div>
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => {
                const w = defaultWorkout();
                save(w);
              }}
            >
              New
            </button>
          </div>

          {workouts.length === 0 ? (
            <div className="muted" style={{ fontSize: 13 }}>
              No custom workouts yet.
            </div>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              {workouts.map((w) => (
                <button
                  key={w.id}
                  type="button"
                  onClick={() => setActiveId(w.id)}
                  className="card"
                  style={{
                    textAlign: "left",
                    padding: 12,
                    background: w.id === activeId ? "rgba(60, 242, 176, 0.08)" : "rgba(0,0,0,0.12)",
                    borderColor: w.id === activeId ? "rgba(60, 242, 176, 0.25)" : "rgba(255,255,255,0.08)",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ fontWeight: 800 }}>{w.name}</div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                    {w.rounds} round{w.rounds === 1 ? "" : "s"} · {w.steps.length} step{w.steps.length === 1 ? "" : "s"}
                  </div>
                </button>
              ))}
            </div>
          )}

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <Link className="btn" href="/workout">
              Back to Workout
            </Link>
            <Link className="btn" href="/history">
              History
            </Link>
          </div>
        </div>

        <div className="card stack">
          <div className="card__title">Editor</div>

          {!active ? (
            <div className="muted" style={{ fontSize: 13 }}>
              Create a workout to begin.
            </div>
          ) : (
            <div className="stack">
              <label style={{ display: "grid", gap: 6 }}>
                <div className="muted" style={{ fontSize: 12 }}>
                  Name
                </div>
                <input
                  value={active.name}
                  onChange={(e) => save({ ...active, name: e.target.value })}
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#e6edf6",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 12,
                    padding: "10px 12px",
                    fontSize: 14,
                    outline: "none",
                  }}
                />
              </label>

              <label style={{ display: "grid", gap: 6, maxWidth: 260 }}>
                <div className="muted" style={{ fontSize: 12 }}>
                  Rounds
                </div>
                <input
                  type="number"
                  min={1}
                  value={active.rounds}
                  onChange={(e) => {
                    const n = Math.max(1, Math.floor(Number(e.target.value || 1)));
                    save({ ...active, rounds: n });
                  }}
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#e6edf6",
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 12,
                    padding: "10px 12px",
                    fontSize: 14,
                    outline: "none",
                  }}
                />
              </label>

              <div className="card" style={{ padding: 12, background: "rgba(0,0,0,0.12)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                  <div style={{ fontWeight: 800 }}>Steps</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    <button
                      type="button"
                      className="btn"
                      onClick={() =>
                        save({
                          ...active,
                          steps: [...active.steps, { kind: "work_reps", exercise: "squats", targetReps: 10, label: "Squats" }],
                        })
                      }
                    >
                      Add reps
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() =>
                        save({
                          ...active,
                          steps: [...active.steps, { kind: "work_time", exercise: "jumping_jacks", workSec: 30, label: "Jumping jacks" }],
                        })
                      }
                    >
                      Add timed
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => save({ ...active, steps: [...active.steps, { kind: "rest", restSec: 30, label: "Rest" }] })}
                    >
                      Add rest
                    </button>
                  </div>
                </div>

                {active.steps.length === 0 ? (
                  <div className="muted" style={{ fontSize: 13, marginTop: 10 }}>
                    Add your first step.
                  </div>
                ) : (
                  <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
                    {active.steps.map((s, i) => (
                      <div
                        key={`${active.id}_${i}`}
                        className="card"
                        style={{ padding: 12, background: "rgba(255,255,255,0.02)" }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
                          <div style={{ display: "grid", gap: 2 }}>
                            <div style={{ fontWeight: 800 }}>{`${i + 1}. ${stepLabel(s)}`}</div>
                            <div className="muted" style={{ fontSize: 12 }}>
                              {s.kind === "rest" ? "Rest" : s.kind === "work_reps" ? "Work (reps)" : "Work (timed)"}
                            </div>
                          </div>

                          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            <button
                              type="button"
                              className="btn"
                              onClick={() => {
                                if (i === 0) return;
                                const next = [...active.steps];
                                const tmp = next[i - 1];
                                next[i - 1] = next[i];
                                next[i] = tmp;
                                save({ ...active, steps: next });
                              }}
                              disabled={i === 0}
                            >
                              Up
                            </button>
                            <button
                              type="button"
                              className="btn"
                              onClick={() => {
                                if (i === active.steps.length - 1) return;
                                const next = [...active.steps];
                                const tmp = next[i + 1];
                                next[i + 1] = next[i];
                                next[i] = tmp;
                                save({ ...active, steps: next });
                              }}
                              disabled={i === active.steps.length - 1}
                            >
                              Down
                            </button>
                            <button
                              type="button"
                              className="btn"
                              onClick={() => {
                                const next = active.steps.filter((_, idx) => idx !== i);
                                save({ ...active, steps: next });
                              }}
                            >
                              Delete
                            </button>
                          </div>
                        </div>

                        {s.kind !== "rest" && (
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                              <div className="muted" style={{ fontSize: 12 }}>
                                Exercise
                              </div>
                              <input
                                value={s.exercise}
                                onChange={(e) => {
                                  const next = [...active.steps];
                                  next[i] = { ...s, exercise: e.target.value, label: e.target.value.replaceAll("_", " ") } as CustomWorkoutStep;
                                  save({ ...active, steps: next });
                                }}
                                style={{
                                  background: "rgba(255,255,255,0.06)",
                                  color: "#e6edf6",
                                  border: "1px solid rgba(255,255,255,0.12)",
                                  borderRadius: 12,
                                  padding: "10px 12px",
                                  fontSize: 14,
                                  outline: "none",
                                }}
                              />
                            </label>

                            {s.kind === "work_reps" ? (
                              <label style={{ display: "grid", gap: 6 }}>
                                <div className="muted" style={{ fontSize: 12 }}>
                                  Target reps
                                </div>
                                <input
                                  type="number"
                                  min={1}
                                  value={s.targetReps}
                                  onChange={(e) => {
                                    const n = Math.max(1, Math.floor(Number(e.target.value || 1)));
                                    const next = [...active.steps];
                                    next[i] = { ...s, targetReps: n };
                                    save({ ...active, steps: next });
                                  }}
                                  style={{
                                    background: "rgba(255,255,255,0.06)",
                                    color: "#e6edf6",
                                    border: "1px solid rgba(255,255,255,0.12)",
                                    borderRadius: 12,
                                    padding: "10px 12px",
                                    fontSize: 14,
                                    outline: "none",
                                  }}
                                />
                              </label>
                            ) : (
                              <label style={{ display: "grid", gap: 6 }}>
                                <div className="muted" style={{ fontSize: 12 }}>
                                  Work seconds
                                </div>
                                <input
                                  type="number"
                                  min={5}
                                  value={s.workSec}
                                  onChange={(e) => {
                                    const n = Math.max(5, Math.floor(Number(e.target.value || 5)));
                                    const next = [...active.steps];
                                    next[i] = { ...s, workSec: n };
                                    save({ ...active, steps: next });
                                  }}
                                  style={{
                                    background: "rgba(255,255,255,0.06)",
                                    color: "#e6edf6",
                                    border: "1px solid rgba(255,255,255,0.12)",
                                    borderRadius: 12,
                                    padding: "10px 12px",
                                    fontSize: 14,
                                    outline: "none",
                                  }}
                                />
                              </label>
                            )}
                          </div>
                        )}

                        {s.kind === "rest" && (
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 10 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                              <div className="muted" style={{ fontSize: 12 }}>
                                Label
                              </div>
                              <input
                                value={s.label}
                                onChange={(e) => {
                                  const next = [...active.steps];
                                  next[i] = { ...s, label: e.target.value };
                                  save({ ...active, steps: next });
                                }}
                                style={{
                                  background: "rgba(255,255,255,0.06)",
                                  color: "#e6edf6",
                                  border: "1px solid rgba(255,255,255,0.12)",
                                  borderRadius: 12,
                                  padding: "10px 12px",
                                  fontSize: 14,
                                  outline: "none",
                                }}
                              />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                              <div className="muted" style={{ fontSize: 12 }}>
                                Rest seconds
                              </div>
                              <input
                                type="number"
                                min={5}
                                value={s.restSec}
                                onChange={(e) => {
                                  const n = Math.max(5, Math.floor(Number(e.target.value || 5)));
                                  const next = [...active.steps];
                                  next[i] = { ...s, restSec: n };
                                  save({ ...active, steps: next });
                                }}
                                style={{
                                  background: "rgba(255,255,255,0.06)",
                                  color: "#e6edf6",
                                  border: "1px solid rgba(255,255,255,0.12)",
                                  borderRadius: 12,
                                  padding: "10px 12px",
                                  fontSize: 14,
                                  outline: "none",
                                }}
                              />
                            </label>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    const ok = window.confirm("Delete this workout?");
                    if (!ok) return;
                    const next = deleteCustomWorkout(active.id);
                    setWorkouts(next);
                    setActiveId(next[0]?.id ?? null);
                  }}
                >
                  Delete workout
                </button>

                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={() => {
                    try {
                      localStorage.setItem(
                        "repdetect:runCustomWorkout:v1",
                        JSON.stringify({ workoutId: active.id })
                      );
                    } catch {
                      // ignore
                    }
                    window.location.href = "/workout";
                  }}
                >
                  Run in Workout
                </button>
              </div>

              <div className="muted" style={{ fontSize: 12 }}>
                Tip: exercise IDs match the tracker (e.g. squats, jumping_jacks). Timed steps still track reps, but progression is by time.
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
