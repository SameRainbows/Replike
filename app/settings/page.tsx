"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { loadSettings, saveSettings } from "@/app/lib/settings";

export default function SettingsPage() {
  const [calibrationEnabled, setCalibrationEnabled] = useState(false);
  const [soundOnRep, setSoundOnRep] = useState(true);
  const [soundOnGoal, setSoundOnGoal] = useState(true);

  useEffect(() => {
    const s = loadSettings();
    setCalibrationEnabled(s.calibrationEnabled);
    setSoundOnRep(s.soundOnRep);
    setSoundOnGoal(s.soundOnGoal);

    const onSettings = () => {
      const next = loadSettings();
      setCalibrationEnabled(next.calibrationEnabled);
      setSoundOnRep(next.soundOnRep);
      setSoundOnGoal(next.soundOnGoal);
    };
    window.addEventListener("storage", onSettings);
    window.addEventListener("repdetect:settings", onSettings);
    return () => {
      window.removeEventListener("storage", onSettings);
      window.removeEventListener("repdetect:settings", onSettings);
    };
  }, []);

  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">Settings</h1>
        <p className="lead">
          Optional controls for more accurate rep counting. Most people can start working out immediately.
        </p>
      </header>

      <div className="card stack">
        <div className="card__title">Calibration</div>
        <p className="muted">
          Enable calibration if you want tighter thresholds per exercise. When enabled, the Workout page will guide you
          through hands-free auto-calibration (and optional manual calibration).
        </p>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            padding: 12,
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.10)",
            background: "rgba(0,0,0,0.18)",
          }}
        >
          <div style={{ display: "grid", gap: 2 }}>
            <div style={{ fontWeight: 800 }}>Enable calibration</div>
            <div className="muted" style={{ fontSize: 13 }}>
              Recommended if reps feel inconsistent.
            </div>
          </div>

          <input
            type="checkbox"
            checked={calibrationEnabled}
            onChange={(e) => {
              const next = e.target.checked;
              setCalibrationEnabled(next);
              saveSettings({ calibrationEnabled: next, soundOnRep, soundOnGoal });
            }}
            style={{ width: 20, height: 20 }}
          />
        </label>

        <div style={{ display: "grid", gap: 10 }}>
          <div className="card__title">Sound cues</div>
          <p className="muted">Optional beeps so you don’t have to count reps while working out.</p>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              padding: 12,
              borderRadius: 14,
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(0,0,0,0.18)",
            }}
          >
            <div style={{ display: "grid", gap: 2 }}>
              <div style={{ fontWeight: 800 }}>Beep on rep</div>
              <div className="muted" style={{ fontSize: 13 }}>
                Short beep when a rep is counted.
              </div>
            </div>

            <input
              type="checkbox"
              checked={soundOnRep}
              onChange={(e) => {
                const next = e.target.checked;
                setSoundOnRep(next);
                saveSettings({ calibrationEnabled, soundOnRep: next, soundOnGoal });
              }}
              style={{ width: 20, height: 20 }}
            />
          </label>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              padding: 12,
              borderRadius: 14,
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(0,0,0,0.18)",
            }}
          >
            <div style={{ display: "grid", gap: 2 }}>
              <div style={{ fontWeight: 800 }}>Beep on goal</div>
              <div className="muted" style={{ fontSize: 13 }}>
                Distinct beep when you hit your rep goal.
              </div>
            </div>

            <input
              type="checkbox"
              checked={soundOnGoal}
              onChange={(e) => {
                const next = e.target.checked;
                setSoundOnGoal(next);
                saveSettings({ calibrationEnabled, soundOnRep, soundOnGoal: next });
              }}
              style={{ width: 20, height: 20 }}
            />
          </label>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Link className="btn btn--primary" href="/workout">
            Back to Workout
          </Link>
          <Link className="btn" href="/">
            Home
          </Link>
        </div>
      </div>

      <div className="card stack">
        <div className="card__title">Tip</div>
        <p className="muted">
          Even without calibration, you’ll get the best tracking with good lighting and your full body visible in frame.
        </p>
      </div>
    </section>
  );
}
