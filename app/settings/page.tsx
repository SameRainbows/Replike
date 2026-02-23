"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { loadSettings, saveSettings } from "@/app/lib/settings";

export default function SettingsPage() {
  const [calibrationEnabled, setCalibrationEnabled] = useState(false);

  useEffect(() => {
    setCalibrationEnabled(loadSettings().calibrationEnabled);

    const onSettings = () => setCalibrationEnabled(loadSettings().calibrationEnabled);
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
              saveSettings({ calibrationEnabled: next });
            }}
            style={{ width: 20, height: 20 }}
          />
        </label>

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
