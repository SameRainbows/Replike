"use client";

import { useState, useEffect } from "react";
import { getProfile, logDailyMetrics } from "@/app/lib/profile";

export function DailyLog() {
    const [sleep, setSleep] = useState(7);
    const [water, setWater] = useState(4);
    const [logged, setLogged] = useState(false);

    useEffect(() => {
        const p = getProfile();
        const today = new Date().toLocaleDateString();
        const td = p.metrics.find(m => m.dateStr === today);
        if (td) {
            setSleep(td.sleepHours);
            setWater(td.waterGlasses);
            setLogged(true);
        }
    }, []);

    const handleSave = () => {
        logDailyMetrics(sleep, water);
        setLogged(true);
    };

    return (
        <div className="card stack">
            <div className="card__title">Daily Lifestyle Log</div>
            <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>
                Fitness is 20% working out, 80% recovery. Log your daily stats to help the Smart Coach adjust recommendations.
            </div>

            <div style={{ display: "grid", gap: 16 }}>
                <label className="stack" style={{ gap: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 800 }}>
                        <span>💤 Sleep (Hours)</span>
                        <span style={{ color: "var(--accent)" }}>{sleep} hrs</span>
                    </div>
                    <input
                        type="range"
                        min="2" max="14" step="0.5"
                        value={sleep}
                        onChange={(e) => { setSleep(Number(e.target.value)); setLogged(false); }}
                        style={{ width: "100%", accentColor: "var(--accent)" }}
                    />
                </label>

                <label className="stack" style={{ gap: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 800 }}>
                        <span>💧 Hydration (Glasses)</span>
                        <span style={{ color: "#3daee9" }}>{water} glasses</span>
                    </div>
                    <input
                        type="range"
                        min="0" max="15" step="1"
                        value={water}
                        onChange={(e) => { setWater(Number(e.target.value)); setLogged(false); }}
                        style={{ width: "100%", accentColor: "#3daee9" }}
                    />
                </label>

                <button
                    disabled={logged}
                    className="btn btn--primary"
                    onClick={handleSave}
                    style={{
                        opacity: logged ? 0.3 : 1,
                        cursor: logged ? "default" : "pointer"
                    }}
                >
                    {logged ? "Logged for Today" : "Save Logs"}
                </button>
            </div>
        </div>
    );
}
