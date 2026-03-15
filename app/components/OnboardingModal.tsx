"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { getProfile, saveProfile, type FitnessGoal, type FitnessLevel } from "../lib/profile";

export function OnboardingModal() {
    const [isOpen, setIsOpen] = useState(false);
    const [step, setStep] = useState(1);
    const [goal, setGoal] = useState<FitnessGoal>("stay_active");
    const [level, setLevel] = useState<FitnessLevel>("beginner");

    useEffect(() => {
        // Check if onboarding is needed
        const profile = getProfile();
        if (!profile.hasCompletedOnboarding) {
            setIsOpen(true);
        }
    }, []);

    const handleComplete = () => {
        saveProfile({ goal, level, hasCompletedOnboarding: true });
        setIsOpen(false);
    };

    if (!isOpen) return null;

    return (
        <div className="modalBackdrop" role="dialog" aria-modal="true" style={{ zIndex: 9999 }}>
            <AnimatePresence mode="wait">
                <motion.div
                    key={step}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2 }}
                    className="modalCard"
                    style={{ maxWidth: 460, margin: "0 auto", marginTop: "10vh" }}
                >
                    {step === 1 && (
                        <div className="stack" style={{ gap: 24, padding: "10px 0" }}>
                            <div style={{ textAlign: "center", display: "grid", gap: 8 }}>
                                <h2 className="h2" style={{ fontSize: 28 }}>Welcome to RepDetect</h2>
                                <div className="muted">Let's set up your profile so we can tailor the experience to you.</div>
                            </div>

                            <div className="stack" style={{ gap: 12 }}>
                                <div style={{ fontWeight: 800 }}>What is your primary goal?</div>

                                {(
                                    [
                                        { id: "stay_active", label: "🧘‍♀️ Stay Active & Healthy" },
                                        { id: "build_muscle", label: "💪 Build Muscle & Strength" },
                                        { id: "lose_weight", label: "🔥 Burn Fat & Lose Weight" }
                                    ] as const
                                ).map((g) => (
                                    <button
                                        key={g.id}
                                        onClick={() => setGoal(g.id)}
                                        style={{
                                            background: goal === g.id ? "rgba(232, 132, 94, 0.15)" : "rgba(255,255,255,0.05)",
                                            border: goal === g.id ? "1px solid rgba(232, 132, 94, 0.4)" : "1px solid rgba(255,255,255,0.1)",
                                            color: goal === g.id ? "var(--accent)" : "#fff",
                                            padding: 16,
                                            borderRadius: 12,
                                            textAlign: "left",
                                            fontSize: 16,
                                            fontWeight: 600,
                                            cursor: "pointer",
                                            transition: "all 0.2s"
                                        }}
                                    >
                                        {g.label}
                                    </button>
                                ))}
                            </div>

                            <button className="btn btn--primary" onClick={() => setStep(2)} style={{ padding: 14, marginTop: 10 }}>
                                Next
                            </button>
                        </div>
                    )}

                    {step === 2 && (
                        <div className="stack" style={{ gap: 24, padding: "10px 0" }}>
                            <div style={{ textAlign: "center", display: "grid", gap: 8 }}>
                                <h2 className="h2" style={{ fontSize: 28 }}>Experience Level</h2>
                                <div className="muted">This helps us match the right difficulty and feedback to your level.</div>
                            </div>

                            <div className="stack" style={{ gap: 12 }}>
                                <div style={{ fontWeight: 800 }}>How often do you exercise?</div>

                                {(
                                    [
                                        { id: "beginner", label: "Seedling (Rarely to Never)", sub: "We'll start slow and focus on good form." },
                                        { id: "intermediate", label: "Active (1-3 times a week)", sub: "You'll get nudges to stay on track." },
                                        { id: "advanced", label: "Athlete (4+ times a week)", sub: "Strict form tracking and higher targets." }
                                    ] as const
                                ).map((l) => (
                                    <button
                                        key={l.id}
                                        onClick={() => setLevel(l.id)}
                                        style={{
                                            background: level === l.id ? "rgba(232, 132, 94, 0.15)" : "rgba(255,255,255,0.05)",
                                            border: level === l.id ? "1px solid rgba(232, 132, 94, 0.4)" : "1px solid rgba(255,255,255,0.1)",
                                            color: "#fff",
                                            padding: 16,
                                            borderRadius: 12,
                                            textAlign: "left",
                                            cursor: "pointer",
                                            transition: "all 0.2s",
                                            display: "grid",
                                            gap: 4
                                        }}
                                    >
                                        <div style={{ fontWeight: 800, fontSize: 16, color: level === l.id ? "var(--accent)" : "#fff" }}>{l.label}</div>
                                        <div style={{ fontSize: 13, color: level === l.id ? "rgba(232, 132, 94, 0.8)" : "rgba(255,255,255,0.5)" }}>{l.sub}</div>
                                    </button>
                                ))}
                            </div>

                            <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
                                <button className="btn" onClick={() => setStep(1)} style={{ padding: 14, flex: 1 }}>
                                    Back
                                </button>
                                <button className="btn btn--primary" onClick={handleComplete} style={{ padding: 14, flex: 2 }}>
                                    Finish Setup
                                </button>
                            </div>
                        </div>
                    )}
                </motion.div>
            </AnimatePresence>
        </div>
    );
}
