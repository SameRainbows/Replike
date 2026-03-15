"use client";

import { motion } from "framer-motion";
import type { ExerciseId } from "../exercises/types";

export function ExerciseAnimation({ exercise }: { exercise: ExerciseId }) {
    // We render a sleek, abstract stick-figure representation using Framer Motion
    const renderSquats = () => (
        <div style={{ position: "relative", width: 80, height: 120 }}>
            <motion.div
                animate={{ y: [0, 30, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 35, width: 10, height: 10, borderRadius: 5, background: "var(--accent)" }}
            />
            <motion.div
                animate={{ y: [0, 30, 0], scaleY: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 38, top: 12, width: 4, height: 40, background: "rgba(255,255,255,0.7)" }}
            />
            {/* Legs folding */}
            <motion.div
                animate={{ y: [0, 20, 0], scaleY: [1, 0.4, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 28, top: 52, width: 4, height: 40, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
            <motion.div
                animate={{ y: [0, 20, 0], scaleY: [1, 0.4, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 52, width: 4, height: 40, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
        </div>
    );

    const renderJumpingJacks = () => (
        <div style={{ position: "relative", width: 100, height: 120 }}>
            {/* Head */}
            <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 45, width: 10, height: 10, borderRadius: 5, background: "var(--accent)" }}
            />
            {/* Torso */}
            <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 12, width: 4, height: 40, background: "rgba(255,255,255,0.7)" }}
            />
            {/* Arms */}
            <motion.div
                animate={{ rotate: [30, 150, 30], y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 15, width: 4, height: 35, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
            <motion.div
                animate={{ rotate: [-30, -150, -30], y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 15, width: 4, height: 35, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
            {/* Legs */}
            <motion.div
                animate={{ rotate: [10, 30, 10], y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 52, width: 4, height: 45, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
            <motion.div
                animate={{ rotate: [-10, -30, -10], y: [0, -10, 0] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 48, top: 52, width: 4, height: 45, background: "rgba(255,255,255,0.5)", transformOrigin: "top center" }}
            />
        </div>
    );

    const renderPushUps = () => (
        <div style={{ position: "relative", width: 120, height: 80, marginTop: 40 }}>
            {/* Body */}
            <motion.div
                animate={{ rotate: [-20, -5, -20], y: [0, 20, 0], x: [0, 5, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 20, top: 0, width: 60, height: 4, background: "rgba(255,255,255,0.7)", transformOrigin: "left center" }}
            >
                {/* Head */}
                <div style={{ position: "absolute", right: -15, top: -7, width: 10, height: 10, borderRadius: 5, background: "var(--accent)" }} />
            </motion.div>

            {/* Arm */}
            <motion.div
                animate={{ scaleY: [1, 0.5, 1], y: [0, 8, 0], x: [0, 3, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ position: "absolute", left: 70, top: 0, width: 4, height: 35, background: "rgba(255,255,255,0.5)", transformOrigin: "bottom center" }}
            />
        </div>
    );

    const renderDefault = () => (
        <div style={{ position: "relative", width: 60, height: 60, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <motion.div
                animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ width: 40, height: 40, borderRadius: 20, background: "rgba(232, 132, 94, 0.2)", border: "2px solid var(--accent)" }}
            />
        </div>
    );

    const getAnim = () => {
        if (exercise === "squats" || exercise === "jump_squats") return renderSquats();
        if (exercise === "jumping_jacks" || exercise === "burpees") return renderJumpingJacks();
        if (exercise === "push_ups" || exercise === "plank" || exercise === "sit_ups") return renderPushUps();
        // default dots for others like lunges, high_knees
        return renderDefault();
    };

    return (
        <div style={{
            background: "linear-gradient(145deg, rgba(20, 30, 45, 0.8) 0%, rgba(0,0,0,0.6) 100%)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 16,
            height: 180,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 20
        }}>
            {getAnim()}
        </div>
    );
}
