"use client";

import { motion } from "framer-motion";

export function ModelLoader() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "60px 20px",
        gap: 24,
        background: "rgba(0,0,0,0.2)",
        borderRadius: 20,
        border: "1px dashed var(--border)",
      }}
    >
      <div style={{ position: "relative", width: 80, height: 80 }}>
        {/* AI Breathing Orb - Outer Aura */}
        <motion.div
          animate={{
            rotate: 360,
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{
            position: "absolute",
            inset: "-20%",
            borderRadius: "50%",
            background: "radial-gradient(circle at center, rgba(60, 242, 176, 0.15) 0%, transparent 70%)",
            filter: "blur(12px)",
          }}
        />

        {/* AI Breathing Orb - Middle Layer */}
        <motion.div
          animate={{
            rotate: -360,
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            position: "absolute",
            top: "-10%",
            left: "-10%",
            right: "10%",
            bottom: "10%",
            borderRadius: "50%",
            background: "radial-gradient(circle at top right, rgba(84, 126, 255, 0.4) 0%, transparent 60%)",
            filter: "blur(8px)",
          }}
        />

        {/* AI Breathing Orb - Core */}
        <motion.div
          animate={{
            scale: [1, 1.05, 1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            position: "absolute",
            top: "25%",
            left: "25%",
            right: "25%",
            bottom: "25%",
            borderRadius: "50%",
            background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
            boxShadow: "0 0 20px rgba(60, 242, 176, 0.5), 0 0 40px rgba(84, 126, 255, 0.3)",
          }}
        />
      </div>

      <div style={{ textAlign: "center", display: "grid", gap: 8 }}>
        <h3 className="h3">Preparing your workspace</h3>
        <p className="muted" style={{ maxWidth: 280, margin: "0 auto", fontSize: 13 }}>
          Loading the AI fitness coach and securing your local camera feed. This usually takes just a few seconds.
        </p>
      </div>
    </div>
  );
}
