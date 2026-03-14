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
        {/* Pulsing ring */}
        <motion.div
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.6, 0, 0.6],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "2px solid var(--accent)",
          }}
        />

        {/* Center dot */}
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 24,
            height: 24,
            borderRadius: "50%",
            background: "var(--accent)",
            boxShadow: "0 0 20px rgba(60, 242, 176, 0.8)",
          }}
        />

        {/* Scanning line */}
        <motion.div
          animate={{
            y: [0, 80, 0],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{
            position: "absolute",
            top: 0,
            left: "-10%",
            width: "120%",
            height: 2,
            background: "rgba(60, 242, 176, 0.8)",
            boxShadow: "0 0 10px rgba(60, 242, 176, 0.8)",
          }}
        />
      </div>

      <div style={{ textAlign: "center", display: "grid", gap: 8 }}>
        <h3 className="h3">Warming up Vision AI</h3>
        <p className="muted" style={{ maxWidth: 280, margin: "0 auto", fontSize: 13 }}>
          Downloading the tracking model and activating your camera. This usually takes just a few seconds.
        </p>
      </div>
    </div>
  );
}
