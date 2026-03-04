"use client";

import Link from "next/link";
import { motion, Variants } from "framer-motion";
import { useEffect, useState } from "react";

const STAGGER_DELAY = 0.1;

export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: STAGGER_DELAY,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 100, damping: 12 } },
  };

  return (
    <section className="stack" style={{ gap: 40, paddingBottom: 60, overflow: "hidden" }}>
      {/* Hero Section */}
      <motion.header
        className="hero"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={itemVariants} className="hero__badge" style={{ display: "inline-block", marginBottom: 16 }}>
          Private. In-browser. Real-time.
        </motion.div>

        <motion.h1 variants={itemVariants} className="hero__title" style={{ fontSize: "clamp(2.5rem, 6vw, 4.5rem)", lineHeight: 1.1 }}>
          Your AI Coach, <br /> built into the web.
        </motion.h1>

        <motion.p variants={itemVariants} className="hero__subtitle" style={{ maxWidth: 600, margin: "0 auto 32px auto", fontSize: "1.1rem" }}>
          RepDetect uses your device's camera to analyze form, count reps, and keep you safe.
          Zero downloads. Zero data uploads.
        </motion.p>

        <motion.div variants={itemVariants} className="hero__actions" style={{ justifyContent: "center", display: "flex", gap: 16 }}>
          <Link className="btn btn--primary" href="/workout" style={{ padding: "14px 32px", fontSize: "1.1rem" }}>
            Start Workout
          </Link>
          <Link className="btn" href="/about" style={{ padding: "14px 32px", fontSize: "1.1rem" }}>
            How it works
          </Link>
        </motion.div>
      </motion.header>

      {/* Infinite Scrolling Marquee */}
      {mounted && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 1 }}
          className="marquee-container"
        >
          <div className="marquee-content">
            <span className="marquee-item">JUMPING JACKS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">SQUATS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">LUNGES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">HIGH KNEES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">JUMP SQUATS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">BURPEES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">PUSH-UPS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">SIT-UPS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">PLANKS</span>
            <span className="marquee-item">•</span>
          </div>
          <div className="marquee-content" aria-hidden="true">
            <span className="marquee-item">JUMPING JACKS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">SQUATS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">LUNGES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">HIGH KNEES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">JUMP SQUATS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">BURPEES</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">PUSH-UPS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">SIT-UPS</span>
            <span className="marquee-item">•</span>
            <span className="marquee-item">PLANKS</span>
            <span className="marquee-item">•</span>
          </div>
        </motion.div>
      )}

      {/* Feature Highlight Cards */}
      <motion.div
        className="grid"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6, staggerChildren: 0.1 }}
      >
        <div className="card stack sleek-hover">
          <div className="card__title">Live rep counting</div>
          <div className="muted">
            Counts reps using spatial pose estimation with anti-cheat hysteresis rules.
          </div>
        </div>
        <div className="card stack sleek-hover">
          <div className="card__title">Real-time Form cues</div>
          <div className="muted">
            On-screen AI coaching helps you hit full range-of-motion seamlessly.
          </div>
        </div>
        <div className="card stack sleek-hover">
          <div className="card__title">100% Privacy-first</div>
          <div className="muted">No video uploads. No cloud. Local WebAssembly processing.</div>
        </div>
      </motion.div>

      {/* Getting Started Split */}
      <motion.div
        className="split"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="card stack">
          <h2 className="h2" style={{ color: "var(--accent)" }}>Get Started in Seconds</h2>
          <div className="stack" style={{ gap: 12 }}>
            <p className="p" style={{ display: "flex", gap: 12 }}>
              <span style={{ color: "rgba(255,255,255,0.4)" }}>01</span>
              <span>Navigate to <strong>Workout</strong> and allow camera access.</span>
            </p>
            <p className="p" style={{ display: "flex", gap: 12 }}>
              <span style={{ color: "rgba(255,255,255,0.4)" }}>02</span>
              <span>Step back so your full body sits firmly in frame.</span>
            </p>
            <p className="p" style={{ display: "flex", gap: 12 }}>
              <span style={{ color: "rgba(255,255,255,0.4)" }}>03</span>
              <span>Follow the on-screen form cues to achieve optimal sets.</span>
            </p>
          </div>
        </div>
        <div className="card stack" style={{ background: "transparent", border: "1px dashed rgba(255,255,255,0.15)" }}>
          <h2 className="h2">Built for Real Spaces</h2>
          <div className="stack">
            <p className="p">Optimized for bedrooms, home gyms, and living rooms.</p>
            <p className="p">Featuring Hands-Free Auto-Calibration to instantly adapt to your height and camera distance without touching the screen.</p>
          </div>
          <Link href="/about" className="btn" style={{ alignSelf: "flex-start", marginTop: 16 }}>Read the whitepaper</Link>
        </div>
      </motion.div>
    </section>
  );
}
