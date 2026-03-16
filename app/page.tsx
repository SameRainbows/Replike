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
    <section className="stack" style={{ gap: 40, paddingBottom: 60 }}>
      {/* Hero Section */}
      <motion.header
        className="hero"
        variants={containerVariants}
        initial="hidden"
        animate="show"
        style={{ textAlign: "left" }}
      >
        <motion.h1 variants={itemVariants} className="hero__title" style={{ fontSize: "clamp(2.5rem, 6vw, 4.5rem)", lineHeight: 1.1 }}>
          Count reps. Fix form. <br /> No app needed.
        </motion.h1>

        <motion.p variants={itemVariants} className="hero__subtitle" style={{ maxWidth: 600, fontSize: "1.1rem" }}>
          RepDetect watches your form through your camera and counts every rep — all
          inside your browser. Nothing is uploaded, ever.
        </motion.p>

        <motion.div variants={itemVariants} className="hero__actions" style={{ display: "flex", gap: 16 }}>
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
            Tracks your reps automatically so you can focus on the workout, not the count.
          </div>
        </div>
        <div className="card stack sleek-hover">
          <div className="card__title">Real-time form cues</div>
          <div className="muted">
            Tells you when your form is off before a bad habit sticks.
          </div>
        </div>
        <div className="card stack sleek-hover">
          <div className="card__title">100% private</div>
          <div className="muted">Your camera feed never leaves your device. Not even a single frame.</div>
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
          <Link href="/about" className="btn" style={{ alignSelf: "flex-start", marginTop: 16 }}>Learn more</Link>
        </div>
      </motion.div>
    </section>
  );
}
