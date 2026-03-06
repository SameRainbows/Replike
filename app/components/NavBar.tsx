"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { loadSessions } from "../lib/workoutHistory";
import { calculateStreak } from "../lib/streaks";

export default function NavBar() {
  const pathname = usePathname();
  const [streak, setStreak] = useState(0);

  useEffect(() => {
    // Load explicitly in client to avoid hydration mismatch
    const syncSt = () => {
      const s = loadSessions();
      setStreak(calculateStreak(s).current);
    };
    syncSt();
    // Re-check automatically
    window.addEventListener("repdetect:history", syncSt);
    return () => window.removeEventListener("repdetect:history", syncSt);
  }, []);

  return (
    <header className="nav">
      <div className="container nav__inner">
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <Link className="brand" href="/">
            RepDetect
          </Link>
          {streak > 0 && (
            <div title={`${streak} day streak!`} style={{ display: "flex", alignItems: "center", gap: 4, background: "rgba(255, 100, 80, 0.15)", color: "#ff8c6b", padding: "4px 10px", borderRadius: 12, fontSize: 13, fontWeight: 800 }}>
              <span style={{ fontSize: 16 }}>🔥</span> {streak}
            </div>
          )}
        </div>

        <nav className="nav__links" aria-label="Primary">
          <Link
            className={`nav__link${pathname === "/workout" ? " nav__link--active" : ""}`}
            href="/workout"
            aria-current={pathname === "/workout" ? "page" : undefined}
          >
            Workout
          </Link>
          <Link
            className={`nav__link${pathname === "/builder" ? " nav__link--active" : ""}`}
            href="/builder"
            aria-current={pathname === "/builder" ? "page" : undefined}
          >
            Builder
          </Link>
          <Link
            className={`nav__link${pathname === "/history" ? " nav__link--active" : ""}`}
            href="/history"
            aria-current={pathname === "/history" ? "page" : undefined}
          >
            History
          </Link>
          <Link
            className={`nav__link${pathname === "/trends" ? " nav__link--active" : ""}`}
            href="/trends"
            aria-current={pathname === "/trends" ? "page" : undefined}
          >
            Trends
          </Link>
          <Link
            className={`nav__link${pathname === "/settings" ? " nav__link--active" : ""}`}
            href="/settings"
            aria-current={pathname === "/settings" ? "page" : undefined}
          >
            Settings
          </Link>
          <Link
            className={`nav__link${pathname === "/about" ? " nav__link--active" : ""}`}
            href="/about"
            aria-current={pathname === "/about" ? "page" : undefined}
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
