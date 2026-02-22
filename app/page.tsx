export default function HomePage() {
  return (
    <section className="stack">
      <header className="hero">
        <div className="hero__badge">Private. In-browser. Real-time.</div>
        <h1 className="hero__title">RepDetect</h1>
        <p className="hero__subtitle">
          A camera-based rep counter that runs entirely in your browser. Pick an exercise,
          see form cues, and track clean reps.
        </p>

        <div className="hero__actions">
          <a className="btn btn--primary" href="/workout">
            Start workout
          </a>
          <a className="btn" href="/about">
            Learn more
          </a>
        </div>

        <div className="hero__meta">
          <div className="pill">Hands-free calibration</div>
          <div className="pill">Form feedback</div>
          <div className="pill">No uploads</div>
        </div>
      </header>

      <div className="grid">
        <div className="card stack">
          <div className="card__title">Live rep counting</div>
          <div className="muted">
            Counts reps using pose estimation with smoothing and anti-cheat rules.
          </div>
        </div>
        <div className="card stack">
          <div className="card__title">Form cues</div>
          <div className="muted">
            On-screen feedback helps you hit full range-of-motion and avoid false counts.
          </div>
        </div>
        <div className="card stack">
          <div className="card__title">Privacy-first</div>
          <div className="muted">No uploads. Processing runs locally on your device.</div>
        </div>
      </div>

      <div className="split">
        <div className="card stack">
          <h2 className="h2">Getting started</h2>
          <div className="stack">
            <p className="p">1) Go to Workout and allow camera access.</p>
            <p className="p">2) Step back so your full body is visible.</p>
            <p className="p">3) Move with control for the cleanest counts.</p>
          </div>
        </div>
        <div className="card stack">
          <h2 className="h2">Designed for real rooms</h2>
          <div className="stack">
            <p className="p">Works best with even lighting and a stable camera angle.</p>
            <p className="p">If tracking feels jumpy, pause, reposition, and recalibrate.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
