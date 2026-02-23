import PoseRepCounter from "@/app/pose/PoseRepCounter";
import Link from "next/link";

export default function WorkoutPage() {
  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">Workout</h1>
        <p className="lead">
          Allow camera access to start. Pick an exercise and track clean reps in real time.
        </p>
      </header>

      <div className="card stack">
        <div className="card__title">Setup</div>
        <div className="muted">
          Place your device so your full body fits in frame. For best results, use good lighting and
          keep the camera stable.
        </div>
        <div className="muted" style={{ fontSize: 13 }}>
          Want tighter tracking? Enable calibration in <Link href="/settings">Settings</Link>.
        </div>
      </div>

      <PoseRepCounter />
    </section>
  );
}
