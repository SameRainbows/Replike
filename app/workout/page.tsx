import PoseRepCounter from "@/app/pose/PoseRepCounter";

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
          Place your device so your full body fits in frame. If tracking feels off, pause and
          reposition, then recalibrate.
        </div>
      </div>

      <PoseRepCounter />
    </section>
  );
}
