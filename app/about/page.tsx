export default function AboutPage() {
  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">About</h1>
        <p className="lead">
          RepDetect counts reps using in-browser pose estimation. No videos are uploaded.
        </p>
      </header>

      <div className="card stack">
        <h2 className="h2">Tips for accuracy</h2>
        <div className="stack">
          <p className="p">
            Keep your full body visible, especially ankles for jumping jacks and knees for
            squats/lunges.
          </p>
          <p className="p">Use good lighting and keep the camera stable.</p>
          <p className="p">Move with control. Very fast movements can be ignored as noise.</p>
        </div>
      </div>

      <div className="card stack">
        <h2 className="h2">Privacy</h2>
        <p className="p">
          The app runs completely in your browser. Pose processing happens locally.
        </p>
      </div>
    </section>
  );
}
