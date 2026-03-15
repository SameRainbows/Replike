export default function AboutPage() {
  return (
    <section className="stack">
      <header className="stack">
        <h1 className="h1">About</h1>
        <p className="lead">
          RepDetect counts reps using in-browser pose estimation. No videos are uploaded.
        </p>
      </header>

      <div className="grid grid--2">
        <div className="card stack">
          <h2 className="h2">What it does</h2>
          <p className="p">
            You pick an exercise, optionally calibrate in Settings for tighter tracking, and RepDetect
            counts clean reps while showing live form cues.
          </p>
        </div>
        <div className="card stack">
          <h2 className="h2">What it doesn’t do</h2>
          <p className="p">
            It doesn’t record or upload your video. It’s a lightweight tool for practice and
            consistency, not a medical or coaching device.
          </p>
        </div>
      </div>

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
        <h2 className="h2">How it Works (Privacy Built-in)</h2>
        <div className="stack">
          <p className="p">
            <strong>RepDetect is 100% private.</strong> We understand that having a camera active can be intimidating, so we built this application to run entirely inside your browser. No video frames, images, or pose coordinates ever leave your device.
          </p>
          <ul className="muted" style={{ paddingLeft: 20, fontSize: 14 }}>
            <li style={{ marginBottom: 8 }}>
              <strong>Zero Server Uploads:</strong> When you grant camera access, the video feed is routed strictly to a local HTML <code>&lt;canvas&gt;</code> element on your screen. 
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Google MediaPipe Vision:</strong> We use an advanced AI Vision model that is downloaded directly into your browser's memory (via WebAssembly). 
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Coordinates Only:</strong> The AI mathematically scans the video locally to find 33 coordinates (your joints, eyes, and shoulders). It discards the video frame immediately and only uses those X, Y, Z numbers to calculate the "angle" of your reps.
            </li>
            <li>
              <strong>Local Storage:</strong> Your workout history and settings are saved securely on your own hard drive using standard web <code>localStorage</code>.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
