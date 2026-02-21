import PoseRepCounter from "@/app/pose/PoseRepCounter";

export default function HomePage() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        gap: 16,
        padding: 16,
        maxWidth: 980,
        margin: "0 auto",
      }}
    >
      <header style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <h1 style={{ margin: 0, fontSize: 28, letterSpacing: -0.4 }}>RepDetect</h1>
        <p style={{ margin: 0, color: "#a7b4c7", lineHeight: 1.4 }}>
          Camera-based pose detection that counts reps in real time (all processing runs in
          your browser).
        </p>
      </header>

      <PoseRepCounter />

      <footer style={{ color: "#7f8ba0", fontSize: 12, lineHeight: 1.4 }}>
        This is an MVP. Works best when your full body is visible and the camera is stable.
      </footer>
    </main>
  );
}
