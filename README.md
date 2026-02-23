# Replike (RepDetect)

Browser-based workout rep counting using real-time pose detection.

## Live demo

https://replike.vercel.app/

## What it does

Replike uses your device camera and an on-device pose model to track body landmarks, guide calibration, and count reps for multiple exercises.
It also supports guided workout plans and stores workout history locally in your browser.

## Key features

- **Hands-free calibration**
  The app automatically detects when you hold a stable pose and captures calibration frames without button clicks.
- **Multiple exercises + per-exercise state machines**
  Includes jumping jacks, squats, lunges, high knees, jump squats, and burpees.
- **Guided workout plans**
  Preset plans with work/rest steps, timers, and step-by-step progression.
- **Workout history**
  Completed plan sessions are auto-saved; free workouts can be saved manually.
  Sessions are stored in `localStorage`.
- **In-video overlays**
  Landmarks and prompts are rendered to a canvas overlay aligned to the displayed video.

## How to use

1. Go to **Workout**.
2. Allow camera access.
3. Follow the on-screen calibration prompts.
4. Choose:
   - **Free workout** to pick an exercise and count reps.
   - **Guided plan** to run a preset plan.
5. View saved sessions under **History**.

## Tech stack

- Next.js (App Router)
- React + TypeScript
- MediaPipe Tasks Vision (`PoseLandmarker`)

## Privacy & permissions

- **Camera permission is required** to run pose detection.
- **No video is uploaded by default.** Pose detection runs locally in the browser.
- Workout sessions are saved in your browser storage (`localStorage`) unless you clear them.

## Run locally

1. Install dependencies

```bash
npm install
```

2. Start the dev server

```bash
npm run dev
```

3. Open

http://localhost:3000

## Notes / troubleshooting

- Camera access typically requires **HTTPS** in production (Vercel provides this automatically).
- For best results, use good lighting and keep your full body in frame.

## Roadmap ideas

- Custom plan builder
- Session summaries and trends (weekly volume, PRs)
- More exercises and stricter form scoring
