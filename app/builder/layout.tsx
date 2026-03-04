import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Workout Builder | RepDetect",
    description: "Build custom workouts with intervals, reps, and rest periods.",
};

export default function BuilderLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
