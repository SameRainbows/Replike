import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Workout | RepDetect",
    description: "Live camera-based rep counter for your browser.",
};

export default function WorkoutLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
