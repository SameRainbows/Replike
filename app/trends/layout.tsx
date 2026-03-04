import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Trends | RepDetect",
    description: "Track your workout volume and personal records over time.",
};

export default function TrendsLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
