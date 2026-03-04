import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "History | RepDetect",
    description: "View your past workout sessions and rep quality.",
};

export default function HistoryLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
