import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Settings | RepDetect",
    description: "Configure rep counting, form tracking, and sounds.",
};

export default function SettingsLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return <>{children}</>;
}
