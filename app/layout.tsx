import type { Metadata } from "next";
import "./globals.css";
import AppShell from "@/app/components/AppShell";

export const metadata: Metadata = {
  title: "RepDetect",
  description: "In-browser pose detection + rep counting",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
