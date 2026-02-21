import type { Metadata } from "next";
import "./globals.css";

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
      <body>{children}</body>
    </html>
  );
}
