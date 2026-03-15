import NavBar from "@/app/components/NavBar";
import { OnboardingModal } from "@/app/components/OnboardingModal";
import { ThemeProvider } from "@/app/components/ThemeProvider";

export default function AppShell({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="app">
      <ThemeProvider />
      <NavBar />
      <main className="container app__main">
        <div className="page">{children}</div>
      </main>
      <footer className="footer">
        <div className="container footer__inner">
          <div className="muted">All processing runs locally in your browser.</div>
          <div className="muted">Best results with a steady camera and your full body visible.</div>
        </div>
      </footer>
      <OnboardingModal />
      {/* Animated nebula background glow */}
      <div className="nebula" aria-hidden="true">
        <div className="nebula__layer nebula__layer--1" />
        <div className="nebula__layer nebula__layer--2" />
        <div className="nebula__layer nebula__layer--3" />
      </div>
    </div>
  );
}
