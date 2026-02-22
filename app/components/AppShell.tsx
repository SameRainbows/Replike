import NavBar from "@/app/components/NavBar";

export default function AppShell({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="app">
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
    </div>
  );
}
