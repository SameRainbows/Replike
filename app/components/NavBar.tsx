import Link from "next/link";

export default function NavBar() {
  return (
    <header className="nav">
      <div className="container nav__inner">
        <Link className="brand" href="/">
          RepDetect
        </Link>

        <nav className="nav__links" aria-label="Primary">
          <Link className="nav__link" href="/workout">
            Workout
          </Link>
          <Link className="nav__link" href="/about">
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
