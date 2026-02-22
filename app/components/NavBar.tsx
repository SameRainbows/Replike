"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="nav">
      <div className="container nav__inner">
        <Link className="brand" href="/">
          RepDetect
        </Link>

        <nav className="nav__links" aria-label="Primary">
          <Link
            className={`nav__link${pathname === "/workout" ? " nav__link--active" : ""}`}
            href="/workout"
            aria-current={pathname === "/workout" ? "page" : undefined}
          >
            Workout
          </Link>
          <Link
            className={`nav__link${pathname === "/about" ? " nav__link--active" : ""}`}
            href="/about"
            aria-current={pathname === "/about" ? "page" : undefined}
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
