import { Link } from "react-router-dom";
import { navLinks } from "./data";
import { BrandMark } from "../brand/Brand";
import ThemeToggle from "../ThemeToggle";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-[var(--line)] bg-[var(--bg)]/90 backdrop-blur-md">
      <nav className="mx-auto flex max-w-6xl items-center gap-1 px-3 sm:gap-2 sm:px-6 h-16" aria-label="Main">
        <BrandMark className="mr-auto" />
        {navLinks.map((l) => (
          <a
            key={l.label}
            href={l.href}
            className="hidden sm:block rounded-full px-3 py-1.5 text-[15px] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            {l.label}
          </a>
        ))}
        <Link
          to="/login"
          className="rounded-full px-3 py-1.5 text-[15px] font-semibold text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
        >
          Log in
        </Link>
        <Link
          to="/signup"
          className="rounded-full bg-[var(--text)] px-4 py-2 text-[15px] font-semibold text-[var(--bg)] transition-opacity hover:opacity-90"
        >
          Sign up
        </Link>
        <ThemeToggle className="ml-1" />
      </nav>
    </header>
  );
}
