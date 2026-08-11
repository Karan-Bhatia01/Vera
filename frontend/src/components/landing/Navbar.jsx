import { Link } from "react-router-dom";
import { navLinks } from "./data";
import { useState, useEffect } from "react";

export default function Navbar() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("app-theme") || "dark",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("app-theme", theme);
  }, [theme]);
  return (
    <header className="sticky top-4 z-50 flex justify-center px-6">
      <nav className="flex items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--surface)]/90 backdrop-blur px-3 py-2 shadow-xl">
        <span className="flex items-center gap-2 pl-3 pr-4 font-bold tracking-wide text-sm">
          <span className="h-2 w-2 rounded-full bg-[var(--blue)]" />
          VERA
        </span>
        {navLinks.map((l) => (
          <a
            key={l.label}
            href={l.href}
            className="hidden sm:block px-3 py-1.5 text-sm text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            {l.label}
          </a>
        ))}
        <Link
          to="/login"
          className="ml-1 rounded-full border border-[var(--line)] px-4 py-1.5 text-sm font-semibold text-[var(--text)] hover:bg-[var(--line)] transition-colors"
        >
          Login
        </Link>
        <Link
          to="/signup"
          className="rounded-full bg-[var(--blue)] px-4 py-1.5 text-sm font-semibold text-[var(--bg)]"
        >
          Signup
        </Link>
        <button
          onClick={() =>
            setTheme((value) => (value === "dark" ? "light" : "dark"))
          }
          className="ml-2 flex h-8 w-8 items-center justify-center rounded-full bg-[var(--line)] text-[var(--text)] transition-colors hover:bg-[var(--blue)] hover:text-[var(--bg)]"
          aria-label="Toggle Theme"
        >
          {theme === "dark" ? "☀" : "☾"}
        </button>
      </nav>
    </header>
  );
}
