import { Link } from "react-router-dom";
import { navLinks } from "./data";

export default function Navbar() {
  return (
    <header className="sticky top-4 z-50 flex justify-center px-6">
      <nav className="flex items-center gap-2 rounded-full border border-white/10 bg-[#0d1117]/90 backdrop-blur px-3 py-2 shadow-[0_8px_30px_rgba(0,0,0,0.4)]">
        <span className="flex items-center gap-2 pl-3 pr-4 font-bold tracking-wide text-sm">
          <span className="h-2 w-2 rounded-full bg-[#b56126]" />
          VERA
        </span>
        {navLinks.map((l) => (
          <a
            key={l}
            href="#"
            onClick={(e) => e.preventDefault()}
            className="hidden sm:block px-3 py-1.5 text-sm text-[#9a9a93] hover:text-[#f0ece2] transition-colors"
          >
            {l}
          </a>
        ))}
        <Link
          to="/login"
          className="ml-1 rounded-full border border-white/15 px-4 py-1.5 text-sm font-semibold text-[#f0ece2] hover:border-white/30 transition-colors"
        >
          Login
        </Link>
        <Link
          to="/signup"
          className="rounded-full bg-[#b56126] px-4 py-1.5 text-sm font-semibold text-[#0d1117]"
        >
          Signup
        </Link>
      </nav>
    </header>
  );
}