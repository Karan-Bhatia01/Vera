import { Link } from "react-router-dom";
import { BrandMark } from "../brand/Brand";
import ConnectionStatus from "../ConnectionStatus";

export default function Footer() {
  return (
    <footer className="border-t border-[var(--line)]">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 sm:px-6 md:flex-row md:items-center">
        <div className="md:mr-auto">
          <BrandMark />
          <p className="mt-2 text-sm text-[var(--muted)]">© {new Date().getFullYear()} Karan Bhatia. All rights reserved.</p>
        </div>
        <ConnectionStatus />
        <nav className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-[var(--muted)]" aria-label="Legal">
          <Link to="/privacy-policy" className="hover:text-[var(--text)] transition-colors">Privacy policy</Link>
          <Link to="/terms-of-service" className="hover:text-[var(--text)] transition-colors">Terms of service</Link>
          <Link to="/cookie-policy" className="hover:text-[var(--text)] transition-colors">Cookie policy</Link>
        </nav>
      </div>
    </footer>
  );
}
