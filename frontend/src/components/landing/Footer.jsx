import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="px-6 py-12 border-t border-[var(--text)]/[0.08]">
      <div className="mx-auto max-w-5xl flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-col gap-2">
          <span className="font-extrabold text-2xl tracking-tight">Vera.</span>
          <p className="text-sm font-mono text-[var(--muted)]">
            © {new Date().getFullYear()} Karan Bhatia. All rights reserved.
          </p>
        </div>
        
        <div className="flex flex-wrap gap-4 text-sm font-medium text-[var(--muted)]">
          <Link to="/privacy-policy" className="hover:text-[var(--text)] transition-colors">Privacy Policy</Link>
          <Link to="/terms-of-service" className="hover:text-[var(--text)] transition-colors">Terms of Service</Link>
          <Link to="/cookie-policy" className="hover:text-[var(--text)] transition-colors">Cookie Policy</Link>
        </div>
      </div>
    </footer>
  );
}
