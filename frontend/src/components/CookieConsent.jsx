import { useState } from "react";
import { Link } from "react-router-dom";

export default function CookieConsent() {
  const [show, setShow] = useState(() => !localStorage.getItem("cookie-consent"));

  const accept = () => {
    localStorage.setItem("cookie-consent", "true");
    setShow(false);
  };

  if (!show) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 pointer-events-none flex justify-center sm:justify-end">
      <div className="pointer-events-auto w-full max-w-sm rounded-2xl border border-[var(--text)]/[0.1] bg-[var(--bg)] shadow-2xl p-6 relative overflow-hidden">
        {/* Subtle glow */}
        <div className="absolute -top-10 -right-10 h-32 w-32 rounded-full bg-[var(--text)] opacity-5 blur-[40px]" />
        
        <h3 className="text-lg font-bold tracking-tight mb-2 relative z-10">We value your privacy</h3>
        <p className="text-sm text-[var(--muted)] mb-6 relative z-10 leading-relaxed">
          We use your browser's local storage to keep you signed in and save your settings. We don't use tracking or advertising cookies. 
          Read our <Link to="/cookie-policy" className="underline hover:text-[var(--text)]">Cookie Policy</Link> for more details.
        </p>
        <button
          onClick={accept}
          className="w-full rounded-xl bg-[var(--text)] py-3 text-sm font-bold text-[var(--bg)] transition hover:opacity-90 relative z-10"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
