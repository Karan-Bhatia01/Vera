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
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex justify-center p-4 sm:justify-end">
      <div role="region" aria-label="Storage notice" className="pointer-events-auto w-full max-w-sm rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-xl">
        <h2 className="mb-2 text-lg font-extrabold tracking-tight">How Vera uses your browser</h2>
        <p className="mb-5 text-[15px] leading-relaxed text-[var(--muted)]">
          We use your browser&apos;s local storage to keep you signed in and save your settings. We don&apos;t use tracking or
          advertising cookies. Read the <Link to="/cookie-policy" className="underline underline-offset-2 hover:text-[var(--text)]">cookie policy</Link>.
        </p>
        <button
          onClick={accept}
          className="w-full rounded-full bg-[var(--text)] py-3 font-bold text-[var(--bg)] transition-opacity hover:opacity-90"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
