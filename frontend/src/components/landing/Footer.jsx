export default function Footer() {
  return (
    <footer className="border-t border-[var(--line)] px-6 py-8">
      <div className="mx-auto max-w-5xl flex flex-col items-center gap-2 text-center sm:flex-row sm:justify-between">
        <span className="font-bold text-[var(--blue)]">Vera Intelligence.</span>
        <p className="text-xs text-[#6e6e66]">
          © 2026 Karan Bhatia. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
