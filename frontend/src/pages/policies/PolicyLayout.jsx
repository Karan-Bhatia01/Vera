import Navbar from "../../components/landing/Navbar";
import Footer from "../../components/landing/Footer";

// Policy clauses are numbered because they are referenced by number.
export function PolicySection({ number, title, children }) {
  return (
    <section className="grid gap-2 border-t border-[var(--line)] pt-8 sm:grid-cols-[3rem_1fr]">
      <span className="font-mono text-lg text-[var(--muted)]" aria-hidden="true">{number}.</span>
      <div>
        <h2 className="mb-3 text-2xl font-extrabold tracking-tight">{title}</h2>
        <div className="max-w-[65ch] space-y-4 text-lg leading-relaxed text-[var(--muted)]">{children}</div>
      </div>
    </section>
  );
}

export default function PolicyLayout({ title, lastUpdated, children }) {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-16 sm:px-6 md:py-24">
        <h1 className="text-5xl font-extrabold tracking-tight md:text-6xl">{title}</h1>
        <p className="mt-4 mb-14 text-[var(--muted)]">Last updated {lastUpdated}</p>
        <div className="space-y-10">{children}</div>
      </main>
      <Footer />
    </div>
  );
}
