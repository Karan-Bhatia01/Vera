import { faqs } from "./data";

// Native <details>: keyboard and screen-reader friendly with no extra state.
export default function FAQ() {
  return (
    <section id="faq" className="scroll-mt-20 border-t border-[var(--line)]">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-24 sm:px-6 lg:grid-cols-[1fr_2fr]">
        <h2 className="text-4xl font-extrabold tracking-tight sm:text-5xl">Questions</h2>
        <div className="border-t border-[var(--line)]">
          {faqs.map((item) => (
            <details key={item.q} className="group border-b border-[var(--line)]">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-6 py-5 text-lg font-bold [&::-webkit-details-marker]:hidden">
                {item.q}
                <span
                  className="shrink-0 text-2xl leading-none text-[var(--muted)] transition-transform duration-200 group-open:rotate-45"
                  aria-hidden="true"
                >
                  +
                </span>
              </summary>
              <p className="max-w-[62ch] pb-6 text-lg leading-relaxed text-[var(--muted)]">{item.a}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
