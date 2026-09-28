import { useState } from "react";
import { faqs, faqSubtitle } from "./data";

function FAQItem({ q, a }) {
  const [open, setOpen] = useState(false);

  return (
    <button
      onClick={() => setOpen((v) => !v)}
      className="w-full text-left rounded-2xl border border-[var(--line)] bg-[var(--surface-2)] backdrop-blur-sm transition-all duration-200 hover:border-[var(--blue)]/40"
    >
      <div className="flex items-center justify-between gap-4 px-6 py-5">
        <span className="text-[15px] font-semibold leading-snug">{q}</span>
        <span
          className={`shrink-0 text-lg text-[var(--muted)] transition-transform duration-200 ${open ? "rotate-45" : ""}`}
        >
          +
        </span>
      </div>

      <div
        className={`grid transition-all duration-200 ${open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
      >
        <div className="overflow-hidden">
          <p className="px-6 pb-5 text-sm leading-relaxed text-[var(--muted)]">
            {a}
          </p>
        </div>
      </div>
    </button>
  );
}

export default function FAQ() {
  return (
    <section id="faq" className="px-6 pb-24 scroll-mt-24">
      <div className="mx-auto max-w-3xl">
        <div className="text-center">
          <p className="font-['JetBrains_Mono'] text-xs tracking-widest text-[#6e6e66]">
            FAQ
          </p>
          <h2 className="mt-3 text-4xl font-extrabold">Got questions?</h2>
          <p className="mt-3 text-[var(--muted)]">{faqSubtitle}</p>
        </div>

        <div className="mt-12 grid gap-3">
          {faqs.map((item) => (
            <FAQItem key={item.q} q={item.q} a={item.a} />
          ))}
        </div>
      </div>
    </section>
  );
}
