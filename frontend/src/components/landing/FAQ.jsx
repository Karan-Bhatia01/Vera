import { useState } from "react";
import { motion } from "framer-motion";
import { faqs, faqSubtitle } from "./data";

function FAQItem({ q, a }) {
  const [open, setOpen] = useState(false);

  return (
    <button
      onClick={() => setOpen((v) => !v)}
      className="group relative w-full text-left overflow-hidden rounded-3xl border border-[var(--text)]/[0.08] bg-[var(--text)]/[0.02] transition-all duration-300 hover:border-[var(--text)]/[0.2] hover:bg-[var(--text)]/[0.04]"
    >
      <div className="absolute inset-0 m-auto h-16 w-full bg-[var(--text)] opacity-0 blur-[60px] transition-opacity duration-500 group-hover:opacity-[0.03]" />
      
      <div className="relative z-10 flex items-center justify-between gap-4 py-8 px-8 sm:px-10">
        <span className="text-xl sm:text-2xl font-bold tracking-tight">{q}</span>
        <span
          className={`shrink-0 text-3xl font-light text-[var(--muted)] transition-all duration-300 group-hover:text-[var(--text)] ${open ? "rotate-45" : ""}`}
        >
          +
        </span>
      </div>

      <div
        className={`relative z-10 grid transition-all duration-300 ${open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
      >
        <div className="overflow-hidden">
          <p className="px-8 sm:px-10 pb-8 text-lg leading-relaxed text-[var(--muted)] max-w-3xl">
            {a}
          </p>
        </div>
      </div>
    </button>
  );
}

export default function FAQ() {
  return (
    <section id="faq" className="px-6 py-32 scroll-mt-24">
      <div className="mx-auto max-w-4xl">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.5 }}
          className="mb-24 text-center relative"
        >
          <p className="font-mono text-xs tracking-widest text-[var(--muted)] uppercase">
            FAQ
          </p>
          <h2 className="mt-4 text-5xl sm:text-7xl font-extrabold tracking-tight leading-none">
            Got questions?
          </h2>
          <p className="mt-6 text-xl text-[var(--muted)] max-w-2xl mx-auto leading-relaxed">
            {faqSubtitle}
          </p>
        </motion.div>

        <div className="flex flex-col gap-4">
          {faqs.map((item, i) => (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
              key={item.q}
            >
              <FAQItem q={item.q} a={item.a} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
