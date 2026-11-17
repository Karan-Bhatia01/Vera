import { motion } from "framer-motion";
import { stats } from "./data";

export default function Stats() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 sm:gap-6">
          {stats.map((s, i) => (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              key={s.label}
              className="group relative overflow-hidden flex flex-col items-center justify-center rounded-3xl border border-[var(--text)]/[0.08] bg-[var(--text)]/[0.02] p-8 sm:p-10 text-center transition-all duration-300 hover:border-[var(--text)]/[0.2] hover:bg-[var(--text)]/[0.04] hover:-translate-y-1"
            >
              {/* Center glow on hover */}
              <div className="absolute inset-0 m-auto h-32 w-32 rounded-full bg-[var(--text)] opacity-0 blur-[50px] transition-opacity duration-500 group-hover:opacity-10" />
              
              <div className="relative z-10">
                <div className="text-4xl sm:text-6xl font-extrabold tracking-tight">
                  {s.value}
                </div>
                <div className="mt-4 font-mono text-xs tracking-widest text-[var(--muted)] uppercase transition-colors duration-300 group-hover:text-[var(--text)]">
                  {s.label}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
