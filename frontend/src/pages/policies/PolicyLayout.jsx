import { motion } from "framer-motion";
import Navbar from "../../components/landing/Navbar";
import Footer from "../../components/landing/Footer";

export function PolicySection({ number, title, children }) {
  return (
    <section className="relative group">
      <div className="flex flex-col md:flex-row gap-4 md:gap-8 rounded-3xl border-2 border-[var(--text)]/[0.15] bg-[var(--text)]/[0.02] p-8 sm:p-10 transition-all duration-300 hover:border-[var(--text)]/[0.25] hover:bg-[var(--text)]/[0.04]">
        <div className="shrink-0 pt-1.5 md:w-12">
          <span className="font-mono text-sm text-[var(--text)] opacity-40 transition-opacity duration-300 group-hover:opacity-100">
            {number < 10 ? `0${number}` : number}
          </span>
        </div>
        <div>
          <h2 className="text-2xl font-bold tracking-tight mb-4 text-[var(--text)]">{title}</h2>
          <div className="text-lg leading-relaxed text-[var(--muted)] space-y-4">
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}

export default function PolicyLayout({ title, lastUpdated, children }) {
  return (
    <div className="min-h-screen relative flex flex-col bg-[var(--bg)] text-[var(--text)] selection:bg-[var(--text)] selection:text-[var(--bg)]">
      {/* Subtle ambient light from top */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-[400px] bg-[var(--text)] opacity-[0.02] blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 flex flex-col flex-1">
        <Navbar />
        
        <main className="flex-1 px-6 py-32 md:py-48 max-w-4xl mx-auto w-full relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="font-mono text-xs tracking-widest text-[var(--muted)] uppercase mb-8">
              Legal & Compliance
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight leading-none mb-12">
              {title}
            </h1>
            <div className="font-mono text-sm text-[var(--text)] opacity-40 mb-20 pb-8 border-b border-[var(--text)]/[0.08]">
              Last updated: {lastUpdated}
            </div>
            
            <div className="space-y-16">
              {children}
            </div>
          </motion.div>
        </main>
        
        <Footer />
      </div>
    </div>
  );
}
