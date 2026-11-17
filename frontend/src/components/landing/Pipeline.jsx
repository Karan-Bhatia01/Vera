import { motion } from "framer-motion";
import { pipelineSteps } from "./data";

export default function Pipeline() {
  return (
    <section id="pipeline" className="px-6 py-32 scroll-mt-24">
      <div className="mx-auto max-w-5xl">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.5 }}
          className="text-center mb-24"
        >
          <p className="font-mono text-xs tracking-widest text-[var(--muted)] uppercase">
            The Engine
          </p>
          <h2 className="mt-4 text-5xl sm:text-7xl font-extrabold tracking-tight leading-none">
            High-fidelity automation.
          </h2>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-[var(--muted)] leading-relaxed">
            Every upload runs a 3-stage pipeline: profile, explore, model.
          </p>
        </motion.div>

        <ol className="space-y-16 mt-16">
          {pipelineSteps.map((step, i) => {
            const teddies = [
              "/Lime_Plush_Monster_with_Backpack-removebg-preview.png",
              "/Purple_Plush_DJ_with_Smartphone-removebg-preview.png",
              "/Plush_Yellow_Scholar_with_Beret-removebg-preview.png",
              "/Blue_Beret_Plush_Enjoying_Coffee-removebg-preview.png"
            ];

            return (
              <motion.li 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                key={step.n} 
                className="group relative overflow-visible flex flex-col md:flex-row gap-8 md:gap-16 items-start rounded-3xl border border-[var(--text)]/[0.08] bg-[var(--text)]/[0.02] p-8 md:p-12 pt-16 md:pt-12 transition-all duration-300 hover:border-[var(--text)]/[0.2] hover:bg-[var(--text)]/[0.04]"
              >
                {/* Teddy sitting on top left */}
                <img 
                  src={teddies[i % teddies.length]} 
                  alt="Teddy" 
                  className="absolute -top-16 left-8 w-32 h-32 object-contain drop-shadow-2xl z-20 pointer-events-none transform transition-transform duration-500 group-hover:scale-110 group-hover:-rotate-3 origin-bottom"
                />

                {/* Subtle side glow on hover */}
                <div className="absolute -left-12 top-1/2 -translate-y-1/2 h-48 w-48 rounded-full bg-[var(--text)] opacity-0 blur-[60px] transition-opacity duration-500 group-hover:opacity-10 pointer-events-none z-0" />

                <span className="relative z-10 font-mono text-4xl text-[var(--text)] opacity-30 shrink-0 mt-1 transition-opacity duration-300 group-hover:opacity-60">
                  {step.n}
                </span>
                <div className="relative z-10">
                  <h3 className="text-3xl font-extrabold tracking-tight">{step.title}</h3>
                  <p className="mt-4 text-lg text-[var(--muted)] max-w-2xl leading-relaxed">
                    {step.body}
                  </p>
                </div>
              </motion.li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}
