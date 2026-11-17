import { motion } from "framer-motion";
import { features } from "./data";

export default function Features() {
  return (
    <section id="features" className="px-6 py-32">
      <div className="mx-auto max-w-6xl">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.5 }}
          className="text-center mb-24"
        >
          <p className="font-mono text-xs tracking-widest text-[var(--muted)] uppercase">
            Capabilities
          </p>
          <h2 className="mt-4 text-5xl sm:text-7xl font-extrabold tracking-tight leading-none">
            What Vera does.
          </h2>
        </motion.div>

        <div className="grid gap-x-6 gap-y-16 sm:grid-cols-2 lg:grid-cols-4 mt-16">
          {features.map((f, i) => {
            const teddies = [
              "/Plush_Yellow_Scholar_with_Beret-removebg-preview.png",
              "/Blue_Beret_Plush_Enjoying_Coffee-removebg-preview.png",
              "/Lime_Plush_Monster_with_Backpack-removebg-preview.png",
              "/Purple_Plush_DJ_with_Smartphone-removebg-preview.png"
            ];
            
            return (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                key={f.title}
                className="group relative overflow-visible rounded-3xl border border-[var(--text)]/[0.08] bg-[var(--text)]/[0.02] p-8 pt-12 transition-all duration-300 hover:border-[var(--text)]/[0.2] hover:bg-[var(--text)]/[0.04] hover:-translate-y-1 mt-8 lg:mt-0"
              >
                {/* Teddy sitting on top left */}
                <img 
                  src={teddies[i % teddies.length]} 
                  alt="Teddy" 
                  className="absolute -top-16 left-4 w-28 h-28 object-contain drop-shadow-xl z-20 pointer-events-none transform transition-transform duration-500 group-hover:scale-110 group-hover:rotate-6 origin-bottom"
                />

                {/* Subtle top glow on hover */}
                <div className="absolute -top-12 -left-12 h-48 w-48 rounded-full bg-[var(--text)] opacity-0 blur-[60px] transition-opacity duration-500 group-hover:opacity-20 pointer-events-none z-0" />
                
                <div className="relative z-10">
                  <span className="font-mono text-sm text-[var(--text)] opacity-30 mb-6 block transition-opacity duration-300 group-hover:opacity-60">
                    0{i + 1}
                  </span>
                  <h3 className="text-2xl font-extrabold tracking-tight mb-4">
                    {f.title}
                  </h3>
                  <p className="text-[var(--muted)] leading-relaxed">
                    {f.body}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
