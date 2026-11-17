import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function CTA() {
  return (
    <section className="px-6 py-40 text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true, margin: "-50px" }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <h2 className="text-6xl sm:text-8xl font-extrabold tracking-tight">Begin.</h2>
        <p className="mt-6 text-xl sm:text-2xl text-[var(--muted)]">
          Start with a free upload. No credit card required.
        </p>
        <Link
          to="/signup"
          className="mt-12 inline-block rounded-md bg-[var(--text)] px-10 py-4 font-bold text-[var(--bg)] transition hover:opacity-80 text-lg"
        >
          Upload Your First Dataset
        </Link>
      </motion.div>
    </section>
  );
}
