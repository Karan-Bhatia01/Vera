import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const fade = {
  hidden: { opacity: 0, y: 14 },
  show: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.18, duration: 0.7, ease: "easeOut" },
  }),
};

export default function Hero() {
  return (
    <section className="hero">
      {/* Abstract geometric background elements */}
      <div className="hero__graphics">
        {/* Top Left: Orbital Rings (Image 2 style) */}
        <motion.div
          className="graphic graphic--orbit"
          animate={{ rotate: 360 }}
          transition={{ duration: 160, repeat: Infinity, ease: "linear" }}
        >
          <svg viewBox="0 0 600 600" fill="none">
            {/* Outer warm ring */}
            <circle cx="300" cy="300" r="260" stroke="var(--orb-outer)" strokeWidth="1.5" />
            {/* Inner neutral ring */}
            <circle cx="300" cy="300" r="140" stroke="var(--orb-inner)" strokeWidth="1" />
            {/* Crisp white dot on outer ring */}
            <circle cx="40" cy="300" r="7" fill="var(--orb-dot)" />
            {/* Dim solid circle */}
            <circle cx="460" cy="420" r="65" fill="var(--orb-fill)" />
          </svg>
        </motion.div>

        {/* Bottom Right: Pill & Circle intersection (Image 1/3 style) */}
        <motion.div
          className="graphic graphic--pill"
          animate={{ y: [0, -25, 0], x: [0, 15, 0], rotate: [0, -3, 0] }}
          transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
        >
          <svg viewBox="0 0 500 300" fill="none">
            <g transform="rotate(-15 250 150)">
              {/* Pill outline (blue tint) */}
              <rect x="50" y="75" width="400" height="150" rx="75" stroke="var(--pill-stroke)" strokeWidth="1.5" />
              {/* Solid intersecting circle */}
              <circle cx="110" cy="200" r="60" fill="var(--pill-fill)" />
            </g>
          </svg>
        </motion.div>

        {/* Top Right: Technical dashed circle (Gold/yellow tint) */}
        <motion.div
          className="graphic graphic--circle"
          animate={{ scale: [1, 1.05, 1], rotate: [0, 90, 0] }}
          transition={{ duration: 35, repeat: Infinity, ease: "easeInOut" }}
        >
          <svg viewBox="0 0 400 400" fill="none">
            <circle cx="200" cy="200" r="180" stroke="var(--circ-stroke)" strokeWidth="1" strokeDasharray="4 12" />
            <circle cx="200" cy="20" r="4" fill="var(--circ-fill)" />
          </svg>
        </motion.div>
      </div>

      {/* Ambient glow */}
      <div className="hero__glow" />

      <div className="hero__content">
        <motion.h1
          className="hero__wordmark"
          variants={fade}
          initial="hidden"
          animate="show"
          custom={0}
        >
          Vera
        </motion.h1>

        <motion.p
          className="hero__tagline"
          variants={fade}
          initial="hidden"
          animate="show"
          custom={1}
        >
          Vera profiles your data, charts what matters, and trains the <br/> best-fit models for your target, with no code.
        </motion.p>

        <motion.div
          className="hero__actions"
          variants={fade}
          initial="hidden"
          animate="show"
          custom={2}
        >
          <Link to="/signup" className="hero__btn hero__btn--primary">
            Start for free
          </Link>
          <a href="#pipeline" className="hero__btn hero__btn--ghost">
            See the pipeline
          </a>
        </motion.div>
      </div>

      <style>{`
        .hero {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          padding: 0 1.5rem;
          overflow: hidden;
        }

        /* --- Graphics --- */
        .hero__graphics {
          position: absolute;
          inset: 0;
          overflow: hidden;
          pointer-events: none;
          z-index: 0;
        }
        .graphic {
          position: absolute;
        }
        .graphic--orbit {
          top: -15%;
          left: -10%;
          width: max(500px, 45vw);
          height: max(500px, 45vw);
        }
        .graphic--pill {
          bottom: 0%;
          right: -5%;
          width: max(450px, 40vw);
          height: max(270px, 24vw);
        }
        .graphic--circle {
          top: 10%;
          right: 5%;
          width: max(300px, 25vw);
          height: max(300px, 25vw);
        }

        .hero__glow {
          position: absolute;
          z-index: 0;
          width: min(550px, 80vw);
          height: min(550px, 80vw);
          border-radius: 50%;
          background: radial-gradient(
            circle,
            color-mix(in srgb, var(--blue) 8%, transparent) 0%,
            transparent 70%
          );
          pointer-events: none;
        }

        .hero__content {
          position: relative;
          z-index: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0;
          text-align: center;
        }

        .hero__wordmark {
          font-family: "Manrope", system-ui, sans-serif;
          font-size: clamp(8rem, 22vw, 20rem);
          font-weight: 800;
          letter-spacing: -0.045em;
          line-height: 0.85;
          color: var(--text);
          user-select: none;
          position: relative;
          z-index: 2;
        }

        .hero__tagline {
          margin-top: 1.8rem;
          font-size: clamp(0.95rem, 1.4vw, 1.15rem);
          color: var(--muted);
          letter-spacing: 0.015em;
          font-weight: 400;
          opacity: 0.7;
          line-height: 1.5;
        }

        .hero__actions {
          display: flex;
          gap: 0.75rem;
          margin-top: 2.8rem;
        }

        .hero__btn {
          display: inline-block;
          padding: 0.65rem 1.6rem;
          font-size: 0.85rem;
          font-weight: 500;
          letter-spacing: 0.01em;
          border-radius: 5px;
          transition: opacity 0.2s, background 0.2s, border-color 0.2s;
          text-decoration: none;
        }

        .hero__btn--primary {
          background: var(--text);
          color: var(--bg);
        }
        .hero__btn--primary:hover {
          opacity: 0.8;
        }

        .hero__btn--ghost {
          border: 1px solid color-mix(in srgb, var(--text) 15%, transparent);
          color: color-mix(in srgb, var(--text) 70%, transparent);
        }
        .hero__btn--ghost:hover {
          background: color-mix(in srgb, var(--text) 5%, transparent);
          border-color: color-mix(in srgb, var(--text) 30%, transparent);
          color: var(--text);
        }
      `}</style>
    </section>
  );
}
