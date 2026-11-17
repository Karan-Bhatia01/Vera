import { Link } from "react-router-dom";
import { STAGES, STAGE_ORDER } from "../brand/stages";

// The page's one orchestrated moment (CSS keyframes in index.css): the shelf draws,
// then each mascot lands in pipeline order. Reduced-motion users see the final state.
const SHELF_DRAW_MS = 500;
const LAND_GAP_MS = 140;

export default function Hero() {
  return (
    <section className="mx-auto grid max-w-6xl items-center gap-14 px-4 pt-16 pb-20 sm:px-6 lg:grid-cols-[1.05fr_1fr] lg:pt-24 lg:pb-28">
      <div>
        <h1 className="max-w-[16ch] text-[clamp(2.5rem,5.4vw,4.25rem)] font-extrabold leading-[1.0] tracking-[-0.025em]">
          Drop in a spreadsheet. Get charts, insights and a trained model.
        </h1>
        <p className="mt-6 max-w-[52ch] text-lg leading-relaxed text-[var(--muted)]">
          Vera reads your CSV, picks the charts worth looking at, and trains models to predict
          the column you care about. No code and no setup.
        </p>
        <div className="mt-9 flex flex-wrap items-center gap-x-6 gap-y-4">
          <Link
            to="/signup"
            className="rounded-full bg-[var(--text)] px-7 py-3.5 text-base font-bold text-[var(--bg)] transition-opacity hover:opacity-90"
          >
            Upload your first CSV
          </Link>
          <a href="#how" className="text-base font-semibold underline decoration-2 underline-offset-4 decoration-[var(--line)] hover:decoration-[var(--text)]">
            See how it works
          </a>
        </div>
        <p className="mt-6 text-sm text-[var(--muted)]">Free to use. CSVs up to 25,000 rows and 20 columns.</p>
      </div>

      <MascotShelf />
    </section>
  );
}

/** The pipeline as a shelf: one plush per stage, standing in order. */
function MascotShelf() {
  return (
    <figure className="w-full" aria-label="Vera's four steps: upload, profile, explore, model">
      <div className="grid grid-cols-4 items-end gap-2 sm:gap-4">
        {STAGE_ORDER.map((key, i) => (
          <img
            key={key}
            src={STAGES[key].mascot}
            alt={STAGES[key].alt}
            className="mascot-land mx-auto w-full max-w-[150px] select-none drop-shadow-[0_14px_14px_rgba(30,36,82,0.18)]"
            style={{ animationDelay: `${SHELF_DRAW_MS + i * LAND_GAP_MS}ms` }}
            draggable="false"
          />
        ))}
      </div>

      <div className="shelf-draw h-[6px] rounded-full bg-[var(--text)]" style={{ animationDuration: `${SHELF_DRAW_MS}ms` }} />

      <figcaption className="mt-4 grid grid-cols-4 gap-2 sm:gap-4">
        {STAGE_ORDER.map((key) => (
          <span key={key} className="flex items-center justify-center gap-2 text-sm font-semibold sm:text-base">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: STAGES[key].color }} aria-hidden="true" />
            {STAGES[key].label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
