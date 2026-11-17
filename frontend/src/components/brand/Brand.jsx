import { Link } from "react-router-dom";
import { STAGES, STAGE_ORDER } from "./stages";

/** Wordmark: "Vera" with four felt dots, one per pipeline stage. */
export function BrandMark({ to = "/", className = "" }) {
  return (
    <Link to={to} className={`inline-flex items-center gap-2 text-xl font-extrabold tracking-tight ${className}`}>
      <span className="grid grid-cols-2 gap-[3px]" aria-hidden="true">
        {STAGE_ORDER.map((key) => (
          <span key={key} className="h-[7px] w-[7px] rounded-full" style={{ background: STAGES[key].color }} />
        ))}
      </span>
      Vera
    </Link>
  );
}

/** A stage's plush mascot. Decorative unless `decorative={false}`. */
export function Mascot({ stage, size = 56, decorative = true, className = "" }) {
  const { mascot, alt } = STAGES[stage];
  return (
    <img
      src={mascot}
      alt={decorative ? "" : alt}
      width={size}
      height={size}
      className={`object-contain select-none ${className}`}
      draggable="false"
    />
  );
}

/** Page title for app screens: the stage's mascot, the title, and an optional subtitle line. */
export function PageHeader({ stage, title, children }) {
  return (
    <header className="mb-8 flex items-center gap-3 sm:gap-4 md:mb-10 md:pr-14">
      {stage && <Mascot stage={stage} size={64} className="-my-2 h-12 w-12 shrink-0 sm:h-16 sm:w-16" />}
      <div className="min-w-0">
        <h1 className="text-3xl font-extrabold tracking-tight leading-tight sm:text-4xl">{title}</h1>
        {/* Wrap on phones; one tidy line on wider screens */}
        {children && <p className="mt-1 text-base text-[var(--muted)] break-words md:truncate">{children}</p>}
      </div>
    </header>
  );
}
