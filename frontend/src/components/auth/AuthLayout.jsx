import Navbar from "../landing/Navbar";
import { Mascot } from "../brand/Brand";
import { STAGES, STAGE_ORDER } from "../brand/stages";

const STAGE_LINES = {
  upload: "Drop in any CSV",
  info: "Every column profiled",
  eda: "Ten charts, explained",
  ml: "Models trained and compared",
};

/** Shared Login/Signup shell: form on the left, the four Vera steps on the right. */
export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto grid max-w-6xl gap-12 px-4 py-14 sm:px-6 lg:grid-cols-[minmax(0,26rem)_1fr] lg:gap-20 lg:py-20">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">{title}</h1>
          <p className="mt-3 mb-10 text-lg text-[var(--muted)]">{subtitle}</p>
          {children}
        </div>

        <aside className="hidden self-start rounded-3xl border border-[var(--line)] bg-[var(--surface)] p-10 lg:block">
          <p className="max-w-[24ch] text-2xl font-extrabold leading-snug tracking-tight">
            Upload a CSV and Vera handles the rest.
          </p>
          <ul className="mt-8 space-y-5">
            {STAGE_ORDER.map((key) => (
              <li key={key} className="flex items-center gap-4">
                <span className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full border-[3px]" style={{ borderColor: STAGES[key].color }}>
                  <Mascot stage={key} size={50} />
                </span>
                <span>
                  <span className="block font-bold">{STAGES[key].label}</span>
                  <span className="text-[var(--muted)]">{STAGE_LINES[key]}</span>
                </span>
              </li>
            ))}
          </ul>
        </aside>
      </main>
    </div>
  );
}

/** Labelled text inputs for an auth form. */
export function AuthFields({ fields, form, onChange }) {
  return fields.map((field) => (
    <div key={field.name}>
      <label htmlFor={field.name} className="block text-[15px] font-semibold mb-2">{field.label}</label>
      <input
        id={field.name}
        name={field.name}
        type={field.type}
        autoComplete={field.autoComplete}
        value={form[field.name]}
        onChange={onChange}
        placeholder={field.placeholder}
        className="w-full rounded-xl border-2 border-[var(--line)] bg-[var(--surface)] px-4 py-3 text-base text-[var(--text)] outline-none transition-colors placeholder:text-[var(--muted)]/70 hover:border-[var(--muted)]/50 focus:border-[var(--sky-ink)]"
      />
    </div>
  ));
}

export function AuthError({ message }) {
  if (!message) return null;
  return (
    <p role="alert" className="rounded-xl border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-4 py-3 text-[15px] font-semibold text-[var(--danger)]">
      {message}
    </p>
  );
}

/** Primary submit button for the auth forms. */
export function AuthSubmit({ loading, children, loadingText }) {
  return (
    <button
      type="submit"
      disabled={loading}
      className="w-full rounded-full bg-[var(--text)] py-3.5 text-base font-bold text-[var(--bg)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading ? loadingText : children}
    </button>
  );
}
