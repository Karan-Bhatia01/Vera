import Navbar from "../landing/Navbar";

/** Shared Login/Signup shell: floating navbar, image on the left, form column on the right. */
export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="flex min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <div className="fixed top-4 left-0 right-0 flex justify-center z-50 pointer-events-none">
        <div className="pointer-events-auto">
          <Navbar />
        </div>
      </div>

      <div className="hidden lg:block lg:w-1/2 relative bg-black overflow-hidden">
        <img
          src="/teddy-bg.png"
          alt="Vera"
          className="absolute inset-0 w-full h-full object-cover opacity-90 mix-blend-lighten"
        />
        {/* Gradient fade to merge smoothly with the right side */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent from-85% to-[var(--bg)]" />
      </div>

      <div className="w-full lg:w-1/2 flex flex-col relative z-10 pt-20">
        <div className="flex-1 flex items-center justify-center w-full px-6 sm:px-12">
          <div className="w-full max-w-sm">
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-2">{title}</h1>
            <p className="text-[var(--muted)] mb-10 text-lg">{subtitle}</p>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Labelled text inputs for an auth form. */
export function AuthFields({ fields, form, onChange }) {
  return fields.map((field) => (
    <div key={field.name}>
      <label className="block text-sm font-semibold mb-2">{field.label}</label>
      <input
        name={field.name}
        type={field.type}
        value={form[field.name]}
        onChange={onChange}
        placeholder={field.placeholder}
        className="w-full rounded-xl border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] px-4 py-3.5 text-base text-[var(--text)] outline-none transition-all focus:border-[var(--text)]/[0.3] focus:bg-transparent placeholder:text-[var(--muted)]/50 hover:border-[var(--text)]/[0.2]"
      />
    </div>
  ));
}

export function AuthError({ message }) {
  if (!message) return null;
  return (
    <p className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400 font-medium">
      {message}
    </p>
  );
}
