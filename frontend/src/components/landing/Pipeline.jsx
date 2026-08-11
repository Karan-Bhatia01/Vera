import { pipelineSteps } from "./data";

export default function Pipeline() {
  return (
    <section id="pipeline" className="px-6 pb-24 scroll-mt-24">
      <div className="mx-auto max-w-4xl">
        <p className="flex items-center gap-3 font-['JetBrains_Mono'] text-xs tracking-widest text-[#6e6e66]">
          THE ENGINE <span className="h-px w-10 bg-[var(--blue)]" />
        </p>
        <h2 className="mt-3 text-4xl sm:text-5xl font-extrabold">
          High-fidelity automation.
        </h2>
        <p className="mt-4 max-w-xl text-[var(--muted)]">
          Every upload triggers a 9-stage intelligent pipeline that transforms
          raw data into deployable intelligence.
        </p>

        <ol className="mt-12 space-y-8 border-l border-[var(--line)] pl-8">
          {pipelineSteps.map((step) => (
            <li key={step.n} className="relative">
              <span className="absolute -left-[41px] top-0 flex h-8 w-8 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--bg)] font-['JetBrains_Mono'] text-xs text-[var(--muted)]">
                {step.n}
              </span>
              <h3 className="text-xl font-bold">{step.title}</h3>
              <p className="mt-1 max-w-xl text-[var(--muted)]">{step.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
