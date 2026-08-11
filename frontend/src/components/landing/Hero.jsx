import { Link } from "react-router-dom";

export default function Hero() {
  return (
    <section className="px-6 pt-20 pb-16 text-center">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-[2.5rem] sm:text-6xl md:text-7xl font-extrabold leading-[1.05] tracking-tight">
          Upload a CSV.
          <br />
          Walk away with a
          <br />
          <span className="text-[var(--yellow)]">trained model.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg text-[var(--muted)]">
          Vera cleans, transforms, and trains over 10 models in parallel,
          delivering enterprise-grade ML insights without the complexity.
        </p>

        <div className="mt-8 flex flex-wrap justify-center gap-4">
          <Link
            to="#"
            onClick={(e) => e.preventDefault()}
            className="rounded-full bg-[var(--blue)] text-[var(--bg)] py-3 px-7 font-semibold"
          >
            Get Started Free →
          </Link>
          <a
            href="#pipeline"
            className="rounded-full border border-[var(--line)] text-[var(--text)] px-7 py-3 font-semibold hover:bg-[var(--line)] transition-colors"
          >
            Watch the Pipeline
          </a>
        </div>
      </div>
    </section>
  );
}
