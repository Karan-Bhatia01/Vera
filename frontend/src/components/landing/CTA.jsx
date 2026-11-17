import { Link } from "react-router-dom";
import { Mascot } from "../brand/Brand";
import { STAGE_ORDER } from "../brand/stages";

export default function CTA() {
  return (
    <section className="px-4 pb-24 sm:px-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 rounded-3xl bg-[var(--text)] px-8 py-12 text-[var(--bg)] sm:px-12 md:flex-row md:items-center">
        <div className="flex -space-x-3" aria-hidden="true">
          {STAGE_ORDER.map((key) => (
            <Mascot key={key} stage={key} size={72} />
          ))}
        </div>
        <div className="md:mr-auto">
          <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Your first chart is one upload away.</h2>
          <p className="mt-2 text-lg opacity-80">Create a free account and bring any CSV.</p>
        </div>
        <Link
          to="/signup"
          className="self-start rounded-full bg-[var(--bg)] px-7 py-3.5 text-base font-bold text-[var(--text)] transition-opacity hover:opacity-90 md:self-center"
        >
          Create a free account
        </Link>
      </div>
    </section>
  );
}
