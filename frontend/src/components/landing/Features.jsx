import { features } from "./data";

export default function Features() {
  return (
    <section id="features" className="px-6 pb-24">
      <div className="mx-auto max-w-5xl">
        <div className="text-center">
          <p className="font-['JetBrains_Mono'] text-xs tracking-widest text-[#6e6e66]">
            CAPABILITIES
          </p>
          <h2 className="mt-3 text-4xl font-extrabold">What Vera does</h2>
        </div>

        <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur-sm"
            >
              <div className="text-3xl">{f.icon}</div>
              <h3 className="mt-4 text-lg font-bold">{f.title}</h3>
              <p className="mt-2 text-sm text-[#9a9a93]">{f.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}