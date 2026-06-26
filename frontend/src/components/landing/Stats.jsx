import { stats } from "./data";

export default function Stats() {
  return (
    <section className="px-6 pb-24">
      <div className="mx-auto grid max-w-3xl grid-cols-2 gap-4 sm:grid-cols-4">
        {stats.map((s) => (
          <div
            key={s.label}
            className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-6 text-center backdrop-blur-sm"
          >
            <div className="text-3xl font-extrabold">{s.value}</div>
            <div className="mt-1 font-['JetBrains_Mono'] text-[10px] tracking-widest text-[#6e6e66]">
              {s.label}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}