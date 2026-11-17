import { Mascot } from "../brand/Brand";
import { STAGES } from "../brand/stages";

const STEPS = [
  {
    stage: "upload",
    title: "Upload your CSV",
    body: "Drop in a file up to 25,000 rows and 20 columns. Vera works out the delimiter and the text encoding for you.",
    preview: <UploadPreview />,
  },
  {
    stage: "info",
    title: "Get a profile of every column",
    body: "Types, missing values, outliers and correlations, computed exactly. The AI then writes up what stands out and suggests a column to predict.",
    preview: <ProfilePreview />,
  },
  {
    stage: "eda",
    title: "Explore ten charts",
    body: "Vera picks the charts that say something about your data and explains each one in plain English.",
    preview: <ChartPreview />,
  },
  {
    stage: "ml",
    title: "Train and compare models",
    body: "Choose the column to predict. Vera shortlists 3 to 5 models, trains them, and shows which one wins and which columns drive it.",
    preview: <LeaderboardPreview />,
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="scroll-mt-20 border-t border-[var(--line)]">
      <div className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
        <h2 className="text-4xl font-extrabold tracking-tight sm:text-5xl">How it works</h2>
        <p className="mt-4 max-w-[60ch] text-lg text-[var(--muted)]">
          Each step has its own screen, so you can read the results before moving on.
        </p>

        <ol className="relative mt-16 space-y-16">
          {/* The spine joining the four steps */}
          <span className="absolute left-[47px] top-12 bottom-12 hidden w-[3px] rounded-full bg-[var(--line)] md:block" aria-hidden="true" />
          {STEPS.map((step, i) => (
            <li key={step.stage} className="relative grid gap-6 md:grid-cols-[96px_minmax(0,1fr)_minmax(0,340px)] md:gap-10 md:items-center">
              <div className="relative z-10 flex h-24 w-24 items-center justify-center rounded-full border-[3px] bg-[var(--bg)]" style={{ borderColor: STAGES[step.stage].color }}>
                <Mascot stage={step.stage} size={76} />
              </div>
              <div className="max-w-[56ch]">
                <h3 className="text-2xl font-extrabold tracking-tight">
                  <span style={{ color: STAGES[step.stage].ink }}>{i + 1}.</span> {step.title}
                </h3>
                <p className="mt-3 text-lg leading-relaxed text-[var(--muted)]">{step.body}</p>
              </div>
              {step.preview}
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

// Small illustrations of what each screen shows. Example data, not real results.

function PreviewFrame({ label, children }) {
  return (
    <div role="img" aria-label={label} className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 shadow-[0_1px_0_var(--line)]">
      {children}
    </div>
  );
}

function UploadPreview() {
  return (
    <PreviewFrame label="Example: sales_2024.csv uploaded with 18,240 rows and 9 columns">
      <div className="flex items-center gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg" style={{ background: "var(--lime)" }} aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1e2452" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
          </svg>
        </span>
        <div className="min-w-0">
          <p className="truncate font-semibold">sales_2024.csv</p>
          <p className="text-sm text-[var(--muted)] tabular-nums">18,240 rows, 9 columns</p>
        </div>
        <span className="ml-auto text-sm font-semibold" style={{ color: "var(--lime-ink)" }}>Ready</span>
      </div>
    </PreviewFrame>
  );
}

function ProfilePreview() {
  const rows = [["region", "text", "0%"], ["revenue", "number", "2.1%"], ["signup_date", "date", "0%"]];
  return (
    <PreviewFrame label="Example column profile: region, revenue and signup date with their types and missing values">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--muted)]">
            <th className="pb-2 font-semibold">Column</th><th className="pb-2 font-semibold">Type</th><th className="pb-2 text-right font-semibold">Missing</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([col, type, missing]) => (
            <tr key={col} className="border-t border-[var(--line)]">
              <td className="py-2 font-semibold">{col}</td><td className="py-2 text-[var(--muted)]">{type}</td><td className="py-2 text-right font-mono">{missing}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-sm">Suggested column to predict: <strong>churned</strong></p>
    </PreviewFrame>
  );
}

function ChartPreview() {
  const bars = [["North", 82], ["South", 64], ["East", 91], ["West", 47]];
  return (
    <PreviewFrame label="Example bar chart of revenue by region">
      <p className="mb-4 text-sm font-semibold">Revenue by region</p>
      <div className="flex gap-3">
        {bars.map(([name, value]) => (
          <div key={name} className="flex flex-1 flex-col items-center gap-2">
            {/* Fixed-height track so the percentage height has something to resolve against */}
            <div className="flex h-24 w-full items-end">
              <div className="w-full rounded-t-md" style={{ height: `${value}%`, background: "var(--violet)" }} />
            </div>
            <span className="text-xs text-[var(--muted)]">{name}</span>
          </div>
        ))}
      </div>
    </PreviewFrame>
  );
}

function LeaderboardPreview() {
  const models = [["Random forest", "0.91"], ["Gradient boosting", "0.89"], ["Logistic regression", "0.84"]];
  return (
    <PreviewFrame label="Example model leaderboard: random forest wins with 0.91 accuracy">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--muted)]"><th className="pb-2 font-semibold">Model</th><th className="pb-2 text-right font-semibold">Accuracy</th></tr>
        </thead>
        <tbody>
          {models.map(([name, score], i) => (
            <tr key={name} className="border-t border-[var(--line)]">
              <td className="py-2 font-semibold">
                {name}
                {i === 0 && <span className="ml-2 rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-xs font-bold" style={{ color: "var(--sky-ink)" }}>Best</span>}
              </td>
              <td className="py-2 text-right font-mono">{score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </PreviewFrame>
  );
}
