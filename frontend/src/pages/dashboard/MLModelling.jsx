import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../hooks/usePipeline";

export default function MLModelling() {
  const navigate = useNavigate();
  const { ml, filename, addJob } = usePipeline();
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");

  if (!filename) return <Empty message="No dataset selected." onAction={() => navigate("/upload")} actionLabel="Go to Upload" />;

  const targetColumn = localStorage.getItem("targetColumn") || null;
  const isRunning = ml && !["completed","failed"].includes(ml.status);
  const result = ml?.status === "completed" ? ml.result : null;

  const startML = async () => {
    if (!targetColumn) { setStartError("No target column found. Run EDA first from the Dashboard."); return; }
    setStarting(true); setStartError("");
    try {
      const res = await api.post("/api/run_ml", { filename, target_column: targetColumn });
      addJob("ml_job_id", res.data.ml_job_id);
    } catch (err) {
      setStartError(err.response?.data?.message || "Could not start training");
    } finally { setStarting(false); }
  };

  return (
    <div>
      <h1 className="text-3xl font-extrabold">ML Modelling</h1>
      <p className="mt-2 text-[#9a9a93]">
        Training for <span className="text-[#f0ece2] font-semibold">{filename}</span>
        {targetColumn && <> · Target: <span className="text-[#f0ece2] font-semibold">{targetColumn}</span></>}
      </p>

      {/* Start button — always shown unless training is running or done */}
      {!isRunning && !result && (
        <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-8 text-center">
          <p className="text-2xl mb-3">🤖</p>
          <p className="font-semibold mb-2">Ready to train?</p>
          <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono'] mb-6">
            The agent will select the best models, train them, and evaluate feature importance.
          </p>
          {startError && <p className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">✗ {startError}</p>}
          <button onClick={startML} disabled={starting}
            className={`rounded-full px-8 py-3.5 text-base font-semibold ${
              starting ? "bg-white/10 text-[#6e6e66] cursor-not-allowed" : "bg-[#b56126] text-[#0d1117]"
            }`}>
            {starting ? "Starting…" : "Start ML Training →"}
          </button>
        </div>
      )}

      {isRunning && <StatusBar label={ml?.message || "Training models…"} />}
      {ml?.status === "failed" && (
        <>
          <ErrBar msg={ml.error || "Training failed"} />
          <div className="mt-4 flex justify-center">
            <button onClick={startML} className="rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">
              Retry →
            </button>
          </div>
        </>
      )}

      {result && (
        <>
          <div className="mt-6 flex flex-wrap gap-2">
            <Tag label={result.problem_type} />
            <Tag label={`Target: ${result.target_column}`} />
            <Tag label={`Best: ${result.best_model}`} accent />
          </div>

          {result.feature_plan && (
            <Card title="Feature Engineering Plan" className="mt-6">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Group label="Dropped" items={result.feature_plan.drop} color="text-red-400" />
                <Group label="Numeric" items={result.feature_plan.numeric} color="text-[#c97539]" />
                <Group label="One-Hot" items={result.feature_plan.onehot} color="text-emerald-400" />
                <Group label="Ordinal" items={Object.keys(result.feature_plan.ordinal || {})} color="text-amber-400" />
              </div>
            </Card>
          )}

          {result.results && (
            <Card title="Model Comparison" className="mt-4">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[#9a9a93] font-['JetBrains_Mono'] text-xs">
                      <th className="pb-2 pr-4">Model</th>
                      {result.problem_type === "classification"
                        ? <><th className="pb-2 pr-4">Accuracy</th><th className="pb-2 pr-4">F1</th><th className="pb-2 pr-4">Precision</th><th className="pb-2">Recall</th></>
                        : <><th className="pb-2 pr-4">R²</th><th className="pb-2 pr-4">RMSE</th><th className="pb-2">MAE</th></>}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.results).map(([name, data]) => (
                      <tr key={name} className={`border-t border-white/5 ${name === result.best_model ? "bg-[#c9753911]" : ""}`}>
                        <td className="py-2 pr-4 font-semibold">
                          {name}
                          {name === result.best_model && <span className="ml-2 rounded bg-[#b56126] px-1.5 py-0.5 text-[9px] font-bold text-[#0d1117]">BEST</span>}
                        </td>
                        {result.problem_type === "classification"
                          ? <><td className="py-2 pr-4 font-['JetBrains_Mono']">{data.metrics.accuracy}</td><td className="py-2 pr-4 font-['JetBrains_Mono']">{data.metrics.f1}</td><td className="py-2 pr-4 font-['JetBrains_Mono']">{data.metrics.precision}</td><td className="py-2 font-['JetBrains_Mono']">{data.metrics.recall}</td></>
                          : <><td className="py-2 pr-4 font-['JetBrains_Mono']">{data.metrics.r2}</td><td className="py-2 pr-4 font-['JetBrains_Mono']">{data.metrics.rmse}</td><td className="py-2 font-['JetBrains_Mono']">{data.metrics.mae}</td></>}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {result.results?.[result.best_model]?.feature_importance && (
            <Card title={`Feature Importance — ${result.best_model}`} className="mt-4">
              {Object.entries(result.results[result.best_model].feature_importance).slice(0, 10).map(([feat, val]) => {
                const maxVal = Math.max(...Object.values(result.results[result.best_model].feature_importance).map(Math.abs));
                return (
                  <div key={feat} className="mb-3">
                    <div className="flex justify-between text-xs mb-1">
                      <span>{feat}</span>
                      <span className="text-[#c97539] font-['JetBrains_Mono']">{val}</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
                      <div className="h-full bg-[#b56126]" style={{ width: `${Math.round((Math.abs(val)/maxVal)*100)}%` }} />
                    </div>
                  </div>
                );
              })}
            </Card>
          )}



          <div className="mt-6 flex justify-center">
            <button onClick={startML} className="rounded-full border border-white/15 px-6 py-2.5 text-sm font-semibold hover:border-white/30 transition-colors">
              Retrain →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

const Tag = ({ label, accent }) => (
  <span className={`rounded-full px-3 py-1.5 text-xs font-semibold ${accent ? "bg-[#b56126] text-[#0d1117]" : "border border-white/15 text-[#9a9a93]"}`}>{label}</span>
);
const Card = ({ title, children, className = "" }) => (
  <div className={`rounded-2xl border border-white/10 bg-white/[0.03] p-6 ${className}`}>
    {title && <h2 className="text-lg font-bold mb-4">{title}</h2>}
    {children}
  </div>
);
const Group = ({ label, items, color }) => {
  if (!items?.length) return null;
  return (
    <div>
      <p className={`text-xs font-bold ${color} mb-2`}>{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {items.map(i => <span key={i} className="rounded bg-white/5 px-2 py-1 text-[11px] text-[#9a9a93]">{i}</span>)}
      </div>
    </div>
  );
};
const StatusBar = ({ label }) => (
  <div className="mt-6 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
    <Spinner /><p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">{label}</p>
  </div>
);
const ErrBar = ({ msg }) => (
  <p className="mt-6 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">✗ {msg}</p>
);
const Spinner = () => <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-[#c97539]" />;
const Empty = ({ message, onAction, actionLabel }) => (
  <div>
    <h1 className="text-3xl font-extrabold">ML Modelling</h1>
    <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center">
      <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">{message}</p>
      {onAction && <button onClick={onAction} className="mt-4 rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">{actionLabel}</button>}
    </div>
  </div>
);