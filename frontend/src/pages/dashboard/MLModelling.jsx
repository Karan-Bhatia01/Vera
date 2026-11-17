import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";
import { motion } from "framer-motion";
import { Mascot, PageHeader } from "../../components/brand/Brand";

const IDLE_ANALYSIS = { mlJobId: null, jobId: null, data: null, error: "" };

export default function MLModelling() {
  const navigate = useNavigate();
  const { ml, filename, addJob } = usePipeline();
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");
  // The executive analysis runs as a background job; it belongs to one ML run.
  const [analysis, setAnalysis] = useState(IDLE_ANALYSIS);

  // Poll the analysis job every 3s until it finishes.
  useEffect(() => {
    if (!analysis.jobId) return;
    const finish = (patch) => setAnalysis((a) => ({ ...a, jobId: null, ...patch }));
    const id = setInterval(async () => {
      try {
        const { data } = await api.get(`/api/pipeline_status/analysis/${analysis.jobId}`);
        if (data.status === "completed") finish({ data: data.result });
        else if (data.status === "failed") finish({ error: data.error || "Failed to analyze metrics." });
      } catch (err) {
        finish({ error: err.response?.data?.message || "Failed to analyze metrics." });
      }
    }, 3000);
    return () => clearInterval(id);
  }, [analysis.jobId]);

  if (!filename)
    return (
      <Empty
        message="No dataset selected."
        onAction={() => navigate("/upload")}
        actionLabel="Open Upload"
      />
    );

  const targetColumn = localStorage.getItem("targetColumn") || null;
  const isRunning = ml && !["completed", "failed"].includes(ml.status);
  const result = ml?.status === "completed" ? ml.result : null;

  const startML = async () => {
    if (!targetColumn) {
      setStartError("No target column found. Run EDA first from the Dashboard.");
      return;
    }
    setStarting(true);
    setStartError("");
    try {
      const res = await api.post("/api/run_ml", {
        filename,
        target_column: targetColumn,
      });
      addJob("ml_job_id", res.data.ml_job_id);
    } catch (err) {
      setStartError(err.response?.data?.message || "Could not start training");
    } finally {
      setStarting(false);
    }
  };

  // Only show analysis that was generated for the ML run currently on screen.
  const current = analysis.mlJobId === ml?.job_id ? analysis : IDLE_ANALYSIS;
  const analysisResult = current.data;
  const analysisLoading = Boolean(current.jobId);

  const getSimpleAnalysis = async () => {
    if (!ml?.job_id) return;
    const mlJobId = ml.job_id;
    try {
      const res = await api.post(`/api/analyze_ml/${mlJobId}`);
      setAnalysis({ ...IDLE_ANALYSIS, mlJobId, jobId: res.data.analysis_job_id });
    } catch (err) {
      setAnalysis({ ...IDLE_ANALYSIS, mlJobId, error: err.response?.data?.message || "Failed to analyze metrics." });
    }
  };

  return (
    <div className="min-h-full pb-20">
      <PageHeader stage="ml" title="ML modelling">
        {filename}
        {targetColumn && <>, predicting <strong className="text-[var(--text)]">{targetColumn}</strong></>}
      </PageHeader>

      {/* Start Action State */}
      {!isRunning && !result && (
        <div className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 sm:p-10">
          <h2 className="text-3xl font-extrabold tracking-tight mb-3">Train models for this dataset</h2>
          <p className="max-w-[60ch] text-lg text-[var(--muted)] mb-8">
            Vera prepares the columns, shortlists 3 to 5 models, trains each one and ranks them by how well they predict your target.
          </p>

          {startError && (
            <p role="alert" className="mb-6 max-w-md rounded-xl border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-4 py-3 font-semibold text-[var(--danger)]">
              {startError}
            </p>
          )}

          <button
            onClick={startML}
            disabled={starting}
            className="inline-flex items-center gap-3 rounded-full bg-[var(--text)] px-8 py-3.5 text-base font-bold text-[var(--bg)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {starting ? (
              <>
                <Spinner /> Starting…
              </>
            ) : (
              "Start training"
            )}
          </button>
        </div>
      )}

      {/* Loading State */}
      {isRunning && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-2xl border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.03] p-5 sm:p-8"
        >
          <div className="flex items-center gap-6">
            <div className="relative flex h-12 w-12 items-center justify-center rounded-full bg-[var(--blue)]/[0.1]">
              <div className="absolute inset-0 animate-ping rounded-full border border-[var(--blue)]/[0.4]" />
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--blue)] border-t-transparent" />
            </div>
            <div>
              <h3 className="text-xl font-bold mb-1">Training models</h3>
              <p className="text-[var(--muted)]">{ml?.message || "Training models…"}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Error State */}
      {ml?.status === "failed" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="rounded-2xl border border-[var(--danger)]/30 bg-[var(--danger)]/10 p-8">
            <h3 className="text-xl font-bold text-[var(--danger)] mb-2">Training failed</h3>
            <p className="text-[var(--danger)] mb-6">{ml.error}</p>
            <button
              onClick={startML}
              className="rounded-full bg-[var(--text)] px-7 py-3 font-bold text-[var(--bg)] transition-opacity hover:opacity-90"
            >
              Try again
            </button>
          </div>
        </motion.div>
      )}

      {/* Results Dashboard */}
      {result && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-8 mt-8"
        >
          {/* Top Metadata Row */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="rounded-full border border-[var(--blue)]/[0.4] bg-[var(--blue)]/[0.1] px-5 py-2 text-sm font-bold text-[var(--blue)] capitalize">
              {result.problem_type}
            </div>
            <div className="rounded-full border border-[var(--line)] bg-[var(--surface)] px-5 py-2 text-sm font-medium">
              Target: <span className="font-bold">{result.target_column}</span>
            </div>
            <div className="flex-1" />
            <div className="rounded-full border border-[var(--line)] bg-[var(--text)] px-5 py-2 text-sm font-bold text-[var(--bg)]">
              Best Model: {result.best_model}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
            {/* Left Column: Feature Plan & Models */}
            <div className="min-w-0 lg:col-span-2 space-y-8">
              {result.feature_plan && (
                <GlassCard title="How each column was used">
                  <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                    <FeatureGroup label="Dropped" items={result.feature_plan.drop} color="text-[var(--danger)]" border="border-[var(--danger)]/30" />
                    <FeatureGroup label="Numeric" items={result.feature_plan.numeric} color="text-[var(--sky-ink)]" border="border-[var(--sky)]/40" />
                    <FeatureGroup label="One-Hot" items={result.feature_plan.onehot} color="text-[var(--lime-ink)]" border="border-[var(--lime)]/60" />
                    <FeatureGroup label="Ordinal" items={Object.keys(result.feature_plan.ordinal || {})} color="text-[var(--sun-ink)]" border="border-[var(--sun)]/50" />
                  </div>
                </GlassCard>
              )}

              {result.results && (
                <GlassCard title="Model leaderboard">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-[var(--line)] text-left text-[var(--muted)] text-sm">
                          <th className="pb-4 font-normal whitespace-nowrap pr-4">Model</th>
                          {result.problem_type === "classification" ? (
                            <>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">Accuracy</th>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">F1 Score</th>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">Precision</th>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">Recall</th>
                            </>
                          ) : (
                            <>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">R² Score</th>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">RMSE</th>
                              <th className="pb-4 font-normal whitespace-nowrap pr-4">MAE</th>
                            </>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.results).map(([name, data]) => {
                          const isBest = name === result.best_model;
                          return (
                            <tr
                              key={name}
                              className={`border-b border-[var(--line)] transition-colors hover:bg-[var(--surface)] ${
                                isBest ? "bg-[var(--text)]/[0.03]" : ""
                              }`}
                            >
                              <td className="py-4 pr-4 font-medium flex items-center gap-2">
                                {name}
                                {isBest && (
                                  <span className="rounded-md bg-[var(--text)] px-2 py-0.5 text-sm font-bold text-[var(--bg)]">Best</span>
                                )}
                              </td>
                              {result.problem_type === "classification" ? (
                                <>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.accuracy}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.f1}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.precision}</td>
                                  <td className={`py-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.recall}</td>
                                </>
                              ) : (
                                <>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.r2}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.rmse}</td>
                                  <td className={`py-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--muted)]'}`}>{data.metrics.mae}</td>
                                </>
                              )}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>
              )}

              {/* AI Analysis Button / Card */}
              {analysisLoading ? (
                <div className="rounded-2xl border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] p-5 sm:p-8">
                  <div className="flex items-center gap-5">
                    <div className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-[var(--blue)]/[0.1]">
                      <div className="absolute inset-0 animate-ping rounded-full border border-[var(--blue)]/[0.4]" />
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--blue)] border-t-transparent" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold mb-1 text-[var(--blue)]">Generating executive analysis…</h3>
                      <p className="text-sm text-[var(--muted)]">
                        The AI is turning these metrics into a plain-English summary. This can take a minute.
                      </p>
                    </div>
                  </div>
                </div>
              ) : !analysisResult ? (
                <div className="relative overflow-hidden rounded-2xl border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] p-5 sm:p-8 text-center transition-all hover:bg-[var(--blue)]/[0.05]">
                  <Mascot stage="ml" size={64} className="mb-3" />
                  <h3 className="text-lg font-bold mb-2">Explain these results</h3>
                  <p className="text-sm text-[var(--muted)] mb-6">Get a plain-English summary of what the scores mean for your target.</p>

                  {current.error && <p role="alert" className="mb-4 font-semibold text-[var(--danger)]">{current.error}</p>}

                  <button
                    onClick={getSimpleAnalysis}
                    className="w-full rounded-full bg-[var(--blue)]/[0.1] border border-[var(--blue)]/[0.2] px-6 py-3 text-sm font-bold text-[var(--blue)] transition-colors hover:bg-[var(--blue)]/[0.2]"
                  >
                    Explain the results
                  </button>
                </div>
              ) : (
                <GlassCard className="border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] relative overflow-hidden">
                  <h2 className="text-xl font-bold mb-6 text-[var(--blue)]">What these results mean</h2>
                  
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-bold text-[var(--muted)] mb-2">Summary</h3>
                      <p className="text-sm font-medium leading-relaxed">{analysisResult.executive_summary}</p>
                    </div>
                    
                    <div className="h-[1px] w-full bg-[var(--text)]/[0.05]" />
                    
                    <div>
                      <h3 className="text-sm font-bold text-[var(--muted)] mb-2">Business impact</h3>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">{analysisResult.business_impact}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-bold text-[var(--muted)] mb-2">How the model works</h3>
                      <p className="text-sm text-[var(--muted)] leading-relaxed italic border-l-2 border-[var(--blue)]/[0.3] pl-3">{analysisResult.simple_explanation}</p>
                    </div>

                    <div className="h-[1px] w-full bg-[var(--text)]/[0.05]" />

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h3 className="text-sm font-bold text-[var(--muted)] mb-2">Top drivers</h3>
                        <ul className="text-xs text-[var(--muted)] space-y-1.5">
                          {analysisResult.top_features?.map((f, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-[var(--blue)] mt-0.5">•</span>
                              <span>{f}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-[var(--muted)] mb-2">Confidence</h3>
                        <div className="inline-flex items-center rounded-lg border border-[var(--line)] bg-[var(--surface)] px-3 py-1.5">
                          <div className={`h-2 w-2 rounded-full mr-2 ${
                            analysisResult.confidence_level?.toLowerCase() === 'high' ? 'bg-[var(--lime)]' 
                            : analysisResult.confidence_level?.toLowerCase() === 'medium' ? 'bg-[var(--sun)]' 
                            : 'bg-[var(--danger)]'
                          }`} />
                          <span className="text-xs font-bold">{analysisResult.confidence_level}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              )}
            </div>

            {/* Right Column: Feature importance & Analysis */}
            <div className="min-w-0 space-y-8">
              {result.results?.[result.best_model]?.feature_importance && (
                <GlassCard title="Feature importance" subtitle={`Based on ${result.best_model}`}>
                  <div className="space-y-4">
                    {Object.entries(result.results[result.best_model].feature_importance)
                      .slice(0, 8)
                      .map(([feat, val]) => {
                        const maxVal = Math.max(...Object.values(result.results[result.best_model].feature_importance).map(Math.abs));
                        const pct = Math.round((Math.abs(val) / maxVal) * 100);
                        return (
                          <div key={feat} className="group">
                            <div className="flex justify-between text-xs mb-1.5 font-medium">
                              <span className="truncate pr-4">{feat}</span>
                              <span className="font-mono text-[var(--muted)]">{val}</span>
                            </div>
                            <div className="h-1.5 w-full rounded-full bg-[var(--text)]/[0.05] overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ duration: 1, ease: "easeOut" }}
                                className="h-full bg-[var(--text)] rounded-full"
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </GlassCard>
              )}

            </div>
          </div>

          <div className="mt-12 flex justify-center pb-12">
            <button
              onClick={startML}
              className="rounded-full border border-[var(--line)] bg-transparent px-8 py-3 text-sm font-bold transition-all hover:bg-[var(--text)]/[0.05]"
            >
              Train again
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
}

const GlassCard = ({ title, subtitle, children, className = "" }) => (
  <div className={`rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 sm:p-8 ${className}`}>
    {(title || subtitle) && (
      <div className="mb-8">
        {title && <h2 className="text-xl font-bold tracking-tight">{title}</h2>}
        {subtitle && <p className="text-sm text-[var(--muted)] mt-1">{subtitle}</p>}
      </div>
    )}
    {children}
  </div>
);

const FeatureGroup = ({ label, items, color, border }) => {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className={`text-sm font-bold ${color} mb-3`}>{label}</h3>
      <div className="flex flex-wrap gap-2">
        {items.map((i) => (
          <span
            key={i}
            className={`rounded-lg border ${border} bg-[var(--surface)] px-2.5 py-1 text-[11px] font-medium text-[var(--muted)]`}
          >
            {i}
          </span>
        ))}
      </div>
    </div>
  );
};

const Spinner = () => (
  <svg className="animate-spin h-5 w-5 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const Empty = ({ message, onAction, actionLabel }) => (
  <div className="flex flex-col items-start gap-4 rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 sm:flex-row sm:items-center sm:gap-6 sm:p-8">
    <Mascot stage="ml" size={88} className="shrink-0" />
    <div>
      <h2 className="mb-2 text-xl font-bold tracking-tight">{message}</h2>
      <p className="mb-6 max-w-[56ch] text-[var(--muted)]">
        Upload a dataset and pick the column to predict on the Dashboard. Your models will appear here.
      </p>
      {onAction && (
        <button
          onClick={onAction}
          className="rounded-full bg-[var(--text)] px-7 py-3 font-bold text-[var(--bg)] transition-opacity hover:opacity-90"
        >
          {actionLabel}
        </button>
      )}
    </div>
  </div>
);
