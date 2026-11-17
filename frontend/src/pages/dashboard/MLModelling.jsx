import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";
import { motion } from "framer-motion";

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
        actionLabel="Go to Upload"
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
      <header className="mb-12">
        <h1 className="text-5xl font-extrabold tracking-tighter mb-4">ML Modelling</h1>
        <div className="inline-flex items-center gap-3 rounded-full border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] px-4 py-2 backdrop-blur-md">
          <span className="text-[var(--text)]/[0.6] text-sm">Dataset:</span>
          <span className="text-[var(--text)] font-semibold text-sm">{filename}</span>
          {targetColumn && (
            <>
              <div className="h-4 w-[1px] bg-[var(--text)]/[0.1]" />
              <span className="text-[var(--text)]/[0.6] text-sm">Target:</span>
              <span className="text-[var(--text)] font-semibold text-sm">{targetColumn}</span>
            </>
          )}
        </div>
      </header>

      {/* Start Action State */}
      {!isRunning && !result && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-[2rem] border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] p-12 text-center backdrop-blur-xl"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-[var(--blue)]/[0.05] to-transparent pointer-events-none" />
          <h2 className="text-4xl font-bold tracking-tight mb-4">Automate the Intelligence</h2>
          <p className="mx-auto max-w-xl text-lg text-[var(--text)]/[0.6] font-light mb-10">
            Our pipeline will clean your data, engineer optimal features, and competitively train multiple machine learning models to find the absolute best predictor for your target.
          </p>
          
          {startError && (
            <div className="mx-auto mb-8 max-w-md rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400">
              <p className="font-mono text-sm">✗ {startError}</p>
            </div>
          )}

          <button
            onClick={startML}
            disabled={starting}
            className={`group relative inline-flex items-center justify-center gap-3 rounded-full px-10 py-5 text-lg font-bold transition-all overflow-hidden ${
              starting
                ? "border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.05] text-[var(--text)]/[0.4] cursor-not-allowed"
                : "bg-[var(--text)] text-[var(--bg)] hover:scale-105 hover:shadow-[0_0_40px_var(--text)]/[0.3]"
            }`}
          >
            {starting ? (
              <span className="flex items-center gap-2">
                <Spinner /> Initializing Pipeline...
              </span>
            ) : (
              <>
                <span className="relative z-10">Start ML Training</span>
                <span className="relative z-10 transition-transform group-hover:translate-x-1">→</span>
              </>
            )}
          </button>
        </motion.div>
      )}

      {/* Loading State */}
      {isRunning && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-[2rem] border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.03] p-8 backdrop-blur-xl"
        >
          <div className="flex items-center gap-6">
            <div className="relative flex h-12 w-12 items-center justify-center rounded-full bg-[var(--blue)]/[0.1]">
              <div className="absolute inset-0 animate-ping rounded-full border border-[var(--blue)]/[0.4]" />
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--blue)] border-t-transparent" />
            </div>
            <div>
              <h3 className="text-xl font-bold mb-1">Training Models</h3>
              <p className="font-mono text-sm text-[var(--text)]/[0.6]">{ml?.message || "Training models..."}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Error State */}
      {ml?.status === "failed" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="rounded-[2rem] border border-red-500/30 bg-red-500/10 p-8">
            <h3 className="text-xl font-bold text-red-400 mb-2">Pipeline Failed</h3>
            <p className="font-mono text-sm text-red-400/80 mb-6">{ml.error}</p>
            <button
              onClick={startML}
              className="rounded-full bg-red-500 px-8 py-3 text-sm font-bold text-white hover:bg-red-600 transition-colors"
            >
              Retry Pipeline
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
            <div className="rounded-full border border-[var(--blue)]/[0.4] bg-[var(--blue)]/[0.1] px-5 py-2 text-sm font-bold text-[var(--blue)]">
              {result.problem_type.toUpperCase()}
            </div>
            <div className="rounded-full border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] px-5 py-2 text-sm font-medium">
              Target: <span className="font-bold">{result.target_column}</span>
            </div>
            <div className="flex-1" />
            <div className="rounded-full border border-[var(--text)]/[0.1] bg-[var(--text)] px-5 py-2 text-sm font-bold text-[var(--bg)] shadow-[0_0_30px_var(--text)]/[0.2]">
              Best Model: {result.best_model}
            </div>
          </div>

          <div className="grid gap-8 lg:grid-cols-3">
            {/* Left Column: Feature Plan & Models */}
            <div className="lg:col-span-2 space-y-8">
              {result.feature_plan && (
                <GlassCard title="Feature Engineering Blueprint">
                  <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                    <FeatureGroup label="Dropped" items={result.feature_plan.drop} color="text-red-400" border="border-red-400/20" />
                    <FeatureGroup label="Numeric" items={result.feature_plan.numeric} color="text-blue-400" border="border-blue-400/20" />
                    <FeatureGroup label="One-Hot" items={result.feature_plan.onehot} color="text-emerald-400" border="border-emerald-400/20" />
                    <FeatureGroup label="Ordinal" items={Object.keys(result.feature_plan.ordinal || {})} color="text-amber-400" border="border-amber-400/20" />
                  </div>
                </GlassCard>
              )}

              {result.results && (
                <GlassCard title="Model Leaderboard">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-[var(--text)]/[0.1] text-left text-[var(--text)]/[0.5] font-mono text-xs">
                          <th className="pb-4 font-normal">Model</th>
                          {result.problem_type === "classification" ? (
                            <>
                              <th className="pb-4 font-normal">Accuracy</th>
                              <th className="pb-4 font-normal">F1 Score</th>
                              <th className="pb-4 font-normal">Precision</th>
                              <th className="pb-4 font-normal">Recall</th>
                            </>
                          ) : (
                            <>
                              <th className="pb-4 font-normal">R² Score</th>
                              <th className="pb-4 font-normal">RMSE</th>
                              <th className="pb-4 font-normal">MAE</th>
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
                              className={`border-b border-[var(--text)]/[0.05] transition-colors hover:bg-[var(--text)]/[0.02] ${
                                isBest ? "bg-[var(--text)]/[0.03]" : ""
                              }`}
                            >
                              <td className="py-4 pr-4 font-medium flex items-center gap-2">
                                {name}
                                {isBest && (
                                  <span className="rounded-md bg-[var(--text)] px-2 py-0.5 text-[10px] font-bold text-[var(--bg)] uppercase tracking-wider">
                                    Winner
                                  </span>
                                )}
                              </td>
                              {result.problem_type === "classification" ? (
                                <>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.accuracy}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.f1}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.precision}</td>
                                  <td className={`py-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.recall}</td>
                                </>
                              ) : (
                                <>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.r2}</td>
                                  <td className={`py-4 pr-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.rmse}</td>
                                  <td className={`py-4 font-mono ${isBest ? 'text-[var(--text)] font-bold' : 'text-[var(--text)]/[0.7]'}`}>{data.metrics.mae}</td>
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
                <div className="rounded-[2rem] border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] p-8 backdrop-blur-md">
                  <div className="flex items-center gap-5">
                    <div className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-[var(--blue)]/[0.1]">
                      <div className="absolute inset-0 animate-ping rounded-full border border-[var(--blue)]/[0.4]" />
                      <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--blue)] border-t-transparent" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold mb-1 text-[var(--blue)]">Generating executive analysis…</h3>
                      <p className="font-mono text-xs text-[var(--text)]/[0.6]">
                        The AI is turning these metrics into a plain-English summary. This can take a minute.
                      </p>
                    </div>
                  </div>
                </div>
              ) : !analysisResult ? (
                <div className="relative overflow-hidden rounded-[2rem] border border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] p-8 text-center backdrop-blur-md transition-all hover:bg-[var(--blue)]/[0.05]">
                  <div className="absolute inset-0 bg-gradient-to-t from-[var(--blue)]/[0.05] to-transparent pointer-events-none" />
                  <p className="text-3xl mb-4">✨</p>
                  <h3 className="text-lg font-bold mb-2">Need a Translation?</h3>
                  <p className="text-sm text-[var(--text)]/[0.6] mb-6">Convert these raw metrics into a simple, executive business analysis.</p>

                  {current.error && <p className="mb-4 text-xs text-red-400 font-mono">{current.error}</p>}

                  <button
                    onClick={getSimpleAnalysis}
                    className="w-full rounded-full bg-[var(--blue)]/[0.1] border border-[var(--blue)]/[0.2] px-6 py-3 text-sm font-bold text-[var(--blue)] transition-colors hover:bg-[var(--blue)]/[0.2]"
                  >
                    Generate Simple Analysis
                  </button>
                </div>
              ) : (
                <GlassCard className="border-[var(--blue)]/[0.3] bg-[var(--blue)]/[0.02] relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-6 opacity-20 text-4xl">✨</div>
                  <h2 className="text-xl font-bold mb-6 text-[var(--blue)]">Executive Analysis</h2>
                  
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-[10px] font-bold text-[var(--text)]/[0.5] uppercase tracking-widest mb-2">Executive Summary</h3>
                      <p className="text-sm font-medium leading-relaxed">{analysisResult.executive_summary}</p>
                    </div>
                    
                    <div className="h-[1px] w-full bg-[var(--text)]/[0.05]" />
                    
                    <div>
                      <h3 className="text-[10px] font-bold text-[var(--text)]/[0.5] uppercase tracking-widest mb-2">Business Impact</h3>
                      <p className="text-sm text-[var(--text)]/[0.7] leading-relaxed">{analysisResult.business_impact}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-[10px] font-bold text-[var(--text)]/[0.5] uppercase tracking-widest mb-2">How it Works</h3>
                      <p className="text-sm text-[var(--text)]/[0.7] leading-relaxed italic border-l-2 border-[var(--blue)]/[0.3] pl-3">{analysisResult.simple_explanation}</p>
                    </div>

                    <div className="h-[1px] w-full bg-[var(--text)]/[0.05]" />

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h3 className="text-[10px] font-bold text-[var(--text)]/[0.5] uppercase tracking-widest mb-2">Top Drivers</h3>
                        <ul className="text-xs text-[var(--text)]/[0.7] space-y-1.5">
                          {analysisResult.top_features?.map((f, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-[var(--blue)] mt-0.5">•</span>
                              <span>{f}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h3 className="text-[10px] font-bold text-[var(--text)]/[0.5] uppercase tracking-widest mb-2">AI Confidence</h3>
                        <div className="inline-flex items-center rounded-lg border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] px-3 py-1.5">
                          <div className={`h-2 w-2 rounded-full mr-2 ${
                            analysisResult.confidence_level?.toLowerCase() === 'high' ? 'bg-emerald-400' 
                            : analysisResult.confidence_level?.toLowerCase() === 'medium' ? 'bg-amber-400' 
                            : 'bg-red-400'
                          }`} />
                          <span className="text-xs font-bold">{analysisResult.confidence_level}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              )}
            </div>

            {/* Right Column: Feature Importance & Analysis */}
            <div className="space-y-8">
              {result.results?.[result.best_model]?.feature_importance && (
                <GlassCard title="Feature Importance" subtitle={`Based on ${result.best_model}`}>
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
                              <span className="font-mono text-[var(--text)]/[0.6]">{val}</span>
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
              className="rounded-full border border-[var(--text)]/[0.1] bg-transparent px-8 py-3 text-sm font-bold transition-all hover:bg-[var(--text)]/[0.05]"
            >
              Start New Training Run
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
}

const GlassCard = ({ title, subtitle, children, className = "" }) => (
  <div className={`rounded-[2rem] border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.01] p-8 backdrop-blur-xl ${className}`}>
    {(title || subtitle) && (
      <div className="mb-8">
        {title && <h2 className="text-xl font-bold tracking-tight">{title}</h2>}
        {subtitle && <p className="text-sm text-[var(--text)]/[0.5] mt-1">{subtitle}</p>}
      </div>
    )}
    {children}
  </div>
);

const FeatureGroup = ({ label, items, color, border }) => {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className={`text-[10px] font-bold uppercase tracking-widest ${color} mb-3`}>{label}</h3>
      <div className="flex flex-wrap gap-2">
        {items.map((i) => (
          <span
            key={i}
            className={`rounded-lg border ${border} bg-[var(--text)]/[0.02] px-2.5 py-1 text-[11px] font-medium text-[var(--text)]/[0.8]`}
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
  <div className="flex h-full min-h-[400px] flex-col items-center justify-center rounded-[2rem] border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] backdrop-blur-xl">
    <div className="mb-6 rounded-full bg-[var(--blue)]/[0.1] p-6 text-4xl">
      📊
    </div>
    <h2 className="mb-2 text-xl font-bold tracking-tight">{message}</h2>
    <p className="mb-8 max-w-md text-center text-sm text-[var(--text)]/[0.6]">
      Upload a dataset and configure the pipeline in the dashboard to view insights here.
    </p>
    {onAction && (
      <button
        onClick={onAction}
        className="rounded-full bg-[var(--text)] px-8 py-3 text-sm font-bold text-[var(--bg)] transition-transform hover:scale-105 shadow-[0_0_30px_var(--text)]/[0.2]"
      >
        {actionLabel}
      </button>
    )}
  </div>
);
