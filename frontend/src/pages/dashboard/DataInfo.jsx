import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "../../context/PipelineContext";

export default function DataInfo() {
  const navigate = useNavigate();
  const {
    info,
    gone,
    filename,
    storedInsights,
    insightsLoading,
    insightsError,
    loadInsights,
  } = usePipeline();

  const stored = filename ? storedInsights[filename] : null;

  useEffect(() => {
    if (!filename) return;
    if (info?.status === "completed") return;

    let jobs = null;
    try {
      jobs = JSON.parse(localStorage.getItem("pipelineJobs") || "null");
    } catch {
      /* ignore */
    }
    const hookOwnsIt =
      jobs?.filename === filename &&
      jobs?.info_job_id &&
      !gone.info &&
      info?.status !== "failed";
    if (hookOwnsIt) return;

    loadInsights(filename);
  }, [filename, info?.status, gone.info, loadInsights]);

  if (!filename)
    return (
      <EmptyState
        title="Data Info"
        message="No dataset selected."
        onAction={() => navigate("/upload")}
        actionLabel="Go to Upload"
      />
    );

  const data = info?.status === "completed" ? info.result : stored;
  const isRunning = info && !["completed", "failed"].includes(info.status);
  const analysis = data?.analysis;
  const aiInsights = data?.ai_insights;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold">Data Info</h1>
          <p className="mt-2 text-[var(--muted)]">
            Summary for{" "}
            <span className="text-[var(--text)] font-semibold">{filename}</span>
          </p>
        </div>
      </div>

      {(isRunning || insightsLoading) && (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-[var(--line)] bg-[var(--surface-2)] px-4 py-3">
          <Spinner />
          <p className="text-sm text-[var(--muted)] font-mono">
            {info?.message || "Loading…"}
          </p>
        </div>
      )}

      {info?.status === "failed" && (
        <p className="mt-6 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-400">
          ✗ {info.error || "Analysis failed"}
        </p>
      )}

      {insightsError && (
        <p className="mt-6 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-400">
          ✗ {insightsError}
        </p>
      )}

      {!isRunning && !insightsLoading && !data && !insightsError && (
        <EmptyState
          title=""
          message="No analysis found for this dataset. It may not have been processed yet — re-upload it to run analysis."
          onAction={() => navigate("/upload")}
          actionLabel="Go to Upload"
        />
      )}

      {analysis && (
        <>
          {/* Quality Alerts */}
          {analysis.cardinality_flags && (analysis.cardinality_flags.constant_columns.length > 0 || analysis.cardinality_flags.id_columns.length > 0) && (
            <div className="mt-8 flex flex-col gap-2">
              {analysis.cardinality_flags.constant_columns.map(col => (
                <div key={col} className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 flex items-start gap-3">
                  <span className="text-red-400 mt-0.5">⚠️</span>
                  <div>
                    <h4 className="text-sm font-bold text-red-400">Constant Column Detected</h4>
                    <p className="text-xs text-[var(--muted)]">The column <span className="text-[var(--text)] font-semibold">{col}</span> contains only a single unique value. Consider dropping it.</p>
                  </div>
                </div>
              ))}
              {analysis.cardinality_flags.id_columns.map(col => (
                <div key={col} className="rounded-xl border border-yellow-500/20 bg-yellow-500/10 px-4 py-3 flex items-start gap-3">
                  <span className="text-yellow-400 mt-0.5">ℹ️</span>
                  <div>
                    <h4 className="text-sm font-bold text-yellow-400">High Cardinality Detected</h4>
                    <p className="text-xs text-[var(--muted)]">The column <span className="text-[var(--text)] font-semibold">{col}</span> has entirely unique values (likely an ID). It may cause overfitting.</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Stats Readout */}
          <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-[1px] bg-[var(--text)]/[0.08] rounded-3xl overflow-hidden border border-[var(--text)]/[0.08]">
            {[
              { 
                label: "Rows",
                value: analysis.shape?.[0]?.toLocaleString() ?? "—" 
              },
              { 
                label: "Columns",
                value: analysis.shape?.[1] ?? "—" 
              },
              { 
                label: "Duplicates",
                value: analysis.duplicate_rows ?? "—" 
              },
              { 
                label: "Memory (MB)", 
                value: analysis.memory_usage_mb ?? "—" 
              },
            ].map((s) => (
              <div key={s.label} className="bg-[var(--bg)] p-6 md:p-8 flex flex-col justify-between">
                <div className="font-mono text-xs text-[var(--muted)] opacity-60">
                  {s.label}
                </div>
                <div className="text-4xl md:text-5xl font-extrabold tracking-tight mt-4 text-[var(--text)]">
                  {s.value}
                </div>
              </div>
            ))}
          </div>

          {/* Grid wrapper for Tables */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mt-8">
            {/* Columns table */}
            <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-6 sm:p-8 flex flex-col h-[32rem]">
              <h2 className="text-2xl font-bold mb-6 tracking-tight shrink-0">Columns</h2>
              <div className="overflow-x-auto overflow-y-auto custom-scrollbar flex-1 pr-4 -mr-4 pb-2">
                <table className="w-full text-left border-collapse">
                  <thead className="sticky top-0 bg-[var(--bg)] z-10 shadow-[0_1px_0_0_var(--text)]/10">
                    <tr className="text-[var(--text)] opacity-50 font-mono text-xs tracking-widest uppercase">
                      <th className="py-4 pr-6 font-medium">Column</th>
                      <th className="py-4 pr-6 font-medium">Type</th>
                      <th className="py-4 pr-6 font-medium">Null %</th>
                      <th className="py-4 font-medium">Unique</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--text)]/[0.05]">
                    {analysis.columns?.map((col) => (
                      <tr key={col} className="transition-colors hover:bg-[var(--text)]/[0.03]">
                        <td className="py-4 pr-6 font-semibold">{col}</td>
                        <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                          {analysis.dtypes?.[col] ?? "—"}
                        </td>
                        <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                          {analysis.null_percentages?.[col] ?? 0}%
                        </td>
                        <td className="py-4 text-[var(--text)] opacity-70 font-mono text-sm">
                          {analysis.unique_counts?.[col] ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Numeric Statistics */}
            {analysis.numeric_stats &&
              Object.keys(analysis.numeric_stats).length > 0 && (
                <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-6 sm:p-8 flex flex-col h-[32rem]">
                  <h2 className="text-2xl font-bold mb-6 tracking-tight shrink-0">Numeric Statistics</h2>
                  <div className="overflow-x-auto overflow-y-auto custom-scrollbar flex-1 pr-4 -mr-4 pb-2">
                    <table className="w-full text-left border-collapse">
                      <thead className="sticky top-0 bg-[var(--bg)] z-10 shadow-[0_1px_0_0_var(--text)]/10">
                        <tr className="text-[var(--text)] opacity-50 font-mono text-xs tracking-widest uppercase">
                          <th className="py-4 pr-6 font-medium">Column</th>
                          <th className="py-4 pr-6 font-medium">Min</th>
                          <th className="py-4 pr-6 font-medium">Mean</th>
                          <th className="py-4 pr-6 font-medium">Median</th>
                          <th className="py-4 pr-6 font-medium">Max</th>
                          <th className="py-4 pr-6 font-medium">Std</th>
                          <th className="py-4 pr-6 font-medium">Outliers</th>
                          <th className="py-4 font-medium">Skew</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[var(--text)]/[0.05]">
                        {Object.entries(analysis.numeric_stats).map(
                          ([col, s]) => (
                            <tr key={col} className="transition-colors hover:bg-[var(--text)]/[0.03]">
                              <td className="py-4 pr-6 font-semibold">{col}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">{s.min}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">{s.mean}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">{s.median}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">{s.max}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">{s.std}</td>
                              <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                                {analysis.advanced_numeric?.[col]?.outliers ?? 0}
                              </td>
                              <td className="py-4 text-[var(--text)] opacity-70 font-mono text-sm">
                                {analysis.advanced_numeric?.[col]?.skewness ?? 0}
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
          </div>

          {/* Advanced Categorical Statistics */}
          {analysis.advanced_categorical &&
            Object.keys(analysis.advanced_categorical).length > 0 && (
              <div className="mt-8 rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-6 sm:p-8">
                <h2 className="text-2xl font-bold mb-6 tracking-tight">Text & Categorical Insights</h2>
                <div className="overflow-x-auto overflow-y-auto max-h-80 custom-scrollbar pr-4 -mr-4">
                  <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-[var(--bg)] z-10 shadow-[0_1px_0_0_var(--text)]/10">
                      <tr className="text-[var(--text)] opacity-50 font-mono text-xs tracking-widest uppercase">
                        <th className="py-4 pr-6 font-medium">Column</th>
                        <th className="py-4 pr-6 font-medium">Inferred Date</th>
                        <th className="py-4 pr-6 font-medium">Min Length</th>
                        <th className="py-4 pr-6 font-medium">Mean Length</th>
                        <th className="py-4 font-medium">Max Length</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--text)]/[0.05]">
                      {Object.entries(analysis.advanced_categorical).map(
                        ([col, stats]) => (
                          <tr key={col} className="transition-colors hover:bg-[var(--text)]/[0.03]">
                            <td className="py-4 pr-6 font-semibold">{col}</td>
                            <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                              {stats.inferred_date ? "✅ Yes" : "❌ No"}
                            </td>
                            <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                              {stats.text_length?.min ?? "—"}
                            </td>
                            <td className="py-4 pr-6 text-[var(--text)] opacity-70 font-mono text-sm">
                              {stats.text_length?.mean ?? "—"}
                            </td>
                            <td className="py-4 text-[var(--text)] opacity-70 font-mono text-sm">
                              {stats.text_length?.max ?? "—"}
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

          {/* Grid for Top Categories & Correlations */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mt-8">
            {/* Top Categories */}
            {analysis.top_categories &&
              Object.keys(analysis.top_categories).length > 0 && (
                <div className="flex flex-col h-[28rem] rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-6 sm:p-8">
                  <h2 className="text-2xl font-bold mb-6 shrink-0 tracking-tight">
                    Top Categories
                  </h2>
                  <div className="overflow-y-auto custom-scrollbar flex-1 pr-4 -mr-4 pb-2">
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
                      {Object.entries(analysis.top_categories).map(
                        ([col, vals]) => (
                          <div
                            key={col}
                            className="rounded-2xl border border-[var(--text)]/[0.08] bg-[var(--bg)] px-6 py-5 transition-colors hover:border-[var(--text)]/[0.2]"
                          >
                            <p className="text-sm font-semibold text-[var(--yellow)] mb-3">
                              {col}
                            </p>
                            <div className="space-y-2">
                              {vals.map((v, i) => (
                                <div
                                  key={i}
                                  className="flex items-center justify-between text-sm"
                                >
                                  <span className="text-[var(--text)] truncate pr-2 font-medium">
                                    {v.value}
                                  </span>
                                  <span className="text-[var(--text)] opacity-50 font-mono">
                                    {v.count}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  </div>
                </div>
              )}

            {/* Correlations */}
            {analysis.correlations?.length > 0 && (
              <div className="flex flex-col h-[28rem] rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-6 sm:p-8">
                <h2 className="text-2xl font-bold mb-6 shrink-0 tracking-tight">
                  Strongest Correlations
                </h2>
                <div className="overflow-y-auto custom-scrollbar flex-1 pr-4 -mr-4 pb-2">
                  <div className="grid gap-3">
                    {analysis.correlations.map((c, i) => {
                      const r = c.correlation;
                      const strong = Math.abs(r) >= 0.7;
                      const moderate = Math.abs(r) >= 0.4;
                      return (
                        <div
                          key={i}
                          className="flex items-center justify-between rounded-2xl border border-[var(--text)]/[0.08] bg-[var(--bg)] px-6 py-4 transition-colors hover:border-[var(--text)]/[0.2]"
                        >
                          <span className="text-sm font-mono text-[var(--text)] opacity-80">
                            {c.columns[0]}{" "}
                            <span className="text-[var(--text)] opacity-30 px-2">↔</span>{" "}
                            {c.columns[1]}
                          </span>
                          <span
                            className={`font-mono text-sm font-bold ${
                              strong
                                ? "text-red-400"
                                : moderate
                                  ? "text-amber-400"
                                  : "text-[var(--text)] opacity-50"
                            }`}
                          >
                            {r > 0 ? "+" : ""}
                            {r}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>

          <AITabs aiInsights={aiInsights} />
        </>
      )}
    </div>
  );
}

function AITabs({ aiInsights }) {
  const [activeTab, setActiveTab] = useState("overview");
  if (!aiInsights) return null;

  return (
    <div className="mt-16 border-t border-[var(--text)]/[0.08] pt-12">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
        <h2 className="text-3xl font-extrabold tracking-tight">AI Insights</h2>
        <div className="flex flex-wrap gap-2 p-1.5 bg-[var(--text)]/[0.05] rounded-full border border-[var(--text)]/[0.05] self-start md:self-auto">
          {[
            { id: "overview", label: "Overview & Targets" },
            { id: "health", label: "Data Health & Prep" },
            { id: "insights", label: "Deep Insights" },
            { id: "business", label: "Business Insights" }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-2.5 rounded-full text-sm font-bold transition-all duration-300 ${
                activeTab === tab.id 
                  ? "bg-[var(--text)] text-[var(--bg)] shadow-md" 
                  : "text-[var(--text)] opacity-60 hover:opacity-100 hover:bg-[var(--text)]/[0.08]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-[300px]">
        {activeTab === "overview" && (
          <div className="grid gap-6 sm:grid-cols-2">
            {aiInsights.summary && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8 sm:col-span-2">
                <h3 className="text-xl font-bold mb-4">AI Summary</h3>
                <p className="text-lg text-[var(--muted)] leading-relaxed">
                  {aiInsights.summary}
                </p>
              </div>
            )}
            {aiInsights.recommended_target?.column && (
              <div className="rounded-3xl border-2 border-[var(--yellow)]/[0.3] bg-[var(--yellow)]/[0.05] p-8">
                <h3 className="text-xl font-bold mb-4 text-[var(--yellow)]">
                  🎯 Recommended Target
                </h3>
                <p className="text-2xl font-bold mb-3">
                  {aiInsights.recommended_target.column}
                </p>
                {aiInsights.recommended_target.reason && (
                  <p className="text-[var(--muted)] leading-relaxed">
                    {aiInsights.recommended_target.reason}
                  </p>
                )}
              </div>
            )}
            {aiInsights.next_steps?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-xl font-bold mb-6">→ Next Steps</h3>
                <div className="grid gap-4">
                  {aiInsights.next_steps.map((step, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5"
                    >
                      <p className="text-base font-semibold text-emerald-400 mb-2">
                        {step.title}
                      </p>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">
                        {step.detail}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "health" && (
          <div className="grid gap-6 sm:grid-cols-2">
            {aiInsights.quality_flags?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8 sm:col-span-2">
                <h3 className="text-xl font-bold mb-6">⚠ Quality Flags</h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  {aiInsights.quality_flags.map((f, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-5"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <span className="font-bold text-base">
                          {f.column}
                        </span>
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider ${
                            f.severity === "high"
                              ? "bg-red-500/20 text-red-400"
                              : f.severity === "medium"
                                ? "bg-amber-500/20 text-amber-400"
                                : "bg-[var(--text)]/[0.1] text-[var(--text)] opacity-70"
                          }`}
                        >
                          {f.severity}
                        </span>
                      </div>
                      <p className="text-sm text-[var(--text)] opacity-90 leading-relaxed font-medium mb-1">{f.issue}</p>
                      {f.detail && (
                        <p className="text-sm text-[var(--muted)] leading-relaxed">
                          {f.detail}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {aiInsights.preprocessing?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-xl font-bold mb-6">
                  ⚙ Preprocessing Steps
                </h3>
                <div className="grid gap-4">
                  {aiInsights.preprocessing.map((p, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-5"
                    >
                      <p className="text-base font-semibold text-violet-400 mb-2">
                        {p.title}
                      </p>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">
                        {p.detail}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {aiInsights.uncertainty_notes && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-xl font-bold mb-4">Uncertainty Notes</h3>
                <p className="text-lg text-[var(--muted)] leading-relaxed">
                  {aiInsights.uncertainty_notes}
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "insights" && (
          <div className="grid gap-6 sm:grid-cols-2">
            {aiInsights.feature_engineering?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8 sm:col-span-2">
                <h3 className="text-xl font-bold mb-6">
                  🛠 Feature Engineering Ideas
                </h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  {aiInsights.feature_engineering.map((f, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-sky-500/20 bg-sky-500/5 p-5"
                    >
                      <p className="text-base font-semibold text-sky-400 mb-2">
                        {f.title}
                      </p>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">
                        {f.detail}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {aiInsights.column_insights?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-xl font-bold mb-6">Column Insights</h3>
                <div className="grid gap-4">
                  {aiInsights.column_insights.map((item, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-[var(--text)]/[0.08] bg-[var(--bg)] p-5"
                    >
                      <p className="text-base font-semibold text-[var(--yellow)] mb-2">
                        {item.column}
                      </p>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">
                        {item.insight}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {aiInsights.correlations?.length > 0 && (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-xl font-bold mb-6">
                  Relationship Insights
                </h3>
                <div className="grid gap-4">
                  {aiInsights.correlations.map((c, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-[var(--text)]/[0.08] bg-[var(--bg)] p-5"
                    >
                      <p className="text-sm font-semibold text-[var(--yellow)] font-mono mb-2">
                        {Array.isArray(c.columns) ? c.columns.join(" ↔ ") : ""}
                      </p>
                      <p className="text-sm text-[var(--muted)] leading-relaxed">
                        {c.detail}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "business" && (
          <div className="grid gap-6">
            {aiInsights.business_insights ? (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8">
                <h3 className="text-2xl font-bold mb-6 text-[var(--yellow)]">
                  Executive Business Evaluation
                </h3>
                <p className="text-lg text-[var(--text)] opacity-80 leading-relaxed whitespace-pre-wrap font-medium">
                  {aiInsights.business_insights}
                </p>
              </div>
            ) : (
              <div className="rounded-3xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] p-8 text-center text-[var(--text)] opacity-50 font-bold">
                No business insights were generated for this dataset.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const Spinner = () => (
  <div className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--blue)]" />
);

function EmptyState({ title, message, onAction, actionLabel }) {
  return (
    <div>
      {title && <h1 className="text-3xl font-extrabold">{title}</h1>}
      <div className="mt-8 surface p-6 text-center">
        <p className="text-sm text-[var(--muted)] font-mono">
          {message}
        </p>
        {onAction && (
          <button
            onClick={onAction}
            className="mt-4 rounded-full bg-[var(--blue)] px-6 py-2.5 text-sm font-semibold text-[var(--bg)]"
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}
