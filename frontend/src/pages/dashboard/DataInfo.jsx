import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "../../hooks/usePipeline";

export default function DataInfo() {
  const navigate = useNavigate();
  const {
    info, gone, filename,
    storedInsights, insightsLoading, insightsError, loadInsights,
  } = usePipeline();

  const stored = filename ? storedInsights[filename] : null;
  const fetching = insightsLoading;
  const fetchError = insightsError;

  // Mongo is the durable source of truth. Load from it (cached per filename in
  // context) whenever the live job can't give us the answer: no job running, the
  // job 404'd (server restart / expired id), or it finished without usable data.
  useEffect(() => {
    if (!filename) return;
    if (info?.status === "completed") return; // hook already has live data

    let jobs = null;
    try { jobs = JSON.parse(localStorage.getItem("pipelineJobs") || "null"); } catch { /* ignore */ }
    // While a non-stale job id is on file, usePipeline owns this — don't race it.
    // The hook drops the id (and sets gone.info) on 404, which lets us fall through.
    const hookOwnsIt = jobs?.filename === filename && jobs?.info_job_id && !gone.info
      && info?.status !== "failed";
    if (hookOwnsIt) return;

    loadInsights(filename);
  }, [filename, info?.status, gone.info, loadInsights]);

  if (!filename) return (
    <EmptyState title="Data Info" message="No dataset selected." onAction={() => navigate("/upload")} actionLabel="Go to Upload" />
  );

  // Resolve data: prefer live job result, fallback to Mongo
  const data = info?.status === "completed" ? info.result : stored;
  const isRunning = info && !["completed","failed"].includes(info.status);
  const analysis = data?.analysis;
  const aiInsights = data?.ai_insights;

  return (
    <div>
      <h1 className="text-3xl font-extrabold">Data Info</h1>
      <p className="mt-2 text-[#9a9a93]">
        Summary for <span className="text-[#f0ece2] font-semibold">{filename}</span>
      </p>

      {(isRunning || fetching) && (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
          <Spinner />
          <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">
            {info?.message || "Loading…"}
          </p>
        </div>
      )}

      {info?.status === "failed" && (
        <p className="mt-6 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">
          ✗ {info.error || "Analysis failed"}
        </p>
      )}

      {fetchError && (
        <p className="mt-6 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">
          ✗ {fetchError}
        </p>
      )}

      {!isRunning && !fetching && !data && !fetchError && (
        <EmptyState
          title=""
          message="No analysis found for this dataset. It may not have been processed yet — re-upload it to run analysis."
          onAction={() => navigate("/upload")}
          actionLabel="Go to Upload"
        />
      )}

      {analysis && (
        <>
          {/* Stats */}
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Rows",        value: analysis.shape?.[0] ?? "—" },
              { label: "Columns",     value: analysis.shape?.[1] ?? "—" },
              { label: "Duplicates",  value: analysis.duplicate_rows ?? "—" },
              { label: "Memory (MB)", value: analysis.memory_usage_mb ?? "—" },
            ].map(s => (
              <div key={s.label} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-5 text-center">
                <div className="text-2xl font-extrabold">{s.value}</div>
                <div className="mt-1 font-['JetBrains_Mono'] text-[10px] tracking-widest text-[#6e6e66]">{s.label.toUpperCase()}</div>
              </div>
            ))}
          </div>

          {/* Columns table */}
          <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
            <h2 className="text-lg font-bold mb-4">Columns</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[#9a9a93] font-['JetBrains_Mono'] text-xs border-b border-white/10">
                    <th className="pb-2 pr-6">Column</th>
                    <th className="pb-2 pr-6">Type</th>
                    <th className="pb-2 pr-6">Null %</th>
                    <th className="pb-2">Unique</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.columns?.map(col => (
                    <tr key={col} className="border-t border-white/5">
                      <td className="py-2 pr-6 font-medium">{col}</td>
                      <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{analysis.dtypes?.[col] ?? "—"}</td>
                      <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{analysis.null_percentages?.[col] ?? 0}%</td>
                      <td className="py-2 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{analysis.unique_counts?.[col] ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Numeric Statistics */}
          {analysis.numeric_stats && Object.keys(analysis.numeric_stats).length > 0 && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">Numeric Statistics</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[#9a9a93] font-['JetBrains_Mono'] text-xs border-b border-white/10">
                      <th className="pb-2 pr-6">Column</th>
                      <th className="pb-2 pr-6">Min</th>
                      <th className="pb-2 pr-6">Mean</th>
                      <th className="pb-2 pr-6">Median</th>
                      <th className="pb-2 pr-6">Max</th>
                      <th className="pb-2">Std</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.numeric_stats).map(([col, s]) => (
                      <tr key={col} className="border-t border-white/5">
                        <td className="py-2 pr-6 font-medium">{col}</td>
                        <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{s.min}</td>
                        <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{s.mean}</td>
                        <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{s.median}</td>
                        <td className="py-2 pr-6 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{s.max}</td>
                        <td className="py-2 text-[#9a9a93] font-['JetBrains_Mono'] text-xs">{s.std}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Top Categories */}
          {analysis.top_categories && Object.keys(analysis.top_categories).length > 0 && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">Top Categories</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {Object.entries(analysis.top_categories).map(([col, vals]) => (
                  <div key={col} className="rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
                    <p className="text-sm font-semibold text-[#c97539] mb-2">{col}</p>
                    <div className="space-y-1">
                      {vals.map((v, i) => (
                        <div key={i} className="flex items-center justify-between text-xs">
                          <span className="text-[#f0ece2] truncate pr-2">{v.value}</span>
                          <span className="text-[#9a9a93] font-['JetBrains_Mono']">{v.count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Correlations */}
          {analysis.correlations?.length > 0 && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">Strongest Correlations</h2>
              <div className="grid gap-2">
                {analysis.correlations.map((c, i) => {
                  const r = c.correlation;
                  const strong = Math.abs(r) >= 0.7;
                  const moderate = Math.abs(r) >= 0.4;
                  return (
                    <div key={i} className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] px-4 py-2">
                      <span className="text-sm font-['JetBrains_Mono']">
                        {c.columns[0]} <span className="text-[#6e6e66]">↔</span> {c.columns[1]}
                      </span>
                      <span className={`font-['JetBrains_Mono'] text-sm font-bold ${
                        strong ? "text-red-400" : moderate ? "text-amber-400" : "text-[#9a9a93]"
                      }`}>{r > 0 ? "+" : ""}{r}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* AI Summary */}
          {aiInsights?.summary && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-2">AI Summary</h2>
              <p className="text-sm text-[#9a9a93] leading-relaxed">{aiInsights.summary}</p>
            </div>
          )}

          {/* Recommended Target */}
          {aiInsights?.recommended_target?.column && (
            <div className="mt-4 rounded-2xl border border-[#c97539]/30 bg-[#c97539]/[0.06] p-6">
              <h2 className="text-lg font-bold mb-2">🎯 Recommended Target</h2>
              <p className="text-sm font-semibold text-[#c97539]">{aiInsights.recommended_target.column}</p>
              {aiInsights.recommended_target.reason && (
                <p className="mt-1 text-sm text-[#9a9a93] leading-relaxed">{aiInsights.recommended_target.reason}</p>
              )}
            </div>
          )}

          {/* Quality Flags */}
          {aiInsights?.quality_flags?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">⚠ Quality Flags</h2>
              <div className="grid gap-3">
                {aiInsights.quality_flags.map((f, i) => (
                  <div key={i} className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-sm">{f.column}</span>
                      <span className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                        f.severity === "high" ? "bg-red-500/20 text-red-400" :
                        f.severity === "medium" ? "bg-amber-500/20 text-amber-400" :
                        "bg-white/10 text-[#9a9a93]"
                      }`}>{f.severity}</span>
                    </div>
                    <p className="text-sm text-[#9a9a93]">{f.issue}</p>
                    {f.detail && <p className="mt-1 text-xs text-[#6e6e66]">{f.detail}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Column Insights */}
          {aiInsights?.column_insights?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">Column Insights</h2>
              <div className="grid gap-3">
                {aiInsights.column_insights.map((item, i) => (
                  <div key={i} className="rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
                    <p className="text-sm font-semibold text-[#c97539]">{item.column}</p>
                    <p className="mt-1 text-sm text-[#9a9a93]">{item.insight}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Correlation Insights (AI commentary) */}
          {aiInsights?.correlations?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">Relationship Insights</h2>
              <div className="grid gap-3">
                {aiInsights.correlations.map((c, i) => (
                  <div key={i} className="rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
                    <p className="text-sm font-semibold text-[#c97539] font-['JetBrains_Mono']">
                      {Array.isArray(c.columns) ? c.columns.join(" ↔ ") : ""}
                    </p>
                    <p className="mt-1 text-sm text-[#9a9a93]">{c.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Feature Engineering */}
          {aiInsights?.feature_engineering?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">🛠 Feature Engineering Ideas</h2>
              <div className="grid gap-3">
                {aiInsights.feature_engineering.map((f, i) => (
                  <div key={i} className="rounded-lg border border-sky-500/20 bg-sky-500/5 px-4 py-3">
                    <p className="text-sm font-semibold text-sky-400">{f.title}</p>
                    <p className="mt-1 text-sm text-[#9a9a93]">{f.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Preprocessing */}
          {aiInsights?.preprocessing?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">⚙ Preprocessing Steps</h2>
              <div className="grid gap-3">
                {aiInsights.preprocessing.map((p, i) => (
                  <div key={i} className="rounded-lg border border-violet-500/20 bg-violet-500/5 px-4 py-3">
                    <p className="text-sm font-semibold text-violet-400">{p.title}</p>
                    <p className="mt-1 text-sm text-[#9a9a93]">{p.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Uncertainty Notes */}
          {aiInsights?.uncertainty_notes && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-2">Uncertainty Notes</h2>
              <p className="text-sm text-[#9a9a93] leading-relaxed">{aiInsights.uncertainty_notes}</p>
            </div>
          )}

          {/* Next Steps */}
          {aiInsights?.next_steps?.length > 0 && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <h2 className="text-lg font-bold mb-4">→ Next Steps</h2>
              <div className="grid gap-3">
                {aiInsights.next_steps.map((step, i) => (
                  <div key={i} className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3">
                    <p className="text-sm font-semibold text-emerald-400">{step.title}</p>
                    <p className="mt-1 text-sm text-[#9a9a93]">{step.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

const Spinner = () => (
  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-[#c97539]" />
);

function EmptyState({ title, message, onAction, actionLabel }) {
  return (
    <div>
      {title && <h1 className="text-3xl font-extrabold">{title}</h1>}
      <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center">
        <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">{message}</p>
        {onAction && (
          <button onClick={onAction} className="mt-4 rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
}