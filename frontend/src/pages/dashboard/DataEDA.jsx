import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";

const PAGE_SIZE = 5;

export default function DataEDA() {
  const navigate = useNavigate();
  const { eda, filename } = usePipeline();
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  if (!filename) return <Empty message="No dataset selected." onAction={() => navigate("/upload")} actionLabel="Go to Upload" />;

  const jobs = JSON.parse(localStorage.getItem("pipelineJobs") || "null");
  const hasEdaJob = jobs?.filename === filename && jobs?.eda_job_id;

  if (!hasEdaJob && !eda) return (
    <div>
      <h1 className="text-3xl font-extrabold">Data EDA</h1>
      <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-8 text-center">
        <p className="text-2xl mb-3">📊</p>
        <p className="font-semibold mb-2">Submit the pipeline first</p>
        <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono'] mb-6">
          Go to Dashboard, select your target column, and click Run EDA.
        </p>
        <button onClick={() => navigate("/dashboard")}
          className="rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">
          Go to Dashboard →
        </button>
      </div>
    </div>
  );

  const isRunning = eda && !["completed","failed"].includes(eda.status);
  const charts = eda?.status === "completed" ? (eda.result?.charts || {}) : null;
  const chartEntries = charts ? Object.entries(charts) : [];

  return (
    <div>
      <h1 className="text-3xl font-extrabold">Data EDA</h1>
      <p className="mt-2 text-[#9a9a93]">Charts for <span className="text-[#f0ece2] font-semibold">{filename}</span></p>

      {isRunning && <StatusBar label={eda?.message || "Generating charts…"} />}
      {eda?.status === "failed" && <ErrBar msg={eda.error || "EDA failed"} />}
      {!isRunning && !charts && hasEdaJob && <StatusBar label="Waiting for results…" />}

      {chartEntries.length > 0 && (
        <>
          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            {chartEntries.slice(0, visibleCount).map(([title, b64]) => (
              <ChartCard key={title} title={title} imageB64={b64} filename={filename} />
            ))}
          </div>
          {visibleCount < chartEntries.length && (
            <div className="mt-8 flex justify-center">
              <button onClick={() => setVisibleCount(c => c + PAGE_SIZE)}
                className="rounded-full border border-white/15 px-7 py-3 text-sm font-semibold hover:border-white/30 transition-colors">
                Show next {Math.min(PAGE_SIZE, chartEntries.length - visibleCount)} →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ChartCard({ title, imageB64, filename }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const requestInsights = async () => {
    setLoading(true); setErr("");
    try {
      const res = await api.post("/api/analyse_chart", { filename, image_b64: imageB64, chart_title: title });
      if (res.data.error) setErr(res.data.error);
      else setAnalysis(res.data);
    } catch (e) { setErr(e.response?.data?.error || "Analysis unavailable"); }
    finally { setLoading(false); }
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-[#0d1117] flex flex-col overflow-hidden">
      <div className="border-b border-white/10 bg-white/[0.03] px-4 py-3">
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <div className="flex items-center justify-center p-4 min-h-[240px]">
        <img src={`data:image/png;base64,${imageB64}`} alt={title} loading="lazy"
          className="max-w-full max-h-[300px] object-contain rounded-md" />
      </div>
      <div className="border-t border-white/10 p-4">
        <button onClick={requestInsights} disabled={loading || !!analysis}
          className={`w-full rounded-md py-2 text-sm font-semibold border transition-colors ${
            analysis ? "border-white/10 text-[#6e6e66] cursor-default" : "border-white/15 hover:border-white/30"
          }`}>
          {loading ? "Analysing…" : analysis ? "✓ Analysed" : "✦ AI Insights"}
        </button>
        {err && <p className="mt-2 text-xs text-red-400 font-['JetBrains_Mono']">✗ {err}</p>}
        {analysis && (
          <div className="mt-4 space-y-3 text-sm">
            {analysis.represents && <p className="text-[#f0ece2]">{analysis.represents}</p>}
            {analysis.key_findings?.length > 0 && <List label="Key Findings" items={analysis.key_findings} color="text-[#c97539]" />}
            {analysis.anomalies?.length > 0 && <List label="Anomalies" items={analysis.anomalies} color="text-amber-400" />}
            {analysis.recommendations?.length > 0 && <List label="Recommendations" items={analysis.recommendations} color="text-emerald-400" />}
          </div>
        )}
      </div>
    </div>
  );
}

const List = ({ label, items, color }) => (
  <div>
    <p className={`text-xs font-bold ${color} mb-1`}>{label}</p>
    <ul className="space-y-1 text-[#9a9a93] text-xs list-disc list-inside">
      {items.map((item, i) => <li key={i}>{item}</li>)}
    </ul>
  </div>
);
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
    <h1 className="text-3xl font-extrabold">Data EDA</h1>
    <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-center">
      <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">{message}</p>
      {onAction && <button onClick={onAction} className="mt-4 rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">{actionLabel}</button>}
    </div>
  </div>
);