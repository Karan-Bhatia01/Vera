import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "../../context/PipelineContext";
import AestheticChart from "../../components/charts/AestheticChart";

export default function DataEDA() {
  const navigate = useNavigate();
  const { eda, filename } = usePipeline();
  const [page, setPage] = useState(0);
  const [activeChart, setActiveChart] = useState(null);

  const chartsPerPage = 5;

  // Reset states if the dataset changes
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    setPage(0);
    setActiveChart(null);
  }, [filename]);

  if (!filename)
    return (
      <Empty
        message="No dataset selected."
        onAction={() => navigate("/upload")}
        actionLabel="Go to Upload"
      />
    );

  const jobs = JSON.parse(localStorage.getItem("pipelineJobs") || "null");
  const hasEdaJob = jobs?.filename === filename && jobs?.eda_job_id;

  if (!hasEdaJob && !eda)
    return (
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight">Data EDA</h1>
        <div className="mt-8 surface border-2 border-[var(--text)]/[0.1] rounded-3xl p-12 text-center bg-[var(--text)]/[0.015]">
          <p className="text-4xl mb-4">📊</p>
          <p className="text-xl font-bold mb-2">Submit the pipeline first</p>
          <p className="text-base text-[var(--muted)] mb-8">
            Go to Dashboard, select your target column, and click Run EDA.
          </p>
          <button
            onClick={() => navigate("/dashboard")}
            className="rounded-full bg-[var(--text)] text-[var(--bg)] px-8 py-3 text-sm font-bold shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all"
          >
            Go to Dashboard →
          </button>
        </div>
      </div>
    );

  const isRunning = eda && !["completed", "failed"].includes(eda.status);
  const chartsData = eda?.status === "completed" ? eda.result?.charts || [] : null;
  const chartList = Array.isArray(chartsData) ? chartsData : [];
  
  const totalPages = Math.ceil(chartList.length / chartsPerPage);
  const currentCharts = chartList.slice(page * chartsPerPage, (page + 1) * chartsPerPage);

  const handleNext = () => setPage(p => Math.min(totalPages - 1, p + 1));
  const handlePrev = () => setPage(p => Math.max(0, p - 1));

  // Helper to safely format the AI's markdown storytelling
  const formatStory = (text) => {
    if (!text) return null;
    const blocks = text.split('\n\n').filter(b => b.trim());
    return blocks.map((block, idx) => {
      if (block.trim().startsWith('- ') || block.trim().startsWith('* ')) {
        const items = block.split('\n').map(l => l.replace(/^[-*]\s/, ''));
        return (
          <ul key={idx} className="list-disc list-outside ml-5 mb-4 text-sm text-[var(--muted)] space-y-2 leading-relaxed">
            {items.map((item, i) => <li key={i}>{item}</li>)}
          </ul>
        );
      }
      return <p key={idx} className="mb-4 text-sm text-[var(--muted)] leading-relaxed">{block}</p>;
    });
  };

  return (
    <div>
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Data EDA</h1>
          <p className="mt-2 text-lg text-[var(--muted)]">
            AI Curated Visualizations for{" "}
            <span className="text-[var(--text)] font-semibold">{filename}</span>
          </p>
        </div>
        
        {/* Pagination Controls */}
        {chartList.length > chartsPerPage && !activeChart && (
          <div className="flex items-center gap-4 bg-[var(--text)]/[0.03] px-4 py-2 rounded-full border border-[var(--text)]/[0.1] mr-14">
            <button 
              onClick={handlePrev} 
              disabled={page === 0}
              className="text-[var(--text)] disabled:opacity-30 hover:scale-110 transition-transform"
            >
              ←
            </button>
            <span className="text-sm font-bold font-mono">
              {page + 1} / {totalPages}
            </span>
            <button 
              onClick={handleNext} 
              disabled={page === totalPages - 1}
              className="text-[var(--text)] disabled:opacity-30 hover:scale-110 transition-transform"
            >
              →
            </button>
          </div>
        )}
      </div>

      {isRunning && <StatusBar label={eda?.message || "Generating interactive charts…"} />}
      {eda?.status === "failed" && <ErrBar msg={eda.error || "EDA failed"} />}
      {!isRunning && !chartsData && hasEdaJob && (
        <StatusBar label="Waiting for results…" />
      )}

      {/* Expanded Chart View */}
      {activeChart && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <button 
            onClick={() => setActiveChart(null)}
            className="mb-6 flex items-center gap-2 text-sm font-bold hover:text-[var(--muted)] transition-colors"
          >
            ← Back to Grid
          </button>
          <div className="flex flex-col xl:flex-row gap-8">
            <div className="xl:w-2/3 bg-[var(--text)]/[0.015] border-2 border-[var(--text)]/[0.1] rounded-3xl p-6 sm:p-8">
              <div className="mb-6 flex justify-between items-start">
                <h3 className="text-2xl font-extrabold tracking-tight capitalize">
                  {activeChart.columns ? activeChart.columns.join(' vs ') : 'Chart'}
                </h3>
                <span className="px-3 py-1 bg-[var(--text)] text-[var(--bg)] rounded-full text-xs font-bold uppercase tracking-wider">
                  {activeChart.chart_type}
                </span>
              </div>
              <AestheticChart data={activeChart} height="500px" />
            </div>

            <div className="xl:w-1/3 bg-[var(--yellow)]/[0.05] border-2 border-[var(--yellow)]/[0.2] rounded-3xl p-6 sm:p-8 overflow-y-auto max-h-[600px]">
              <h4 className="text-sm font-bold uppercase tracking-wider text-[var(--yellow)] mb-4 sticky top-0 bg-[var(--bg)] py-2">Business & Statistical Insights</h4>
              <div className="prose prose-sm dark:prose-invert max-w-none mt-4">
                {formatStory(activeChart.takeaway)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bento Grid View */}
      {chartList.length > 0 && !activeChart && (
        <div className="grid grid-cols-1 md:grid-cols-6 gap-6 animate-in fade-in duration-500">
          {currentCharts.map((chart, idx) => {
            const isTopRow = idx < 3;
            const colSpan = isTopRow ? "md:col-span-2" : "md:col-span-3";
            const titleText = chart.columns ? chart.columns.join(' vs ') : 'Chart';
            
            return (
              <div 
                key={idx} 
                onClick={() => setActiveChart(chart)}
                className={`${colSpan} bg-[var(--text)]/[0.015] border-2 border-[var(--text)]/[0.1] rounded-3xl p-6 cursor-pointer hover:-translate-y-2 hover:shadow-2xl transition-all duration-300 flex flex-col group`}
              >
                <div className="mb-4 flex justify-between items-start">
                  <h3 className="text-lg font-bold capitalize truncate pr-4 group-hover:text-[var(--yellow)] transition-colors">
                    {titleText}
                  </h3>
                  <span className="text-[10px] px-2 py-1 bg-[var(--text)]/[0.1] rounded-full uppercase text-[var(--text)] font-bold tracking-wider">
                    {chart.chart_type}
                  </span>
                </div>
                <div className="flex-1 w-full flex items-center justify-center">
                   <AestheticChart data={chart} height="220px" />
                </div>
                <div className="mt-4 pt-4 border-t-2 border-[var(--text)]/[0.05] flex justify-between items-center transition-colors group-hover:text-[var(--yellow)]">
                  <span className="text-xs font-bold text-[var(--muted)] group-hover:text-[var(--yellow)]">View Insights</span>
                  <span>→</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {!isRunning && chartsData && chartList.length === 0 && (
         <div className="mt-8 p-12 border-2 border-[var(--text)]/[0.1] rounded-3xl text-center bg-[var(--text)]/[0.015]">
            <p className="font-bold text-xl mb-2">No Charts Recommended</p>
            <p className="text-[var(--muted)]">The AI did not find any highly significant visual relationships to plot.</p>
         </div>
      )}
    </div>
  );
}

const StatusBar = ({ label }) => (
  <div className="mt-8 flex items-center gap-4 rounded-2xl border-2 border-[var(--text)]/[0.1] bg-[var(--text)]/[0.015] px-6 py-5">
    <Spinner />
    <p className="text-base text-[var(--text)] font-medium font-mono">
      {label}
    </p>
  </div>
);

const ErrBar = ({ msg }) => (
  <div className="mt-8 rounded-2xl border-2 border-red-500/30 bg-red-500/10 px-6 py-5">
    <p className="font-mono text-sm text-red-400 font-semibold">
      ✗ {msg}
    </p>
  </div>
);

const Spinner = () => (
  <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--text)]" />
);

const Empty = ({ message, onAction, actionLabel }) => (
  <div>
    <h1 className="text-3xl font-extrabold tracking-tight">Data EDA</h1>
    <div className="mt-8 surface border-2 border-[var(--text)]/[0.1] rounded-3xl p-12 text-center bg-[var(--text)]/[0.015]">
      <p className="text-base text-[var(--muted)] font-mono mb-6">
        {message}
      </p>
      {onAction && (
        <button
          onClick={onAction}
          className="rounded-full bg-[var(--text)] text-[var(--bg)] px-8 py-3 text-sm font-bold shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all"
        >
          {actionLabel}
        </button>
      )}
    </div>
  </div>
);
