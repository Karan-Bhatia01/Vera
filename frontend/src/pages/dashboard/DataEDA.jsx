import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "../../context/PipelineContext";
import AestheticChart from "../../components/charts/AestheticChart";
import { Mascot, PageHeader } from "../../components/brand/Brand";

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
        actionLabel="Open Upload"
      />
    );

  const jobs = JSON.parse(localStorage.getItem("pipelineJobs") || "null");
  const hasEdaJob = jobs?.filename === filename && jobs?.eda_job_id;

  if (!hasEdaJob && !eda)
    return (
      <div>
        <PageHeader stage="eda" title="Data EDA" />
        <div className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-10">
          <p className="text-xl font-bold mb-2">No charts yet</p>
          <p className="text-base text-[var(--muted)] mb-6 max-w-[56ch]">
            On the Dashboard, choose the column to predict and select Run analysis. Your charts will appear here.
          </p>
          <button
            onClick={() => navigate("/dashboard")}
            className="rounded-full bg-[var(--text)] text-[var(--bg)] px-8 py-3 text-sm font-bold transition-all"
          >
            Open Dashboard
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
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <PageHeader stage="eda" title="Data EDA">Charts picked by the AI for {filename}</PageHeader>
        
        {/* Pagination Controls */}
        {chartList.length > chartsPerPage && !activeChart && (
          <div className="flex items-center gap-4 bg-[var(--text)]/[0.03] px-4 py-2 rounded-full border border-[var(--line)] self-start sm:self-auto sm:mr-14">
            <button 
              onClick={handlePrev} 
              disabled={page === 0}
              aria-label="Previous charts"
              className="text-[var(--text)] disabled:opacity-30"
            >
              ←
            </button>
            <span className="text-sm font-bold font-mono" aria-live="polite">
              {page + 1} / {totalPages}
            </span>
            <button 
              onClick={handleNext} 
              disabled={page === totalPages - 1}
              aria-label="Next charts"
              className="text-[var(--text)] disabled:opacity-30"
            >
              →
            </button>
          </div>
        )}
      </div>

      {isRunning && <StatusBar label={eda?.message || "Generating interactive charts…"} />}
      {eda?.status === "failed" && <ErrBar msg={eda.error || "EDA failed"} />}
      {!isRunning && !chartsData && hasEdaJob && (
        <StatusBar label="Waiting for the results…" />
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
            <div className="xl:w-2/3 min-w-0 bg-[var(--surface)] border border-[var(--line)] rounded-2xl p-4 sm:p-8">
              <div className="mb-6 flex justify-between items-start">
                <h3 className="text-2xl font-extrabold tracking-tight capitalize">
                  {activeChart.columns ? activeChart.columns.join(' vs ') : 'Chart'}
                </h3>
                <span className="px-3 py-1 bg-[var(--text)] text-[var(--bg)] rounded-full text-xs font-bold capitalize">
                  {activeChart.chart_type}
                </span>
              </div>
              <AestheticChart data={activeChart} height="clamp(300px, 60vh, 500px)" />
            </div>

            {/* Fixed heading + scrolling body, so text never slides behind the title */}
            <div className="xl:w-1/3 flex max-h-[600px] flex-col overflow-hidden rounded-2xl border border-[var(--violet)]/30 bg-[var(--violet)]/[0.06]">
              <h4 className="shrink-0 border-b border-[var(--violet)]/20 px-6 py-4 font-bold text-[var(--violet-ink)] sm:px-8">What this chart shows</h4>
              <div className="flex-1 overflow-y-auto px-6 pt-5 pb-2 sm:px-8">
                {formatStory(activeChart.takeaway)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bento Grid View */}
      {chartList.length > 0 && !activeChart && (
        <div className="grid grid-cols-1 md:grid-cols-6 gap-6">
          {currentCharts.map((chart, idx) => {
            const isTopRow = idx < 3;
            const colSpan = isTopRow ? "md:col-span-2" : "md:col-span-3";
            const titleText = chart.columns ? chart.columns.join(' vs ') : 'Chart';
            
            return (
              <div 
                key={idx} 
                onClick={() => setActiveChart(chart)}
                className={`${colSpan} bg-[var(--surface)] border border-[var(--line)] rounded-2xl p-5 sm:p-6 cursor-pointer min-w-0 transition-all duration-300 flex flex-col group`}
              >
                <div className="mb-4 flex justify-between items-start">
                  <h3 className="text-lg font-bold capitalize truncate pr-4 group-hover:text-[var(--violet-ink)] transition-colors">
                    {titleText}
                  </h3>
                  <span className="text-[10px] px-2 py-1 bg-[var(--text)]/[0.1] rounded-full text-xs text-[var(--text)] font-bold capitalize bg-[var(--surface-2)]">
                    {chart.chart_type}
                  </span>
                </div>
                <div className="flex-1 w-full flex items-center justify-center">
                   <AestheticChart data={chart} height="220px" />
                </div>
                <div className="mt-4 pt-4 border-t border-[var(--line)] flex justify-between items-center transition-colors group-hover:text-[var(--violet-ink)]">
                  <span className="text-xs font-bold text-[var(--muted)] group-hover:text-[var(--violet-ink)]">Read the explanation</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {!isRunning && chartsData && chartList.length === 0 && (
         <div className="mt-8 flex items-center gap-5 p-8 border border-[var(--line)] rounded-2xl bg-[var(--surface)]">
            <Mascot stage="eda" size={64} />
            <div>
              <p className="font-bold text-xl mb-1">No charts to show</p>
              <p className="text-[var(--muted)] max-w-[60ch]">The AI didn&apos;t recommend any charts. Make sure Data info finished for this dataset, then run the analysis again from the Dashboard.</p>
            </div>
         </div>
      )}
    </div>
  );
}

const StatusBar = ({ label }) => (
  <div className="mt-8 flex items-center gap-4 rounded-2xl border border-[var(--line)] bg-[var(--surface)] px-6 py-5">
    <Spinner />
    <p className="text-base font-semibold">
      {label}
    </p>
  </div>
);

const ErrBar = ({ msg }) => (
  <div className="mt-8 rounded-2xl border-2 border-[var(--danger)]/30 bg-[var(--danger)]/10 px-6 py-5">
    <p className="text-[var(--danger)] font-semibold">
      {msg}
    </p>
  </div>
);

const Spinner = () => (
  <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--text)]" />
);

const Empty = ({ message, onAction, actionLabel }) => (
  <div>
    <PageHeader stage="eda" title="Data EDA" />
    <div className="mt-8 surface border border-[var(--line)] rounded-2xl p-12 text-center bg-[var(--surface)]">
      <p className="text-base text-[var(--muted)] mb-6">
        {message}
      </p>
      {onAction && (
        <button
          onClick={onAction}
          className="rounded-full bg-[var(--text)] text-[var(--bg)] px-8 py-3 text-sm font-bold transition-all"
        >
          {actionLabel}
        </button>
      )}
    </div>
  </div>
);
