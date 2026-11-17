import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";

const MAX_DATASETS = 3;

const pipelineSteps = [
  { label: "Upload", path: "/upload" },
  { label: "Info", path: "/dashboard/info" },
  { label: "EDA", path: "/dashboard" },
  { label: "Modelling", path: "/dashboard/ml" },
];

function getActiveStep(selectedFilename, submitted) {
  if (!selectedFilename) return 0;
  if (submitted) return 3;
  return 2;
}

export default function DashboardHome() {
  const navigate = useNavigate();
  const {
    polling,
    filename: pipelineFile,
    addJob,
    selectDataset: selectPipelineDataset,
  } = usePipeline();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFilename, setSelectedFilename] = useState(
    () => localStorage.getItem("selectedDataset") || null,
  );
  const [columns, setColumns] = useState([]);
  const [columnsLoading, setColumnsLoading] = useState(false);
  const [targetColumn, setTargetColumn] = useState("");
  const [columnsToDrop, setColumnsToDrop] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  async function fetchDatasets() {
    setLoading(true);
    try {
      const res = await api.get("/api/stored_datasets");
      setDatasets(res.data.datasets || []);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }

  async function fetchColumns(fname) {
    setColumnsLoading(true);
    try {
      const res = await api.get(`/api/info/${encodeURIComponent(fname)}`);
      setColumns(res.data.data?.columns || []);
    } catch {
      setColumns([]);
    } finally {
      setColumnsLoading(false);
    }
  }

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    fetchDatasets();
  }, []);

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    if (!selectedFilename) {
      setColumns([]);
      return;
    }
    setSubmitted(false);
    setSubmitError("");
    setTargetColumn("");
    setColumnsToDrop([]);
    fetchColumns(selectedFilename);
  }, [selectedFilename]);

  const selectDataset = (fname) => {
    setSelectedFilename(fname);
    selectPipelineDataset(fname);
  };

  const toggleDrop = (col) =>
    setColumnsToDrop((p) =>
      p.includes(col) ? p.filter((c) => c !== col) : [...p, col],
    );

  const submitEDA = async () => {
    if (!selectedFilename || !targetColumn) return;
    setSubmitting(true);
    setSubmitError("");
    try {
      const res = await api.post("/api/run_eda", {
        filename: selectedFilename,
        target_column: targetColumn,
        columns_to_drop: columnsToDrop,
      });
      localStorage.setItem("targetColumn", targetColumn);
      addJob("eda_job_id", res.data.eda_job_id);
      setSubmitted(true);
    } catch (err) {
      setSubmitError(err.response?.data?.message || "Could not start EDA");
    } finally {
      setSubmitting(false);
    }
  };

  const activeStep = getActiveStep(selectedFilename, submitted);

  return (
    <div>
      {/* Header */}
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <p className="mt-1 text-sm text-[var(--muted)]">
        Select a dataset and configure your analysis.
      </p>

      {/* Pipeline running banner */}
      {polling && pipelineFile === selectedFilename && (
        <div className="mt-4 flex items-center gap-2 rounded-xl border border-[var(--blue)]/30 bg-[var(--blue)]/5 px-4 py-2.5">
          <Spinner />
          <p className="text-xs font-semibold text-[var(--blue)] font-mono">
            Pipeline running…
          </p>
        </div>
      )}

      {/* Pipeline Steps */}
      <div className="mt-6 flex items-center gap-0">
        {pipelineSteps.map((step, i) => {
          const isCompleted = i < activeStep;
          const isCurrent = i === activeStep;
          return (
            <div key={step.label} className="flex items-center flex-1 last:flex-none">
              <button
                onClick={() => navigate(step.path)}
                className="flex items-center gap-2 group"
              >
                <div
                  className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-all ${
                    isCompleted
                      ? "bg-[var(--text)] text-[var(--bg)]"
                      : isCurrent
                        ? "border-2 border-[var(--text)] text-[var(--text)]"
                        : "border border-[var(--text)]/20 text-[var(--muted)]"
                  }`}
                >
                  {isCompleted ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </div>
                <span
                  className={`text-xs font-semibold transition-colors ${
                    isCompleted || isCurrent ? "text-[var(--text)]" : "text-[var(--muted)]"
                  }`}
                >
                  {step.label}
                </span>
              </button>
              {i < pipelineSteps.length - 1 && (
                <div className="flex-1 mx-3">
                  <div
                    className={`h-[1.5px] w-full rounded-full transition-colors ${
                      i < activeStep ? "bg-[var(--text)]" : "bg-[var(--text)]/10"
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Dataset History + Usage Card */}
      <div className="mt-8 grid gap-4 lg:grid-cols-[2.5fr_1fr]">
        {/* Dataset History */}
        <div className="rounded-2xl border border-[var(--text)]/[0.15] p-5">
          <h2 className="text-sm font-bold mb-4">Dataset History</h2>
          {loading ? (
            <p className="text-xs text-[var(--muted)] font-mono">Loading…</p>
          ) : datasets.length === 0 ? (
            <div className="text-center py-6">
              <p className="text-xs text-[var(--muted)]">No datasets yet.</p>
              <button
                onClick={() => navigate("/upload")}
                className="mt-4 rounded-xl bg-[var(--text)] px-5 py-2 text-xs font-bold text-[var(--bg)]"
              >
                Upload a dataset
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {datasets.map((d) => {
                const isSelected = d.filename === selectedFilename;
                return (
                  <button
                    key={d.filename}
                    onClick={() => selectDataset(d.filename)}
                    className={`w-full flex items-center gap-4 rounded-xl border p-3.5 text-left transition-all ${
                      isSelected
                        ? "border-[var(--text)]/30 bg-[var(--text)]/[0.04]"
                        : "border-[var(--text)]/[0.06] hover:border-[var(--text)]/20"
                    }`}
                  >
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--text)]/[0.05]">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--muted)]">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-xs truncate">{d.filename}</h3>
                    </div>
                    {isSelected && (
                      <span className="flex items-center gap-1.5 shrink-0 text-[11px] font-bold text-[var(--text)]">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--green)]">
                          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                          <polyline points="22 4 12 14.01 9 11.01" />
                        </svg>
                        Selected
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Usage Card */}
        <div className="rounded-2xl border border-[var(--text)]/[0.15] p-5 flex flex-col items-start justify-between w-full">
          <div>
            <p className="text-xs text-[var(--muted)] mb-1">Storage</p>
            <p className="text-lg font-bold">
              {datasets.length} <span className="text-sm font-normal text-[var(--muted)]">of {MAX_DATASETS}</span>
            </p>
            <p className="text-[11px] text-[var(--muted)] mt-1">datasets used</p>
            {/* Progress bar */}
            <div className="mt-3 h-1.5 w-full rounded-full bg-[var(--text)]/[0.08] overflow-hidden">
              <div
                className="h-full rounded-full bg-[var(--blue)] transition-all duration-500"
                style={{ width: `${(datasets.length / MAX_DATASETS) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* EDA Config Form */}
      {selectedFilename && (
        <div className="mt-6 rounded-2xl border border-[var(--text)]/[0.15] p-5">
          <div className="flex items-center gap-2 mb-1">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--muted)]">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            <h2 className="text-sm font-bold">Configure Analysis</h2>
          </div>
          <p className="text-xs text-[var(--muted)] mb-5">
            For <span className="font-semibold text-[var(--text)]">{selectedFilename}</span>, select target column, then submit to start Analysis.
          </p>

          {columnsLoading ? (
            <p className="text-xs text-[var(--muted)] font-mono">Loading columns…</p>
          ) : submitted ? (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-4 py-3">
              <p className="text-xs text-emerald-400 font-medium">
                ✓ Analysis started. Check <strong>Data EDA</strong> in the sidebar for results.
              </p>
            </div>
          ) : (
            <>
              {/* Target Column */}
              <div className="mb-5">
                <div className="flex items-center justify-between mb-2">
                  <label className="flex items-center gap-2 text-xs font-semibold">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--blue)]">
                      <circle cx="12" cy="12" r="10" />
                      <circle cx="12" cy="12" r="6" />
                      <circle cx="12" cy="12" r="2" />
                    </svg>
                    Target Column
                  </label>
                </div>
                <CustomSelect
                  value={targetColumn}
                  onChange={setTargetColumn}
                  options={columns}
                  placeholder="Select target column..."
                />
                <p className="text-[10px] text-[var(--muted)] mt-1.5 flex items-center gap-1">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                  This column will be used as the prediction target.
                </p>
              </div>

              {/* Columns to Drop */}
              {columns.length > 0 && (
                <div className="mb-5">
                  <label className="block text-xs font-semibold mb-3">
                    Columns to Drop
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {columns
                      .filter((c) => c !== targetColumn)
                      .map((col) => {
                        const dropped = columnsToDrop.includes(col);
                        return (
                          <button
                            key={col}
                            onClick={() => toggleDrop(col)}
                            className={`group flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[11px] font-medium transition-all ${
                              dropped
                                ? "border-red-500/30 bg-red-500/10 text-[var(--muted)] line-through hover:border-red-500/50"
                                : "border-[var(--text)]/[0.15] text-[var(--text)] hover:border-[var(--text)]/40 hover:bg-[var(--text)]/[0.02]"
                            }`}
                          >
                            {dropped ? (
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-red-400">
                                <line x1="18" y1="6" x2="6" y2="18" />
                                <line x1="6" y1="6" x2="18" y2="18" />
                              </svg>
                            ) : (
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--text)]/40 group-hover:text-[var(--text)]">
                                <line x1="12" y1="5" x2="12" y2="19" />
                                <line x1="5" y1="12" x2="19" y2="12" />
                              </svg>
                            )}
                            {col}
                          </button>
                        );
                      })}
                  </div>
                </div>
              )}

              {submitError && (
                <p className="mb-4 rounded-xl border border-red-500/20 bg-red-500/5 px-3 py-2 text-[11px] text-red-400 font-medium">
                  {submitError}
                </p>
              )}

              <button
                onClick={submitEDA}
                disabled={!targetColumn || submitting}
                className={`rounded-xl px-6 py-2.5 text-xs font-bold transition-all ${
                  targetColumn && !submitting
                    ? "bg-[var(--text)] text-[var(--bg)] hover:opacity-90"
                    : "bg-[var(--text)]/[0.05] text-[var(--muted)] cursor-not-allowed"
                }`}
              >
                {submitting ? "Starting…" : "Run Analysis →"}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const Spinner = () => (
  <div className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--blue)] border-t-transparent" />
);

function CustomSelect({ value, onChange, options, placeholder }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative w-full">
      <div
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between rounded-xl border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] text-[var(--text)] px-4 py-2.5 text-xs cursor-pointer hover:border-[var(--text)]/20 transition-colors"
      >
        <span>
          {value ? (
            value
          ) : (
            <span className="text-[var(--muted)]">{placeholder}</span>
          )}
        </span>
        <svg
          className={`w-3.5 h-3.5 transition-transform text-[var(--muted)] ${open ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </div>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          ></div>
          <div className="absolute z-50 w-full mt-1 bg-[var(--surface)] border border-[var(--text)]/[0.1] rounded-xl shadow-2xl shadow-black/40 max-h-60 overflow-y-auto custom-scrollbar p-1.5 flex flex-col gap-0.5">
            {options.map((opt) => (
              <div
                key={opt}
                onClick={() => {
                  onChange(opt);
                  setOpen(false);
                }}
                className={`px-3 py-2 text-xs rounded-lg cursor-pointer transition-colors ${
                  value === opt
                    ? "bg-[var(--text)] text-[var(--bg)] font-bold"
                    : "text-[var(--text)] hover:bg-[var(--text)]/[0.06]"
                }`}
              >
                {opt}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
