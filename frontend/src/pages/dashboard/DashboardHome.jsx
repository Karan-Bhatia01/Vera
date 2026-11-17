import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";
import { Mascot, PageHeader } from "../../components/brand/Brand";
import { STAGES } from "../../components/brand/stages";
import Select from "../../components/Select";

const MAX_DATASETS = 3;

const pipelineSteps = [
  { stage: "upload", label: "Upload", path: "/upload" },
  { stage: "info", label: "Data info", path: "/dashboard/info" },
  { stage: "eda", label: "EDA", path: "/dashboard/eda" },
  { stage: "ml", label: "Modelling", path: "/dashboard/ml" },
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
      <PageHeader title="Dashboard">Pick a dataset and the column you want to predict.</PageHeader>

      {polling && pipelineFile === selectedFilename && (
        <p role="status" className="mb-6 flex items-center gap-3 rounded-xl border border-[var(--line)] bg-[var(--surface)] px-4 py-3 font-semibold">
          <Spinner />
          Vera is working on this dataset. Results appear on each page as they finish.
        </p>
      )}

      <Stepper activeStep={activeStep} onNavigate={navigate} />

      <div className="mt-10 grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6">
          <h2 className="mb-4 text-xl font-extrabold tracking-tight">Your datasets</h2>
          {loading ? (
            <p className="text-[var(--muted)]" role="status">Loading…</p>
          ) : datasets.length === 0 ? (
            <div className="py-4">
              <p className="text-[var(--muted)]">No datasets yet.</p>
              <button
                onClick={() => navigate("/upload")}
                className="mt-4 rounded-full bg-[var(--text)] px-5 py-2.5 font-bold text-[var(--bg)] hover:opacity-90"
              >
                Upload a CSV
              </button>
            </div>
          ) : (
            <ul className="space-y-2">
              {datasets.map((d) => {
                const isSelected = d.filename === selectedFilename;
                return (
                  <li key={d.filename}>
                    <button
                      onClick={() => selectDataset(d.filename)}
                      aria-pressed={isSelected}
                      className={`flex w-full items-center gap-3 rounded-xl border-2 px-4 py-3 text-left transition-colors ${
                        isSelected
                          ? "border-[var(--sun)] bg-[var(--sun)]/10"
                          : "border-[var(--line)] hover:border-[var(--muted)]/50"
                      }`}
                    >
                      <span className="truncate font-semibold">{d.filename}</span>
                      {isSelected && <span className="ml-auto shrink-0 text-sm font-bold text-[var(--sun-ink)]">Selected</span>}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6">
          <h2 className="text-xl font-extrabold tracking-tight">Storage</h2>
          <p className="mt-3 text-3xl font-extrabold">
            <span className="font-mono">{datasets.length}</span>
            <span className="text-lg font-semibold text-[var(--muted)]"> of {MAX_DATASETS} datasets</span>
          </p>
          <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-[var(--surface-2)]" aria-hidden="true">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${(datasets.length / MAX_DATASETS) * 100}%`, background: "var(--lime)" }}
            />
          </div>
        </section>
      </div>

      {selectedFilename && (
        <section className="mt-6 rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6">
          <h2 className="text-xl font-extrabold tracking-tight">Set up the analysis</h2>
          <p className="mt-1 mb-6 text-[var(--muted)]">
            For <strong className="text-[var(--text)]">{selectedFilename}</strong>. Choose a target, leave out any columns you don&apos;t need, then run.
          </p>

          {columnsLoading ? (
            <p className="text-[var(--muted)]" role="status">Loading columns…</p>
          ) : submitted ? (
            <p role="status" className="rounded-xl border border-[var(--lime-ink)]/30 bg-[var(--lime)]/15 px-4 py-3 font-semibold text-[var(--lime-ink)]">
              Analysis started. Open{" "}
              <button onClick={() => navigate("/dashboard/eda")} className="underline underline-offset-2">Data EDA</button>{" "}
              to see the charts as they arrive.
            </p>
          ) : (
            <>
              <div className="mb-6 max-w-md">
                <label id="target-column-label" htmlFor="target-column" className="mb-2 block font-semibold">Column to predict</label>
                <Select
                  id="target-column"
                  labelledBy="target-column-label"
                  value={targetColumn}
                  onChange={setTargetColumn}
                  options={columns}
                  placeholder="Choose a column…"
                />
                <p className="mt-2 text-sm text-[var(--muted)]">This is the column the models learn to predict.</p>
              </div>

              {columns.length > 0 && (
                <fieldset className="mb-6">
                  <legend className="mb-3 font-semibold">Columns to leave out</legend>
                  <div className="flex flex-wrap gap-2">
                    {columns
                      .filter((c) => c !== targetColumn)
                      .map((col) => {
                        const dropped = columnsToDrop.includes(col);
                        return (
                          <button
                            key={col}
                            type="button"
                            onClick={() => toggleDrop(col)}
                            aria-pressed={dropped}
                            className={`rounded-full border-2 px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                              dropped
                                ? "border-[var(--danger)]/40 bg-[var(--danger)]/10 text-[var(--danger)] line-through"
                                : "border-[var(--line)] text-[var(--text)] hover:border-[var(--muted)]/50"
                            }`}
                          >
                            {col}
                          </button>
                        );
                      })}
                  </div>
                </fieldset>
              )}

              {submitError && (
                <p role="alert" className="mb-4 rounded-xl border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-4 py-3 font-semibold text-[var(--danger)]">
                  {submitError}
                </p>
              )}

              <button
                onClick={submitEDA}
                disabled={!targetColumn || submitting}
                className="rounded-full bg-[var(--text)] px-7 py-3 font-bold text-[var(--bg)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {submitting ? "Starting…" : "Run analysis"}
              </button>
            </>
          )}
        </section>
      )}
    </div>
  );
}

/** The four stages as mascots: done ones are tinted, the current one is ringed, later ones are faded. */
function Stepper({ activeStep, onNavigate }) {
  return (
    <ol className="grid grid-cols-4 gap-2 rounded-2xl border border-[var(--line)] bg-[var(--surface)] px-3 py-5 sm:px-6" aria-label="Progress">
      {pipelineSteps.map((step, i) => {
        const state = i < activeStep ? "done" : i === activeStep ? "current" : "next";
        const { color } = STAGES[step.stage];
        return (
          <li key={step.stage}>
            <button
              onClick={() => onNavigate(step.path)}
              aria-current={state === "current" ? "step" : undefined}
              className="flex w-full flex-col items-center gap-2 rounded-xl py-1 text-center"
            >
              <span
                className={`flex h-16 w-16 items-center justify-center rounded-full border-[3px] ${state === "next" ? "opacity-45 grayscale" : ""}`}
                style={{
                  borderColor: state === "next" ? "var(--line)" : color,
                  background: state === "done" ? `color-mix(in srgb, ${color} 22%, transparent)` : "transparent",
                }}
              >
                <Mascot stage={step.stage} size={50} />
              </span>
              <span className={`text-sm font-semibold ${state === "next" ? "text-[var(--muted)]" : ""}`}>
                {step.label}
                {state === "done" && <span className="sr-only"> (done)</span>}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

const Spinner = () => (
  <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--sky)]" aria-hidden="true" />
);
