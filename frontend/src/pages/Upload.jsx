import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client";
import { usePipeline } from "../context/PipelineContext";
import { Mascot, PageHeader } from "../components/brand/Brand";

const MAX_DATASETS = 3;

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const { addJob, selectDataset: selectPipelineDataset, forgetDataset } = usePipeline();
  const [deleting, setDeleting] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [navigating, setNavigating] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedFilename, setSelectedFilename] = useState(
    () => localStorage.getItem("selectedDataset") || null,
  );

  async function fetchDatasets() {
    setLoadingList(true);
    try {
      const res = await api.get("/api/stored_datasets");
      setDatasets(res.data.datasets || []);
    } catch (err) {
      setError(err.response?.data?.message || "Could not load datasets");
    } finally {
      setLoadingList(false);
    }
  }

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    if (!localStorage.getItem("token")) {
      navigate("/login");
      return;
    }
    fetchDatasets();
  }, [navigate]);

  const selectDataset = (filename) => {
    setSelectedFilename(filename);
    selectPipelineDataset(filename);
  };

  const uploadFile = async (file) => {
    setError("");
    setSuccess("");
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only CSV files are supported");
      return;
    }
    if (datasets.length >= MAX_DATASETS) {
      setError(`Dataset limit reached (${MAX_DATASETS} max).`);
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const res = await api.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSuccess(`"${res.data.data.filename}" uploaded successfully.`);
      await fetchDatasets();
    } catch (err) {
      setError(err.response?.data?.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const deleteDataset = async (filename) => {
    if (!window.confirm(`Delete "${filename}"? This removes the file and its analysis.`)) return;
    setError("");
    setSuccess("");
    setDeleting(filename);
    try {
      await api.delete(`/api/datasets/${encodeURIComponent(filename)}`);
      forgetDataset(filename);
      if (selectedFilename === filename) setSelectedFilename(null);
      setSuccess(`"${filename}" deleted.`);
      await fetchDatasets();
    } catch (err) {
      setError(err.response?.data?.message || "Could not delete dataset");
    } finally {
      setDeleting(null);
    }
  };

  const goToDashboard = async () => {
    if (!selectedFilename) return;
    setNavigating(true);
    setError("");
    try {
      // Trigger Data Info analysis job before navigating
      const res = await api.post("/api/run_info", {
        filename: selectedFilename,
      });
      addJob("info_job_id", res.data.info_job_id);
      navigate("/dashboard");
    } catch (err) {
      setError(err.response?.data?.message || "Could not start analysis");
      setNavigating(false);
    }
  };

  const atLimit = datasets.length >= MAX_DATASETS;

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (atLimit) return;
    const f = e.dataTransfer.files?.[0];
    if (f) uploadFile(f);
  };

  return (
    <div>
      <PageHeader stage="upload" title="Upload a dataset">
        Add a CSV, then pick the dataset you want to work on.
      </PageHeader>

      {/* Dropzone: a <label> around a visually hidden file input, so it works by click, drag and keyboard. */}
      <label
        onDragOver={(e) => {
          e.preventDefault();
          if (!atLimit) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        className={`relative flex min-h-[260px] flex-col items-center justify-center rounded-3xl border-[3px] border-dashed px-6 py-12 text-center transition-colors has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-solid has-[:focus-visible]:outline-[var(--sky)] ${
          atLimit
            ? "cursor-not-allowed border-[var(--line)] opacity-60"
            : dragActive
              ? "cursor-copy border-[var(--lime-ink)] bg-[var(--lime)]/15"
              : "cursor-pointer border-[var(--line)] bg-[var(--surface)] hover:border-[var(--lime-ink)]"
        }`}
      >
        <Mascot stage="upload" size={88} className="mb-4" />
        <span className="text-2xl font-extrabold tracking-tight">
          {atLimit ? "Your library is full" : dragActive ? "Drop it here" : "Drop a CSV here, or click to choose one"}
        </span>
        <span className="mt-2 text-[var(--muted)]">
          {atLimit
            ? `You can keep ${MAX_DATASETS} datasets. Delete one below to upload another.`
            : "Up to 25,000 rows and 20 columns."}
        </span>

        {uploading && (
          <span className="absolute inset-0 flex items-center justify-center rounded-3xl bg-[var(--surface)]/90 text-lg font-bold" role="status">
            Uploading…
          </span>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) uploadFile(f);
            e.target.value = "";
          }}
          disabled={atLimit || uploading}
        />
      </label>

      <div className="mt-4 min-h-[3rem]" aria-live="polite">
        {error && (
          <p role="alert" className="rounded-xl border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-4 py-3 font-semibold text-[var(--danger)]">
            {error}
          </p>
        )}
        {success && (
          <p className="rounded-xl border border-[var(--lime-ink)]/30 bg-[var(--lime)]/15 px-4 py-3 font-semibold text-[var(--lime-ink)]">
            {success}
          </p>
        )}
      </div>

      <section className="mt-8">
        <div className="mb-4 flex items-baseline justify-between gap-4">
          <h2 className="text-2xl font-extrabold tracking-tight">Your library</h2>
          <span className="text-[var(--muted)]">
            <span className="tabular-nums">{datasets.length} of {MAX_DATASETS} used</span>
          </span>
        </div>

        {loadingList ? (
          <p className="py-10 text-[var(--muted)]" role="status">Loading your datasets…</p>
        ) : datasets.length === 0 ? (
          <p className="rounded-2xl border border-dashed border-[var(--line)] px-6 py-10 text-[var(--muted)]">
            Nothing here yet. Upload a CSV above to get started.
          </p>
        ) : (
          <ul className="divide-y divide-[var(--line)] rounded-2xl border border-[var(--line)] bg-[var(--surface)]">
            {datasets.map((d) => {
              const isSelected = d.filename === selectedFilename;
              return (
                <li key={d.filename} className="flex items-center gap-2 pr-3">
                  <button
                    onClick={() => selectDataset(d.filename)}
                    aria-pressed={isSelected}
                    className={`flex min-w-0 flex-1 items-center gap-4 border-l-[5px] px-5 py-4 text-left transition-colors hover:bg-[var(--surface-2)] ${
                      isSelected ? "border-[var(--lime)]" : "border-transparent"
                    }`}
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 text-[var(--muted)]" aria-hidden="true">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                    </svg>
                    <span className="truncate font-semibold">{d.filename}</span>
                    {isSelected && <span className="ml-auto hidden shrink-0 text-sm font-bold text-[var(--lime-ink)] sm:inline">Selected</span>}
                  </button>
                  <button
                    onClick={() => deleteDataset(d.filename)}
                    disabled={deleting === d.filename}
                    aria-label={`Delete ${d.filename}`}
                    title="Delete dataset"
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[var(--muted)] transition-colors hover:bg-[var(--danger)]/10 hover:text-[var(--danger)] disabled:opacity-40"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                      <path d="M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                    </svg>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <div className="mt-10 flex flex-wrap items-center gap-4">
        <button
          onClick={goToDashboard}
          disabled={!selectedFilename || navigating}
          className="rounded-full bg-[var(--text)] px-8 py-3.5 text-base font-bold text-[var(--bg)] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {navigating ? "Starting analysis…" : "Analyze this dataset"}
        </button>
        {!selectedFilename && datasets.length > 0 && (
          <span className="text-[var(--muted)]">Select a dataset first.</span>
        )}
      </div>
    </div>
  );
}
