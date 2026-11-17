import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client";
import { usePipeline } from "../context/PipelineContext";

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

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 flex flex-col items-center min-h-screen">
      <header className="text-center mb-16">
        <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight text-[var(--text)]">
          Upload Dataset.
        </h1>
        <p className="mt-4 text-[var(--muted)] text-lg">
          Drop your CSV file to unlock automated model-ready insights.
        </p>
      </header>

      {/* Main Upload Dropzone */}
      <div className="w-full max-w-2xl relative group mb-16">
        {/* Subtle border glow on hover */}
        <div className="absolute -inset-[1px] rounded-[2.5rem] bg-[var(--text)]/[0.06] opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
        
        <div
          onClick={() => !atLimit && fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            if (!atLimit) setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={
            atLimit
              ? undefined
              : (e) => {
                  e.preventDefault();
                  setDragActive(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) uploadFile(f);
                }
          }
          className={`relative flex min-h-[320px] w-full flex-col items-center justify-center rounded-[2rem] border border-[var(--text)]/[0.1] bg-[var(--bg)] px-6 py-10 text-center transition-all duration-300 ${
            atLimit
              ? "opacity-50 cursor-not-allowed"
              : dragActive
                ? "border-[var(--blue)] bg-[var(--blue)]/5 cursor-pointer scale-[1.02]"
                : "cursor-pointer hover:border-[var(--text)]/[0.3] hover:bg-[var(--text)]/[0.02]"
          }`}
        >
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[var(--text)]/[0.05] mb-6 transition-transform duration-300 group-hover:scale-110">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--text)]">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          
          <h3 className="text-2xl font-bold mb-2">
            {atLimit ? "Storage Full" : "Click or drag CSV here"}
          </h3>
          <p className="text-sm text-[var(--muted)] font-mono">
            {atLimit
              ? `You have reached the ${MAX_DATASETS} dataset limit.`
              : "Max 25,000 rows · 20 columns"}
          </p>

          {uploading && (
            <div className="absolute inset-0 flex items-center justify-center rounded-[2rem] bg-[var(--bg)]/80 backdrop-blur-sm">
              <p className="font-mono text-sm font-bold animate-pulse text-[var(--blue)]">
                Uploading...
              </p>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) uploadFile(f);
              e.target.value = "";
            }}
            disabled={atLimit}
          />
        </div>
        
        {/* Alerts */}
        <div className="absolute -bottom-16 left-0 right-0 flex justify-center">
          {error && (
            <p className="rounded-full border border-red-500/20 bg-red-500/10 px-6 py-2 text-sm text-red-400 font-medium animate-in fade-in slide-in-from-bottom-2">
              {error}
            </p>
          )}
          {success && (
            <p className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-6 py-2 text-sm text-emerald-400 font-medium animate-in fade-in slide-in-from-bottom-2">
              {success}
            </p>
          )}
        </div>
      </div>

      {/* Dataset Selection */}
      <div className="w-full max-w-3xl mt-8 rounded-[2rem] border border-[var(--text)]/[0.1] bg-[var(--text)]/[0.02] p-8 sm:p-10">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold">Your Library</h2>
            <p className="text-sm text-[var(--muted)] mt-1">Select a dataset to begin analysis</p>
          </div>
          <span className="mt-4 sm:mt-0 inline-flex items-center rounded-full bg-[var(--text)]/[0.05] px-4 py-1.5 text-xs font-bold font-mono text-[var(--text)]">
            {datasets.length} / {MAX_DATASETS} USED
          </span>
        </div>

        {loadingList ? (
          <div className="flex justify-center py-12">
            <p className="text-sm text-[var(--muted)] font-mono animate-pulse">Loading datasets...</p>
          </div>
        ) : datasets.length === 0 ? (
          <div className="rounded-2xl border border-[var(--text)]/[0.05] border-dashed py-12 text-center">
            <p className="text-[var(--muted)]">No files found. Upload your first CSV above.</p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {datasets.map((d) => {
              const isSelected = d.filename === selectedFilename;
              return (
                <div key={d.filename} className="group relative transition-all duration-300 hover:-translate-y-1">
                  <button
                    onClick={() => selectDataset(d.filename)}
                    className={`relative w-full h-full overflow-hidden text-left rounded-2xl border p-5 pb-12 flex flex-col gap-3 transition-all duration-300 ${
                      isSelected
                        ? "border-[var(--text)] bg-[var(--text)] text-[var(--bg)] shadow-xl"
                        : "border-[var(--text)]/[0.20] bg-[var(--surface)] hover:border-[var(--text)]/[0.3]"
                    }`}
                  >
                    <div className="flex items-start justify-between w-full">
                      <h3 className={`font-bold text-base line-clamp-2 pr-2 ${isSelected ? "text-[var(--bg)]" : "text-[var(--text)]"}`}>
                        {d.filename}
                      </h3>
                    </div>
                    {isSelected && (
                      <div className="absolute top-4 right-4 h-2 w-2 rounded-full bg-[var(--blue)] animate-pulse" />
                    )}
                  </button>
                  <button
                    onClick={() => deleteDataset(d.filename)}
                    disabled={deleting === d.filename}
                    aria-label={`Delete ${d.filename}`}
                    title="Delete dataset"
                    className={`absolute bottom-3 right-3 flex h-8 w-8 items-center justify-center rounded-full transition-colors disabled:opacity-40 hover:bg-red-500/15 hover:text-red-400 ${
                      isSelected ? "text-[var(--bg)]/60" : "text-[var(--muted)]"
                    }`}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                      <path d="M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Action Button */}
      <div className="mt-16 mb-8 w-full max-w-sm">
        <button
          onClick={goToDashboard}
          disabled={!selectedFilename || navigating}
          className={`w-full rounded-2xl px-8 py-5 text-lg font-bold transition-all duration-300 ${
            selectedFilename && !navigating
              ? "bg-[var(--text)] text-[var(--bg)] hover:scale-[1.02] hover:shadow-2xl hover:shadow-[var(--text)]/20"
              : "bg-[var(--text)]/[0.05] text-[var(--muted)] cursor-not-allowed"
          }`}
        >
          {navigating ? "Initializing Pipeline..." : "Enter Workspace →"}
        </button>
      </div>
    </div>
  );
}
