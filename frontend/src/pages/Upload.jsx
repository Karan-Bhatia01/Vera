import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client";
import { usePipeline } from "../context/PipelineContext";

const MAX_DATASETS = 3;

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const { addJob, selectDataset: selectPipelineDataset } = usePipeline();
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

  useEffect(() => {
    if (!localStorage.getItem("token")) {
      navigate("/login");
      return;
    }
    fetchDatasets();
  }, []);

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
    <div className="mx-auto max-w-5xl px-2 py-8">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-extrabold text-[var(--text)]">
          Upload Your Dataset
        </h1>
        <p className="mt-2 text-[var(--muted)] font-['JetBrains_Mono']">
          Transform raw data into model-ready insights
        </p>
      </header>

      <div className="grid gap-6 sm:grid-cols-2">
        <section className="surface flex flex-col">
          <h2 className="text-lg font-bold mb-1">Upload CSV File</h2>
          <p className="text-sm text-[var(--muted)] mb-6 font-['JetBrains_Mono']">
            Drag & drop or click to select
          </p>
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
            className={`flex-1 min-h-[200px] flex flex-col items-center justify-center rounded-[var(--radius-md)] border-2 border-dashed px-6 py-10 text-center transition-colors ${
              atLimit
                ? "border-[var(--line)] opacity-50 cursor-not-allowed"
                : dragActive
                  ? "border-[var(--blue)] bg-[var(--blue-dark)] cursor-pointer"
                  : "border-[var(--line)] cursor-pointer hover:border-gray-500"
            }`}
          >
            <div className="text-4xl mb-3">📁</div>
            <p className="font-semibold">
              {atLimit ? "Dataset limit reached" : "Click or drag CSV here"}
            </p>
            <p className="mt-1 text-sm text-[var(--muted)] font-['JetBrains_Mono']">
              {atLimit
                ? "Delete a dataset first"
                : "Max 50,000 rows · 20 columns · 50MB"}
            </p>
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
          {uploading && (
            <p className="mt-4 text-center font-['JetBrains_Mono'] text-xs text-[var(--muted)]">
              Uploading…
            </p>
          )}
          {error && (
            <p className="mt-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">
              ✗ {error}
            </p>
          )}
          {success && (
            <p className="mt-4 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-emerald-400">
              ✓ {success}
            </p>
          )}
        </section>

        <section className="surface">
          <h2 className="text-lg font-bold mb-4">What Vera Does</h2>
          <div className="space-y-5">
            {[
              {
                icon: "🔍",
                title: "Smart Data Analysis",
                body: "AI identifies patterns, flags quality issues automatically.",
              },
              {
                icon: "⚙️",
                title: "Auto ML Pipeline",
                body: "Preprocessing, feature engineering, and training — all automated.",
              },
              {
                icon: "💬",
                title: "AI Assistant",
                body: "Ask questions about your data in plain language.",
              },
            ].map((item) => (
              <div key={item.title}>
                <div className="text-xl mb-1">{item.icon}</div>
                <h3 className="font-semibold text-sm">{item.title}</h3>
                <p className="mt-1 text-sm text-[var(--muted)] font-['JetBrains_Mono']">
                  {item.body}
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="mt-8 surface">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-lg font-bold">Your Datasets</h2>
          <span className="rounded-md bg-[var(--blue-dark)] px-3 py-1 text-xs font-semibold text-[var(--blue)]">
            {datasets.length} / {MAX_DATASETS}
          </span>
        </div>
        <p className="text-sm text-[var(--muted)] mb-1 font-['JetBrains_Mono']">
          Click a dataset to select it
        </p>
        <p className="text-xs text-[var(--muted)] opacity-70 font-['JetBrains_Mono'] mb-6">
          CSV files only · max {MAX_DATASETS}
        </p>

        {loadingList ? (
          <p className="text-sm text-[var(--muted)] font-['JetBrains_Mono']">
            Loading…
          </p>
        ) : datasets.length === 0 ? (
          <p className="text-sm text-[var(--muted)] font-['JetBrains_Mono']">
            No datasets uploaded yet.
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {datasets.map((d) => {
              const isSelected = d.filename === selectedFilename;
              return (
                <button
                  key={d.filename}
                  onClick={() => selectDataset(d.filename)}
                  className={`text-left rounded-[var(--radius-md)] border p-4 flex flex-col gap-3 transition-colors ${
                    isSelected
                      ? "border-[var(--blue)] bg-[var(--surface-2)]"
                      : "border-[var(--line)] bg-[var(--surface)] hover:border-gray-500"
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <h3 className="font-semibold break-words text-sm truncate pr-2">
                      📄 {d.filename}
                    </h3>
                    {isSelected && (
                      <span className="shrink-0 rounded-full bg-[var(--blue)] px-2 py-0.5 text-[10px] font-bold text-[var(--bg)]">
                        SELECTED
                      </span>
                    )}
                  </div>
                  {d.shape && (
                    <p className="text-xs text-[var(--muted)] font-['JetBrains_Mono']">
                      {d.shape[0]} rows · {d.shape[1]} cols
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </section>

      <div className="mt-8 flex flex-col items-center gap-2">
        {!selectedFilename && (
          <p className="text-xs text-[var(--muted)] font-['JetBrains_Mono']">
            Select a dataset above to continue
          </p>
        )}
        <button
          onClick={goToDashboard}
          disabled={!selectedFilename || navigating}
          className={`rounded-full px-8 py-3.5 font-semibold transition-opacity ${
            selectedFilename && !navigating
              ? "bg-[var(--blue)] text-[var(--bg)] hover:opacity-90"
              : "bg-[var(--line)] text-[var(--muted)] cursor-not-allowed"
          }`}
        >
          {navigating ? "Starting analysis…" : "Go to Dashboard →"}
        </button>
      </div>
    </div>
  );
}
