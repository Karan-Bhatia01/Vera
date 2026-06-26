import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Background from "../components/Background";
import api from "../api/client";

const MAX_DATASETS = 3;

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [navigating, setNavigating] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedFilename, setSelectedFilename] = useState(
    () => localStorage.getItem("selectedDataset") || null
  );

  useEffect(() => {
    if (!localStorage.getItem("token")) { navigate("/login"); return; }
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    setLoadingList(true);
    try {
      const res = await api.get("/api/stored_datasets");
      setDatasets(res.data.datasets || []);
    } catch (err) {
      setError(err.response?.data?.message || "Could not load datasets");
    } finally { setLoadingList(false); }
  };

  const selectDataset = (filename) => {
    setSelectedFilename(filename);
    localStorage.setItem("selectedDataset", filename);
  };

  const uploadFile = async (file) => {
    setError(""); setSuccess("");
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only CSV files are supported"); return; }
    if (datasets.length >= MAX_DATASETS) { setError(`Dataset limit reached (${MAX_DATASETS} max).`); return; }
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const res = await api.post("/api/upload", formData, { headers: { "Content-Type": "multipart/form-data" } });
      setSuccess(`"${res.data.data.filename}" uploaded successfully.`);
      await fetchDatasets();
    } catch (err) {
      setError(err.response?.data?.message || "Upload failed");
    } finally { setUploading(false); }
  };

  const goToDashboard = async () => {
    if (!selectedFilename) return;
    setNavigating(true);
    setError("");
    try {
      // Trigger Data Info analysis job before navigating
      const res = await api.post("/api/run_info", { filename: selectedFilename });
      const existingJobs = JSON.parse(localStorage.getItem("pipelineJobs") || "{}");
      localStorage.setItem("pipelineJobs", JSON.stringify({
        ...existingJobs,
        filename: selectedFilename,
        info_job_id: res.data.info_job_id,
      }));
      navigate("/dashboard");
    } catch (err) {
      setError(err.response?.data?.message || "Could not start analysis");
      setNavigating(false);
    }
  };

  const atLimit = datasets.length >= MAX_DATASETS;

  return (
    <div className="relative min-h-screen bg-[#0d1117] text-[#f0ece2] font-['Space_Grotesk']">
      <Background />
      <div className="relative z-10 mx-auto max-w-5xl px-6 py-16">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-extrabold">Upload Your Dataset</h1>
          <p className="mt-2 text-[#9a9a93]">Transform raw data into model-ready insights</p>
        </header>

        <div className="grid gap-6 sm:grid-cols-2">
          <section className="rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm p-6 flex flex-col">
            <h2 className="text-lg font-bold mb-1">Upload CSV File</h2>
            <p className="text-sm text-[#9a9a93] mb-6">Drag & drop or click to select</p>
            <div
              onClick={() => !atLimit && fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); if (!atLimit) setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={atLimit ? undefined : (e) => { e.preventDefault(); setDragActive(false); const f = e.dataTransfer.files?.[0]; if (f) uploadFile(f); }}
              className={`flex-1 min-h-[200px] flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
                atLimit ? "border-white/10 opacity-50 cursor-not-allowed"
                : dragActive ? "border-[#c97539] bg-[#c9753911] cursor-pointer"
                : "border-white/20 cursor-pointer hover:border-white/30"
              }`}
            >
              <div className="text-4xl mb-3">📁</div>
              <p className="font-semibold">{atLimit ? "Dataset limit reached" : "Click or drag CSV here"}</p>
              <p className="mt-1 text-sm text-[#9a9a93]">{atLimit ? "Delete a dataset first" : "Max 50,000 rows · 20 columns · 50MB"}</p>
              <input ref={fileInputRef} type="file" accept=".csv" className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = ""; }}
                disabled={atLimit} />
            </div>
            {uploading && <p className="mt-4 text-center font-['JetBrains_Mono'] text-xs text-[#9a9a93]">Uploading…</p>}
            {error && <p className="mt-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">✗ {error}</p>}
            {success && <p className="mt-4 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-emerald-400">✓ {success}</p>}
          </section>

          <section className="rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm p-6">
            <h2 className="text-lg font-bold mb-4">What Vera Does</h2>
            <div className="space-y-5">
              {[
                { icon: "🔍", title: "Smart Data Analysis", body: "AI identifies patterns, flags quality issues automatically." },
                { icon: "⚙️", title: "Auto ML Pipeline", body: "Preprocessing, feature engineering, and training — all automated." },
                { icon: "💬", title: "AI Assistant", body: "Ask questions about your data in plain language." },
              ].map(item => (
                <div key={item.title}>
                  <div className="text-xl mb-1">{item.icon}</div>
                  <h3 className="font-semibold text-sm">{item.title}</h3>
                  <p className="mt-1 text-sm text-[#9a9a93]">{item.body}</p>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm p-6">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-lg font-bold">Your Datasets</h2>
            <span className="rounded-md bg-[#b56126] px-3 py-1 text-xs font-semibold text-[#0d1117]">
              {datasets.length} / {MAX_DATASETS}
            </span>
          </div>
          <p className="text-sm text-[#9a9a93] mb-1">Click a dataset to select it</p>
          <p className="text-xs text-[#6e6e66] font-['JetBrains_Mono'] mb-6">CSV files only · max {MAX_DATASETS}</p>

          {loadingList ? (
            <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">Loading…</p>
          ) : datasets.length === 0 ? (
            <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">No datasets uploaded yet.</p>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {datasets.map(d => {
                const isSelected = d.filename === selectedFilename;
                return (
                  <button key={d.filename} onClick={() => selectDataset(d.filename)}
                    className={`text-left rounded-xl border p-4 flex flex-col gap-3 transition-colors ${
                      isSelected ? "border-[#c97539] bg-[#c9753911]" : "border-white/10 bg-[#0d1117] hover:border-white/25"
                    }`}>
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold break-words text-sm">📄 {d.filename}</h3>
                      {isSelected && <span className="shrink-0 rounded-full bg-[#b56126] px-2 py-0.5 text-[10px] font-bold text-[#0d1117]">SELECTED</span>}
                    </div>
                    {d.shape && <p className="text-xs text-[#9a9a93] font-['JetBrains_Mono']">{d.shape[0]} rows · {d.shape[1]} cols</p>}
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <div className="mt-8 flex flex-col items-center gap-2">
          {!selectedFilename && (
            <p className="text-xs text-[#9a9a93] font-['JetBrains_Mono']">Select a dataset above to continue</p>
          )}
          <button onClick={goToDashboard} disabled={!selectedFilename || navigating}
            className={`rounded-full px-8 py-3.5 font-semibold transition-opacity ${
              selectedFilename && !navigating ? "bg-[#b56126] text-[#0d1117]" : "bg-white/10 text-[#6e6e66] cursor-not-allowed"
            }`}>
            {navigating ? "Starting analysis…" : "Go to Dashboard →"}
          </button>
        </div>
      </div>
    </div>
  );
}