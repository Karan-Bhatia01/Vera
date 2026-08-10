import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/client";
import { usePipeline } from "../../context/PipelineContext";

const MAX_DATASETS = 3;

export default function DashboardHome() {
  const navigate = useNavigate();
  const { polling, filename: pipelineFile, addJob, selectDataset: selectPipelineDataset } = usePipeline();
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFilename, setSelectedFilename] = useState(
    () => localStorage.getItem("selectedDataset") || null
  );
  const [columns, setColumns] = useState([]);
  const [columnsLoading, setColumnsLoading] = useState(false);
  const [targetColumn, setTargetColumn] = useState("");
  const [columnsToDrop, setColumnsToDrop] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => { fetchDatasets(); }, []);

  useEffect(() => {
    if (!selectedFilename) { setColumns([]); return; }
    setSubmitted(false); setSubmitError(""); setTargetColumn(""); setColumnsToDrop([]);
    fetchColumns(selectedFilename);
  }, [selectedFilename]);

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const res = await api.get("/api/stored_datasets");
      setDatasets(res.data.datasets || []);
    } catch {} finally { setLoading(false); }
  };

  const fetchColumns = async (fname) => {
    setColumnsLoading(true);
    try {
      const res = await api.get(`/api/info/${encodeURIComponent(fname)}`);
      setColumns(res.data.data?.columns || []);
    } catch { setColumns([]); }
    finally { setColumnsLoading(false); }
  };

  const selectDataset = (fname) => {
    setSelectedFilename(fname);
    selectPipelineDataset(fname); // updates localStorage + shared pipeline filename
  };

  const toggleDrop = (col) =>
    setColumnsToDrop(p => p.includes(col) ? p.filter(c => c !== col) : [...p, col]);

  const submitEDA = async () => {
    if (!selectedFilename || !targetColumn) return;
    setSubmitting(true); setSubmitError("");
    try {
      const res = await api.post("/api/run_eda", {
        filename: selectedFilename,
        target_column: targetColumn,
        columns_to_drop: columnsToDrop,
      });
      // Store target_column for ML page to reuse
      localStorage.setItem("targetColumn", targetColumn);
      addJob("eda_job_id", res.data.eda_job_id);
      setSubmitted(true);
    } catch (err) {
      setSubmitError(err.response?.data?.message || "Could not start EDA");
    } finally { setSubmitting(false); }
  };

  return (
    <div>
      <h1 className="text-3xl font-extrabold">Dashboard</h1>
      <p className="mt-2 text-[#9a9a93]">Select a dataset and configure your analysis.</p>

      {polling && pipelineFile === selectedFilename && (
        <div className="mt-4 flex items-center gap-2 rounded-xl border border-[#c97539]/30 bg-[#c9753911] px-4 py-2.5">
          <Spinner /><p className="text-sm text-[#c97539] font-['JetBrains_Mono']">Pipeline running…</p>
        </div>
      )}

      {/* Dataset history */}
      <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-lg font-bold">Dataset History</h2>
          <span className="rounded-md bg-[#b56126] px-3 py-1 text-xs font-semibold text-[#0d1117]">
            {datasets.length} / {MAX_DATASETS}
          </span>
        </div>
        <p className="text-xs text-[#6e6e66] font-['JetBrains_Mono'] mb-4">CSV files only · max {MAX_DATASETS}</p>
        {loading ? <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">Loading…</p>
        : datasets.length === 0 ? (
          <div className="text-center py-6">
            <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">No datasets yet.</p>
            <button onClick={() => navigate("/upload")}
              className="mt-4 rounded-full bg-[#b56126] px-6 py-2.5 text-sm font-semibold text-[#0d1117]">
              Upload a dataset
            </button>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {datasets.map(d => (
              <button key={d.filename} onClick={() => selectDataset(d.filename)}
                className={`text-left rounded-xl border p-4 flex flex-col gap-2 transition-colors ${
                  d.filename === selectedFilename
                    ? "border-[#c97539] bg-[#c9753911]"
                    : "border-white/10 bg-[#0d1117] hover:border-white/25"
                }`}>
                <h3 className="font-semibold text-sm break-words">📄 {d.filename}</h3>
                {d.shape && <p className="text-xs text-[#9a9a93] font-['JetBrains_Mono']">{d.shape[0]} rows · {d.shape[1]} cols</p>}
                {d.stored_at && <p className="text-xs text-[#6e6e66] font-['JetBrains_Mono']">{d.stored_at}</p>}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* EDA config form */}
      {selectedFilename && (
        <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <h2 className="text-lg font-bold mb-1">Configure EDA Pipeline</h2>
          <p className="text-sm text-[#9a9a93] mb-6">
            For <span className="font-semibold text-[#f0ece2]">{selectedFilename}</span>.
            Select target column, then submit to start EDA.
          </p>

          {columnsLoading ? (
            <p className="text-sm text-[#9a9a93] font-['JetBrains_Mono']">Loading columns…</p>
          ) : submitted ? (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-4">
              <p className="text-sm text-emerald-400">
                ✓ EDA started. Check <strong>Data EDA</strong> in the sidebar for results.
              </p>
            </div>
          ) : (
            <>
              <div className="mb-5">
                <label className="block text-sm font-semibold mb-2">Target Column</label>
                <select value={targetColumn} onChange={e => setTargetColumn(e.target.value)}
                  className="w-full rounded-md border border-white/20 bg-[#0d1117] px-4 py-2.5 text-sm outline-none focus:border-[#c97539]">
                  <option value="">— Select target column —</option>
                  {columns.map(col => <option key={col} value={col}>{col}</option>)}
                </select>
              </div>

              {columns.length > 0 && (
                <div className="mb-5">
                  <label className="block text-sm font-semibold mb-2">
                    Columns to Drop <span className="text-[#6e6e66] font-normal">(optional)</span>
                  </label>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {columns.filter(c => c !== targetColumn).map(col => (
                      <label key={col} className="flex items-center gap-2 text-sm text-[#9a9a93] cursor-pointer">
                        <input type="checkbox" checked={columnsToDrop.includes(col)}
                          onChange={() => toggleDrop(col)} className="accent-[#b56126]" />
                        {col}
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {submitError && (
                <p className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">
                  ✗ {submitError}
                </p>
              )}

              <button onClick={submitEDA} disabled={!targetColumn || submitting}
                className={`rounded-full px-7 py-3 font-semibold ${
                  targetColumn && !submitting
                    ? "bg-[#b56126] text-[#0d1117]"
                    : "bg-white/10 text-[#6e6e66] cursor-not-allowed"
                }`}>
                {submitting ? "Starting…" : "Run EDA →"}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const Spinner = () => <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[#c97539]/30 border-t-[#c97539]" />;