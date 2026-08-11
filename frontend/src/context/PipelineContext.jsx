import {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  useCallback,
} from "react";
import api from "../api/client";

const DONE = ["completed", "failed"];
const PipelineContext = createContext(null);

function readJobs() {
  try {
    return JSON.parse(localStorage.getItem("pipelineJobs") || "null");
  } catch {
    return null;
  }
}

// Server no longer knows this job (restart / TTL expiry). Drop the stale id so
// we stop hammering 404s; consumers fall back to Mongo via the `gone` flag.
function dropStaleJob(idKey) {
  const j = readJobs();
  if (!j || !(idKey in j)) return;
  delete j[idKey];
  localStorage.setItem("pipelineJobs", JSON.stringify(j));
}

/**
 * Mounted once at the dashboard layout so every page shares ONE polling loop
 * and ONE result cache. Navigating between sidebar tabs no longer remounts the
 * hook, so it never restarts polling or re-fetches what it already has.
 */
export function PipelineProvider({ children }) {
  const [filename, setFilename] = useState(
    () => localStorage.getItem("selectedDataset") || null,
  );
  const [info, setInfo] = useState(null);
  const [eda, setEda] = useState(null);
  const [ml, setMl] = useState(null);
  const [polling, setPolling] = useState(false);
  const [gone, setGone] = useState({ info: false, eda: false, ml: false });

  // Mongo-insights cache, keyed by filename, so revisiting Data Info is free.
  const [storedInsights, setStoredInsights] = useState({});
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState("");
  const loadedRef = useRef({});

  const intervalRef = useRef(null);
  const cancelledRef = useRef(false);
  const doneRef = useRef({ info: true, eda: true, ml: true });

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  }, []);

  const setterFor = (key) =>
    key === "info" ? setInfo : key === "eda" ? setEda : setMl;

  useEffect(() => {
    cancelledRef.current = false;
    doneRef.current = { info: true, eda: true, ml: true };

    const jobs = readJobs();
    if (!jobs || jobs.filename !== filename) return;

    const steps = [
      { key: "info", idKey: "info_job_id" },
      { key: "eda", idKey: "eda_job_id" },
      { key: "ml", idKey: "ml_job_id" },
    ].filter((s) => jobs[s.idKey]);

    if (steps.length === 0) return;
    steps.forEach((s) => {
      doneRef.current[s.key] = false;
    });

    const pollOne = async (s) => {
      const jobId = readJobs()?.[s.idKey];
      if (!jobId) {
        doneRef.current[s.key] = true;
        return;
      }
      try {
        const r = await api.get(`/api/pipeline_status/${s.key}/${jobId}`);
        setterFor(s.key)(r.data);
        if (DONE.includes(r.data.status)) doneRef.current[s.key] = true;
      } catch (e) {
        if (e.response?.status === 404) {
          doneRef.current[s.key] = true;
          dropStaleJob(s.idKey);
          setGone((g) => (g[s.key] ? g : { ...g, [s.key]: true }));
        }
      }
    };

    const allDone = () => steps.every((s) => doneRef.current[s.key]);

    const tick = async () => {
      await Promise.all(
        steps.filter((s) => !doneRef.current[s.key]).map(pollOne),
      );
      if (allDone()) stop();
    };

    (async () => {
      await tick();
      if (cancelledRef.current || allDone()) return;
      if (intervalRef.current) return;
      setPolling(true);
      let attempts = 0;
      intervalRef.current = setInterval(async () => {
        if (++attempts > 120 || !readJobs()) {
          stop();
          return;
        }
        await tick();
      }, 3000);
    })();

    return () => {
      cancelledRef.current = true;
      stop();
    };
  }, [filename, stop]);

  // key is the full storage key, e.g. "eda_job_id" / "ml_job_id".
  const addJob = useCallback(
    (key, jobId) => {
      const shortKey = key.replace("_job_id", "");
      const j = readJobs() || {};
      j.filename = filename;
      j[key] = jobId;
      localStorage.setItem("pipelineJobs", JSON.stringify(j));

      doneRef.current[shortKey] = false;
      setGone((g) => (g[shortKey] ? { ...g, [shortKey]: false } : g));
      stop();
      setPolling(true);
      let attempts = 0;
      intervalRef.current = setInterval(async () => {
        if (++attempts > 120) {
          stop();
          return;
        }
        try {
          const r = await api.get(`/api/pipeline_status/${shortKey}/${jobId}`);
          setterFor(shortKey)(r.data);
          if (DONE.includes(r.data.status)) {
            doneRef.current[shortKey] = true;
            stop();
          }
        } catch (e) {
          if (e.response?.status === 404) {
            dropStaleJob(key);
            setGone((g) => ({ ...g, [shortKey]: true }));
            stop();
          }
        }
      }, 3000);
    },
    [filename, stop],
  );

  const clearJobs = useCallback(() => {
    localStorage.removeItem("pipelineJobs");
    setInfo(null);
    setEda(null);
    setMl(null);
    setGone({ info: false, eda: false, ml: false });
    stop();
  }, [stop]);

  const selectDataset = useCallback((name) => {
    localStorage.setItem("selectedDataset", name);
    setFilename(name);
  }, []);

  // Fetch stored insights once per filename; cached for the session.
  const loadInsights = useCallback(async (fname) => {
    if (!fname || loadedRef.current[fname]) return;
    loadedRef.current[fname] = true;
    setInsightsLoading(true);
    setInsightsError("");
    try {
      const res = await api.get(`/api/insights/${encodeURIComponent(fname)}`);
      setStoredInsights((prev) => ({
        ...prev,
        [fname]: res.data.insights || null,
      }));
    } catch (e) {
      loadedRef.current[fname] = false; // allow retry after an error
      setInsightsError(e.response?.data?.message || "Could not load insights");
    } finally {
      setInsightsLoading(false);
    }
  }, []);

  const value = {
    info,
    eda,
    ml,
    polling,
    gone,
    filename,
    addJob,
    clearJobs,
    selectDataset,
    storedInsights,
    insightsLoading,
    insightsError,
    loadInsights,
  };

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  );
}

export function usePipeline() {
  const ctx = useContext(PipelineContext);
  if (!ctx)
    throw new Error("usePipeline must be used within <PipelineProvider>");
  return ctx;
}
