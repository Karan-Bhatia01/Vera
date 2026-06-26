// Pipeline state is shared via context (mounted at the dashboard layout) so
// navigating between sidebar tabs doesn't restart polling or re-fetch results.
export { usePipeline, PipelineProvider } from "../context/PipelineContext";
