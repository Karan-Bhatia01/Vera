export const navLinks = ["Home", "Engine", "Features"];

export const stats = [
  { value: "9", label: "STAGES" },
  { value: "10+", label: "MODELS" },
  { value: "12+", label: "CHARTS" },
  { value: "50MB", label: "FILE LIMIT" },
];

export const pipelineSteps = [
  {
    n: "01",
    title: "Ingest",
    body: "Drop a CSV. Vera profiles every column, infers types, and flags quality issues before anything else runs.",
  },
  {
    n: "02",
    title: "Clean & Transform",
    body: "Missing values, outliers, and encoding handled automatically — with every decision logged and reversible.",
  },
  {
    n: "03",
    title: "Train in Parallel",
    body: "Ten-plus candidate models trained and scored simultaneously, no manual loop required.",
  },
  {
    n: "04",
    title: "Explain",
    body: "SHAP-backed feature importance and a RAG assistant that answers questions about your own data.",
  },
  {
    n: "05",
    title: "Export",
    body: "Walk away with a .pkl model, a full Jupyter notebook, or both.",
  },
];

export const features = [
  {
    icon: "📊",
    title: "Smart Analysis",
    body: "AI-powered data understanding. Vera analyzes your dataset, identifies patterns, and flags quality issues automatically.",
  },
  {
    icon: "⚙️",
    title: "Auto ML",
    body: "End-to-end preprocessing and model training. From feature engineering to tuning, Vera handles it all.",
  },
  {
    icon: "✨",
    title: "RAG Assistant",
    body: "Ask questions about your data in plain language and get answers grounded in the dataset itself.",
  },
  {
    icon: "📈",
    title: "Visual Insights",
    body: "Automatic visualizations with AI-generated explanations for every chart, pattern, and anomaly.",
  },
  {
    icon: "🎯",
    title: "Model Comparison",
    body: "Compare models head-to-head with confusion matrices, feature importance, and performance scores.",
  },
  {
    icon: "💾",
    title: "Export & Deploy",
    body: "Download trained models as .pkl files or export the full analysis as a Jupyter notebook.",
  },
];