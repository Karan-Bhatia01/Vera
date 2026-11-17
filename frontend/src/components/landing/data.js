export const navLinks = [
  { label: "Home", href: "/#" },
  { label: "Engine", href: "/#pipeline" },
  { label: "Features", href: "/#features" },
  { label: "FAQ", href: "/#faq" },
];

export const stats = [
  { value: "3", label: "STAGES" },
  { value: "3–5", label: "MODELS" },
  { value: "10", label: "AI CHARTS" },
  { value: "25K", label: "ROW LIMIT" },
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
    body: "Missing values are filled and categories encoded automatically, with the AI choosing a strategy for each column.",
  },
  {
    n: "03",
    title: "Train & Compare",
    body: "An AI-picked shortlist of 3–5 models, each trained and scored on a held-out test split.",
  },
  {
    n: "04",
    title: "Explain",
    body: "Feature importance for the best model, plus a plain-English AI summary of what the results mean.",
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
    body: "End-to-end preprocessing and model training. From feature engineering to model selection, Vera handles it all.",
  },
  {
    icon: "📈",
    title: "Visual Insights",
    body: "Automatic visualizations with AI-generated explanations for every chart, pattern, and anomaly.",
  },
  {
    icon: "🎯",
    title: "Model Comparison",
    body: "Compare models head-to-head on accuracy, F1, R² and feature importance.",
  },
];

export const faqSubtitle = "Questions About an AutoML Tool. Ask Away.";

export const faqs = [
  {
    q: "Is it really free?",
    a: "Yes! Unless you count the hours you'll spend staring at confusion matrices.",
  },
  {
    q: "Do I need to know ML?",
    a: "Nope. Vera does the hard part. You just pick a target column and look smart.",
  },
  {
    q: "What if my model gets 100% accuracy?",
    a: "Congrats, or more likely, data leakage. Go check your columns. We'll wait.",
  },
  {
    q: "What if I find a bug?",
    a: "Tell us. Bugs are just surprise features we didn't ask for.",
  },
    {
    q: "Can Vera help me impress my crush?",
    a: "Absolutely! Just casually mention your model's F1 score and watch the sparks fly. Or not. No guarantees.",
  },
  {
    q: "Can Vera predict my exam results?",
    a: "Upload your attendance and study hours, and it'll try. The answer may hurt. Don't blame the model.",
  },
  {
    q: "Does Vera get things wrong?",
    a: "Sometimes. It's a model, not a wizard. Garbage data in, confident garbage out.",
  },
  {
    q: "Can I train it on my Netflix history?",
    a: "You can. It will conclude you need sleep. Very accurate.",
  },
  {
    q: "What if all the models perform badly?",
    a: "Then the data is the problem, not you. Add better features and try again.",
  },
  {
    q: "Can I show Vera's results to my boss?",
    a: "Yes. Just say you built it. We won't tell.",
  },
];