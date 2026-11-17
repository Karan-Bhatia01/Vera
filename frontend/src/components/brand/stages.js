// The four pipeline stages. Each one is owned by a plush mascot and its colour.
export const STAGES = {
  upload: {
    label: "Upload",
    mascot: "/mascots/lime.png",
    alt: "Lime plush monster carrying a backpack",
    color: "var(--lime)",
    ink: "var(--lime-ink)",
  },
  info: {
    label: "Profile",
    mascot: "/mascots/sun.png",
    alt: "Yellow plush scholar holding a book",
    color: "var(--sun)",
    ink: "var(--sun-ink)",
  },
  eda: {
    label: "Explore",
    mascot: "/mascots/violet.png",
    alt: "Purple plush DJ wearing headphones",
    color: "var(--violet)",
    ink: "var(--violet-ink)",
  },
  ml: {
    label: "Model",
    mascot: "/mascots/sky.png",
    alt: "Blue plush in a beret holding a coffee",
    color: "var(--sky)",
    ink: "var(--sky-ink)",
  },
};

export const STAGE_ORDER = ["upload", "info", "eda", "ml"];
