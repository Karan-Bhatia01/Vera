import { useEffect, useState } from "react";

/** Dark/light theme persisted in localStorage and applied as <html data-theme>. */
export default function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem("app-theme") || "light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("app-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((value) => (value === "dark" ? "light" : "dark"));
  return [theme, toggleTheme];
}
