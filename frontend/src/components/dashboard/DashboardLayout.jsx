import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import { useState, useEffect } from "react";
import Background from "../Background";
import { PipelineProvider } from "../../context/PipelineContext";

// Access control is handled by <ProtectedRoute> in App.jsx.
export default function DashboardLayout() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("app-theme") || "dark",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("app-theme", theme);
  }, [theme]);

  return (
    <PipelineProvider>
      <div className="relative min-h-screen">
        <Background />
        <div className="relative z-10 flex min-h-screen">
          <Sidebar />
          <main className="flex-1 px-8 py-10 overflow-y-auto relative">
            <button
              onClick={() =>
                setTheme((value) => (value === "dark" ? "light" : "dark"))
              }
              className="absolute top-8 right-8 flex h-9 w-9 items-center justify-center rounded-full bg-[var(--surface-2)] border border-[var(--line)] text-[var(--text)] transition-colors hover:bg-[var(--line)] shadow-sm"
              aria-label="Toggle Theme"
            >
              {theme === "dark" ? "☀" : "☾"}
            </button>
            <Outlet />
          </main>
        </div>
      </div>
    </PipelineProvider>
  );
}
