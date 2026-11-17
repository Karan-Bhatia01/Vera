import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import ThemeToggle from "../ThemeToggle";
import { PipelineProvider } from "../../context/PipelineContext";

// Access control is handled by <ProtectedRoute> in App.jsx.
export default function DashboardLayout() {
  return (
    <PipelineProvider>
      <div className="flex min-h-screen flex-col md:flex-row">
        <Sidebar />
        <main className="relative min-w-0 flex-1 px-4 py-7 sm:px-8 md:px-10 md:py-10">
          {/* On phones the theme switch lives in the menu */}
          <div className="absolute top-8 right-10 hidden md:block">
            <ThemeToggle />
          </div>
          <div className="mx-auto max-w-6xl">
            <Outlet />
          </div>
        </main>
      </div>
    </PipelineProvider>
  );
}
