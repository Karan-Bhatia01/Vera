import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import Background from "../Background";
import { PipelineProvider } from "../../context/PipelineContext";

// Access control is handled by <ProtectedRoute> in App.jsx.
export default function DashboardLayout() {
  return (
    <PipelineProvider>
      <div className="relative min-h-screen bg-[#0d1117] text-[#f0ece2] font-['Space_Grotesk']">
        <Background />
        <div className="relative z-10 flex min-h-screen">
          <Sidebar />
          <main className="flex-1 px-8 py-10 overflow-y-auto">
            <Outlet />
          </main>
        </div>
      </div>
    </PipelineProvider>
  );
}