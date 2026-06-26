import { NavLink, useNavigate } from "react-router-dom";

const navItems = [
  { to: "/upload", label: "Upload", icon: "⬆️" },
  { to: "/dashboard", label: "Dashboard", icon: "🏠", end: true },
  { to: "/dashboard/info", label: "Data Info", icon: "📋" },
  { to: "/dashboard/eda", label: "Data EDA", icon: "📊" },
  { to: "/dashboard/ml", label: "ML Modelling", icon: "⚙️" },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const email = localStorage.getItem("email");

  const signOut = () => {
    // Clear the session + any dataset/pipeline state tied to this user so the
    // next login starts clean.
    [
      "token", "email", "selectedDataset", "pipelineJobs",
      "targetColumn",
    ].forEach((k) => localStorage.removeItem(k));
    navigate("/login");
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-white/10 bg-[#0d1117]/80 backdrop-blur-sm">
      <div className="px-6 py-6">
        <span className="flex items-center gap-2 font-bold tracking-wide">
          <span className="h-2 w-2 rounded-full bg-[#b56126]" />
          VERA
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-[#b56126] text-[#0d1117]"
                  : "text-[#9a9a93] hover:bg-white/5 hover:text-[#f0ece2]"
              }`
            }
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-white/10 p-3">
        {email && (
          <p className="px-3 pb-2 text-xs text-[#6e6e66] font-['JetBrains_Mono'] truncate" title={email}>
            {email}
          </p>
        )}
        <button
          onClick={signOut}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-[#9a9a93] transition-colors hover:bg-red-500/10 hover:text-red-400"
        >
          <span>🚪</span>
          Sign out
        </button>
      </div>
    </aside>
  );
}