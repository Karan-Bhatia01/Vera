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
      "token",
      "email",
      "selectedDataset",
      "pipelineJobs",
      "targetColumn",
    ].forEach((k) => localStorage.removeItem(k));
    navigate("/login");
  };

  return (
    <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col border-r border-[var(--line)] surface overflow-y-auto">
      <div className="px-6 py-6">
        <span className="flex items-center gap-2 font-bold tracking-wide">
          <span className="h-2 w-2 rounded-full bg-[var(--blue)]" />
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
                  ? "bg-[var(--blue)] text-[var(--bg)]"
                  : "text-[var(--muted)] hover:bg-white/5 hover:text-[var(--text)]"
              }`
            }
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-[var(--line)] p-3">
        {email && (
          <p
            className="px-3 pb-2 text-xs text-[var(--muted)] eyebrow truncate"
            title={email}
          >
            {email}
          </p>
        )}
        <button
          onClick={signOut}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-[var(--muted)] transition-colors hover:bg-red-500/10 hover:text-red-400"
        >
          <span>🚪</span>
          Sign out
        </button>
      </div>
    </aside>
  );
}
