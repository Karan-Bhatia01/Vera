import { useState, useRef, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { clearSession } from "../../api/client";

const navItems = [
  {
    to: "/upload",
    label: "Import Data",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
  {
    to: "/dashboard",
    label: "Dashboard",
    end: true,
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="9" rx="1" />
        <rect x="14" y="3" width="7" height="5" rx="1" />
        <rect x="14" y="12" width="7" height="9" rx="1" />
        <rect x="3" y="16" width="7" height="5" rx="1" />
      </svg>
    ),
  },
  {
    to: "/dashboard/info",
    label: "Data Info",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
  },
  {
    to: "/dashboard/eda",
    label: "Data EDA",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
  {
    to: "/dashboard/ml",
    label: "ML Modelling",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
];

const popupMenuItems = [
  {
    label: "Settings",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
  {
    label: "Terms & Conditions",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    ),
  },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const email = localStorage.getItem("email");
  const [popupOpen, setPopupOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const popupRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (popupRef.current && !popupRef.current.contains(e.target)) {
        setPopupOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(""), 2500);
  };

  const signOut = () => {
    clearSession();
    ["selectedDataset", "pipelineJobs", "targetColumn"].forEach((k) => localStorage.removeItem(k));
    navigate("/login");
  };

  const username = email ? email.split("@")[0] : "User";
  const initial = email ? email[0].toUpperCase() : "U";

  return (
    <>
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 rounded-2xl border border-[var(--blue)]/40 bg-[var(--surface)]/90 backdrop-blur-md px-5 py-3 text-xs font-semibold text-[var(--text)] shadow-2xl animate-in fade-in slide-in-from-bottom-2">
          ⚡ {toastMessage}
        </div>
      )}

      <aside className="sticky top-0 flex h-screen w-[220px] shrink-0 flex-col bg-[var(--sidebar)] border-r border-[var(--line)] overflow-y-auto">
        {/* Logo */}
        <div className="px-4 pt-6 pb-6">
          <span className="flex items-center gap-2 text-sm font-bold tracking-tight">
            <span className="flex h-1.5 w-1.5 rounded-full bg-[var(--blue)]" />
            VERA
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex flex-1 flex-col gap-0.5 px-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-2.5 rounded-lg px-3 py-2 text-[12.5px] font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-[var(--text)] text-[var(--bg)]"
                    : "text-[var(--muted)] hover:bg-[var(--text)]/[0.05] hover:text-[var(--text)]"
                }`
              }
            >
              <span className="flex shrink-0">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* Profile Section */}
        <div className="relative p-3 mt-auto" ref={popupRef}>
          {/* Popup Menu */}
          {popupOpen && (
            <div className="absolute bottom-full left-3 right-3 mb-2 rounded-2xl border border-[var(--text)]/[0.1] bg-[var(--surface)] shadow-2xl shadow-black/40 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200 z-50">
              {/* Popup Header */}
              <div className="flex items-center gap-3 px-4 py-4 border-b border-[var(--text)]/[0.06]">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--blue)] font-bold text-[var(--bg)] text-sm">
                  {initial}
                </div>
                <div className="flex flex-col overflow-hidden">
                  <span className="truncate text-sm font-semibold text-[var(--text)]">
                    {username}
                  </span>
                  <span className="truncate text-xs text-[var(--muted)] font-mono">
                    {email || "user@vera.ai"}
                  </span>
                </div>
              </div>

              {/* Menu Items */}
              <div className="py-1.5">
                {popupMenuItems.map((item) => (
                  <button
                    key={item.label}
                    onClick={() => {
                      setPopupOpen(false);
                      showToast(`${item.label} — coming soon`);
                    }}
                    className="flex w-full items-center gap-3 px-4 py-2.5 text-[13px] font-medium text-[var(--muted)] transition-colors hover:bg-[var(--text)]/[0.05] hover:text-[var(--text)] cursor-pointer"
                  >
                    <span className="flex shrink-0">{item.icon}</span>
                    {item.label}
                  </button>
                ))}
              </div>

              {/* Divider + Log out */}
              <div className="border-t border-[var(--text)]/[0.06] py-1.5">
                <button
                  onClick={signOut}
                  className="flex w-full items-center gap-3 px-4 py-2.5 text-[13px] font-medium text-[var(--muted)] transition-colors hover:bg-red-500/10 hover:text-red-400 cursor-pointer"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  Log out
                </button>
              </div>
            </div>
          )}

          {/* Profile Card (clickable) */}
          <button
            onClick={() => setPopupOpen((prev) => !prev)}
            className="w-full flex items-center gap-3 rounded-xl p-3 transition-all duration-200 hover:bg-[var(--text)]/[0.05] group cursor-pointer"
          >
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[var(--blue)] font-bold text-[var(--bg)] text-sm ring-2 ring-transparent group-hover:ring-[var(--blue)]/30 transition-all">
              {initial}
            </div>
            <div className="flex flex-col overflow-hidden text-left">
              <span className="truncate text-sm font-semibold text-[var(--text)]">
                {username}
              </span>
              <span className="truncate text-[10px] font-mono text-[var(--muted)]">
                {email}
              </span>
            </div>
            {/* Chevron */}
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`ml-auto text-[var(--muted)] transition-transform duration-200 ${popupOpen ? "rotate-180" : ""}`}
            >
              <polyline points="18 15 12 9 6 15" />
            </svg>
          </button>
        </div>
      </aside>
    </>
  );
}
