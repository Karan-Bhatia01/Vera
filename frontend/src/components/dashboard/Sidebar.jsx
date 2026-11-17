import { useState, useRef, useEffect } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { clearSession } from "../../api/client";
import { BrandMark } from "../brand/Brand";
import { STAGES, STAGE_ORDER } from "../brand/stages";
import ThemeToggle from "../ThemeToggle";

// Stage pages carry their mascot's colour; the dashboard shows all four.
const navItems = [
  { to: "/upload", label: "Upload data", stage: "upload" },
  { to: "/dashboard", label: "Dashboard", end: true },
  { to: "/dashboard/info", label: "Data info", stage: "info" },
  { to: "/dashboard/eda", label: "Data EDA", stage: "eda" },
  { to: "/dashboard/ml", label: "ML modelling", stage: "ml" },
];

const menuRow = "flex w-full px-4 py-3 text-left text-[15px] text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]";

function NavMarker({ stage }) {
  if (stage) {
    return <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: STAGES[stage].color }} aria-hidden="true" />;
  }
  return (
    <span className="grid shrink-0 grid-cols-2 gap-[2px]" aria-hidden="true">
      {STAGE_ORDER.map((key) => (
        <span key={key} className="h-[4px] w-[4px] rounded-full" style={{ background: STAGES[key].color }} />
      ))}
    </span>
  );
}

function NavList({ onNavigate, large = false }) {
  return navItems.map((item) => (
    <NavLink
      key={item.to}
      to={item.to}
      end={item.end}
      onClick={onNavigate}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-xl px-3 font-semibold transition-colors ${large ? "py-3.5 text-lg" : "py-2.5 text-[15px]"} ${
          isActive
            ? "bg-[var(--surface)] text-[var(--text)] shadow-[0_0_0_1px_var(--line)]"
            : "text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
        }`
      }
    >
      <NavMarker stage={item.stage} />
      {item.label}
    </NavLink>
  ));
}

function Avatar({ initial }) {
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full font-bold text-[#1e2452]" style={{ background: "var(--sun)" }}>
      {initial}
    </span>
  );
}

export default function Sidebar() {
  const navigate = useNavigate();
  const email = localStorage.getItem("email");
  const [popupOpen, setPopupOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const popupRef = useRef(null);
  const mobilePanelRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (popupRef.current && !popupRef.current.contains(e.target)) setPopupOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Mobile menu: Esc closes it, the page behind doesn't scroll, focus moves into the panel.
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e) => e.key === "Escape" && setMobileOpen(false);
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    mobilePanelRef.current?.querySelector("a, button")?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [mobileOpen]);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(""), 2500);
  };

  const openSettings = () => {
    setPopupOpen(false);
    setMobileOpen(false);
    showToast("Settings aren't available yet.");
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
        <div role="status" className="fixed bottom-6 left-4 right-4 z-[60] rounded-xl border border-[var(--line)] bg-[var(--surface)] px-5 py-3 text-[15px] font-semibold shadow-lg sm:left-auto sm:right-6">
          {toastMessage}
        </div>
      )}

      {/* ── Phones: top bar + full-screen menu ── */}
      <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-[var(--line)] bg-[var(--sidebar)] px-4 md:hidden">
        <BrandMark to="/dashboard" />
        <button
          onClick={() => setMobileOpen((v) => !v)}
          aria-expanded={mobileOpen}
          aria-controls="mobile-menu"
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
          className="-mr-2 flex h-11 w-11 items-center justify-center rounded-full text-[var(--text)] hover:bg-[var(--surface-2)]"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
            {mobileOpen ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </header>

      {mobileOpen && (
        <div
          id="mobile-menu"
          ref={mobilePanelRef}
          className="fixed inset-x-0 top-14 bottom-0 z-40 flex flex-col overflow-y-auto bg-[var(--bg)] px-4 pt-4 pb-8 md:hidden"
        >
          <nav className="flex flex-col gap-1" aria-label="Workspace">
            <NavList large onNavigate={() => setMobileOpen(false)} />
          </nav>

          <div className="mt-6 flex items-center justify-between rounded-xl border border-[var(--line)] bg-[var(--surface)] px-4 py-3">
            <span className="font-semibold">Theme</span>
            <ThemeToggle />
          </div>

          <div className="mt-6 overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface)]">
            <div className="flex items-center gap-3 border-b border-[var(--line)] px-4 py-3">
              <Avatar initial={initial} />
              <span className="min-w-0">
                <span className="block truncate font-semibold">{username}</span>
                <span className="block truncate text-sm text-[var(--muted)]">{email}</span>
              </span>
            </div>
            <button onClick={openSettings} className={menuRow}>Settings</button>
            <Link to="/terms-of-service" className={menuRow}>Terms of service</Link>
            <button onClick={signOut} className={`${menuRow} border-t border-[var(--line)] font-semibold !text-[var(--danger)]`}>
              Log out
            </button>
          </div>
        </div>
      )}

      {/* ── Tablets and up: sidebar ── */}
      <aside className="sticky top-0 hidden h-screen w-[232px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--sidebar)] md:flex">
        <div className="px-5 pt-7 pb-8">
          <BrandMark to="/dashboard" />
        </div>

        <nav className="flex flex-1 flex-col gap-1 px-3" aria-label="Workspace">
          <NavList />
        </nav>

        <div className="relative mt-auto p-3" ref={popupRef}>
          {popupOpen && (
            <div className="absolute bottom-full left-3 right-3 z-50 mb-2 overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] shadow-xl">
              <div className="border-b border-[var(--line)] px-4 py-3">
                <p className="truncate font-semibold">{username}</p>
                <p className="truncate text-sm text-[var(--muted)]">{email || "Signed in"}</p>
              </div>
              <div className="py-1.5">
                <button onClick={openSettings} className={menuRow}>Settings</button>
                <Link to="/terms-of-service" className={menuRow}>Terms of service</Link>
              </div>
              <div className="border-t border-[var(--line)] py-1.5">
                <button onClick={signOut} className={`${menuRow} font-semibold !text-[var(--danger)]`}>Log out</button>
              </div>
            </div>
          )}

          <button
            onClick={() => setPopupOpen((prev) => !prev)}
            aria-expanded={popupOpen}
            aria-label="Account menu"
            className="flex w-full items-center gap-3 rounded-xl p-2.5 transition-colors hover:bg-[var(--surface-2)]"
          >
            <Avatar initial={initial} />
            <span className="flex min-w-0 flex-col text-left">
              <span className="truncate text-[15px] font-semibold">{username}</span>
              <span className="truncate text-xs text-[var(--muted)]">{email}</span>
            </span>
          </button>
        </div>
      </aside>
    </>
  );
}
