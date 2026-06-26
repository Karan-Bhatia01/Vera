import { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import api from "../api/client";

/** Decode a JWT payload without a library. Returns null if missing/malformed. */
function decodeToken(token) {
  if (!token) return null;
  try {
    const base64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(escape(atob(base64))));
  } catch {
    return null;
  }
}

/** Quick client-side gate: a token must exist and not be expired. */
function hasUnexpiredToken() {
  const payload = decodeToken(localStorage.getItem("token"));
  if (!payload) return false;
  if (payload.exp && payload.exp * 1000 <= Date.now()) return false;
  return true;
}

/**
 * Gate for pages that require login.
 *
 * Two layers:
 *  1. Instant client check — no/expired token → redirect immediately.
 *  2. Server check — call /api/me so a forged or server-rejected token can't
 *     reach a protected page. While verifying we render nothing (brief).
 *
 * The server check runs once per mount of the guard (i.e. on entering /upload
 * or the dashboard), not on every tab switch.
 */
export default function ProtectedRoute({ children }) {
  const location = useLocation();
  // Resolve the instant client check synchronously: no/expired token → denied
  // up front; otherwise "checking" until the server confirms.
  const [status, setStatus] = useState(() => (hasUnexpiredToken() ? "checking" : "denied"));

  useEffect(() => {
    if (status !== "checking") return;
    let active = true;
    api.get("/api/me")
      .then(() => active && setStatus("ok"))
      .catch(() => {
        // 401/expired/forged → drop creds and bounce to login.
        localStorage.removeItem("token");
        localStorage.removeItem("email");
        if (active) setStatus("denied");
      });
    return () => { active = false; };
  }, [status]);

  if (status === "denied") {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (status === "checking") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0d1117]">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-[#c97539]" />
      </div>
    );
  }
  return children;
}
