import { useEffect, useState } from "react";
import api from "../api/client";

/**
 * "Service online" line for the footer. Polls the health endpoint and only
 * renders while the database reports itself as connected.
 */
export default function ConnectionStatus() {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        const { data } = await api.get("/api/health");
        if (active) setConnected(data?.database === "connected");
      } catch {
        if (active) setConnected(false);
      }
    };

    check();
    const id = setInterval(check, 20000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  if (!connected) return null;

  return (
    <p className="flex items-center gap-2 text-sm text-[var(--muted)]">
      <span className="h-2 w-2 rounded-full" style={{ background: "var(--lime)" }} aria-hidden="true" />
      Service online
    </p>
  );
}
