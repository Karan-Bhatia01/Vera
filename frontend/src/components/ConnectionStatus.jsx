import { useEffect, useState } from "react";
import api from "../api/client";

/**
 * A small glassy "Connected" badge fixed to the top-right corner.
 * It polls the backend health endpoint and only renders while the
 * database reports itself as connected — otherwise it shows nothing.
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
    <div className="fixed top-4 right-4 z-50 flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-1.5 text-sm font-medium text-[#f0ece2] shadow-[0_8px_30px_rgba(0,0,0,0.35)] backdrop-blur-md">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
      </span>
      Connected
    </div>
  );
}
