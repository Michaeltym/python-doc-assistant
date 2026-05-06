import { useEffect, useState } from "react";

type HealthStatus = "checking" | "ok" | "down";

function useHealth(): HealthStatus {
  const [status, setStatus] = useState<HealthStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const r = await fetch("/health", { cache: "no-store" });
        if (cancelled) return;
        setStatus(r.ok ? "ok" : "down");
      } catch {
        if (!cancelled) setStatus("down");
      }
    };
    check();
    const id = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return status;
}

function HealthPill({ status }: { status: HealthStatus }) {
  const config: Record<HealthStatus, { label: string; dot: string; text: string }> = {
    checking: { label: "checking…", dot: "bg-slate-500", text: "text-slate-400" },
    ok: { label: "online", dot: "bg-emerald-500", text: "text-emerald-400" },
    down: { label: "offline", dot: "bg-red-500", text: "text-red-400" },
  };
  const c = config[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/80 px-2.5 py-1 text-[11px] font-medium ${c.text}`}
    >
      <span
        className={`relative flex h-1.5 w-1.5 ${
          status === "checking"
            ? ""
            : "after:absolute after:inset-0 after:animate-ping after:rounded-full after:opacity-50"
        }`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
      </span>
      {c.label}
    </span>
  );
}

export function HeaderBar({ docsVersion }: { docsVersion: string }) {
  const status = useHealth();
  return (
    <header className="sticky top-0 z-10 border-b border-slate-800/60 bg-slate-950/70 px-4 py-3 backdrop-blur">
      <div className="mx-auto flex max-w-3xl items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 shadow-md shadow-amber-500/20">
            <span className="font-mono text-sm font-bold text-slate-900">py</span>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-100">python-doc-assistant</h1>
            <p className="font-mono text-[11px] text-slate-500">
              docs {docsVersion} · grounded · local
            </p>
          </div>
        </div>
        <HealthPill status={status} />
      </div>
    </header>
  );
}
