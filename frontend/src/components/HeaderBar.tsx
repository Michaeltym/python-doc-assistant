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
  // Semantic dot colors stay green / red so users read the signal
  // instantly. Pill bg + text use the palette (cream on forest).
  const config: Record<HealthStatus, { label: string; dot: string; text: string }> = {
    checking: { label: "checking…", dot: "bg-cream-300", text: "text-cream-200" },
    ok: { label: "online", dot: "bg-emerald-400", text: "text-emerald-300" },
    down: { label: "offline", dot: "bg-red-400", text: "text-red-300" },
  };
  const c = config[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border border-olive-700 bg-forest-900/80 px-2.5 py-1 text-[11px] font-medium ${c.text}`}
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
    <header className="sticky top-0 z-10 border-b border-olive-700/60 bg-forest-950/70 px-4 py-3 backdrop-blur">
      <div className="mx-auto flex max-w-3xl items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cream-50 shadow-md shadow-cream-50/10">
            <span className="font-mono text-sm font-bold text-forest-900">py</span>
          </div>
          <div>
            <h1 className="font-display text-sm font-bold tracking-wider text-cream-50 uppercase">
              python_doc_assistant
            </h1>
            <p className="font-mono text-[11px] tracking-wide text-cream-200/70">
              docs {docsVersion} · grounded · local
            </p>
          </div>
        </div>
        <HealthPill status={status} />
      </div>
    </header>
  );
}
