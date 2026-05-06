import { useEffect, useRef, useState } from "react";
import type { ModelInfo } from "../types";

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedId: string | null;
  onChange: (id: string) => void;
}

function ChevronIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-3 w-3"
    >
      <path
        fillRule="evenodd"
        d="M5.23 7.21a.75.75 0 011.06.02L10 11.06l3.71-3.83a.75.75 0 111.08 1.04l-4.25 4.4a.75.75 0 01-1.08 0l-4.25-4.4a.75.75 0 01.02-1.06z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/** Compact dropdown rendered to the right of the status pill. */
export function ModelSelector({ models, selectedId, onChange }: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (models.length === 0) return null;
  const current = models.find((m) => m.id === selectedId) ?? models[0];

  // With a single registered model there is nothing to switch between —
  // render a static chip instead of an interactive dropdown.
  if (models.length === 1) {
    return (
      <span className="rounded-full border border-olive-700 bg-forest-900/80 px-2.5 py-1 font-mono text-[11px] text-cream-200/70">
        {current.label}
      </span>
    );
  }

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1.5 rounded-full border border-olive-700 bg-forest-900/80 px-2.5 py-1 font-mono text-[11px] text-cream-100 transition hover:border-sand-500 hover:text-sand-400"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>{current.label}</span>
        <ChevronIcon />
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute right-0 z-20 mt-1 w-72 overflow-hidden rounded-lg border border-olive-700 bg-forest-950/95 shadow-xl shadow-black/40 backdrop-blur"
        >
          {models.map((m) => {
            const isSelected = m.id === selectedId;
            return (
              <button
                key={m.id}
                type="button"
                role="option"
                aria-selected={isSelected}
                onClick={() => {
                  onChange(m.id);
                  setOpen(false);
                }}
                className={[
                  "flex w-full flex-col items-start gap-0.5 border-b border-olive-700/50 px-3 py-2 text-left transition last:border-b-0",
                  isSelected ? "bg-forest-900 text-sand-400" : "text-cream-100 hover:bg-forest-900",
                ].join(" ")}
              >
                <span className="text-[13px] font-medium">{m.label}</span>
                <span className="font-mono text-[10.5px] text-cream-200/60">{m.id}</span>
                <span className="text-[11px] text-cream-200/60">{m.description}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
