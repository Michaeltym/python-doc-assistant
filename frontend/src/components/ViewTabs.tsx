import type { View } from "../types";

interface ViewTabsProps {
  view: View;
  onChange: (v: View) => void;
}

const TABS: { id: View; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "playground", label: "Playground" },
  { id: "compare", label: "Compare" },
];

export function ViewTabs({ view, onChange }: ViewTabsProps) {
  return (
    <div className="inline-flex items-center rounded-full border border-olive-700 bg-forest-950/60 p-0.5">
      {TABS.map((t) => {
        const active = t.id === view;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onChange(t.id)}
            className={[
              "rounded-full px-3 py-1 text-[11px] font-medium tracking-wide transition",
              active
                ? "bg-cream-50 text-forest-900 shadow shadow-cream-50/15"
                : "text-cream-200/70 hover:text-cream-50",
            ].join(" ")}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
