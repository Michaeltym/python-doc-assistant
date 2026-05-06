import { useEffect, useState } from "react";
import type { ModelInfo } from "../types";

const STORAGE_KEY = "pdr.selectedModel";

interface UseModelsResult {
  models: ModelInfo[];
  defaultModel: string | null;
  selectedModel: string | null;
  setSelectedModel: (id: string) => void;
  loading: boolean;
}

/**
 * Fetch /api/models on mount, surface the list + the active selection.
 *
 * The selection persists to localStorage so refreshing the tab keeps the
 * user's pick. Falls back to the server's default when no preference is
 * stored or the stored id is no longer registered.
 */
export function useModels(): UseModelsResult {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [selectedModel, setSelectedModelState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/models")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data) => {
        if (cancelled) return;
        const ms: ModelInfo[] = data.models;
        const def: string = data.default;
        const stored = localStorage.getItem(STORAGE_KEY);
        const initial = stored && ms.some((m) => m.id === stored) ? stored : def;
        setModels(ms);
        setDefaultModel(def);
        setSelectedModelState(initial);
      })
      .catch(() => {
        if (cancelled) return;
        setModels([]);
        setDefaultModel(null);
        setSelectedModelState(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const setSelectedModel = (id: string) => {
    setSelectedModelState(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* localStorage may be disabled (private mode); silently ignore. */
    }
  };

  return { models, defaultModel, selectedModel, setSelectedModel, loading };
}
