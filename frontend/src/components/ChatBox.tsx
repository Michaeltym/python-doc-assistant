import {
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type Ref,
} from "react";

interface ChatBoxProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
  onCancel?: () => void;
  inputRef?: Ref<ChatBoxHandle>;
}

export interface ChatBoxHandle {
  setValue: (v: string) => void;
  focus: () => void;
}

function SendIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-4 w-4"
    >
      <path d="M2.5 2.5l15 7.5-15 7.5 3-7.5-3-7.5z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-4 w-4"
    >
      <rect x="5" y="5" width="10" height="10" rx="1.5" />
    </svg>
  );
}

export function ChatBox({ onSubmit, disabled = false, onCancel, inputRef }: ChatBoxProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(
    inputRef,
    () => ({
      setValue: (v: string) => {
        setValue(v);
        textareaRef.current?.focus();
      },
      focus: () => textareaRef.current?.focus(),
    }),
    [],
  );

  // Autosize textarea (capped at ~6 lines).
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [value]);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    submit();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form onSubmit={handleSubmit} className="px-4 pb-4">
      <div
        className={[
          "flex items-end gap-2 rounded-2xl border bg-slate-900/80 p-2 shadow-lg shadow-black/30 backdrop-blur transition",
          disabled
            ? "border-slate-800"
            : "border-slate-800 focus-within:border-amber-500/60 focus-within:shadow-amber-500/5",
        ].join(" ")}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Ask about pathlib, asyncio, json…"
          className="flex-1 resize-none bg-transparent px-3 py-2 text-[15px] text-slate-100 placeholder-slate-500 focus:outline-none"
        />
        {disabled && onCancel ? (
          <button
            type="button"
            onClick={onCancel}
            aria-label="Stop"
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-800 text-slate-300 transition hover:bg-red-700 hover:text-white"
          >
            <StopIcon />
          </button>
        ) : (
          <button
            type="submit"
            disabled={disabled || !value.trim()}
            aria-label="Send"
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-amber-500 text-slate-900 shadow-md shadow-amber-500/30 transition hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none"
          >
            <SendIcon />
          </button>
        )}
      </div>
      <p className="mt-2 px-1 text-[11px] text-slate-500">
        <kbd className="rounded border border-slate-800 bg-slate-900 px-1.5 py-0.5 font-mono text-[10px]">
          Enter
        </kbd>{" "}
        to send ·{" "}
        <kbd className="rounded border border-slate-800 bg-slate-900 px-1.5 py-0.5 font-mono text-[10px]">
          Shift + Enter
        </kbd>{" "}
        for newline
      </p>
    </form>
  );
}
