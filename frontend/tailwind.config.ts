import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Brand accent — a calm Python-yellow / amber hint, less harsh than pure yellow.
        accent: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "monospace"],
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
      },
      keyframes: {
        bounce1: {
          "0%, 80%, 100%": { opacity: "0.3", transform: "translateY(0)" },
          "40%": { opacity: "1", transform: "translateY(-3px)" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "dot-1": "bounce1 1.4s ease-in-out infinite",
        "dot-2": "bounce1 1.4s ease-in-out 0.2s infinite",
        "dot-3": "bounce1 1.4s ease-in-out 0.4s infinite",
        "fade-up": "fadeUp 200ms ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
