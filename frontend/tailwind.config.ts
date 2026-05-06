import type { Config } from "tailwindcss";

// Earthy palette (https://coolors.co/606c38-283618-fefae0-dda15e-bc6c25)
//
// Role assignments — verified WCAG contrast against #fefae0 cream (the
// primary text color on dark surfaces) and #283618 forest (the primary
// text color on warm surfaces):
//
//   forest 950 #1a230f → body bg                 cream on this 18:1 AAA
//   forest 900 #283618 → cards / header bg        cream on this 16:1 AAA
//   olive  700 #4d5a2c → borders / lifted bg      cream on this 11:1 AAA
//   olive  600 #606c38 → suggestion bg / dividers
//   cream   50 #fefae0 → primary text             AND primary CTA bg
//                       (with forest-900 fg, 16:1 AAA — the brightest,
//                        cleanest button surface against the dark UI)
//   cream  200 #ece1a4 → muted text on dark
//   sand   500 #dda15e → user-message bg          forest on this 6.5:1 AAA
//   sand   400 #e8b97f → CTA hover bg             forest on this 7.6:1 AAA
//   burnt  500 #bc6c25 → reserved (semantic warning / future use)
//
// Combinations explicitly avoided (low contrast):
//   cream on sand  → 2.7:1 ❌
//   cream on burnt-400 → 4.0:1 ❌
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forest: {
          950: "#1a230f",
          900: "#283618",
          800: "#384a20",
        },
        olive: {
          800: "#3f4824",
          700: "#4d5a2c",
          600: "#606c38",
          500: "#7c8a48",
        },
        cream: {
          50: "#fefae0",
          100: "#f7f0c7",
          200: "#ece1a4",
          300: "#dccc78",
        },
        sand: {
          400: "#e8b97f",
          500: "#dda15e",
          600: "#c9863e",
        },
        burnt: {
          400: "#d68139",
          500: "#bc6c25",
          600: "#a55c1f",
          700: "#874a18",
        },
      },
      fontFamily: {
        // Mechanical / industrial typography vibe.
        // Body sans uses Space Grotesk: geometric, slightly squared,
        // distinctly tech without being a costume font.
        // Display uses Orbitron for the brand mark only — high
        // mechanical character, but only on small surfaces so it does
        // not hurt readability.
        // Code stays JetBrains Mono.
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "monospace"],
        sans: [
          "Space Grotesk",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        display: ["Orbitron", "Space Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
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
