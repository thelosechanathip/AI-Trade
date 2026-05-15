import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          deep:  "#050a14",
          card:  "#0d1424",
          raise: "#111d2e",
          line:  "#1e2d40",
        },
        brand: {
          green:  "#00e896",
          red:    "#ff4d6a",
          blue:   "#4da6ff",
          gold:   "#f5a623",
          yellow: "#ffd166",
          purple: "#a78bfa",
        },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "blink":      "blink 1.2s step-end infinite",
        "slide-up":   "slideUp 0.2s ease-out",
        "flash-green": "flashGreen 0.6s ease-out",
        "flash-red":   "flashRed 0.6s ease-out",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0" },
        },
        slideUp: {
          from: { transform: "translateY(4px)", opacity: "0" },
          to:   { transform: "translateY(0)",   opacity: "1" },
        },
        flashGreen: {
          "0%":   { backgroundColor: "rgba(0,232,150,0.25)" },
          "100%": { backgroundColor: "transparent" },
        },
        flashRed: {
          "0%":   { backgroundColor: "rgba(255,77,106,0.25)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
