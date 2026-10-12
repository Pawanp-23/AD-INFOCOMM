/** Design tokens live in src/styles/index.css as CSS variables (light + dark). */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        canvas: token("canvas"),
        surface: token("surface"),
        subtle: token("subtle"),
        border: token("border"),
        fg: token("fg"),
        muted: token("muted"),
        faint: token("faint"),
        accent: token("accent"),
        "accent-fg": token("accent-fg"),
        danger: token("danger"),
        warning: token("warning"),
        success: token("success"),
        info: token("info"),
      },
      borderRadius: { DEFAULT: "6px", md: "6px", lg: "8px", xl: "10px" },
      fontSize: { "2xs": ["0.6875rem", "1rem"] },
      keyframes: {
        pulseDot: { "0%,100%": { opacity: "1" }, "50%": { opacity: ".35" } },
      },
      animation: { "pulse-dot": "pulseDot 1.4s ease-in-out infinite" },
    },
  },
  plugins: [],
};
