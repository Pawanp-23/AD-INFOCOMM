// Applies the saved or system colour theme before first paint (no flash of the wrong theme).
// A separate file rather than an inline script, so the Content-Security-Policy can forbid
// inline scripts.
try {
  const t = localStorage.getItem("ramei-theme");
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  if (dark) document.documentElement.classList.add("dark");
} catch {}
