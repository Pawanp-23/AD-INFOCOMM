import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies the API (HTTP + WebSocket) so the browser talks to one origin (TRD §9).
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": "/src" } },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, ws: true },
    },
  },
});
