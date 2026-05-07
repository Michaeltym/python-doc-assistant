import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// During `npm run dev` Vite serves at :5173 and proxies /api to the
// FastAPI backend at :8000. In production the FastAPI app is expected
// to serve `frontend/dist/` itself, so the proxy is dev-only.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // SSE: force the upstream to send raw bytes (no gzip / br)
        // so the proxy does not buffer chunks waiting for the next
        // compression boundary. Without this, /api/ask streams the
        // whole token list at once when the frontend goes through
        // `npm run dev`.
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("Accept-Encoding", "identity");
          });
        },
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
