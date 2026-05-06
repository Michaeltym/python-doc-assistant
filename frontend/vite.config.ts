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
