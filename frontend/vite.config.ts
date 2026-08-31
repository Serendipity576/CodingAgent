import { defineConfig } from "vite";

export default defineConfig({
  // FastAPI exposes the production bundle from this fixed same-origin path.
  base: "/static/",
  // Vite's esbuild transform supports React's automatic JSX runtime directly.
  esbuild: { jsx: "automatic" },
  build: {
    // The Python package serves this directory and includes it in distributions.
    outDir: "../src/agent/web/static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      // Local development keeps the browser on one origin while FastAPI owns
      // the conversation REST endpoints and the SSE event stream.
      "/api": "http://127.0.0.1:8765",
    },
  },
});
