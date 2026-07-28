import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are emitted straight into the Python package, so `uv run
// ffmpeg-mcp-ui` serves the UI with no Node toolchain present.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/ffmpeg_mcp/ui/static",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8756",
      "/ws": { target: "ws://127.0.0.1:8756", ws: true },
    },
  },
});
