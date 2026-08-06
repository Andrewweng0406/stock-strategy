import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: [path.resolve(__dirname, "..")],
    },
  },
  // Railway (and most PaaS hosts) proxy requests through a generated
  // hostname — allow it rather than only localhost-style Host headers.
  preview: {
    host: true,
    allowedHosts: true,
  },
});
