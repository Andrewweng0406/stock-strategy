import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Railway (and most PaaS hosts) proxy requests through a generated
  // hostname — allow it rather than only localhost-style Host headers.
  preview: {
    host: true,
    allowedHosts: true,
  },
});
