import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";
import svgLoader from "vite-svg-loader";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), svgLoader()],
  base: "",
  resolve: {
    // When bumps-webview-client is linked from a bumps checkout (bun/npm link), its imports
    // would otherwise resolve to that checkout's own node_modules, bundling a second copy.
    // bumps patches asyncEmit onto socket.io's Socket class, and shares reactive state
    // through vue, so both sides must use the same copy of these.
    dedupe: ["vue", "socket.io-client", "plotly.js"],
  },
  define: {
    // By default, Vite doesn't include shims for NodeJS.
    // Plotly fails to load without this shim.
    global: {},
  },
  optimizeDeps: {
    include: ["plotly.js/lib/core", "plotly.js/lib/heatmap", "plotly.js/lib/bar", "json-difference"],
  },
});
