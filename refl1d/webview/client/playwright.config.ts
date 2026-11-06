import { existsSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";

/** Port for the test server; avoids the default port (and the bumps e2e port) so a running session isn't hit. */
const PORT = 5180;

/** Use the repo's virtualenv when present (local dev), otherwise whatever `python` is on PATH (CI). */
const PYTHON =
  process.env.REFL1D_PYTHON ?? (existsSync("../../../.venv/bin/python") ? "../../../.venv/bin/python" : "python");

/** Ni film on Si, measured at four angles: small, fast to load, with data and fittable parameters. */
const MODEL = "../../../doc/examples/ex1/nifilm-fit.py";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  // All tests share one server (and so one fit state), so run them one at a time.
  workers: 1,
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1400, height: 900 } } }],
  webServer: {
    // Build first so the server never serves a stale client from dist/.
    command: `bun run build && ${PYTHON} -m refl1d.webview.server.cli --headless --port ${PORT} ${MODEL}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
