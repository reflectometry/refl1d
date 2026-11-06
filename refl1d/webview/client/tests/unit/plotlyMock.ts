import { vi } from "vitest";

/**
 * Replacement for plotly.js/lib/core, which needs a real browser to draw. Use it with
 *   vi.mock("plotly.js/lib/core", () => import("./plotlyMock"));
 * and check what a panel would have drawn with `vi.mocked(Plotly.react).mock.lastCall`.
 */
export const react = vi.fn(async () => {});
export const purge = vi.fn(async () => {});
export const downloadImage = vi.fn(async () => {});
export const Plots = { resize: vi.fn(async () => {}) };
export const Icons = { "camera-retro": {} };
