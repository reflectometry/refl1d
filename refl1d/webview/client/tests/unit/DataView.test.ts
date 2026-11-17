import type { VueWrapper } from "@vue/test-utils";
import * as Plotly from "plotly.js/lib/core";
import { afterEach, expect, test, vi } from "vitest";
import DataPanel from "../../src/components/DataView.vue";
import { emitted } from "./fakeSocket";
import { mountPanel } from "./mountPanel";

vi.mock("plotly.js/lib/core", () => import("./plotlyMock"));

/** One cross-section, as sent by get_plot_data. */
const xs = (polarization: string, theory: number[], R: number[], extra: object = {}) => ({
  label: "film",
  polarization,
  Q: [0.01, 0.02, 0.03],
  dQ: [0.001, 0.001, 0.001],
  theory,
  fresnel: [2, 2, 2],
  intensity: 1,
  background: 0,
  R,
  dR: [0.1, 0.1, 0.1],
  ...extra,
});

const UNPOLARIZED = [[xs("", [0.5, 0.25, 0.125], [0.6, 0.2, 0.1])]];
const POLARIZED = [[xs("++", [0.6, 0.6, 0.6], [0.8, 0.8, 0.8]), xs("--", [0.2, 0.2, 0.2], [0.2, 0.2, 0.2])]];

type Trace = { name: string; x: number[]; y: number[]; yaxis?: string; error_x?: object; error_y?: object };
type Layout = Record<string, any>;

let wrapper: VueWrapper | undefined;
afterEach(() => {
  wrapper?.unmount();
  vi.mocked(Plotly.react).mockClear();
  vi.mocked(Plotly.purge).mockClear();
});

async function setup(plotdata: unknown = UNPOLARIZED) {
  const mounted = mountPanel(DataPanel, { get_plot_data: { plotdata, chisq: "1.23(4)" } });
  wrapper = mounted.wrapper;
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
  return mounted;
}

/** Traces (by name) and layout of the most recent draw. */
function lastPlot() {
  const [, traces, layout] = vi.mocked(Plotly.react).mock.lastCall as unknown as [unknown, Trace[], Layout];
  return { trace: (name: string) => traces.find((t) => t.name === name)!, names: traces.map((t) => t.name), layout };
}

async function check(id: string, checked: boolean) {
  vi.mocked(Plotly.react).mockClear();
  await wrapper!.get(`#${id}`).setValue(checked);
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
}

async function selectPlot(type: string) {
  vi.mocked(Plotly.react).mockClear();
  await wrapper!.get("#plot_mode").setValue(type);
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
}

test("Draws theory and data, with error bars, on a log scale with chisq shown", async () => {
  const { socket } = await setup();
  expect(emitted(socket, "get_plot_data")).toEqual([["linear"]]);
  const { names, trace, layout } = lastPlot();
  expect(names).toEqual(["film  theory", "film  data"]);
  expect(trace("film  theory").y).toEqual([0.5, 0.25, 0.125]);
  expect(trace("film  data")).toMatchObject({
    y: [0.6, 0.2, 0.1],
    error_x: { array: [0.001, 0.001, 0.001] },
    error_y: { array: [0.1, 0.1, 0.1] },
  });
  expect(layout.yaxis.type).toBe("log");
  expect(layout.annotations[0].text).toContain("1.23(4)");
});

test("Clears the plot when the server has no data", async () => {
  mountPanel(DataPanel, { get_plot_data: { plotdata: null } });
  await vi.waitFor(() => expect(Plotly.purge).toHaveBeenCalled());
  expect(Plotly.react).not.toHaveBeenCalled();
});

test("R_corr applies background and intensity to the data rather than the theory", async () => {
  await setup([[xs("", [1.1, 0.3, 0.1], [1.1, 0.3, 0.1], { intensity: 2, background: 0.1 })]]);
  // (theory - background) / intensity
  const corrected = lastPlot().trace("film  theory").y;
  [0.5, 0.1, 0].forEach((y, i) => {
    expect(corrected[i]).toBeCloseTo(y);
  });

  await check("apply_corrections", false);
  expect(lastPlot().trace("film  theory").y).toEqual([1.1, 0.3, 0.1]);
});

test("Turning off dQ removes the horizontal error bars", async () => {
  await setup();
  await check("show_resolution", false);
  expect(lastPlot().trace("film  data").error_x).toBeUndefined();
});

test("Residuals are drawn on a second axis below the plot", async () => {
  await setup();
  await check("show_residuals", true);
  const { trace, layout } = lastPlot();
  // (R - theory) / dR
  const residuals = trace("film  residuals");
  expect(residuals.yaxis).toBe("y2");
  [1, -0.5, -0.25].forEach((y, i) => {
    expect(residuals.y[i]).toBeCloseTo(y);
  });
  expect(layout.yaxis2.domain).toEqual([0, 0.25]);
});

test("Fresnel divides theory and data by the substrate reflectivity", async () => {
  await setup();
  await selectPlot("Fresnel (R/R_substrate)");
  const { trace, layout } = lastPlot();
  expect(trace("film  theory").y).toEqual([0.25, 0.125, 0.0625]);
  expect(trace("film  data").y).toEqual([0.3, 0.1, 0.05]);
  expect(layout.yaxis.title.text).toBe("Fresnel (R/R_substrate)");
});

test("Spin Asymmetry combines ++ and -- on a fixed linear axis", async () => {
  await setup(POLARIZED);
  await selectPlot("Spin Asymmetry");
  const { names, trace, layout } = lastPlot();
  expect(names).toEqual(["film theory", "film data"]);
  // (pp - mm) / (pp + mm)
  expect(trace("film theory").y.map((y) => y.toFixed(6))).toEqual(Array(3).fill("0.500000"));
  expect(trace("film data").y.map((y) => y.toFixed(6))).toEqual(Array(3).fill("0.600000"));
  expect(layout.yaxis).toMatchObject({ type: "linear", range: [-1.5, 1.5], autorange: false });
  expect(wrapper!.get<HTMLInputElement>("#log_y").element.checked).toBe(false);
});

const spinGrids = [
  { name: "aligned", Q: [0.01, 0.02, 0.03], theory: [0.15, 0.25, 0.35] },
  { name: "shifted", Q: [0.005, 0.025, 0.045], theory: [0.1, 0.3, 0.5] },
  { name: "shorter", Q: [0.005, 0.045], theory: [0.1, 0.5] },
  { name: "longer", Q: [0.005, 0.015, 0.025, 0.035, 0.045], theory: [0.1, 0.2, 0.3, 0.4, 0.5] },
];

test.each(spinGrids.flatMap((grid) => [true, false].map((corrections) => ({ ...grid, corrections }))))(
  "Spin Asymmetry aligns the $name -- grid (corrections=$corrections)",
  async ({ Q, theory, corrections }) => {
    const pp = xs("++", [1.7, 1.7, 1.7], [1.7, 1.7, 1.7], { intensity: 3, background: 0.2 });
    const measured = theory.map((r) => 2 * r + 0.1);
    const mm = xs("--", measured, measured, {
      Q,
      dQ: Q.map(() => 0.001),
      dR: Q.map(() => 0.1),
      fresnel: Q.map(() => 2),
      intensity: 2,
      background: 0.1,
    });
    await setup([[pp, mm]]);
    await selectPlot("Spin Asymmetry");
    await check("show_residuals", true);
    if (!corrections) await check("apply_corrections", false);

    // Every grid samples the same linear -- signal: [0.15, 0.25, 0.35]
    // at the ++ Q values, or [0.4, 0.6, 0.8] before beam corrections.
    const expected = corrections ? [7 / 13, 1 / 3, 3 / 17] : [13 / 21, 11 / 23, 9 / 25];
    const { trace } = lastPlot();
    for (const kind of ["theory", "data"]) {
      const plotted = trace(`film ${kind}`);
      expect(plotted.x).toEqual(pp.Q);
      expect(plotted.y).toHaveLength(expected.length);
      expected.forEach((value, i) => {
        expect(plotted.y[i]).toBeCloseTo(value, 10);
      });
    }
    // Measured data equal the theory on each channel's own grid.
    trace("film residuals").y.forEach((value) => {
      expect(value).toBeCloseTo(0, 10);
    });
  }
);

test("Plot offset separates the cross-sections", async () => {
  await setup(POLARIZED);
  vi.mocked(Plotly.react).mockClear();
  await wrapper!.get("#plot_offset_control").setValue("1");
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
  const { trace } = lastPlot();
  // on a log scale, the second cross-section is multiplied by 10^offset
  expect(trace("film ++ theory").y).toEqual([0.6, 0.6, 0.6]);
  trace("film -- theory").y.forEach((y) => {
    expect(y).toBeCloseTo(2);
  });
});

test.each([true, false])("Spin Asymmetry residuals ignore plot offsets (corrections=%s)", async (corrections) => {
  const first = POLARIZED[0].map((section) => ({ ...section, label: "first" }));
  const second = POLARIZED[0].map((section) => ({ ...section, label: "second", intensity: 2, background: 0.1 }));
  await setup([first, second]);
  await selectPlot("Spin Asymmetry");
  await check("show_residuals", true);
  if (!corrections) await check("apply_corrections", false);

  const original = lastPlot();
  const residuals = [...original.trace("second residuals").y];
  // Corrected SA: data=3/4, theory=2/3, variance=0.048828125.
  // Uncorrected SA: data=3/5, theory=1/2, variance=0.0272.
  const expected = corrections ? (0.75 - 2 / 3) / Math.sqrt(0.048828125) : 0.1 / Math.sqrt(0.0272);
  residuals.forEach((value) => {
    expect(value).toBeCloseTo(expected);
  });

  for (const offset of [0.7, 0.3, 0]) {
    vi.mocked(Plotly.react).mockClear();
    await wrapper!.get("#plot_offset_control").setValue(String(offset));
    await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
    const shifted = lastPlot();
    // The offset still moves the displayed curves, with error bars unchanged.
    for (const kind of ["theory", "data"]) {
      shifted.trace(`second ${kind}`).y.forEach((value, i) => {
        expect(value).toBeCloseTo(original.trace(`second ${kind}`).y[i] + offset);
      });
      expect(shifted.trace(`first ${kind}`).y).toEqual(original.trace(`first ${kind}`).y);
    }
    expect(shifted.trace("second data").error_y).toEqual(original.trace("second data").error_y);
    expect(shifted.trace("first residuals").y).toEqual(original.trace("first residuals").y);
    expect(shifted.trace("second residuals").y).toEqual(residuals);
  }
});
