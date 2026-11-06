import type { VueWrapper } from "@vue/test-utils";
import * as Plotly from "plotly.js/lib/core";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import ProfileUncertaintyView from "../../src/components/ProfileUncertaintyView.vue";
import { emitted } from "./fakeSocket";
import { mountPanel } from "./mountPanel";

vi.mock("plotly.js/lib/core", () => import("./plotlyMock"));

const TIMESTAMP = "2026-01-01T00:00:00";

/** What get_profile_uncertainty_plot returns: one model, with 68% and 95% contours of rho. */
const PAYLOAD = {
  fig: { data: [{ x: [0, 1.5], y: [1, 2] }], layout: { title: "contours" } },
  contour_data: {
    M1: { z: [0, 1.5], data: { rho: { "68%": [1, 2], "95%": [0.5, 2.5] } } },
  },
  contours: [68, 95],
};

let wrapper: VueWrapper | undefined;
/** Links the panel clicked to download a file (jsdom can't follow them). */
let downloads: HTMLAnchorElement[];

beforeEach(() => {
  downloads = [];
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push(this);
  });
  vi.spyOn(console, "log").mockImplementation(() => {});
});
afterEach(() => {
  wrapper?.unmount();
  vi.restoreAllMocks();
  vi.mocked(Plotly.react).mockClear();
});

async function setup(payload: unknown = PAYLOAD) {
  const mounted = mountPanel(ProfileUncertaintyView, { get_profile_uncertainty_plot: payload });
  wrapper = mounted.wrapper;
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
  return mounted;
}

test("Requests the plot with the default settings and draws it", async () => {
  const { socket } = await setup();
  // autoAlign, align, nshown, npoints, random, show_residuals, timestamp
  expect(emitted(socket, "get_profile_uncertainty_plot")).toEqual([[true, 0, 50, 200, true, false, TIMESTAMP]]);
  const [, data, layout] = vi.mocked(Plotly.react).mock.lastCall as unknown[];
  expect(data).toEqual(PAYLOAD.fig.data);
  expect(layout).toEqual(PAYLOAD.fig.layout);
});

test("Download CSV saves the contours, one column per model and contour", async () => {
  await setup();
  await wrapper!.get("button.btn-primary").trigger("click");

  expect(downloads).toHaveLength(1);
  const [link] = downloads;
  expect(link!.download).toBe("contours.csv");
  const [prefix, body] = link!.href.split(",", 2);
  expect(prefix).toBe("data:text/csv;charset=utf-8");
  expect(decodeURIComponent(body!).split("\n")).toEqual([
    `"M1 z","M1 rho 68%","M1 rho 95%"`,
    "0.00000,1.00000,0.500000",
    "1.50000,2.00000,2.50000",
  ]);
});

test("Download CSV does nothing before there are contours", async () => {
  await setup({ ...PAYLOAD, contour_data: {} });
  await wrapper!.get("button.btn-primary").trigger("click");
  expect(downloads).toEqual([]);
});

test("Clicking the alignment input turns off auto-align and redraws", async () => {
  const { socket } = await setup();
  const input = wrapper!.get<HTMLInputElement>("#align-interface");
  expect(input.element.disabled).toBe(true);

  // the disabled input is covered by an overlay that catches the click
  input.element.parentElement!.querySelector("span")!.click();

  await vi.waitFor(() => expect(emitted(socket, "get_profile_uncertainty_plot")).toHaveLength(2));
  expect(emitted(socket, "get_profile_uncertainty_plot")[1]![0]).toBe(false);
  expect(wrapper!.get<HTMLInputElement>("#align-interface").element.disabled).toBe(false);
  expect(wrapper!.get<HTMLInputElement>("#auto-align").element.checked).toBe(false);
});
