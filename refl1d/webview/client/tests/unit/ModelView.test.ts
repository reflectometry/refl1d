import { flushPromises, type VueWrapper } from "@vue/test-utils";
import * as Plotly from "plotly.js/lib/core";
import { afterEach, expect, test, vi } from "vitest";
import ModelView from "../../src/components/ModelView.vue";
import { emitted } from "./fakeSocket";
import { mountPanel } from "./mountPanel";

vi.mock("plotly.js/lib/core", () => import("./plotlyMock"));

const MODEL_NAMES = [
  { name: "film", part_name: "", model_index: 0, part_index: 0 },
  { name: "film", part_name: "mixture", model_index: 0, part_index: 1 },
  { name: "bilayer", part_name: "", model_index: 1, part_index: 0 },
];

let wrapper: VueWrapper | undefined;
afterEach(() => {
  wrapper?.unmount();
  vi.mocked(Plotly.react).mockClear();
});

async function setup() {
  const mounted = mountPanel(ModelView, {
    get_model_names: MODEL_NAMES,
    get_profile_plots: { data: [{ name: "rho" }], layout: {} },
  });
  wrapper = mounted.wrapper;
  await vi.waitFor(() => expect(Plotly.react).toHaveBeenCalled());
  await flushPromises();
  return mounted;
}

const profileRequests = (socket: Awaited<ReturnType<typeof setup>>["socket"]) =>
  emitted(socket, "get_profile_plots").map(([specs]) => specs);

test("Lists the models and draws the profile of the first", async () => {
  const { socket } = await setup();
  expect(wrapper!.findAll("#model option").map((o) => o.text())).toEqual([
    "0:0 --- film:",
    "0:1 --- film:mixture",
    "1:0 --- bilayer:",
  ]);
  expect(profileRequests(socket)).toEqual([[{ model_index: 0, sample_index: 0 }]]);
  expect(vi.mocked(Plotly.react).mock.lastCall![1]).toEqual([{ name: "rho" }]);
});

test("Choosing another model draws its profile", async () => {
  const { socket } = await setup();
  const select = wrapper!.get<HTMLSelectElement>("#model");
  select.element.selectedIndex = 2; // "1:0 --- bilayer:"
  await select.trigger("change");
  await vi.waitFor(() => expect(profileRequests(socket)).toHaveLength(2));
  expect(profileRequests(socket)[1]).toEqual([{ model_index: 1, sample_index: 0 }]);
});

test("Show Multiple draws several models at once", async () => {
  const { socket } = await setup();
  await wrapper!.get("#multiple").setValue(true);
  const select = wrapper!.get<HTMLSelectElement>("#model");
  expect(select.element.multiple).toBe(true);
  for (const option of select.element.options) {
    option.selected = option.text !== "0:1 --- film:mixture";
  }
  await select.trigger("change");
  await vi.waitFor(() => expect(profileRequests(socket)).toHaveLength(2));
  expect(profileRequests(socket)[1]).toEqual([
    { model_index: 0, sample_index: 0 },
    { model_index: 1, sample_index: 0 },
  ]);
});

test("Fetches the model names again when a new model is loaded", async () => {
  const { socket } = await setup();
  expect(emitted(socket, "get_model_names")).toHaveLength(1);
  socket.serverEmit("model_loaded");
  expect(emitted(socket, "get_model_names")).toHaveLength(2);
});
