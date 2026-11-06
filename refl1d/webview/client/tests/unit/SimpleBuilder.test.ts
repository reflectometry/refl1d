import { flushPromises, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import SimpleBuilder from "../../src/components/SimpleBuilder.vue";
import { emitted, type FakeSocket } from "./fakeSocket";
import { mountPanel } from "./mountPanel";

const parameter = (id: string, name: string, value: number) => ({
  id,
  name,
  fixed: true,
  slot: { value, __class__: "bumps.parameter.Variable" },
  limits: ["-inf", "inf"],
  bounds: null,
  tags: [],
  __class__: "bumps.parameter.Parameter",
});

const reference = (id: string) => ({ id, __class__: "Reference" });

/** A slab whose parameters are references into the model's `references`, as get_model sends them. */
const slab = (name: string) => ({
  name,
  material: {
    name,
    rho: reference(`${name}-rho`),
    irho: reference(`${name}-irho`),
    __class__: "refl1d.sample.material.SLD",
  },
  thickness: reference(`${name}-thickness`),
  interface: reference(`${name}-interface`),
  magnetism: null,
  __class__: "refl1d.sample.layers.Slab",
});

/** Si | Ni (100 Å, rho 9.4) | Vacuum, serialized as by the server's get_model. */
function serializedModel() {
  const layers = [
    ["Si", 2.07, 0, 0, 3],
    ["Ni", 9.4, 0, 100, 5],
    ["Vacuum", 0, 0, 0, 0],
  ] as const;
  const references: Record<string, ReturnType<typeof parameter>> = {};
  for (const [name, rho, irho, thickness, interface_] of layers) {
    const values = { rho, irho, thickness, interface: interface_ };
    for (const [key, value] of Object.entries(values)) {
      references[`${name}-${key}`] = parameter(`${name}-${key}`, key, value);
    }
  }
  return JSON.stringify({
    references,
    object: {
      __class__: "bumps.fitproblem.FitProblem",
      models: [
        {
          __class__: "refl1d.experiment.Experiment",
          sample: { __class__: "refl1d.sample.layers.Stack", layers: layers.map(([name]) => slab(name)) },
          probe: {},
        },
      ],
    },
    $schema: "bumps-draft-02",
  });
}

let wrapper: VueWrapper | undefined;

beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "debug").mockImplementation(() => {});
});
afterEach(() => {
  wrapper?.unmount();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function setup(model = serializedModel()) {
  const mounted = mountPanel(SimpleBuilder, { get_model: model });
  wrapper = mounted.wrapper;
  await flushPromises();
  return mounted;
}

const rows = () => wrapper!.findAll("tbody tr");
const button = (text: string) => wrapper!.findAll("button").find((b) => b.text() === text)!;

/** The model sent by the most recent set_serialized_problem, parsed, with the other arguments. */
function lastSent(socket: FakeSocket) {
  const [json, isNew, name] = emitted(socket, "set_serialized_problem").at(-1)!;
  const model = JSON.parse(json as string);
  const resolve = (p: { id: string; __class__: string }) => (p.__class__ === "Reference" ? model.references[p.id] : p);
  const layers = model.object.models[0].sample.layers.map((layer: ReturnType<typeof slab>) => ({
    name: layer.name,
    thickness: resolve(layer.thickness),
    rho: resolve(layer.material.rho),
  }));
  return { model, layers, isNew, name };
}

test("Asks for data when no model is loaded", async () => {
  await setup("");
  expect(wrapper!.text()).toContain("Load data to start building a model");
  expect(wrapper!.find("table").exists()).toBe(false);
});

test("Lists the layers of the loaded model", async () => {
  await setup();
  expect(rows().map((row) => row.get<HTMLTextAreaElement>("textarea").element.value)).toEqual(["Si", "Ni", "Vacuum"]);
  expect(rows()[1]!.get<HTMLInputElement>("#layer-thickness").element.value).toBe("100");
  expect(rows()[1]!.get<HTMLInputElement>("#layer-rho").element.value).toBe("9.4");
});

test("New Model sends a Si/Vacuum model and shows it once the server reloads", async () => {
  const { socket, responses } = await setup("");
  await button("New Model").trigger("click");
  const sent = lastSent(socket);
  expect(sent.isNew).toBe(true);
  expect(sent.name).toBe("Simple Builder Model");
  expect(sent.layers.map((l: { name: string }) => l.name)).toEqual(["Si", "Vacuum"]);
  expect(sent.layers[0].rho).toMatchObject({ name: "Si rho", slot: { value: 2.07 } });

  const probe = sent.model.object.models[0].probe;
  expect(probe.Q.values).toHaveLength(250);
  expect(probe.Q.values[0]).toBe(0);
  expect(Math.max(...probe.Q.values)).toBeLessThanOrEqual(0.1);

  // the server announces the new model, and the builder fetches it to show the layers
  responses.get_model = JSON.stringify(sent.model);
  socket.serverEmit("model_loaded");
  await flushPromises();
  expect(rows()).toHaveLength(2);
});

test.each([
  [false, 0],
  [true, 1],
])("New Model replaces a loaded model only if confirmed (%s)", async (confirmed, sends) => {
  vi.stubGlobal(
    "confirm",
    vi.fn(() => confirmed)
  );
  const { socket } = await setup();
  await button("New Model").trigger("click");
  expect(confirm).toHaveBeenCalledOnce();
  expect(emitted(socket, "set_serialized_problem")).toHaveLength(sends);
});

test("Editing a layer sends the model, naming parameters after the layer and setting fit ranges", async () => {
  const { socket } = await setup();
  const ni = rows()[1]!;
  await ni.get("textarea").setValue("Nickel");
  await ni.get("#layer-thickness").setValue("150");
  await ni.get("#layer-thickness").trigger("blur");

  const nickel = lastSent(socket).layers[1];
  expect(nickel.name).toBe("Nickel");
  expect(nickel.thickness).toMatchObject({ name: "Nickel thickness", slot: { value: 150 }, bounds: [75, 225] });
  expect(nickel.rho.name).toBe("Nickel rho");
  expect(nickel.rho.bounds[0]).toBeCloseTo(4.7);
  expect(nickel.rho.bounds[1]).toBeCloseTo(14.1);
  // zero-valued parameters get a small range around zero
  expect(lastSent(socket).layers[2].rho.bounds).toEqual([-0.1, 0.1]);
});

test("+ inserts a new layer below its row", async () => {
  const { socket } = await setup();
  await rows()[0]!.get("button.add-layer-after").trigger("click");
  const { layers } = lastSent(socket);
  expect(layers.map((l: { name: string }) => l.name)).toEqual(["Si", "sld", "Ni", "Vacuum"]);
  expect(layers[1].thickness).toMatchObject({ name: "sld thickness", slot: { value: 25 } });
});

test("Delete removes the layer", async () => {
  const { socket } = await setup();
  await rows()[1]!.get("button.btn-danger").trigger("click");
  expect(lastSent(socket).layers.map((l: { name: string }) => l.name)).toEqual(["Si", "Vacuum"]);
});

test("Apply new Q sends a probe over the new Q range", async () => {
  const { socket } = await setup();
  await button("Edit Q-range").trigger("click");
  await wrapper!.get("#qmin").setValue("0.01");
  await wrapper!.get("#qmax").setValue("0.2");
  await wrapper!.get("#qsteps").setValue("10");
  await button("Apply new Q").trigger("click");

  const probe = lastSent(socket).model.object.models[0].probe;
  expect(probe.__class__).toBe("refl1d.probe.QProbe");
  expect(probe.Q.values).toHaveLength(10);
  expect(probe.Q.values[0]).toBeCloseTo(0.01);
  expect(Math.max(...probe.Q.values)).toBeLessThanOrEqual(0.2);
  expect(probe.dQ.values).toEqual(Array(10).fill(0.0001));
});

test("Fetches the model again when parameters change on the server", async () => {
  const { socket } = await setup();
  expect(emitted(socket, "get_model")).toHaveLength(1);
  socket.serverEmit("updated_parameters");
  expect(emitted(socket, "get_model")).toHaveLength(2);
});
