<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from "vue";
import type { ComputedRef } from "vue";
import type { AsyncSocket } from "bumps-webview-client/src/asyncSocket";
import { v4 as uuidv4 } from "uuid";
import { dqIsFWHM } from "../app_state";
import type {
  BoundsValue,
  Magnetism,
  Parameter,
  ParameterLike,
  QProbe,
  SerializedModel,
  Slab,
  SLD,
  Stack,
} from "../model";

// const title = "Builder";
// @ts-ignore: intentionally infinite type recursion
const modelJson = shallowRef<json>({});
const showInstructions = ref(false);
const showEditQRange = ref(false);
const editQmin = ref(0);
const editQmax = ref(0.1);
const editQsteps = ref(250);
const activeModel = ref(0);
const insertIndex = ref(-1);
const parametersById = ref<{ [key: string]: Parameter }>({});
const dictionaryLoaded = ref(false);
// Builder options
const showImaginary = ref(false);

const props = defineProps<{
  socket: AsyncSocket;
}>();

const sortedLayers: ComputedRef<Slab[]> = computed(() => {
  return modelJson.value["object"]["models"][activeModel.value]["sample"]["layers"];
});

props.socket.on("updated_parameters", fetchModel);
props.socket.on("model_loaded", fetchModel);

function createParameter(
  name: string,
  value: number,
  limits: [BoundsValue, BoundsValue] = ["-inf", "inf"],
  fixed: boolean = true,
  tags: string[] = []
) {
  const par: Parameter = {
    id: uuidv4(),
    name,
    fixed,
    slot: { value, __class__: "bumps.parameter.Variable" },
    limits,
    bounds: null,
    tags,
    __class__: "bumps.parameter.Parameter",
  };
  return par;
}

function createLayer(
  name: string,
  rho: number,
  irho: number,
  thickness: number,
  interface_: number,
  magnetism: Magnetism | null = null
) {
  const rhoParam = createParameter("rho", rho, ["-inf", "inf"], true, ["sample"]);
  const irhoParam = createParameter("irho", irho, ["-inf", "inf"], true, ["sample"]);
  const thicknessParam = createParameter("thickness", thickness, [0, "inf"], true, ["sample"]);
  const interfaceParam = createParameter("interface", interface_, [0, "inf"], true, ["sample"]);
  const material: SLD = { name, rho: rhoParam, irho: irhoParam, __class__: "refl1d.sample.material.SLD" };
  const layer: Slab = {
    name,
    material,
    thickness: thicknessParam,
    interface: interfaceParam,
    magnetism,
    __class__: "refl1d.sample.layers.Slab",
  };
  return layer;
}

function createModel(): SerializedModel {
  return {
    references: {},
    object: {
      __class__: "bumps.fitproblem.FitProblem",
      models: [
        {
          __class__: "refl1d.experiment.Experiment",
          sample: {
            __class__: "refl1d.sample.layers.Stack",
            layers: [createLayer("Si", 2.07, 0.0, 0.0, 1.0), createLayer("Vacuum", 0.0, 0.0, 0.0, 0.0)],
          },
          probe: generateQProbe(editQmin.value, editQmax.value, editQsteps.value, 0.0001),
        },
      ],
    },
    $schema: "bumps-draft-02",
  };
}

function generateQProbe(qmin: number = 0, qmax: number = 0.1, qsteps: number = 250, dQ: number = 0.00001) {
  const qArr = Array.from({ length: qsteps }, (_, i) => qmin + (i * (qmax - qmin)) / qsteps);
  const dqArr = Array.from({ length: qsteps }, () => dQ);
  const probe: QProbe = {
    Q: { values: qArr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
    dQ: { values: dqArr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
    background: createParameter("background", 0.0, [0, "inf"], true, ["probe"]),
    intensity: createParameter("intensity", 1.0, [0, "inf"], true, ["probe"]),
    back_absorption: createParameter("back_absorption", 0.0, [0, 1.0], true, ["probe"]),
    back_reflectivity: false,
    resolution: "normal",
    __class__: "refl1d.probe.QProbe",
  };
  return probe;
}

async function fetchModel() {
  props.socket.asyncEmit("get_model", (payload: ArrayBuffer) => {
    const json_bytes = new Uint8Array(payload);
    console.debug({ payload });

    if (json_bytes.length < 3) {
      // no model defined...
      modelJson.value = {};
      dictionaryLoaded.value = false;
    } else {
      const jsonValue = JSON.parse(decoder.decode(json_bytes));
      modelJson.value = jsonValue;
      console.log("modelJson", modelJson.value);
      parametersById.value = jsonValue.references;
      dictionaryLoaded.value = true;
    }
  });
}

async function newModel() {
  const model = createModel();
  if (dictionaryLoaded.value) {
    const confirmation = confirm("This will overwrite your current model...");
    if (!confirmation) {
      return;
    }
  }
  modelJson.value = model;
  sendModel(true, "Simple Builder Model");
}

function getSlot(parameterLike: ParameterLike) {
  // parameterLike can be type: "bumps.parameter.Parameter" or type: "Reference"
  if (parameterLike != null) {
    const parameter = resolveParameter(parameterLike);
    return parameter.slot;
  }
  // TODO: This should really not return null.
  // return null;
  return { value: 0.0, __class__: "bumps.parameter.Variable" };
}

function resolveParameter(parameterLike: ParameterLike): Parameter {
  // parameterLike can be type: "bumps.parameter.Parameter" or type: "Reference"
  if (parameterLike.__class__ === "bumps.parameter.Parameter") {
    return parameterLike as Parameter;
  } else if (parameterLike.__class__ === "Reference" && parameterLike.id in parametersById.value) {
    return parametersById.value[parameterLike.id] as Parameter;
  } else {
    throw new Error(`Parameter with id ${parameterLike.id} not found in parametersById`);
  }
}

function setParameterNames(stack: Stack) {
  for (const layer of stack.layers) {
    if (layer.__class__ === "refl1d.sample.layers.Repeat") {
      setParameterNames(layer.stack);
    } else {
      const l = layer as Slab;
      const { material, thickness, interface: interface_ } = l;
      const { name, rho, irho } = material;
      const thicknessParam = resolveParameter(thickness);
      const interfaceParam = resolveParameter(interface_);
      const rhoParam = resolveParameter(rho);
      const irhoParam = resolveParameter(irho);
      l.name = name;
      thicknessParam.name = `${material.name} thickness`;
      interfaceParam.name = `${material.name} interface`;
      rhoParam.name = `${material.name} rho`;
      irhoParam.name = `${material.name} irho`;
    }
  }
}

function setParameterBounds(stack: Stack) {
  // set the bounds of the fixed parameters
  const boundsSetter = (p: Parameter) => {
    if (!p.fixed) {
      return;
    }
    const value = p.slot.value;
    p.bounds = value === 0.0 ? [-0.1, 0.1] : [value * 0.5, value * 1.5];
    p.bounds.sort((a, b) => {
      let c =
        a === "-inf" ? Number.NEGATIVE_INFINITY
        : a === "inf" ? Number.POSITIVE_INFINITY
        : a;
      let d =
        b === "-inf" ? Number.NEGATIVE_INFINITY
        : b === "inf" ? Number.POSITIVE_INFINITY
        : b;
      return c - d;
    });
  };
  for (const layer of stack.layers) {
    if (layer.__class__ === "refl1d.sample.layers.Repeat") {
      setParameterBounds(layer.stack);
    } else {
      const l = layer as Slab;
      const { material, thickness, interface: interface_ } = l;
      const { rho, irho } = material;
      const thicknessParam = resolveParameter(thickness);
      const interfaceParam = resolveParameter(interface_);
      const rhoParam = resolveParameter(rho);
      const irhoParam = resolveParameter(irho);
      [thicknessParam, interfaceParam, rhoParam, irhoParam].forEach(boundsSetter);
    }
  }
}

async function sendModel(is_new: boolean = false, name: string | null = null) {
  for (const model of modelJson.value["object"]["models"]) {
    setParameterNames(model["sample"]);
    setParameterBounds(model["sample"]);
  }
  const json_model = JSON.stringify(modelJson.value);
  await props.socket.asyncEmit("set_serialized_problem", json_model, is_new, name);
}

// Adding and deleting layers
function deleteLayer(index: number) {
  sortedLayers.value.splice(index, 1);
  sendModel();
}

function addLayer(after_index: number = -1) {
  const newLayer: Slab = createLayer("sld", 2.5, 0.0, 25.0, 1.0);
  sortedLayers.value.splice(after_index, 0, newLayer);
  sendModel();
}

function setQProbe() {
  modelJson.value["object"]["models"][activeModel.value]["probe"] = generateQProbe(
    editQmin.value,
    editQmax.value,
    editQsteps.value,
    0.0001
  );
  sendModel();
}

const decoder = new TextDecoder("utf-8");

onMounted(() => {
  fetchModel();
});

// Code for draggable rows
const dragData = ref<number | null>(null);

function dragStart(index: number, event: DragEvent) {
  if (event.dataTransfer === null) {
    return;
  }
  dragData.value = index;
  event.dataTransfer.setData("text/plain", `${index}`);
}

function dragOver(index: number, event: DragEvent) {
  event.preventDefault();
}

function drop(index: number) {
  if (dragData.value !== null) {
    const draggedDict = sortedLayers.value[dragData.value];
    sortedLayers.value.splice(dragData.value, 1);
    sortedLayers.value.splice(index, 0, draggedDict);
    dragData.value = null;
  }
  sendModel();
}

function dragEnd() {
  dragData.value = null;
}
</script>

<template>
  <div id="builder">
    <div class="container m-2">
      <div class="card">
        <div class="card-body">
          <h5 class="card-title d-inline-block px-2 my-0 align-middle">Simple Slab Model Builder</h5>
          <button
            class="btn btn-primary"
            type="button"
            aria-expanded="false"
            aria-controls="builderInstructions"
            @click="showInstructions = !showInstructions"
          >
            Instructions
          </button>

          <div id="builderInstructions" class="card-text collapse" :class="{ show: showInstructions }">
            <ul>
              <li>Click the "Add layer" button to add a new layer.</li>
              <li>Drag and drop the rows to change the order of the layers.</li>
              <li>You can toggle the imaginary SLD by clicking the checkbox below.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    <div class="container mt-4">
      <div class="row justify-content-end">
        <div class="col">
          <button class="btn btn-primary m-2" @click="showEditQRange = !showEditQRange">Edit Q-range</button>
        </div>
        <div class="col-auto align-self-center">
          <div class="form-check form-switch m-2">
            <input
              id="showImaginary_input"
              v-model="showImaginary"
              class="form-check-input"
              type="checkbox"
              @click="sendModel()"
            />
            <label class="form-check-label" for="showImaginary_input">Show imaginary SLD</label>
          </div>
          <div class="form-check form-switch m-2">
            <input
              id="dqIsFWHM_input"
              v-model="dqIsFWHM"
              class="form-check-input"
              type="checkbox"
              @click="sendModel()"
            />
            <label class="form-check-label" for="dqIsFWHM_input">Resolution as FWHM</label>
          </div>
        </div>
      </div>
      <div v-if="showEditQRange" class="row mb-2">
        <div class="col">
          <label for="qmin">Q min (Å<sup>-1</sup>)</label>
          <input id="qmin" v-model="editQmin" class="form-control" type="number" step="0.01" />
        </div>
        <div class="col">
          <label for="qmax">Q max (Å<sup>-1</sup>)</label>
          <input id="qmax" v-model="editQmax" class="form-control" type="number" step="0.01" />
        </div>
        <div class="col">
          <label for="qsteps">Q steps</label>
          <input id="qsteps" v-model="editQsteps" class="form-control" type="number" step="1" />
        </div>
        <div class="col-auto align-self-end">
          <button class="btn btn-secondary mx-2" @click="setQProbe">Apply new Q</button>
        </div>
      </div>
      <table v-if="dictionaryLoaded" id="sortable" class="table table-sm">
        <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
          <tr>
            <th></th>
            <th>Layer</th>
            <th>Thickness</th>
            <th>Rho</th>
            <th v-if="showImaginary">iRho</th>
            <th>Interface</th>
            <th></th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(layer, key) in sortedLayers" :key="key" class="align-middle" @blur.capture="sendModel()">
            <td
              draggable="true"
              class="draggable"
              @dragstart="dragStart(key, $event)"
              @dragover="dragOver(key, $event)"
              @drop="drop(key)"
              @dragend="dragEnd"
            >
              <span width="10px" class="badge bg-secondary">:</span>
            </td>
            <td>
              <textarea
                id="layer-name"
                v-model="layer.material.name"
                class="form-control name"
                rows="1"
                cols="40"
                type="text"
                :title="layer.material.name"
              >
                <label for="layer-name">Layer Name</label>
              </textarea>
            </td>
            <td>
              <label for="layer-thickness" class="visually-hidden">>Layer Thickness</label>
              <input
                v-if="getSlot(layer.thickness) !== null"
                id="layer-thickness"
                v-model="getSlot(layer.thickness).value"
                class="form-control"
                type="number"
                step="5"
              />
            </td>
            <td>
              <label for="layer-rho" class="visually-hidden">>Layer Rho</label>
              <input
                v-if="getSlot(layer.material.rho) !== null"
                id="layer-rho"
                v-model="getSlot(layer.material.rho).value"
                class="form-control"
                type="number"
                step="0.01"
              />
            </td>
            <td v-if="showImaginary">
              <label for="layer-irho" class="visually-hidden">>Layer iRho</label>
              <input
                v-if="getSlot(layer.material.irho) !== null"
                id="layer-irho"
                v-model="getSlot(layer.material.irho).value"
                class="form-control"
                type="number"
                step="0.01"
              />
            </td>
            <td>
              <label for="layer-interface" class="visually-hidden">>Layer Interface</label>
              <input
                v-if="getSlot(layer.interface) !== null"
                id="layer-interface"
                v-model="getSlot(layer.interface).value"
                class="form-control"
                type="number"
                step="1"
              />
            </td>
            <td><button class="btn btn-danger btn-sm" @click="deleteLayer(key)">Delete</button></td>
            <td>
              <button class="btn btn-success btn-sm add-layer-after" title="add layer here" @click="addLayer(key + 1)">
                +
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else>Load data to start building a model</p>
    </div>

    <div class="row">
      <div class="col">
        <button class="btn btn-primary m-2" @click="newModel">New Model</button>
      </div>
      <div v-if="dictionaryLoaded" class="col-auto">
        <button class="btn btn-secondary m-2" @click="sendModel()">Apply changes</button>
      </div>
      <div v-if="dictionaryLoaded" class="col-auto">
        <div class="input-group m-2">
          <button class="btn btn-success btn-sm" @click="addLayer(insertIndex)">Add layer at index:</button>
          <label for="insert-index" class="visually-hidden">Insert index</label>
          <input id="insert-index" v-model="insertIndex" class="form-control me-4 insert-index" type="number" />
        </div>
      </div>
    </div>

    <div class="container m-2">
      <div class="card bg-warning">
        <div class="card-body">
          <h5 class="card-title">Limitations and future features</h5>
          <div class="card-text">
            <ul>
              <li>This builder can currently only do non-magnetic models.</li>
              <li>It can only deal with a single model.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 50%;
}
.draggable {
  cursor: move;
}

textarea.name {
  overflow-x: auto;
  white-space: nowrap;
  resize: none;
}
input.insert-index {
  width: 4em;
}
button.add-layer-after {
  margin-bottom: -3em;
  padding: 0.1em 0.3em;
}
</style>
