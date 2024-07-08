<script setup lang="ts">
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import type { ComputedRef } from 'vue';
import { v4 as uuidv4 } from 'uuid';
import type { Slab, Magnetism, Parameter, ParameterLike, QProbe, Reference, SLD, Stack, SerializedModel, BoundsValue } from '../model';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket.ts';

import { dq_is_FWHM } from '../app_state';

const title = "Builder";
// @ts-ignore: intentionally infinite type recursion
const modelJson = shallowRef<json>({});
const showInstructions = ref(false);
const showEditQRange = ref(false);
const editQmin = ref(0);
const editQmax = ref(0.1);
const editQsteps = ref(250);
const activeModel = ref(0);
const insert_index = ref(-1);
const parameters_by_id = ref<{[key: string]: Parameter}>({});
const dictionaryLoaded = ref(false);
// Builder options
const showImaginary = ref(false);

const props = defineProps<{
  socket: AsyncSocket,
}>();

const sortedLayers: ComputedRef<Slab[]> = computed(() => {
  return modelJson.value['object']['models'][activeModel.value]['sample']['layers'];
});

// async function get_model_names() {
//   model_names.value = await props.socket.asyncEmit("get_model_names");
// }

props.socket.on('updated_parameters', fetch_model);
props.socket.on('model_loaded', fetch_model);

function createParameter(name: string, value: number, limits: [BoundsValue, BoundsValue] = ["-inf", "inf"], fixed: boolean = true, tags: string[] = []) {
  const par: Parameter = {
    id: uuidv4(),
    name,
    fixed,
    slot: { value, __class__: "bumps.parameter.Variable" },
    limits,
    bounds: null,
    tags,
    __class__: "bumps.parameter.Parameter"
  }
  return par;
}

function createLayer(name: string, rho: number, irho: number, thickness: number, interface_: number, magnetism: Magnetism | null = null) {
  const rho_param = createParameter("rho", rho, ["-inf", "inf"], true, ["sample"]);
  const irho_param = createParameter("irho", irho, ["-inf", "inf"], true, ["sample"]);
  const thickness_param = createParameter("thickness", thickness, [0, "inf"], true, ["sample"]);
  const interface_param = createParameter("interface", interface_, [0, "inf"], true, ["sample"]);
  const material: SLD = { __class__: "refl1d.material.SLD", name, rho: rho_param, irho: irho_param };
  const layer: Slab = { __class__: "refl1d.model.Slab", name, material, thickness: thickness_param, interface: interface_param, magnetism };
  return layer;
}

function createModel(): SerializedModel {
  return {
    references: {},
    object: {
      __class__: "refl1d.fitproblem.FitProblem",
      models: [
        {
          __class__: "refl1d.experiment.Experiment",
          sample: {
            __class__: "refl1d.model.Stack",
            layers: [
              createLayer("Si", 2.07, 0.0, 0.0, 1.0),
              createLayer("Vacuum", 0.0, 0.0, 0.0, 0.0),
            ]
          },
          probe: generateQProbe(editQmin.value, editQmax.value, editQsteps.value, 0.0001),
        }
      ]
    },
    "$schema": "bumps-draft-02"
  }
}

function generateQProbe(qmin: number = 0, qmax: number = 0.1, qsteps: number = 250, dQ: number = 0.00001) {
    const Q_arr = Array.from({ length: qsteps }, (_, i) => qmin + i * (qmax - qmin) / qsteps);
    const dQ_arr = Array.from({ length: qsteps }, () => dQ);
    const probe: QProbe = {
        Q: { values: Q_arr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
        dQ: { values: dQ_arr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
        background: createParameter("background", 0.0, [0, "inf"], true, ["probe"]),
        intensity: createParameter("intensity", 1.0, [0, "inf"], true, ["probe"]),
        back_absorption: createParameter("back_absorption", 0.0, [0, 1.0], true, ["probe"]),
        back_reflectivity: false,
        resolution: 'normal',
        __class__: "refl1d.probe.QProbe"
    };
    return probe;
}

async function fetch_model() {
  props.socket.asyncEmit('get_model', (payload: ArrayBuffer) => {
    const json_bytes = new Uint8Array(payload);
    if (json_bytes.length < 3) {
      // no model defined...
      modelJson.value = {};
      dictionaryLoaded.value = false;
    }
    else {
      const json_value = JSON.parse(decoder.decode(json_bytes));
      modelJson.value = json_value;
      console.log('modelJson', modelJson.value);
      parameters_by_id.value = json_value.references;
      dictionaryLoaded.value = true;
    }
  });
}

async function new_model() {
  const model = createModel();
  if (dictionaryLoaded.value) {
    const confirmation = confirm('This will overwrite your current model...');
    if (!confirmation) {
      return;
    }
  }
  modelJson.value = model;
  send_model(true, 'Simple Builder Model');
}

function get_slot(parameter_like: ParameterLike) {
    // parameter_like can be type: "bumps.parameter.Parameter" or type: "Reference"
    if (parameter_like == null) {
        return null;
    }
    const parameter = resolve_parameter(parameter_like);
    return parameter.slot;
}

function resolve_parameter(parameter_like: ParameterLike): Parameter {
  // parameter_like can be type: "bumps.parameter.Parameter" or type: "Reference"
  if (parameter_like.__class__ === "bumps.parameter.Parameter") {
    return parameter_like as Parameter;
  }
  else if (parameter_like.__class__ === "Reference" && parameter_like.id in parameters_by_id.value) {
    return parameters_by_id.value[parameter_like.id] as Parameter;
  }
  else {
    throw new Error(`Parameter with id ${parameter_like.id} not found in parameters_by_id`);
  }
}

function set_parameter_names(stack: Stack) {
  for (const layer of stack.layers) {
    if (layer.__class__ === "refl1d.model.Repeat") {
      set_parameter_names(layer.stack);
    }
    else {
      const l = layer as Slab;
      const { material, thickness, interface: interface_ } = l;
      const { name, rho, irho } = material;
      const thickness_param = resolve_parameter(thickness);
      const interface_param = resolve_parameter(interface_);
      const rho_param = resolve_parameter(rho);
      const irho_param = resolve_parameter(irho);
      l.name = name;
      thickness_param.name = `${material.name} thickness`;
      interface_param.name = `${material.name} interface`;
      rho_param.name = `${material.name} rho`;
      irho_param.name = `${material.name} irho`;
    }
  }
}

function set_parameter_bounds(stack: Stack) {
  // set the bounds of the fixed parameters
  const bounds_setter = (p: Parameter) => {
    if (!p.fixed) {
      return;
    }
    const value = p.slot.value;
    p.bounds = (value == 0.0) ? [-0.1, 0.1] : [value * 0.5, value * 1.5];
    p.bounds.sort((a, b) => a - b);

  }
  for (const layer of stack.layers) {
    if (layer.__class__ === "refl1d.model.Repeat") {
      set_parameter_bounds(layer.stack);
    }
    else {
      const l = layer as Slab;
      const { material, thickness, interface: interface_ } = l;
      const { rho, irho } = material;
      const thickness_param = resolve_parameter(thickness);
      const interface_param = resolve_parameter(interface_);
      const rho_param = resolve_parameter(rho);
      const irho_param = resolve_parameter(irho);
      [thickness_param, interface_param, rho_param, irho_param].forEach(bounds_setter);
    }
  }
}

async function send_model(is_new: boolean = false, name: string | null = null) {
  for (const model of modelJson.value['object']['models']) {
    set_parameter_names(model['sample']);
    set_parameter_bounds(model['sample']);
  }
  const json_model = JSON.stringify(modelJson.value);
  await props.socket.asyncEmit('set_serialized_problem', json_model, is_new, name);
}

// Adding and deleting layers
function delete_layer(index) {
    sortedLayers.value.splice(index, 1);
    send_model();
};

function add_layer(after_index: number = -1) {
    const new_layer: Slab = createLayer("sld", 2.5, 0.0, 25.0, 1.0);
    sortedLayers.value.splice(after_index, 0, new_layer);
    send_model();
};

function setQProbe() {
  modelJson.value['object']['models'][activeModel.value]['probe'] = generateQProbe(editQmin.value, editQmax.value, editQsteps.value, 0.0001);
  send_model();
}

const decoder = new TextDecoder('utf-8');

onMounted(() => {
    fetch_model();
})

// Code for draggable rows
const dragData = ref<number | null>(null);

function dragStart(index: number, event) {
    dragData.value = index;
    event.dataTransfer.setData('text/plain', index);
};

function dragOver(index, event) {
    event.preventDefault();
};

function drop(index) {
    if (dragData.value !== null) {
        const draggedDict = sortedLayers.value[dragData.value];
        sortedLayers.value.splice(dragData.value, 1);
        sortedLayers.value.splice(index, 0, draggedDict);
        dragData.value = null;
    }
    send_model();
};

function dragEnd() {
    dragData.value = null;
};

</script>

<template>
<div id="builder" @keyup.enter="send_model()" @keyup.tab="send_model()">
    <div class="container m-2">
        <div class="card">
            <div class="card-body">
              <h5 class="card-title d-inline-block px-2 my-0 align-middle">Simple Slab Model Builder</h5>
                <button class="btn btn-primary" type="button" aria-expanded="false"
                  aria-controls="builderInstructions" @click="showInstructions = !showInstructions">
                  Instructions
                </button>

              <p class="card-text collapse" :class="{ show: showInstructions }" id="builderInstructions">
                  <ul>
                      <li>Click the "Add layer" button to add a new layer.</li>
                      <li>Drag and drop the rows to change the order of the layers.</li>
                      <li>You can toggle the imaginary SLD by clicking the checkbox below.</li>
                  </ul>
              </p>
            </div>
        </div>
    </div>
    <div class="container mt-4">
      <div class="row justify-content-end">
        <div class="col">
          <button class="btn btn-primary m-2" @click="showEditQRange = !showEditQRange">Edit Q-range</button>
        </div>
        <div class="col-auto align-self-center">
          <div class="form-check form-switch m-2" @click="send_model()">
            <input class="form-check-input" type="checkbox" id="showImaginary_input" v-model="showImaginary">
            <label class="form-check-label" for="showImaginary_input">Show imaginary SLD</label>
          </div>
          <div class="form-check form-switch m-2" @click="send_model()">
            <input class="form-check-input" type="checkbox" id="dq_is_FWHM_input" v-model="dq_is_FWHM">
            <label class="form-check-label" for="dq_is_FWHM_input">Resolution as FWHM</label>
          </div>
        </div>
      </div>
      <div class="row mb-2" v-if="showEditQRange">
        <div class="col">
          <label for="qmin">Q min (Å<sup>-1</sup>)</label>
          <input class="form-control" type="number" step="0.01" v-model="editQmin">
        </div>
        <div class="col">
          <label for="qmax">Q max (Å<sup>-1</sup>)</label>
          <input class="form-control" type="number" step="0.01" v-model="editQmax">
        </div>
        <div class="col">
          <label for="qsteps">Q steps</label>
          <input class="form-control" type="number" step="1" v-model="editQsteps">
        </div>
        <div class="col-auto align-self-end">
          <button class="btn btn-secondary mx-2" @click="setQProbe">Apply new Q</button>
        </div>
      </div>
        <table class="table table-sm" v-if="dictionaryLoaded" id="sortable">
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
                <tr v-for="(layer, key) in sortedLayers" :key="key" @blur.capture="send_model()" class="align-middle">
                    <td  
                        @dragstart="dragStart(key, $event)"
                        @dragover="dragOver(key, $event)"
                        @drop="drop(key)"
                        @dragend="dragEnd"
                        draggable="true"
                        class="draggable"><span width=10px class="badge bg-secondary">:</span></td>
                    <td><textarea class="form-control name" rows=1 cols="40" type="text" v-model="layer.material.name" :title="layer.material.name"/></td>
                    <td><input class="form-control" v-if="get_slot(layer.thickness) !== null" type="number" step="5" v-model="get_slot(layer.thickness).value"></td>
                    <td><input class="form-control" v-if="get_slot(layer.material.rho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.rho).value"></td>
                    <td v-if="showImaginary"><input class="form-control" v-if="get_slot(layer.material.irho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.irho).value"></td>
                    <td><input class="form-control" v-if="get_slot(layer.interface) !== null" type="number" step="1" v-model="get_slot(layer.interface).value"></td>
                    <td><button class="btn btn-danger btn-sm" @click="delete_layer(key)">Delete</button></td>
                    <td><button class="btn btn-success btn-sm add-layer-after" @click="add_layer(key+1)" title="add layer here">+</button></td>
                </tr>
            </tbody>
        </table>
    <p v-else>Load data to start building a model</p>
    </div>
       
    <div class="row">
        <div class="col">
            <button class="btn btn-primary m-2" @click="new_model">New model</button>
        </div>
        <div class="col-auto" v-if="dictionaryLoaded">
          <button class="btn btn-secondary m-2" @click="send_model()">Apply changes</button>
        </div>
        <div class="col-auto" v-if="dictionaryLoaded">
          <div class="input-group m-2">
            <button class="btn btn-success btn-sm" @click="add_layer(insert_index)">Add layer at index: </button>
            <input class="form-control me-4 insert-index" v-model="insert_index" type="number"/>
          </div>
        </div>
    </div>
    
    <div class="container m-2">
        <div class="card bg-warning">
            <div class="card-body">
                <h5 class="card-title">Limitations and future features</h5>
                <p class="card-text">
                    <ul>
                        <li>This builder can currently only do non-magnetic models.</li>
                        <li>It can only deal with a single model.</li>
                    </ul>
                    
                </p>
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
      padding: 0.1em 0.3em;
      margin-bottom: -3em;
    }
</style>
