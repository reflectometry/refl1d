<script setup lang="ts">
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import type { ComputedRef } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';


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
const parameters_by_id = ref({});
const dictionaryLoaded = ref(false);
// Builder options
const showImaginary = ref(false);

const props = defineProps<{
  socket: AsyncSocket,
}>();

const sortedLayers: ComputedRef<NonMagneticLayer[]> = computed(() => {
  return modelJson.value['object']['models'][activeModel.value]['sample']['layers'];
});

// async function get_model_names() {
//   model_names.value = await props.socket.asyncEmit("get_model_names");
// }

props.socket.on('update_parameters', fetch_model);
props.socket.on('model_loaded', fetch_model);

interface Variable {
  value: number,
  __class__: "bumps.parameter.Variable"
}

// this is an imcomplete interface definition...
interface Parameter {
  name?: string,
  fixed: boolean,
  slot: Variable | number,
  limits?: (number | "-inf" | "inf")[],
  bounds?: (number | "-inf" | "inf")[],
  tags?: string[],
  __class__: "bumps.parameter.Parameter",
}

interface SLD {
  name: string,
  rho: number | Parameter,
  irho: number | Parameter,
  __class__: "refl1d.material.SLD"
}

interface NonMagneticLayer {
  name: string,
  thickness: number | Parameter,
  magnetism: null,
  material: SLD,
  interface: number | Parameter,
  __class__: "refl1d.model.Slab"
}

const newParameterTemplate = {
  value: 0,
  __class__: "bumps.parameter.Parameter",
}

const newSLDTemplate: SLD = {
  name: "sld",
  rho: 2.5,
  irho: 0,
  __class__: "refl1d.material.SLD"
}

const newLayerTemplate: NonMagneticLayer = {
    name: "new",
    thickness: 25,
    magnetism: null,
    material: newSLDTemplate,
    interface: 1,
    __class__: "refl1d.model.Slab"
}

const newModelTemplate = {
  references: {},
  object: { 
    __class__: "refl1d.fitproblem.FitProblem",
    models: [
      {
        __class__: "refl1d.experiment.Experiment",
        sample: { 
          __class__: "refl1d.model.Stack",
          layers: [
            { ...newLayerTemplate, name: "Substrate", material: { ...newSLDTemplate, name: "Si", rho: 2.07 } },
            { ...newLayerTemplate, name: "Vacuum", material: { ...newSLDTemplate, name: "Vacuum", rho: 0 } },
          ] 
        },
        probe: createQProbe(0, 0.1, 250, 0.00001),
      }
    ] 
  },
  "$schema": "bumps-draft-02"
}

function createQProbe(qmin: number = 0, qmax: number = 0.1, qsteps: number = 250, dQ: number = 0.00001) {
  return {
    __class__: "refl1d.probe.QProbe",
    Q: {
      __class__: "bumps.util.NumpyArray",
      values: Array.from({length: qsteps}).map((_, i) => i * (qmax - qmin) / qsteps),
      dtype: 'float'
    },
    dQ: {
      __class__: "bumps.util.NumpyArray",
      values: Array.from({length: qsteps}).map((_, i) => dQ),
      dtype: 'float'
    },
  }
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
      extract_parameters(json_value);
      dictionaryLoaded.value = true;
    }
  });
}

async function new_model() {
  const model = structuredClone(newModelTemplate);
  if (dictionaryLoaded.value) {
    const confirmation = confirm("This will overwrite your current model...");
    if (!confirmation) {
      return;
    }
  }
  modelJson.value = model;
  send_model();
}

function extract_parameters(model) {
    const new_parameters_by_id = model.references;
    const get_parameters = (obj, parent_obj, path, key) => {
        if (obj && obj?.__class__ === 'bumps.parameter.Parameter') {
            // Replace the first word of the parameter name with the parent object name
            // e.g. "new thickness" -> "layer1 thickness"
            const parent_name = parent_obj?.name ?? '';
            const new_name = `${parent_name} ${obj.name.split(' ').slice(1).join(' ')}`;
            obj.name = new_name;
            new_parameters_by_id[obj.id] = obj;
        }
    }
    walk_object(model.object, null, '', '', get_parameters);
    parameters_by_id.value = new_parameters_by_id;
}

function walk_object(obj, parent_obj, path, key, cb: Function) {
    cb(obj, parent_obj, path, key);
    if (Array.isArray(obj)) {
        obj.forEach((subobj, i) => walk_object(subobj, obj, `${path}/${i}`, i, cb));
    }
    else if (obj instanceof Object) {
        Object.entries(obj).forEach(([subkey, subobj]) => walk_object(subobj, obj, `${path}/${subkey}`, key, cb));
    }
}

function get_slot(parameter_like) {
    // parameter_like can be type: "bumps.parameter.Parameter" or type: "Reference"
    if (parameter_like == null) {
        return null;
    }
    const slot = parameter_like?.slot ?? parameters_by_id.value[parameter_like.id]?.slot;
    return slot;
}

function send_model() {
    // Ensure that the name of the layer is the same as the name of the material
    const renamed_params = new Set();
    for (const [index, item] of Object.entries(sortedLayers.value)) {
      const { material } = item;
      const { name } = material;
      item.name = name;
      for (let param_name of ["rho", "irho"]) {
        const param_ref = material[param_name];
        if ( param_ref instanceof Object && !renamed_params.has(param_ref.id) ) {
          parameters_by_id.value[param_ref.id].name = `${name} ${param_name}`;
          renamed_params.add(param_ref.id);
        }
      }
      for (let param_name of ["thickness", "interface"]) {
        const param_ref = item[param_name];
        if ( param_ref instanceof Object && !renamed_params.has(param_ref.id) ) {
          parameters_by_id.value[param_ref.id].name = `${name} ${param_name}`;
          renamed_params.add(param_ref.id);
        }
      }
    };

    const array_buffer = new TextEncoder().encode(JSON.stringify(modelJson.value));
    props.socket.emit('set_serialized_problem', array_buffer);
}

// Adding and deleting layers
function delete_layer(index) {
    sortedLayers.value.splice(index, 1);
    send_model();
};

function add_layer(after_index: number = -1) {
    const new_layer: NonMagneticLayer = structuredClone(newLayerTemplate);
    sortedLayers.value.splice(after_index, 0, new_layer);
    send_model();
};

function setQProbe() {
  modelJson.value['object']['models'][activeModel.value]['probe'] = createQProbe(editQmin.value, editQmax.value, editQsteps.value);
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
<div id="builder" @keyup.enter="send_model" @keyup.tab="send_model">
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
          <div class="form-check form-switch m-2" @click="send_model">
            <input class="form-check-input" type="checkbox" id="showImaginary_input" v-model="showImaginary">
            <label class="form-check-label" for="showImaginary_input">Show imaginary SLD</label>
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
                  <th>SLD</th>
                  <th v-if="showImaginary">iSLD</th>
                  <th>Interface</th>
                  <th></th>
                  <th></th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(layer, key) in sortedLayers" :key="key" @blur.capture="send_model">
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
          <button class="btn btn-secondary m-2" @click="send_model">Apply changes</button>
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
      margin-bottom: -4em;
    }
</style>
