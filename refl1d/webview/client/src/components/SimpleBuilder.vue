<script setup lang="ts">
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';


const title = "Builder";
// @ts-ignore: intentionally infinite type recursion
const modelJson = ref<json>({});
const parameters_by_id = ref({});
const dictionaryLoaded = ref(false);
const sortedLayers = ref<NonMagneticLayer[]>([]);
// Builder options
const showImaginary = ref(false);

const props = defineProps<{
  socket: AsyncSocket,
}>();

// async function get_model_names() {
//   model_names.value = await props.socket.asyncEmit("get_model_names");
// }

props.socket.on('update_parameters', fetch_model);
props.socket.on('model_loaded', fetch_model);

interface NonMagneticLayer {
  name: string,
  thickness: number,
  magnetism: null,
  material: {
    name: string,
    rho: number,
    irho: number,
    type: string
  },
  interface: number,
  type: "refl1d.model.Slab"
}

const newLayerTemplate: NonMagneticLayer = {
    name: "new",
    thickness: 25,
    magnetism: null,
    material: {name: "new", rho: 2.3, irho: 0, type: "refl1d.material.SLD"},
    interface: 1,
    type: "refl1d.model.Slab"
}

const newModelTemplate = {
  references: {},
  object: { 
    type: "refl1d.fitproblem.FitProblem",
    models: [
      {
        type: "refl1d.experiment.Experiment",
        sample: { 
          type: "refl1d.model.Stack",
          layers: [newLayerTemplate] 
        },
        probe: {
          type: "refl1d.probe.QProbe",
          Q: {
            type: "bumps.util.NumpyArray",
            values: Array.from({length: 250}).map((_, i) => i * 0.1 / 250),
            dtype: 'float'
          },
          dQ: {
            type: "bumps.util.NumpyArray",
            values: Array.from({length: 250}).map((_, i) => 0.00001),
            dtype: 'float'
          },
        },
      }
    ] 
  },
  "$schema": "bumps-draft-02"
}

async function fetch_model() {
  props.socket.asyncEmit('get_model', (payload: ArrayBuffer) => {
    const json_bytes = new Uint8Array(payload);
    const json_value = (json_bytes.length < 3) ? structuredClone(newModelTemplate) : JSON.parse(decoder.decode(json_bytes));
    modelJson.value = json_value;
    extract_parameters(json_value);
    sortedLayers.value = modelJson.value['object']['models'][0]['sample']['layers'];
    dictionaryLoaded.value = true;
  });
}

function extract_parameters(model) {
    const new_parameters_by_id = model.references;
    const get_parameters = (obj, parent_obj, path, key) => {
        if (obj && obj?.type === 'bumps.parameter.Parameter') {
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
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        item.name = item.material.name;
    };

    const array_buffer = new TextEncoder().encode(JSON.stringify(modelJson.value));
    props.socket.emit('set_serialized_problem', array_buffer);
}

// Adding and deleting layers
function delete_layer(index) {
    sortedLayers.value.splice(index, 1);
    send_model();
};

function add_layer() {
    const new_layer: NonMagneticLayer = structuredClone(newLayerTemplate);
    sortedLayers.value.push(new_layer);
    send_model();
};

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
            <div class="card-header">
                <h5 class="card-title">Simple Slab Model Builder</h5>
            </div>
            <div class="card-body">
                <h5 class="card-title">Instructions</h5>
                <p class="card-text">
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
        <table class="table table-sm" v-if="dictionaryLoaded" id="sortable">
            <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
                <tr><th></th>
                    <th>Layer</th>
                    <th>Thickness</th>
                    <th>SLD</th>
                    <th v-if="showImaginary">iSLD</th>
                    <th>Interface</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(layer, key) in sortedLayers" :key="key">
                    <td  
                        @dragstart="dragStart(key, $event)"
                        @dragover="dragOver(key, $event)"
                        @drop="drop(key)"
                        @dragend="dragEnd"
                        draggable="true"
                        class="draggable"><span width=10px class="badge bg-secondary">:</span></td>
                    <td><input type="text" v-model="layer.material.name"></td>
                    <td><input v-if="get_slot(layer.thickness) !== null" type="number" step="5" v-model="get_slot(layer.thickness).value"></td>
                    <td><input v-if="get_slot(layer.material.rho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.rho).value"></td>
                    <td v-if="showImaginary"><input v-if="get_slot(layer.material.irho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.irho).value"></td>
                    <td><input v-if="get_slot(layer.interface) !== null" type="number" step="1" v-model="get_slot(layer.interface).value"></td>
                    <td><button class="btn btn-danger btn-sm" @click="delete_layer(key)">Delete</button></td>
                </tr>
            </tbody>
        </table>
    <p v-else>Load data to start building a model</p>
    </div>
       
    <button class="btn btn-success btn-sm me-2" @click="add_layer">Add layer</button>
    <div class="form-check form-switch m-2" @click="send_model">
        <input class="form-check-input" type="checkbox" id="showImaginary_input" v-model="showImaginary">
        <label class="form-check-label" for="showImaginary_input">Show imaginary SLD</label>
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
</style>
