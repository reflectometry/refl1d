<script setup lang="ts">
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';


const title = "Builder";
// @ts-ignore: intentionally infinite type recursion
const modelJson = ref<json>({});
const parameters_by_id = ref({});
const dictionaryLoaded = ref(false);
const sortedLayers = ref([]);

const props = defineProps<{
  socket: AsyncSocket,
}>();

async function get_model_names() {
  model_names.value = await props.socket.asyncEmit("get_model_names");
}

props.socket.on('update_parameters', fetch_model);
props.socket.on('model_loaded', fetch_model);

async function fetch_model() {
    props.socket.asyncEmit('get_model', (payload: ArrayBuffer) => {
    const json_bytes = new Uint8Array(payload);
    const json_value = JSON.parse(decoder.decode(json_bytes));
    modelJson.value = json_value;
    extract_parameters(json_value);
    //console.log(json_value);
    //console.log(parameters_by_id.value);
    sortedLayers.value = modelJson.value['models'][0]['sample']['layers'];

    update_order_number();
    dictionaryLoaded.value = true;
  });
}

function extract_parameters(model) {
    const new_parameters_by_id = {};
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
    walk_object(model, null, '', '', get_parameters);
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
    // Remove layer if order is -1
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        if (item.order == -1) {
            sortedLayers.value.splice(index, 1);
        }
        item.name = item.material.name;
    };

    // Resort the layers by order
    sortedLayers.value.sort((a, b) => (a.order > b.order) ? 1 : -1);
    // Remove the order key
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        delete item.order;
    };

    const array_buffer = new TextEncoder().encode(JSON.stringify(modelJson.value));
    props.socket.emit('set_serialized_problem', array_buffer);
}

// Adding and deleting layers
function delete_layer(index) {
    sortedLayers.value.splice(index, 1);
    update_order_number();
};

function add_layer() {
    props.socket.emit('add_layer');
};

function update_order_number() {
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        item.order = index;
    };
};

const decoder = new TextDecoder('utf-8');

onMounted(() => {
    fetch_model();
})

// Code for draggable rows
const dragData = ref(null);

function dragStart(index, event) {
    dragData.value = index;
    console.log("grabbing", index);
    event.dataTransfer.setData('text/plain', index);
};
function dragOver(index, event) {
    event.preventDefault();
};
function drop(index) {
    console.log("dropping", index, dragData.value);
    if (dragData.value !== null) {
        const draggedDict = sortedLayers.value[dragData.value];
        sortedLayers.value.splice(dragData.value, 1);
        sortedLayers.value.splice(index, 0, draggedDict);
        dragData.value = null;
    }
    update_order_number();
};
function dragEnd() {
    dragData.value = null;
};

// Builder options
const showImaginary = ref(false);
</script>

<template>
<div id="builder">
    <div class="container m-2">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title">Simple Slab Model Builder</h5>
            </div>
            <div class="card-body">
                <h5 class="card-title">Instructions</h5>
                <p class="card-text">
                    <ul>
                        <li>Click the "Add row" button to add a new layer.</li>
                        <li>Drag and drop the rows to change the order of the layers.</li>
                        <li>Click the "Update model" button to send the model to the server.</li>
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
    <div class="badge bg-secondary p-2 m-2">
        <button class="btn btn-light btn-sm me-2" @click="add_layer">Add row</button>
        <button class="btn btn-success btn-sm" @click="send_model">Update model</button>
    </div>
    <div class="form-check form-switch m-2" @click="send_model">
        <input class="form-check-input" type="checkbox" id="showImaginary" v-model="showImaginary">
        <label class="form-check-label" for="showImaginary">Show imaginary SLD</label>
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
