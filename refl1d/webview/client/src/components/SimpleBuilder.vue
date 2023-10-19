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
    console.log("in fetch_model");
    props.socket.asyncEmit('get_model', (payload: ArrayBuffer) => {
    const json_bytes = new Uint8Array(payload);
    const json_value = JSON.parse(decoder.decode(json_bytes));
    modelJson.value = json_value;
    extract_parameters(json_value);
    console.log(json_value);
    console.log(parameters_by_id.value);
    sortedLayers.value = modelJson.value['models'][0]['sample']['layers'];

    for (const [index, item] of Object.entries(sortedLayers.value)) {
        item.order = index;
    };
    dictionaryLoaded.value = true;
  });
}

function extract_parameters(model) {
    const new_parameters_by_id = {};
    const get_parameters = (obj, parent_obj, path, key) => {
        if (obj && obj?.type === 'bumps.parameter.Parameter') {
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
    console.log(parameter_like, slot);
    return slot;
}

function send_model() {
    // Remove layer if order is -1
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        if (item.order == -1) {
            sortedLayers.value.splice(index, 1);
        }
    };

    // Resort the layers by order
    sortedLayers.value.sort((a, b) => (a.order > b.order) ? 1 : -1);
    // Remove the order key
    for (const [index, item] of Object.entries(sortedLayers.value)) {
        delete item.order;
    };

    const array_buffer = new TextEncoder().encode(JSON.stringify(modelJson.value));
    props.socket.emit('set_model', array_buffer);
}

function add_layer() {
    props.socket.emit('add_layer');
};

const decoder = new TextDecoder('utf-8');

onMounted(() => {
    fetch_model();
})
</script>

<template>
<div id="builder">
    <div class="badge bg-secondary p-1">
        <button class="btn btn-light btn-sm me-2" @click="add_layer">Add row</button>
        <button class="btn btn-success btn-sm" @click="send_model">Update model</button>
    </div>
    <table class="table table-sm" v-if="dictionaryLoaded">
        <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
            <tr>
                <th>Order</th>
                <th>Layer</th>
                <th>Thickness</th>
                <th>SLD</th>
                <th>iSLD</th>
                <th>Interface</th>
            </tr>
        </thead>
        <tbody>
            <tr class="py-1" v-for="(layer, key) in sortedLayers" :key="key">
                <td><input type="number" step="1" v-model="layer.order"></td>
                <td><input type="text" v-model="layer.name"></td>
                <td><input v-if="get_slot(layer.thickness) !== null" type="number" step="5" v-model="get_slot(layer.thickness).value"></td>
                <td><input v-if="get_slot(layer.material.rho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.rho).value"></td>
                <td><input v-if="get_slot(layer.material.irho) !== null" type="number" step="0.01" v-model="get_slot(layer.material.irho).value"></td>
                <td><input v-if="get_slot(layer.interface) !== null" type="number" step="1" v-model="get_slot(layer.interface).value"></td>
            </tr>
        </tbody>
    </table>
    <p v-else>Click Load Dictionary</p>
</div>
</template>

<style scoped>
    svg {
      width: 50%;
    }
</style>
