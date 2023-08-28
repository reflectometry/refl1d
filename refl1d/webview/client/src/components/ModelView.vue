<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from 'bumps-webview-client/src/setupDrawLoop';
import * as Plotly from 'plotly.js/lib/core';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const model_names = ref<{name: string, part_name: string, model_index: number, part_index: number}[]>([]);
const show_multiple = ref(false);
const current_models = ref<Array<[number, number]>>([[0, 0]]);

const props = defineProps<{
  socket: AsyncSocket,
}>();

const { draw_requested } = setupDrawLoop('update_parameters', props.socket, fetch_and_draw, title);

async function get_model_names() {
  model_names.value = await props.socket.asyncEmit("get_model_names");
}

props.socket.on('model_loaded', get_model_names);

onMounted(async () => {
  await get_model_names();
});

async function fetch_and_draw() {
  const specs = current_models.value.map(([model_index, sample_index]) => (
    {model_index, sample_index}
  ));
  const payload = await props.socket.asyncEmit('get_profile_plots', specs);
  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true
    }
  }
  await Plotly.react(plot_div_id.value, [...data], layout, config);
}

function toggle_multiple(value) {
  if (!show_multiple.value) {
    // then we're toggling from multiple to single...
    current_models.value.splice(0, current_models.value.length -1);
    draw_requested.value = true;
  }
}

</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="form-check">
      <label class="form-check-label pe-2" for="multiple">Show multiple</label>
      <input class="form-check-input" type="checkbox" v-model="show_multiple" id="multiple" @change="toggle_multiple"/>
    </div>
    <select v-if="show_multiple"
      v-model="current_models"
      @change="draw_requested = true"
      multiple
      >
      <option v-for="{name, part_name, model_index, part_index} in model_names" :key="`${model_index}:${part_index}`" :value="[model_index, part_index]">
        {{ model_index }}:{{ part_index }} --- {{ name ?? "" }}:{{ part_name ?? "" }}</option>
    </select>
    <select v-else
      v-model="current_models[0]"
      @change="draw_requested = true"
      >
      <option v-for="{name, part_name, model_index, part_index} in model_names" :key="`${model_index}:${part_index}`" :value="[model_index, part_index]">
        {{ model_index }}:{{ part_index }} --- {{ name ?? "" }}:{{ part_name ?? "" }}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
