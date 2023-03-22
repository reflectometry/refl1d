<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import { v4 as uuidv4 } from 'uuid';
import type { Socket } from 'socket.io-client';
import { setupDrawLoop } from '../setupDrawLoop';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);

const props = defineProps<{
  socket: Socket,
}>();

setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

const REFLECTIVITY_PLOTS = [
  "Fresnel",
  "Log Fresnel",
  "Linear",
  "Log",
  "Q4",
  "SA"
] as const;
type ReflectivityPlotEnum = typeof REFLECTIVITY_PLOTS;
type ReflectivityPlot = ReflectivityPlotEnum[number];
const reflectivity_type = ref<ReflectivityPlot>("Log");


async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_plot_data', reflectivity_type.value);
  // console.log(payload);
  const { data, layout } = payload;
  const config = {responsive: true}
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <select v-model="reflectivity_type" @change="fetch_and_draw">
      <option v-for="refl_type in REFLECTIVITY_PLOTS" :key="refl_type" :value="refl_type">{{refl_type}}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
