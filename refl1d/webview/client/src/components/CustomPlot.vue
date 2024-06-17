<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, shallowRef } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket.ts';
import { setupDrawLoop } from 'bumps-webview-client/src/setupDrawLoop';
import { COLORS } from '../colors.mjs';
import { configWithSVGDownloadButton } from 'bumps-webview-client/src/plotly_extras.mjs';

const title = "Custom";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_data = shallowRef<Partial<Plotly.Data>>({});
const plot_layout = shallowRef<Partial<Plotly.Layout>>({});

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('updated_parameters', props.socket, fetch_and_draw);

async function fetch_and_draw() {
  //const payload = await props.socket.asyncEmit('hello_world', 'linear') as {plotdata: Partial<Plotly.PlotData>};
  const payload = await props.socket.asyncEmit('get_custom_plot') as { data: Partial<Plotly.PlotData>[], layout: Partial<Plotly.Layout> };
  let plotdata = { ...payload };
  const { data, layout } = plotdata;

  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true
    }, 
    ...configWithSVGDownloadButton
  }
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);

}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="flex-grow-1" ref="plot_div" id="plot_div">
    </div>
  </div>
</template>

<style scoped>
.plot-mode {
  width: 100%;
}
</style>