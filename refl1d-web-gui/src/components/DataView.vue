<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import Plotly from 'plotly.js/src/core';
import type { Socket } from 'socket.io-client';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement>();

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

props.socket.on('plot_update_ready', () => {
  props.socket.emit('get_plot_data', 'linear', (payload) => {

    let theory_traces = [];
    let data_traces = [];
    for (let model of payload) {
      for (let xs of model) {
        theory_traces.push({ x: xs.Q, y: xs.theory, mode: 'lines', name: xs.label + ' theory', line: {width: 2}});
        data_traces.push({ x: xs.Q, y: xs.R, mode: 'markers', name: xs.label + ' data' });
      }
    }

    const layout = {
      xaxis: {
        title: {
          text: '$Q (Å^{-1})$'
        },
        type: 'linear',
        autorange: true
      },
      yaxis: {
        title: {text: 'Reflectivity'},
        exponentformat: 'e',
        showexponent: 'all',
        type: 'log',
        autorange: true
      }
    };

    Plotly.newPlot("plot_div", [...theory_traces, ...data_traces], layout);

    // mpld3.draw_figure("plot_div", payload);
  });
});

defineExpose({
  title
});
</script>

<template>
  <div class="container h-100">
    <div ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
