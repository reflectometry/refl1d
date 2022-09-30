<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import Plotly from 'plotly.js/src/core';
import type { Socket } from 'socket.io-client';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement>();
const plot = ref(null);

const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

onMounted(() => {
  props.socket.on('plot_update_ready', () => {
    if (props.visible) {
      fetch_and_draw();
    }
  });
});

function fetch_and_draw() {
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

    if (plot.value) {
      const update = {
        x: [...theory_traces.map((t) => t.x), ...data_traces.map((t) => t.x)],
        y: [...theory_traces.map((t) => t.y), ...data_traces.map((t) => t.y)],       
      }
      Plotly.update(plot_div.value, update);
    }
    else {
      plot.value = Plotly.react(plot_div.value, [...theory_traces, ...data_traces], layout);
    }
    // console.log(plot.value);
    // mpld3.draw_figure("plot_div", payload);
  });
}

watch(() => props.visible, (value) => {
  if (value) {
    fetch_and_draw();
  }
});

defineExpose({
  title
});
</script>

<template>
  <div class="container d-flex flex-grow-1">
    <div ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
