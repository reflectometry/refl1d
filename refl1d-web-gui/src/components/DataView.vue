<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import type { Socket } from 'socket.io-client';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement | null>(null);
const plot = ref<Plotly.PlotlyHTMLElement>();

const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

onMounted(() => {
  props.socket.on('update_parameters', () => {
    fetch_and_draw();
  });
});

function fetch_and_draw() {
  if (!props.visible) {
    return
  }
  props.socket.emit('get_plot_data', 'linear', async (payload) => {
    if (plot_div.value === null) {
      return
    }
    let theory_traces: (Plotly.Data & { x: number, y: number })[] = [];
    let data_traces: (Plotly.Data & { x: number, y: number })[] = [];
    for (let model of payload) {
      for (let xs of model) {
        theory_traces.push({ x: xs.Q, y: xs.theory, mode: 'lines', name: xs.label + ' theory', line: { width: 2 } });
        data_traces.push({ x: xs.Q, y: xs.R, mode: 'markers', name: xs.label + ' data' });
      }
    }

    const layout: Partial<Plotly.Layout> = {
      xaxis: {
        title: {
          text: '$Q (Å^{-1})$'
        },
        type: 'linear',
        autorange: true
      },
      yaxis: {
        title: { text: 'Reflectivity' },
        exponentformat: 'e',
        showexponent: 'all',
        type: 'log',
        autorange: true
      },
    };

    if (plot.value !== undefined) {
      const trace_updates = {
        x: [...theory_traces.map((t) => t.x), ...data_traces.map((t) => t.x)],
        y: [...theory_traces.map((t) => t.y), ...data_traces.map((t) => t.y)],
      }
      await Plotly.update(plot_div.value, trace_updates, layout);
    }
    else {
      plot.value = await Plotly.react(plot_div.value, [...theory_traces, ...data_traces], layout);
    }
  });
}

watch(() => props.visible, (value) => {
  if (value) {
    fetch_and_draw();
  }
});

</script>

<template>
  <div class="container d-flex flex-grow-1">
    <div class="flex-grow-1" ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
