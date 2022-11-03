<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import * as Plotly from 'plotly.js/lib/core';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const draw_requested = ref(false);
const plot_div_id = ref(`div-${uuidv4()}`);
const model_names = ref<string[]>([]);
const current_model = ref(0);
const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

props.socket.on('model_loaded', ({message: {model_names: new_model_names}}) => {
  model_names.value = new_model_names;
});

onMounted(() => {
  props.socket.on('update_parameters', () => {
    draw_requested.value = true;
  });
  window.requestAnimationFrame(draw_if_needed);
});

function generate_new_traces(profile_data) {
  let traces: (Plotly.Data & { x: number, y: number })[] = [];
  const { step_profile, smooth_profile } = profile_data;

  traces.push({ x: step_profile.z, y: step_profile.rho, mode: 'lines', name: 'rho', line: {width: 2}})
  traces.push({ x: step_profile.z, y: step_profile.irho, mode: 'lines', name: 'irho', line: {width: 2}});
  if (step_profile.rho_m) {
    traces.push({ x: step_profile.z, y: step_profile.rho_m, mode: 'lines', name: 'rho_m', line: {width: 2}});
  }

  traces.push({ x: smooth_profile.z, y: smooth_profile.rho, mode: 'lines', name: 'rho', line: {width: 2}})
  traces.push({ x: smooth_profile.z, y: smooth_profile.irho, mode: 'lines', name: 'irho', line: {width: 2}});
  if (smooth_profile.rho_m) {
    traces.push({ x: smooth_profile.z, y: smooth_profile.rho_m, mode: 'lines', name: 'rho_m', line: {width: 2}});
  }

  return traces;
}



function fetch_and_draw() {
  if (!props.visible) {
    return
  }
  props.socket.emit('get_profile_data', current_model.value, async (payload) => {
    if (plot_div.value == null) {
      return
    }
    console.log(payload);
    const traces = generate_new_traces(payload);

    const layout: Partial<Plotly.Layout> = {
      uirevision: 1,
      xaxis: {
        title: {
          text: 'z'
        },
        type: 'linear',
        autorange: true,
      },
      yaxis: {
        title: { text: 'SLD (10^6 / A^{-2})' },
        exponentformat: 'e',
        showexponent: 'all',
        type: 'linear',
        autorange: true,
      },
    };

    await Plotly.react(plot_div.value, [...traces], layout);
    
  });
}


function draw_if_needed(timestamp: number) {
  if (draw_requested.value && props.visible) {
    fetch_and_draw();
    draw_requested.value = false;
  }
  window.requestAnimationFrame(draw_if_needed);
}

// watch(() => props.visible, (value) => {
//   if (value) {
//     console.log('visible', value);
//     fetch_and_draw();
//   }
// });

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <select v-model="current_model" @change="draw_requested=true">
      <option v-for="(name, index) in model_names" :key="index" :value="index">{{index}}: {{name ?? ""}}</option>
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