<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import mpld3 from 'mpld3';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const draw_requested = ref(false);
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

onMounted(() => {
  props.socket.on('update_parameters', ({ message, timestamp }) => {
    if (props.visible) {
      draw_requested.value = true;
    }
  });
  window.requestAnimationFrame(draw_if_needed);
});

function fetch_and_draw() {
  props.socket.emit('get_profile_plot', (payload) => {
    if (props.visible) {
      let plotdata = { ...payload };
      console.log(plotdata, plot_div.value);
      plotdata.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
      plotdata.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
      // delete payload.width;
      // delete payload.height;
      /* Data Parsing Functions */
      // mpld3.draw_figure = function(figid, spec, process, clearElem) {}
      mpld3.draw_figure(plot_div_id.value, plotdata, false, true);
    }
  });
}

function draw_if_needed(timestamp: number) {
  if (draw_requested.value) {
    fetch_and_draw();
    draw_requested.value = false;
  }
  window.requestAnimationFrame(draw_if_needed);
}

watch(() => props.visible, (value) => {
  if (value) {
    console.log('visible', value);
    fetch_and_draw();
  }
});

</script>
    
<template>
  <div class="container d-flex flex-grow-1">
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>