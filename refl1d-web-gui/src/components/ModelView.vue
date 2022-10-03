<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import mpld3 from 'mpld3';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const draw_requested = ref(false);

const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

onMounted(() => {
  props.socket.on('plot_update_ready', ({ message, timestamp }) => {
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
      plotdata.width = Math.round(plot_div.value?.clientWidth ?? 640) - 32;
      plotdata.height = Math.round(plot_div.value?.clientHeight ?? 480) - 32;
      // delete payload.width;
      // delete payload.height;
      /* Data Parsing Functions */
      // mpld3.draw_figure = function(figid, spec, process, clearElem) {}
      mpld3.draw_figure("profile_div", plotdata, false, true);
    }
  });
}

function draw_if_needed(timestamp: number) {
  if (draw_requested.value) {
    draw_requested.value = false;
    fetch_and_draw();
  }
  window.requestAnimationFrame(draw_if_needed);
}

watch(() => props.visible, (value) => {
  if (value) {
    console.log('visible', value);
    fetch_and_draw();
  }
});

defineExpose({
  title
});

</script>
    
<template>
  <div class="container d-flex flex-grow-1">
    <div class="flex-grow-1" ref="plot_div" id="profile_div">
    </div>
  </div>
</template>

<style scoped>
  svg {
    width: 100%;
  }
</style>