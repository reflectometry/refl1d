<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import type { Socket } from 'socket.io-client';
import mpld3 from 'mpld3';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

onMounted(() => {
  props.socket.on('plot_update_ready', () => {
    if (props.visible) {
      fetch_and_draw();
    }
  });
});

function fetch_and_draw() {
  props.socket.emit('get_profile_plot', (payload) => {
    console.log(payload, plot_div.value);
    payload.width = plot_div.value?.clientWidth ?? 640;
    payload.height = plot_div.value?.clientHeight ?? 480;
    // delete payload.width;
    // delete payload.height;
    console.log(payload, plot_div.value);
    mpld3.draw_figure("profile_div", payload);
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
    <div class="flex-grow-1" ref="plot_div" id="profile_div">

    </div>
  </div>
</template>

<style scoped>
  svg {
    width: 100%;
  }
</style>