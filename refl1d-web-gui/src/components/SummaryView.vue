<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import type { Socket } from 'socket.io-client';

const title = "Summary";

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

const parameters = shallowRef<{
  name: string,
  value01: number,
  value_str: string,
  min: string,
  max: string
}[]>([]);

const parameters_old = [{
  name: 'background',
  value01: 0.75,
  value_str: '1.7e-6',
  min: '1e-11',
  max: '1e-6'
}];

onMounted(() => {
  props.socket.on('plot_update_ready', () => {
    if (props.visible) {
      fetch_and_draw();
    }
  });
});

function fetch_and_draw() {
  props.socket.emit('get_parameters', (payload) => {
    console.log(payload);
    parameters.value = payload;
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
  <div class="container flex-grow-1">
    <div class="row border-bottom py-1">
      <div class="col-2">Fit Parameter</div>
      <div class="col-6"></div>
      <div class="col-2">Value</div>
      <div class="col-1">Minimum</div>
      <div class="col-1">Maximum</div>
    </div>
    <div class="row" v-for="param in parameters" :key="param.name">
      <div class="col-2">{{param.name}}</div>
      <div class="col-6"><input type="range" class="form-range" min="0" max="1.0" step="0.005" v-model="param.value01"/></div>
      <div class="col-2">{{param.value_str}}</div>
      <div class="col-1">{{param.min}}</div>
      <div class="col-1">{{param.max}}</div>
    </div>
  </div>
</template>
    
<style scoped>
svg {
  width: 100%;
}
</style>