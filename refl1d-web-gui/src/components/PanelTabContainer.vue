<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import type { Component } from 'vue';
import type { Socket } from 'socket.io-client';
import { Tab } from 'bootstrap/dist/js/bootstrap.esm';

type PanelComponent = Component & { title: string };

const tabTriggers = ref<HTMLAnchorElement[]>([]);
const panelComponents = ref<PanelComponent[]>([]);
const panelContainers = ref<HTMLDivElement[]>([]);
const panelVisible = ref<boolean[]>([]);

const props = defineProps<{
  socket: Socket,
  panels: PanelComponent[]
}>();

let triggers: Tab[] = [];

function activateTab(index) {
  triggers?.[index].show();
  panelContainers.value.forEach((el, i) => {
    if (i === index) {
      el.classList.add('active');
      panelVisible.value[i] = true;
    }
    else {
      el.classList.remove('active');
      panelVisible.value[i] = false;
    }
  });
}

onMounted(() => {
  for (let triggerEl of tabTriggers.value) {
    let tabTrigger = new Tab(triggerEl);
    triggers.push(tabTrigger);
  }
  if (triggers.length > 0) {
    activateTab(0);
  }
});
</script>

<template>
  <ul class="nav nav-tabs">
    <li class="nav-item" v-for="(panel, index) in props.panels" :key="index">
      <a ref="tabTriggers" class="nav-link" href="#" @click="activateTab(index)">{{panelComponents[index]?.title}}</a>
    </li>
  </ul>
  <div class="tab-content d-flex flex-column flex-grow-1">
    <div ref="panelContainers" class="tab-pane flex-column flex-grow-1" v-for="(panel, index) in props.panels"
      :key="index">
      <component :is="panel" :socket="props.socket" ref="panelComponents" :visible="panelVisible[index]"></component>
    </div>
  </div>
</template>

<style scoped>
.tab-content>.tab-pane {
  display: none;
}
.tab-content>.tab-pane.active {
  display: flex !important;
}
</style>