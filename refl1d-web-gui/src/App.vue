<script setup lang="ts">
import { Button } from 'bootstrap/dist/js/bootstrap.esm.js';
import { onMounted, ref, shallowRef, VNodeRef } from 'vue';
import { api } from './server_api';
import { io, Socket } from 'socket.io-client';
import FitOptions from './components/FitOptions.vue';
import DataView from './components/DataView.vue';
import ModelView from './components/ModelView.vue';
import PanelTabContainer from './components/PanelTabContainer.vue';
import FileBrowser from './components/FileBrowser.vue';
import SummaryView from './components/SummaryView.vue';
// import { FITTERS as FITTER_DEFAULTS } from './fitter_defaults';

const REFLECTIVITY_PLOTS = [
  "Fresnel",
  "Log Fresnel",
  "Linear",
  "Log",
  "Q4",
  "SA"
] as const;
type ReflectivityPlotEnum = typeof REFLECTIVITY_PLOTS;
type ReflectivityPlot = ReflectivityPlotEnum[number];

const reflectivity_type = ref<ReflectivityPlot>("Linear");
const connected = ref(false);
const menuToggle = ref<HTMLButtonElement>();
const fitOptions = ref<typeof FitOptions>();
const fileBrowser = ref<typeof FileBrowser>();
const fileBrowserSelectCallback = ref((pathlist: string[], filename: string) =>  {});
const file_picker = ref<HTMLInputElement>();

function set_reflectivity(refl_type: ReflectivityPlot) {
  reflectivity_type.value = refl_type;
}

// Create a SocketIO connection, to be passed to child components
// so that they can do their own communications with the host.

const socket = io('', {
  // this is mostly here to test what happens on server fail:
  reconnectionAttempts: 10
});
socket.on('connect', () => {
  console.log(socket.id);
  connected.value = true;
});

socket.on('disconnect', (payload) => {
  console.log("disconnected!", payload);
  // connected.value = false;
})

function disconnect() {
  socket.disconnect();
  connected.value = false;
}

function selectOpenFile() {
  if (fileBrowser.value) {
    fileBrowserSelectCallback.value = (pathlist, filename) => {
      socket.emit("load_model_file", pathlist, filename);
    }
    fileBrowser.value.open();
  }
  // const path = prompt("full path to file:");
  // if (path != null) {
  //   let defaulted_path = (path == '') ? '/home/bbm/dev/refl1d-modelbuilder/ISIS_GGG/GGG_GdIG_MultiFit.py' : path;
  //   socket.emit("load_model_file", defaulted_path);
  // }
  // file_picker.value?.click();
}

function load_model_path(event: Event) {
  console.log(event);
  const { files } = event.target;
  console.log(files);
  // socket.emit("load_model_file", files[0]);
  window.testevent = event;
}


function openFitOptions() {
  fitOptions.value?.open();
}

onMounted(() => {
  const menuToggleButton = new Button(menuToggle.value);
});

</script>

<template>
  <div class="h-100 d-flex flex-column">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark p-2">
      <div class="container-fluid">
        <div class="navbar-brand">
          <img src="./assets/refl1d-icon_256x256x32.png" alt="" height="36" class="d-inline-block align-text-middle">
          Refl1D
        </div>
        <button ref="menuToggle" class="navbar-toggler" type="button" data-bs-toggle="collapse"
          data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false"
          aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarSupportedContent">
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <!-- <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Session
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#" @click="connect">New</a></li>
                <li><a class="dropdown-item" href="#" @click="disconnect">Disconnect</a></li>
                <li><a class="dropdown-item" href="#" @click="reconnect">Existing</a></li>
              </ul>
            </li> -->
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                File
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#" @click="selectOpenFile">Open</a></li>
                <li><a class="dropdown-item" href="#">Save</a></li>
                <li>
                  <hr class="dropdown-divider">
                </li>
                <li><a class="dropdown-item" href="#">Quit</a></li>
              </ul>
            </li>
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Fitting
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#">Start</a></li>
                <li><a class="dropdown-item" href="#">Stop</a></li>
                <li>
                  <hr class="dropdown-divider">
                </li>
                <li><a class="dropdown-item" href="#" @click="openFitOptions">Options...</a></li>
              </ul>
            </li>
            <!-- <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Reflectivity
              </a>
              <ul class="dropdown-menu">
                <li v-for="plot_type in REFLECTIVITY_PLOTS" :key="plot_type">
                  <a :class="{'dropdown-item': true, active: (plot_type === reflectivity_type)}" href="#"
                    @click="set_reflectivity(plot_type)">{{plot_type}}</a>
                </li>
              </ul>
            </li> -->
          </ul>
          <div class="d-flex">
            <div id="connection_status"
              :class="{'btn': true, 'btn-outline-success': connected, 'btn-outline-danger': !connected}">
              {{(connected) ? 'connected' : 'disconnected'}}</div>

          </div>
        </div>
      </div>
    </nav>
    <div class="flex-grow-1 d-flex flex-row">
      <div class="flex-grow-1 d-flex flex-column">
        <PanelTabContainer :panels="[DataView, ModelView, SummaryView]" :socket="socket" />
      </div>
      <div class="flex-grow-1 d-flex flex-column">
        <PanelTabContainer :panels="[DataView, ModelView, SummaryView]" :socket="socket" />
      </div>
    </div>
  </div>
  <FitOptions ref="fitOptions" :socket="socket" />
  <FileBrowser ref="fileBrowser" :socket="socket" title="Load Model File" :callback="fileBrowserSelectCallback" />
  <input ref="file_picker" type="file" multiple="false" id="file_picker" style="display:none;" @change="load_model_path" />
</template>

<style>
html, body, #app {
  height: 100%;
}

div#connection_status {
  pointer-events: none;
}
</style>
