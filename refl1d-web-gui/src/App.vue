<script setup lang="ts">
import { Button } from 'bootstrap/dist/js/bootstrap.esm.js';
import { onMounted, ref, shallowRef } from 'vue';
import { api } from './server_api';
import { io, Socket } from 'socket.io-client';
import FitOptions from './components/FitOptions.vue';
// import { FITTERS as FITTER_DEFAULTS } from './fitter_defaults';
const FITTER_DEFAULTS = {};

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
const connection_id = ref("");
const connection = ref<Socket | null>(null);
const connected = ref(false);
const menuToggle = ref(null);
const fitOptions = ref(null);
const fitter_settings = shallowRef(structuredClone(FITTER_DEFAULTS));
const fitter_defaults = shallowRef(FITTER_DEFAULTS);
const active_fitter = ref('amoeba');

function set_reflectivity(refl_type: ReflectivityPlot) {
  reflectivity_type.value = refl_type;
}

function load_model_path(path: string) {

}

function connect() {
  const socket = io('http://localhost:8080', {
    reconnectionAttempts: 10
  });
  connection.value = socket;
  socket.on('connect', () => {
    console.log(socket.id);
    connected.value = true;
  });

  socket.on('fitter-settings', (payload) => {
    fitter_settings.value = payload;
  });
  socket.on('fitter-defaults', (payload) => {
    fitter_defaults.value = payload;
  });
  socket.on('fitter-active', (payload) => {
    active_fitter.value = payload;
  });
  socket.on('disconnect', (payload) => {
    console.log("disconnected!", payload);
    connected.value = false;
  })
}

function disconnect() {
  connection.value?.disconnect();
  connected.value = false;
}

function openFitOptions() {
  fitOptions?.value?.open();
}

function get_sessions() {
  fetch('http://localhost:8080/get_sessions');
}

function updateActiveFitter(fitter_name) {
  connection.value?.emit("fitter-active", fitter_name);
}

function updateActiveSettings(new_settings) {
  connection.value?.emit("fitter-settings", new_settings);
}

onMounted(() => {
  const menuToggleButton = new Button(menuToggle.value);
  connect();
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
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Session
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#" @click="connect">New</a></li>
                <li><a class="dropdown-item" href="#" @click="disconnect">Disconnect</a></li>
                <li><a class="dropdown-item" href="#" @click="reconnect">Existing</a></li>
              </ul>
            </li>
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                File
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="#">Open</a></li>
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
            <li class="nav-item dropdown">
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
            </li>
          </ul>
          <div class="d-flex">
            <div id="connection_status" :class="{'btn': true, 'btn-outline-success': connected, 'btn-outline-danger': !connected}">
              {{(connected) ? 'connected' : 'disconnected'}}</div>

          </div>
        </div>
      </div>
    </nav>
  </div>
  <FitOptions ref="fitOptions" :fitter_defaults="fitter_defaults" :fitter_settings="fitter_settings"
    :active_fitter="active_fitter" @active-fitter="updateActiveFitter" @active-settings="updateActiveSettings"/>
</template>

<style scoped>
div#connection_status {
  pointer-events: none;
}
</style>
