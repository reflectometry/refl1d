import { createApp, computed } from 'vue'
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'
import { model_file, menu_items, FileBrowserSettings, fileBrowser, socket } from 'bumps-webview-client/src/app_state';
import App from 'bumps-webview-client/src/App.vue';
import { panels } from './panels.mjs';
import { dq_is_FWHM } from './app_state';

const name = "Refl1D";

async function loadProbeFromFile(ev) {
  if (fileBrowser.value) {
    const settings = {
      title: "Load Probe Data from File",
      callback: (pathlist, filename) => {
        socket.value.asyncEmit("load_probe_from_file", pathlist, filename, 0, dq_is_FWHM.value);
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      chosenfile_in: "",
      search_patterns: [".ort", ".orb", ".refl", ".dat", ".txt"]
    };
    fileBrowser.value.open(settings);
  }
}

createApp(App, {panels, name}).mount('#app');
const model_not_loaded = computed(() => model_file.value == null);
menu_items.value.push({ text: "Load Data into Model", action: loadProbeFromFile, disabled: model_not_loaded });