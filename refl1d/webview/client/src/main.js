import { createApp, computed } from 'vue'
import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css'
import { model_loaded, menu_items, fileBrowserSettings, fileBrowser, socket } from 'bumps-webview-client/src/app_state';
import App from 'bumps-webview-client/src/App.vue';
import { panels } from './panels.mjs';

const name = "Refl1D";

const model_not_loaded = computed(() => (!model_loaded.value));
async function loadProbeFromFile(ev) {
    if (fileBrowser.value) {
      const settings = fileBrowserSettings.value;
      settings.title = "Load Probe Data from File"
      settings.callback = (pathlist, filename) => {
        socket.value.asyncEmit("load_probe_from_file", pathlist, filename, 0);
      }
      settings.show_name_input = true;
      settings.name_input_label = "Filename";
      settings.require_name = true;
      settings.show_files = true;
      settings.chosenfile_in = "";
      settings.search_patterns = [".ort", ".orb", ".refl"];
      fileBrowser.value.open();
    }
  }

createApp(App, {panels, name}).mount('#app');
menu_items.value.push({ text: "Load Data into Model", action: loadProbeFromFile, disabled: model_not_loaded });
