import "bootstrap/dist/css/bootstrap.min.css";
import { computed, createApp } from "vue";
import { file_menu_items, fileBrowser, shared_state, socket } from "bumps-webview-client/src/app_state";
import App from "bumps-webview-client/src/App.vue";
import { dqIsFWHM } from "./app_state";
import { panels } from "./panels";
import "./style.css";

const name = "Refl1D";

async function loadProbeFromFile() {
  if (fileBrowser.value) {
    const settings = {
      title: "Load Probe Data from File",
      callback: (pathlist, filename) => {
        socket.value.asyncEmit("load_probe_from_file", pathlist, filename, 0, dqIsFWHM.value);
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      chosenfile_in: "",
      search_patterns: [".ort", ".orb", ".refl", ".dat", ".txt"],
    };
    fileBrowser.value.open(settings);
  }
}

createApp(App, { panels, name }).mount("#app");
const modelNotLoaded = computed(() => shared_state.model_file == null);
file_menu_items.value.push({
  text: "Load Data into Model",
  action: loadProbeFromFile,
  disabled: modelNotLoaded,
});
