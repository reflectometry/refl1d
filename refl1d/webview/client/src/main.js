import "bootstrap/dist/css/bootstrap.min.css";
import { computed, createApp } from "vue";
import { file_menu_items, fileBrowser, shared_state } from "bumps-webview-client/src/app_state";
import App from "bumps-webview-client/src/App.vue";
import { io } from "socket.io-client";
import { dqIsFWHM } from "./app_state";
import { panels } from "./panels";
import "./style.css";

const name = "Refl1D";
const urlParams = new URLSearchParams(window.location.search);
const singlePanel = urlParams.get("single_panel");
const sio_base_path = urlParams.get("base_path") ?? window.location.pathname;
const sio_server = urlParams.get("server") ?? "";

const socket = io(sio_server, {
  path: `${sio_base_path}socket.io`,
});

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

createApp(App, { panels, socket, singlePanel, name }).mount("#app");
const modelNotLoaded = computed(() => shared_state.model_file == null);
file_menu_items.value.push({
  text: "Load Data into Model",
  action: loadProbeFromFile,
  disabled: modelNotLoaded,
});
