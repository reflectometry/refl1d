<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { onMounted, ref } from "vue";
import type { AsyncSocket } from "bumps-webview-client/src/asyncSocket";
import { configWithSVGDownloadButton } from "bumps-webview-client/src/plotly_extras";
import { setupDrawLoop } from "bumps-webview-client/src/setupDrawLoop";
import * as Plotly from "plotly.js/lib/core";

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
interface ModelNameInfo {
  name: string;
  part_name: string;
  model_index: number;
  part_index: number;
}
const model_names = ref<ModelNameInfo[]>([]);
const show_multiple = ref(false);
const current_models = ref<Array<[number, number]>>([[0, 0]]);

const props = defineProps<{
  socket: AsyncSocket;
}>();

const { draw_requested } = setupDrawLoop("updated_parameters", props.socket, fetch_and_draw, title);

async function get_model_names() {
  model_names.value = (await props.socket.asyncEmit("get_model_names")) as ModelNameInfo[];
}

props.socket.on("model_loaded", get_model_names);

onMounted(async () => {
  await get_model_names();
});

async function fetch_and_draw() {
  const specs = current_models.value.map(([model_index, sample_index]) => ({ model_index, sample_index }));
  const payload = (await props.socket.asyncEmit("get_profile_plots", specs)) as {
    data: Partial<Plotly.PlotData>[];
    layout: Partial<Plotly.Layout>;
  };
  const plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true,
    },
    ...configWithSVGDownloadButton,
  };
  if (plot_div.value === undefined) {
    return;
  }
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
}

function toggle_multiple() {
  if (!show_multiple.value) {
    // then we're toggling from multiple to single...
    current_models.value.splice(0, current_models.value.length - 1);
    draw_requested.value = true;
  }
  Plotly.Plots.resize(plot_div.value as HTMLDivElement);
}

function requestRedraw() {
  draw_requested.value = true;
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="form-check">
      <label class="form-check-label pe-2" for="multiple">Show Multiple</label>
      <input id="multiple" v-model="show_multiple" class="form-check-input" type="checkbox" @change="toggle_multiple" />
    </div>
    <label for="model" class="form-label"> Models:</label>
    <select v-if="show_multiple" id="model" v-model="current_models" multiple @change="requestRedraw">
      <option
        v-for="{ name, part_name, model_index, part_index } in model_names"
        :key="`${model_index}:${part_index}`"
        :value="[model_index, part_index]"
      >
        {{ model_index }}:{{ part_index }} --- {{ name ?? "" }}:{{ part_name ?? "" }}
      </option>
    </select>
    <select v-else id="model" v-model="current_models[0]" @change="requestRedraw">
      <option
        v-for="{ name, part_name, model_index, part_index } in model_names"
        :key="`${model_index}:${part_index}`"
        :value="[model_index, part_index]"
      >
        {{ model_index }}:{{ part_index }} --- {{ name ?? "" }}:{{ part_name ?? "" }}
      </option>
    </select>
    <div ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
