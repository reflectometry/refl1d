<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, shallowRef } from "vue";
import type { AsyncSocket } from "bumps-webview-client/src/asyncSocket";
import { configWithSVGDownloadButton } from "bumps-webview-client/src/plotly_extras";
import { setupDrawLoop } from "bumps-webview-client/src/setupDrawLoop";
import * as Plotly from "plotly.js/lib/core";
import { cache } from "../plot_cache";

const title = "Profile Uncertainty";
const plotDiv = ref<HTMLDivElement>();
const hiddenDownload = ref<HTMLAnchorElement>();
const align = ref(0);
const autoAlign = ref(true);
const show_residuals = ref(false);
const nshown = ref(50);
const npoints = ref(200);
const random = ref(true);
// don't use the one from setupDrawLoop because we are calling
// fetch_and_draw locally:
const drawing_busy = ref(false);

const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_uncertainty", props.socket, fetch_and_draw, title);

type PlotData = {
  data: Partial<Plotly.PlotData>[];
  layout: Partial<Plotly.Layout>;
};

type Payload = {
  fig: PlotData;
  contour_data: {
    [model_name: string]: {
      z: number[];
      data: {
        [key: string]: {
          [contour_label: string]: number[];
        };
      };
    };
  };
  contours: number[];
};

const contour_data = shallowRef<Payload["contour_data"]>({});
const contours = ref<number[]>([]);

function get_csv_data() {
  const data = contour_data.value;
  if (Object.keys(data).length === 0) {
    return "";
  }
  const headers: string[] = [];
  const values: number[][] = [];
  for (const [model_name, model_data] of Object.entries(data)) {
    headers.push(`"${model_name} z"`);
    values.push(model_data.z);
    Object.entries(model_data.data).forEach(([key, value]) => {
      Object.entries(value).forEach(([c, v]) => {
        headers.push(`"${model_name} ${key} ${c}"`);
        values.push(v);
      });
    });
  }
  const n = values[0].length;
  const lines = new Array(n + 1);
  lines[0] = headers.join(",");
  for (let i = 0; i < n; i++) {
    lines[i + 1] = values.map((v) => v[i].toPrecision(6)).join(",");
    //keys.map(k => data[k][i].toPrecision(6)).join(',');
  }
  return "data:text/csv;charset=utf-8," + encodeURIComponent(lines.join("\n"));
}

async function download_csv() {
  const a = hiddenDownload.value as HTMLAnchorElement;
  a.href = get_csv_data();
  a.click();
}

async function fetch_and_draw(latest_timestamp?: string) {
  let { timestamp, plotData } = (cache[title] as { timestamp: string; plotData: PlotData }) ?? {};
  const loading_delay = 50; // ms
  // if the plot loads faster than the timeout, don't show spinner
  const show_loader = setTimeout(() => {
    drawing_busy.value = true;
  }, loading_delay);
  if (latest_timestamp === undefined || timestamp !== latest_timestamp) {
    console.log("fetching new profile uncertainty plot", timestamp, latest_timestamp);
    const payload = (await props.socket.asyncEmit(
      "get_profile_uncertainty_plot",
      autoAlign.value,
      align.value,
      nshown.value,
      npoints.value,
      random.value,
      show_residuals.value
    )) as Payload;
    plotData = { ...payload.fig };
    contour_data.value = payload.contour_data;
    contours.value = payload.contours;
    if (latest_timestamp !== undefined) {
      cache[title] = { timestamp: latest_timestamp, plotData };
    }
  }

  const { data, layout } = plotData;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true,
    },
    ...configWithSVGDownloadButton,
  };
  await Plotly.react(plotDiv.value as HTMLDivElement, [...data], layout, config);

  clearTimeout(show_loader);
  drawing_busy.value = false;
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <!-- <details open>
      <summary>Settings</summary> -->
    <div class="row g-3">
      <div class="col-md-2 align-middle text-end">
        <div>
          <label class="form-check-label pe-2" for="auto-align">Auto-align</label>
          <input
            id="auto-align"
            v-model="autoAlign"
            class="form-check-input"
            type="checkbox"
            @change="fetch_and_draw()"
          />
        </div>
        <div>
          <label class="form-check-label pe-2" for="show_residuals">Residuals</label>
          <input
            id="show_residuals"
            v-model="show_residuals"
            class="form-check-input"
            type="checkbox"
            @change="fetch_and_draw()"
          />
        </div>
      </div>
      <div class="col-md-3 align-middle">
        <label class="form-label" for="align-interface">Align interface</label>
        <input
          id="align-interface"
          v-model="align"
          class="form-control"
          type="number"
          :disabled="autoAlign"
          @change="fetch_and_draw()"
        />
      </div>
      <div class="col-md-3 align-middle">
        <label class="form-label" for="n-shown">Num. shown</label>
        <input id="n-shown" v-model="nshown" class="form-control" type="number" @change="fetch_and_draw()" />
      </div>
      <div class="col-md-3 align-middle">
        <label class="form-label" for="n-points">Num. points</label>
        <input id="n-points" v-model="npoints" class="form-control" type="number" @change="fetch_and_draw()" />
      </div>
      <div class="col-md-1 align-middle text-center">
        <label class="form-check-label pe-2" for="randomize">Random draw</label>
        <input id="randomize" v-model="random" class="form-check-input" type="checkbox" @change="fetch_and_draw()" />
      </div>
    </div>
    <div>
      <button class="btn btn-primary btn-sm" @click="download_csv">Download CSV</button>
      <a ref="hiddenDownload" class="hidden" download="contours.csv" type="text/csv">Download CSV</a>
    </div>
    <!-- </details> -->
    <div class="flex-grow-1 position-relative">
      <div ref="plotDiv" class="w-100 h-100 plot-div"></div>
      <div
        v-if="drawing_busy"
        :class="{
          'position-absolute': true,
          'top-0': true,
          'start-0': true,
          'w-100': true,
          'h-100': true,
          'd-flex': true,
          'flex-column': true,
          'align-items-center': true,
          'justify-content-center': true,
          loading: true,
        }"
      >
        <span class="spinner-border text-primary"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
.hidden {
  display: none;
}
span.spinner-border {
  width: 3rem;
  height: 3rem;
}
</style>
