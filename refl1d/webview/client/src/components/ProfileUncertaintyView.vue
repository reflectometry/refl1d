<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, computed, shallowRef } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket.ts';
import { setupDrawLoop } from 'bumps-webview-client/src/setupDrawLoop';
import { configWithSVGDownloadButton } from 'bumps-webview-client/src/plotly_extras.mjs';

import { cache } from '../plotcache';
import * as Plotly from 'plotly.js/lib/core';

const title = "Profile Uncertainty"
const plot_div = ref<HTMLDivElement>();
const hidden_download = ref<HTMLAnchorElement>();
const align = ref(0);
const auto_align = ref(true);
const show_residuals = ref(false);
const nshown = ref(50);
const npoints = ref(200);
const random = ref(true);
// don't use the one from setupDrawLoop because we are calling
// fetch_and_draw locally:
const drawing_busy = ref(false);

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('updated_uncertainty', props.socket, fetch_and_draw, title);

type PlotData = {
  data: Partial<Plotly.PlotData>[],
  layout: Partial<Plotly.Layout>,
}

type Payload  = {
  fig: PlotData,
  contour_data: {
    [model_name: string]: {
      z: number[],
      data: { [key: string]: number[][][] },
    }
  },
  contours: number[],
}

const contour_data = shallowRef<Payload["contour_data"]>({});
const contours = ref<number[]>([]);

function get_csv_data() {
  const data = contour_data.value;
  const contours_value = contours.value;
  if (Object.keys(data).length === 0) {
    return '';
  }
  const headers: string[] = [];
  const values: number[][] = [];
  for (const [model_name, model_data] of Object.entries(data)) {
    headers.push(`"${model_name} z"`);
    values.push(model_data.z);
    Object.keys(model_data.data).forEach((key) => {
      contours_value.forEach((c, ci) => {
        headers.push(`"${model_name} ${key} (${c} lower)"`);
        headers.push(`"${model_name} ${key} (${c} upper)"`);
        values.push(model_data.data[key][ci][0]);
        values.push(model_data.data[key][ci][1]);
      });
    });
  }
  const n = values[0].length;
  const lines = new Array(n + 1);
  lines[0] = headers.join(',');
  for (let i = 0; i < n; i++) {
    lines[i + 1] = values.map((v) => v[i].toPrecision(6)).join(',');
    //keys.map(k => data[k][i].toPrecision(6)).join(',');
  }
  return 'data:text/csv;charset=utf-8,' + encodeURIComponent(lines.join('\n'));
}

async function download_csv() {
  const a = hidden_download.value as HTMLAnchorElement;
  a.href = get_csv_data();
  a.click();
}

async function fetch_and_draw(latest_timestamp?: string) {
  let { timestamp, plotdata } = cache[title] as { timestamp: string, plotdata: PlotData } ?? {};
  const loading_delay = 50; // ms
  // if the plot loads faster than the timeout, don't show spinner
  const show_loader = setTimeout(() => {
    drawing_busy.value = true;
  }, loading_delay);
  if (latest_timestamp === undefined || timestamp !== latest_timestamp) {
    console.log("fetching new profile uncertainty plot", timestamp, latest_timestamp);
    const payload = await props.socket.asyncEmit('get_profile_uncertainty_plot', auto_align.value, align.value, nshown.value, npoints.value, random.value, show_residuals.value) as Payload;
    plotdata = { ...payload.fig };
    contour_data.value = payload.contour_data;
    contours.value = payload.contours;
    if (latest_timestamp !== undefined) {
      cache[title] = {timestamp: latest_timestamp, plotdata};
    }
  }

  const { data, layout } = plotdata;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true
    },
    ...configWithSVGDownloadButton
  }
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);

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
            <input class="form-check-input" type="checkbox" v-model="auto_align" id="auto-align" @change="fetch_and_draw()" />
          </div>
          <div>
            <label class="form-check-label pe-2" for="show_residuals">Residuals</label>
            <input class="form-check-input" type="checkbox" v-model="show_residuals" id="show_residuals" @change="fetch_and_draw()" />
          </div>
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="align-interface">Align interface</label>
          <input class="form-control" type="number" v-model="align" id="align-interface" @change="fetch_and_draw()" 
            :disabled="auto_align"
          />
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="n-shown">Num. shown</label>
          <input class="form-control" type="number" v-model="nshown" id="n-shown" @change="fetch_and_draw()" />
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="n-points">Num. points</label>
          <input class="form-control" type="number" v-model="npoints" id="n-points" @change="fetch_and_draw()" />
        </div>
        <div class="col-md-1 align-middle text-center">
          <label class="form-check-label pe-2" for="randomize">Random draw</label>
          <input class="form-check-input" type="checkbox" v-model="random" id="randomize" @change="fetch_and_draw()" />
        </div>
      </div>
      <div>
        <button class="btn btn-primary btn-sm" @click="download_csv">Download CSV</button>
        <a ref="hidden_download" class="hidden" download='contours.csv' type='text/csv'>Download CSV</a>
      </div>
    <!-- </details> -->
    <div class="flex-grow-1 position-relative">
      <div class="w-100 h-100 plot-div" ref="plot_div"></div>
      <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center loading" v-if="drawing_busy">
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