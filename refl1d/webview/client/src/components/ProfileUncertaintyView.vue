<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from 'bumps-webview-client/src/setupDrawLoop';
import { cache } from '../plotcache';
import mpld3 from 'mpld3';

const title = "Profile Uncertainty"
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const align = ref(0);
const auto_align = ref(true);
const nshown = ref(50);
const npoints = ref(200);
const random = ref(true);


const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('uncertainty_update', props.socket, fetch_and_draw, title);

type MplD3PlotData = {
  width?: number,
  height?: number,
}

async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = cache[title] as { timestamp: string, plotdata: MplD3PlotData } ?? {};
  if (timestamp !== latest_timestamp) {
    console.log("fetching new profile uncertainty plot", timestamp, latest_timestamp);
    const payload = await props.socket.asyncEmit('get_profile_uncertainty_plot', auto_align.value, align.value, nshown.value, npoints.value, random.value) as MplD3PlotData;
    plotdata = { ...payload };
    cache[title] = {timestamp: latest_timestamp, plotdata};
  }
  plotdata.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
  plotdata.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
  mpld3.draw_figure(plot_div_id.value, plotdata, false, true);
}

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <!-- <details open>
      <summary>Settings</summary> -->
        <div class="row g-3">
        <div class="col-md-2 align-middle text-end">
          <label class="form-check-label pe-2" for="auto-align">Auto-align</label>
          <input class="form-check-input" type="checkbox" v-model="auto_align" id="auto-align" @change="fetch_and_draw" />
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="align-interface">Align interface</label>
          <input class="form-control" type="number" v-model="align" id="align-interface" @change="fetch_and_draw" 
            :disabled="auto_align"
          />
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="n-shown">Num. shown</label>
          <input class="form-control" type="number" v-model="nshown" id="n-shown" @change="fetch_and_draw" />
        </div>
        <div class="col-md-3 align-middle">
          <label class="form-label" for="n-points">Num. points</label>
          <input class="form-control" type="number" v-model="npoints" id="n-points" @change="fetch_and_draw" />
        </div>
        <div class="col-md-1 align-middle text-center">
          <label class="form-check-label pe-2" for="randomize">Random draw</label>
          <input class="form-check-input" type="checkbox" v-model="random" id="randomize" @change="fetch_and_draw" />
        </div>
    </div>
    <!-- </details> -->
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>