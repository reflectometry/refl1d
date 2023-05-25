<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import type { AsyncSocket } from 'bumps-webview-client/src/asyncSocket';
import { setupDrawLoop } from 'bumps-webview-client/src/setupDrawLoop';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement | null>(null);

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

const REFLECTIVITY_PLOTS = [
  "Fresnel",
  "Log Fresnel",
  "Linear",
  "Log",
  "Q4",
  "Spin Asymmetry"
] as const;
type ReflectivityPlotEnum = typeof REFLECTIVITY_PLOTS;
type ReflectivityPlot = ReflectivityPlotEnum[number];
const reflectivity_type = ref<ReflectivityPlot>("Log");

const COLORS = [
  '#1f77b4',  // muted blue
  '#ff7f0e',  // safety orange
  '#2ca02c',  // cooked asparagus green
  '#d62728',  // brick red
  '#9467bd',  // muted purple
  '#8c564b',  // chestnut brown
  '#e377c2',  // raspberry yogurt pink
  '#7f7f7f',  // middle gray
  '#bcbd22',  // curry yellow-green
  '#17becf'   // blue-teal
];

function generate_new_traces(model_data, view: ReflectivityPlot) {
  let theory_traces: (Plotly.Data & { x: number, y: number })[] = [];
  let data_traces: (Plotly.Data & { x: number, y: number })[] = [];
  switch (view) {
    case "Log":
    case "Linear": {
      let color_index = 0;
      for (let model of model_data) {
        for (let xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          theory_traces.push({ x: xs.Q, y: xs.theory, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          if (xs.R !== undefined) {
            const data_trace = { x: xs.Q, y: xs.R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }};
            if (xs.dR !== undefined) {
              data_trace.error_y = {type: 'data', array: xs.dR, visible: true};
            }
            data_traces.push(data_trace);
          }
          color_index = (color_index + 1) % COLORS.length;
        }
      }
      break;
    }
    case "Log Fresnel":
    case "Fresnel": {
      let color_index = 0;
      for (let model of model_data) {
        for (let xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const theory = xs.theory.map((y, i) => (y / (xs.fresnel[i])));
          theory_traces.push({ x: xs.Q, y: theory, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          if (xs.R !== undefined) {
            const R = xs.R.map((y, i) => (y / (xs.fresnel[i])));
            data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }});
          }
          color_index = (color_index + 1) % COLORS.length;
        }
      }
      break;
    }
    case "Q4": {
      // Q4 = 1e-8*Q**-4*self.intensity.value + self.background.value
      let color_index = 0;
      for (let model of model_data) {
        for (let xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const {intensity, background} = xs;
          const Q4 = xs.Q.map((qq) => (1e-8*Math.pow(qq, -4)*intensity + background));
          const theory = xs.theory.map((t, i) => (t / Q4[i]));
          const R = xs.R.map((r, i) => (r / Q4[i]));
          theory_traces.push({ x: xs.Q, y: theory, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }});
          color_index = (color_index + 1) % COLORS.length;
        }
      }
      break;
    }
    case "Spin Asymmetry": {
      let color_index = 0;
      for (let model of model_data) {
        const pp = model.find((xs) => xs.polarization === '++');
        const mm = model.find((xs) => xs.polarization === '--');
        if (pp !== undefined && mm !== undefined) {
          const label = pp.label;
          const Rm = interp(pp.Q, mm.Q, mm.R);
          const SA = Rm.map((m, i) => {
            const p = pp.R[i];
            return (p - m) / (p + m);
          });
          const Tm = interp(pp.Q, mm.Q, mm.theory);
          const TSA = Tm.map((m, i) => {
            const p = pp.theory[i];
            return (p - m) / (p + m);
          });
          let dSA: number[] = [];
          if (pp.dR !== undefined && mm.dR !== undefined) {
            const dRm = interp(pp.Q, mm.Q, mm.dR);
            dSA = dRm.map((dm, i) => {
              const dp = pp.dR[i];
              const p = pp.R[i];
              const m = Rm[i];
              return Math.sqrt(4 * ((p*dm)**2 + (m*dm)**2) / (p+m)**4)
            });
          }
      
          theory_traces.push({ x: pp.Q, y: TSA, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          const data_trace = { x: pp.Q, y: SA, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }};
          if (dSA.length > 0) {
              data_trace.error_y = {type: 'data', array: dSA, visible: true};
            }
          data_traces.push(data_trace);
          color_index = (color_index + 1) % COLORS.length;
        }
      }
    }

  }
  return {theory_traces, data_traces};
}

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_plot_data', 'linear');
  // console.log(payload);
  const { theory_traces, data_traces } = generate_new_traces(payload.plotdata, reflectivity_type.value)
  const layout: Partial<Plotly.Layout> = {
    uirevision: reflectivity_type.value,
    xaxis: {
      title: {
        text: 'Q (Å<sup>-1</sup>)'
      },
      type: 'linear',
      autorange: true,
    },
    yaxis: {
      title: { text: 'Reflectivity' },
      exponentformat: 'e',
      showexponent: 'all',
      type: (/^(Log|Q4)/.test(reflectivity_type.value)) ? 'log' : 'linear',
      autorange: true,
    },
    margin: {
      l: 75,
      r: 50,
      t: 25,
      b: 75,
      pad: 4
    },
    annotations: [
      {
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        x: 0.8,
        yanchor: 'top',
        y: -0.05,
        text: `chisq = ${payload.chisq}`,
        showarrow: false,
        font: {size: 16}, 
      }
    ],
    legend: {
      x: 0.95,
      y: 0.95,
      xanchor: 'left',
      yanchor: 'top'
    }
  };

  if (reflectivity_type.value === 'Spin Asymmetry') {
    layout.yaxis.range = [-1.5, 1.5];
    layout.yaxis.autorange = false;
  }

  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true
    }
  }
  await Plotly.react(plot_div.value as HTMLDivElement, [...theory_traces, ...data_traces], layout, config);
}

function interp(x: number[], xp: number[], fp: number[]) {
  // assume x and xp are sorted, monotonically increasing
  if (xp.length != fp.length) {
    throw new Error(`lengths of xp (${xp.length}) and fp (${fp.length}) must match`);
  }

  const xpv = xp.values();
  const fpv = fp.values();

  const lowest_xp = xp[0];
  const lowest_fp = fp[0];
  const highest_xp = xp[xp.length - 1];
  const highest_fp = fp[fp.length - 1];

  let lower_xp = xpv.next();
  let lower_fp = fpv.next();
  let upper_xp = xpv.next();
  let upper_fp = fpv.next();

  if (upper_xp.done) {
    throw new Error("length of xp, fp must be > 2");
  }

  return x.map((xv) => {
    while(xv >= upper_xp.value && !upper_xp.done) {
      lower_xp = upper_xp;
      lower_fp = upper_fp;
      upper_xp = xpv.next();
      upper_fp = fpv.next();
    }

    if (xv < lowest_xp) {
      return lowest_fp;
    }
    else if (upper_xp.done) {
      return highest_fp;
    }
    else {
      // xv < upper_xp.value
      return (upper_fp.value - lower_fp.value) / (upper_xp.value - lower_xp.value) * (xv - lower_xp.value) + lower_fp.value;
    }
  });
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <select v-model="reflectivity_type" @change="fetch_and_draw">
      <option v-for="refl_type in REFLECTIVITY_PLOTS" :key="refl_type" :value="refl_type">{{refl_type}}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
