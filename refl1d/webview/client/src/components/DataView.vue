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

// Colorblind-friendly colors, as per https://gist.github.com/thriveth/8560036
const COLORS = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

const MARKER_OPACITY = 0.5;

function generate_new_traces(model_data, view: ReflectivityPlot) {
  let theory_traces: (Plotly.Data & { x: number, y: number })[] = [];
  let data_traces: (Plotly.Data & { x: number, y: number })[] = [];
  let yaxis_label: string = "Reflectivity";
  let xaxis_label: string = "Q (Å<sup>-1</sup>)";
  switch (view) {
    case "Log":
    case "Linear": {
      let color_index = 0;
      for (let model of model_data) {
        for (let xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          theory_traces.push({ x: xs.Q, y: xs.theory, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          if (xs.R !== undefined) {
            const data_trace = { x: xs.Q, y: xs.R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }, opacity: MARKER_OPACITY};
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
            data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }, opacity: MARKER_OPACITY});
          }
          color_index = (color_index + 1) % COLORS.length;
        }
      }
      yaxis_label = "Fresnel Reflectivity"
      break;
    }
    case "Q4": {
      // Q4 = 1e-8*Q**-4*self.intensity.value + self.background.value
      let color_index = 0;
      for (let model of model_data) {
        for (let xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const {intensity_in, background_in} = xs;
          const intensity = intensity_in ?? 1.0;
          const background = background_in ?? 0.0;
          const Q4 = xs.Q.map((qq) => (1e-8*Math.pow(qq, -4)*intensity + background));
          const theory = xs.theory.map((t, i) => (t / Q4[i]));
          theory_traces.push({ x: xs.Q, y: theory, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});
          if (xs.R !== undefined) {
            const R = xs.R.map((r, i) => (r / Q4[i]));
            data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }, opacity: MARKER_OPACITY});
          }
            color_index = (color_index + 1) % COLORS.length;
        }
      }
      yaxis_label = "Reflectivity / Q<sup>4</sup>";
      break;
    }
    case "Spin Asymmetry": {
      let color_index = 0;
      for (let model of model_data) {
        const pp = model.find((xs) => xs.polarization === '++');
        const mm = model.find((xs) => xs.polarization === '--');

        if (pp !== undefined && mm !== undefined) {
          const label = pp.label;

          const Tm = interp(pp.Q, mm.Q, mm.theory);
          const TSA = Tm.map((m, i) => {
            const p = pp.theory[i];
            return (p - m) / (p + m);
          });

          theory_traces.push({ x: pp.Q, y: TSA, mode: 'lines', name: label + ' theory', line: { width: 2, color: COLORS[color_index] }});

          if (pp.R !== undefined && mm.R !== undefined) {
            const Rm = interp(pp.Q, mm.Q, mm.R);
            const SA = Rm.map((m, i) => {
              const p = pp.R[i];
              return (p - m) / (p + m);
            });
            const data_trace = { x: pp.Q, y: SA, mode: 'markers', name: label + ' data', marker: { color: COLORS[color_index] }, opacity: MARKER_OPACITY};

            if (pp.dR !== undefined && mm.dR !== undefined) {
              const dRm = interp(pp.Q, mm.Q, mm.dR);
              const dSA = dRm.map((dm, i) => {
                const dp = pp.dR[i];
                const p = pp.R[i];
                const m = Rm[i];
                return Math.sqrt(4 * ((p*dm)**2 + (m*dm)**2) / (p+m)**4)
              });
              data_trace.error_y = {type: 'data', array: dSA, visible: true};
            }

            data_traces.push(data_trace);
          }

          color_index = (color_index + 1) % COLORS.length;
        }
      }
      yaxis_label = "Spin Asymmetry (pp - mm) / (pp + mm)"
    }

  }
  return {theory_traces, data_traces, xaxis_label, yaxis_label};
}

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_plot_data', 'linear');
  // console.log(payload);
  const { theory_traces, data_traces, xaxis_label, yaxis_label } = generate_new_traces(payload.plotdata, reflectivity_type.value)
  const layout: Partial<Plotly.Layout> = {
    uirevision: reflectivity_type.value,
    xaxis: {
      title: {
        text: xaxis_label,
      },
      type: 'linear',
      autorange: true,
    },
    yaxis: {
      title: { text: yaxis_label },
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
