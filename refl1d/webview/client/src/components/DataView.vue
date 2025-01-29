<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, shallowRef } from "vue";
import type { AsyncSocket } from "bumps-webview-client/src/asyncSocket";
import { configWithSVGDownloadButton } from "bumps-webview-client/src/plotly_extras";
import { setupDrawLoop } from "bumps-webview-client/src/setupDrawLoop";
import * as Plotly from "plotly.js/lib/core";
import { COLORS } from "../colors";

// const title = "Reflectivity";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_offset = ref(0);
const plot_data = shallowRef<ModelData[][]>([]);
const chisq_str = ref("");

const log_y = ref(true);
const log_x = ref(false);
const show_resolution = ref(true);
const show_residuals = ref(false);

const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_parameters", props.socket, fetch_and_draw);

const REFLECTIVITY_PLOTS = ["Fresnel (R/R_substrate)", "Reflectivity", "RQ^4", "Spin Asymmetry"] as const;
type ReflectivityPlotEnum = typeof REFLECTIVITY_PLOTS;
type ReflectivityPlot = ReflectivityPlotEnum[number];
const reflectivity_type = ref<ReflectivityPlot>("Reflectivity");

const MARKER_OPACITY = 0.5;
type Trace = Partial<Plotly.PlotData>;
type PolarizationString = "" | "--" | "-+" | "+-" | "++" | "unpolarized";
type ModelData = {
  label: string;
  polarization: PolarizationString;
  Q: number[];
  dQ?: number[];
  theory: number[];
  fresnel: number[];
  intensity_in: number;
  background_in: number;
  R?: number[];
  dR?: number[];
};

function generate_new_traces(model_data: ModelData[][], view: ReflectivityPlot, calculate_residuals: boolean = false) {
  if (model_data.length === 0) {
    return { theory_traces: [], data_traces: [], xaxis_label: "Q (Å<sup>-1</sup>)", yaxis_label: "Reflectivity" };
  }
  let theory_traces: Trace[] = [];
  let data_traces: Trace[] = [];
  let residuals_traces: Trace[] = [];
  let yaxis_label: string = "Reflectivity";
  const xaxis_label: string = "Q (Å<sup>-1</sup>)";
  const offset = plot_offset.value;
  switch (view) {
    case "Reflectivity": {
      let plot_index = 0;
      const lin_y = !log_y.value;
      for (const model of model_data) {
        for (const xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const color = COLORS[plot_index % COLORS.length];
          const legendgroup = `group_${plot_index}`;
          const local_offset = lin_y ? plot_index * offset : Math.pow(10, plot_index * offset);
          const y = lin_y ? xs.theory.map((t) => t + local_offset) : xs.theory.map((t) => t * local_offset);
          theory_traces.push({ x: xs.Q, y: y, mode: "lines", name: label + " theory", line: { width: 2, color } });
          if (xs.R !== undefined) {
            const R = lin_y ? xs.R.map((t) => t + local_offset) : xs.R.map((t) => t * local_offset);
            const data_trace: Trace = {
              x: xs.Q,
              y: R,
              mode: "markers",
              name: label + " data",
              marker: { color },
              opacity: MARKER_OPACITY,
              legendgroup,
            };
            if (show_resolution.value && xs.dQ !== undefined) {
              data_trace.error_x = { type: "data", array: xs.dQ, visible: true };
            }
            if (xs.dR !== undefined) {
              const dR = lin_y ? xs.dR : xs.dR.map((t) => t * local_offset);
              data_trace.error_y = { type: "data", array: dR, visible: true };
              if (calculate_residuals) {
                const residuals = xs.R.map((r, i) => (r - xs.theory[i]) / xs.dR![i]);
                const residuals_trace: Trace = {
                  x: xs.Q,
                  y: residuals,
                  mode: "markers",
                  name: label + " residuals",
                  showlegend: false,
                  legendgroup,
                  marker: { color },
                  opacity: MARKER_OPACITY,
                  yaxis: "y2",
                };
                residuals_traces.push(residuals_trace);
              }
            }
            data_traces.push(data_trace);
          }
          plot_index++;
        }
      }
      break;
    }
    case "Fresnel (R/R_substrate)": {
      let plot_index = 0;
      for (const model of model_data) {
        for (const xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const color = COLORS[plot_index % COLORS.length];
          const legendgroup = `group_${plot_index}`;
          const lin_y = !log_y.value;
          const local_offset = lin_y ? plot_index * offset : Math.pow(10, plot_index * offset);
          const theory = xs.theory.map((y, i) => y / xs.fresnel[i]);
          const offset_theory = lin_y ? theory.map((t) => t + local_offset) : theory.map((t) => t * local_offset);
          theory_traces.push({
            x: xs.Q,
            y: offset_theory,
            mode: "lines",
            name: label + " theory",
            line: { width: 2, color },
          });
          if (xs.R !== undefined) {
            const R = xs.R.map((y, i) => y / xs.fresnel[i]);
            const offset_R = lin_y ? R.map((t) => t + local_offset) : R.map((t) => t * local_offset);
            const data_trace: Trace = {
              x: xs.Q,
              y: offset_R,
              mode: "markers",
              name: label + " data",
              marker: { color },
              opacity: MARKER_OPACITY,
              legendgroup,
            };
            if (show_resolution.value && xs.dQ !== undefined) {
              data_trace.error_x = { type: "data", array: xs.dQ, visible: true };
            }
            if (xs.dR !== undefined) {
              const dR = xs.dR.map((dy, i) => dy / xs.fresnel[i]);
              const dR_offset = lin_y ? dR : dR.map((t) => t * local_offset);
              data_trace.error_y = { type: "data", array: dR_offset, visible: true };
              if (calculate_residuals) {
                const residuals = xs.R.map((r, i) => (r - xs.theory[i]) / xs.dR![i]);
                const residuals_trace: Trace = {
                  x: xs.Q,
                  y: residuals,
                  mode: "markers",
                  name: label + " residuals",
                  showlegend: false,
                  legendgroup,
                  marker: { color },
                  opacity: MARKER_OPACITY,
                  yaxis: "y2",
                };
                residuals_traces.push(residuals_trace);
              }
            }
            data_traces.push(data_trace);
          }
          plot_index++;
        }
      }
      yaxis_label = "Fresnel (R/R_substrate)";
      break;
    }
    case "RQ^4": {
      // Q4 = 1e-8*Q**-4*self.intensity.value + self.background.value
      let plot_index = 0;
      for (const model of model_data) {
        for (const xs of model) {
          const label = `${xs.label} ${xs.polarization}`;
          const color = COLORS[plot_index % COLORS.length];
          const legendgroup = `group_${plot_index}`;
          const local_offset = Math.pow(10, plot_index * offset);
          const { intensity_in, background_in } = xs;
          const intensity = intensity_in ?? 1.0;
          const background = background_in ?? 0.0;
          const Q4 = xs.Q.map((qq) => 1e-8 * Math.pow(qq, -4) * intensity + background);
          const theory = xs.theory.map((t, i) => t / Q4[i]);
          const offset_theory = theory.map((t) => t * local_offset);
          theory_traces.push({
            x: xs.Q,
            y: offset_theory,
            mode: "lines",
            name: label + " theory",
            line: { width: 2, color },
          });
          if (xs.R !== undefined) {
            const R = xs.R.map((r, i) => r / Q4[i]);
            const offset_R = R.map((t) => t * local_offset);
            const data_trace: Trace = {
              x: xs.Q,
              y: offset_R,
              mode: "markers",
              name: label + " data",
              marker: { color },
              opacity: MARKER_OPACITY,
              legendgroup,
            };
            if (show_resolution.value && xs.dQ !== undefined) {
              data_trace.error_x = { type: "data", array: xs.dQ, visible: true };
            }
            if (xs.dR !== undefined) {
              const dR = xs.dR.map((dy, i) => dy / Q4[i]);
              const offset_dR = dR.map((t) => t * local_offset);
              data_trace.error_y = { type: "data", array: offset_dR, visible: true };
              if (calculate_residuals) {
                const residuals = xs.R.map((r, i) => (r - xs.theory[i]) / xs.dR![i]);
                const residuals_trace: Trace = {
                  x: xs.Q,
                  y: residuals,
                  mode: "markers",
                  name: label + " residuals",
                  showlegend: false,
                  legendgroup,
                  marker: { color },
                  opacity: MARKER_OPACITY,
                  yaxis: "y2",
                };
                residuals_traces.push(residuals_trace);
              }
            }
            data_traces.push(data_trace);
          }
          plot_index++;
        }
      }
      yaxis_label = "R \u{00B7} Q<sup>4</sup>";
      break;
    }
    case "Spin Asymmetry": {
      let plot_index = 0;
      for (let model of model_data) {
        const pp = model.find((xs) => xs.polarization === "++");
        const mm = model.find((xs) => xs.polarization === "--");
        const local_offset = plot_index * offset;

        if (pp !== undefined && mm !== undefined) {
          const label = pp.label;
          const color = COLORS[plot_index % COLORS.length];
          const legendgroup = `group_${plot_index}`;

          const Tm = interp(pp.Q, mm.Q, mm.theory);
          const TSA = Tm.map((m, i) => {
            const p = pp.theory[i];
            return (p - m) / (p + m) + local_offset;
          });

          theory_traces.push({ x: pp.Q, y: TSA, mode: "lines", name: label + " theory", line: { width: 2, color } });

          if (pp.R !== undefined && mm.R !== undefined) {
            const Rm = interp(pp.Q, mm.Q, mm.R);
            const SA = Rm.map((m, i) => {
              const p = pp.R![i];
              return (p - m) / (p + m);
            });
            const SA_offset = SA.map((v) => v + local_offset);
            const data_trace: Trace = {
              x: pp.Q,
              y: SA_offset,
              mode: "markers",
              name: label + " data",
              marker: { color },
              opacity: MARKER_OPACITY,
              legendgroup,
            };

            if (show_resolution.value && pp.dQ !== undefined) {
              data_trace.error_x = { type: "data", array: pp.dQ, visible: true };
            }
            if (pp.dR !== undefined && mm.dR !== undefined) {
              const dRm = interp(pp.Q, mm.Q, mm.dR);
              const dSA = dRm.map((dm, i) => {
                // const dp = pp.dR[i];
                const p = pp.R![i];
                const m = Rm[i];
                return Math.sqrt((4 * ((p * dm) ** 2 + (m * dm) ** 2)) / (p + m) ** 4);
              });
              data_trace.error_y = { type: "data", array: dSA, visible: true };
              if (calculate_residuals) {
                const residuals = SA.map((v, i) => (v - TSA[i]) / dSA[i]);
                const residuals_trace: Trace = {
                  x: pp.Q,
                  y: residuals,
                  mode: "markers",
                  name: label + " residuals",
                  showlegend: false,
                  legendgroup,
                  marker: { color },
                  opacity: MARKER_OPACITY,
                  yaxis: "y2",
                };
                residuals_traces.push(residuals_trace);
              }
            }

            data_traces.push(data_trace);
          }

          plot_index++;
        }
      }
      yaxis_label = "Spin Asymmetry (pp - mm) / (pp + mm)";
    }
  }
  data_traces.push(...residuals_traces);
  return { theory_traces, data_traces, xaxis_label, yaxis_label };
}

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit("get_plot_data", "linear")) as {
    plotdata: ModelData[][];
    chisq: string;
  };
  plot_data.value = payload.plotdata;
  chisq_str.value = payload.chisq;
  await draw_plot();
}

async function change_plot_type() {
  if (reflectivity_type.value === "Spin Asymmetry") {
    log_y.value = false;
  }
  await fetch_and_draw();
}

async function draw_plot() {
  const { theory_traces, data_traces, xaxis_label, yaxis_label } = generate_new_traces(
    plot_data.value,
    reflectivity_type.value,
    show_residuals.value
  );
  const layout: Partial<Plotly.Layout> = {
    uirevision: reflectivity_type.value,
    xaxis: {
      title: {
        text: xaxis_label,
      },
      type: log_x.value ? "log" : "linear",
      autorange: true,
    },
    yaxis: {
      title: { text: yaxis_label },
      exponentformat: "power",
      showexponent: "all",
      type: log_y.value ? "log" : "linear",
      autorange: true,
    },
    margin: {
      l: 75,
      r: 50,
      t: 25,
      b: 75,
      pad: 4,
    },
    annotations: [
      {
        xref: "paper",
        yref: "paper",
        xanchor: "left",
        x: 0.8,
        yanchor: "top",
        y: -0.05,
        text: `<i>\u{03C7}</i><sup>2</sup> = ${chisq_str.value}`,
        showarrow: false,
        font: { size: 16 },
      },
    ],
    legend: {
      x: 0.95,
      y: 0.95,
      xanchor: "left",
      yanchor: "top",
    },
  };

  if (show_residuals.value) {
    layout.yaxis!.domain = [0.4, 1];
    // layout.yaxis.anchor = 'x';
    layout.yaxis2 = {
      domain: [0, 0.25],
      title: { text: "Residuals" },
      anchor: "x",
    };
  }

  if (reflectivity_type.value === "Spin Asymmetry") {
    layout.yaxis!.range = [-1.5, 1.5];
    layout.yaxis!.autorange = false;
  }

  const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true,
    },
    ...configWithSVGDownloadButton,
  };
  await Plotly.react(plot_div.value as HTMLDivElement, [...theory_traces, ...data_traces], layout, config);
}

/**
 * Performs linear interpolation of the values in fp at the points in xp to the points in x.
 *
 * @param x: the points at which to interpolate
 * @param xp: the points at which the function is known
 * @param fp: the values of the function at the points in xp
 * @returns the interpolated y values at the points in x
 */
function interp(x: number[], xp: number[], fp: number[]): number[] {
  // assume x and xp are sorted, monotonically increasing
  if (xp.length != fp.length) {
    throw new Error(`lengths of xp (${xp.length}) and fp (${fp.length}) must match`);
  }

  const xpv = xp.values();
  const fpv = fp.values();

  const lowest_xp = xp[0];
  const lowest_fp = fp[0];
  // const highest_xp = xp[xp.length - 1];
  const highest_fp = fp[fp.length - 1];

  let lower_xp = xpv.next();
  let lower_fp = fpv.next();
  let upper_xp = xpv.next();
  let upper_fp = fpv.next();

  if (upper_xp.done) {
    throw new Error("length of xp, fp must be > 2");
  }

  return x.map((xv) => {
    while (xv >= upper_xp.value! && !upper_xp.done) {
      lower_xp = upper_xp;
      lower_fp = upper_fp;
      upper_xp = xpv.next();
      upper_fp = fpv.next();
    }

    if (xv < lowest_xp) {
      return lowest_fp;
    } else if (upper_xp.done) {
      return highest_fp;
    } else {
      // xv < upper_xp.value
      return (
        ((upper_fp.value! - lower_fp.value!) / (upper_xp.value - lower_xp.value!)) * (xv - lower_xp.value!) +
        lower_fp.value!
      );
    }
  });
}
</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="row">
      <div class="col">
        <label for="plot_mode" class="col-form-label">Plot mode:</label>
        <select id="plot_mode" v-model="reflectivity_type" class="plot-mode" @change="change_plot_type">
          <option v-for="refl_type in REFLECTIVITY_PLOTS" :key="refl_type" :value="refl_type">
            {{ refl_type }}
          </option>
        </select>
      </div>
      <div class="col-auto form-check">
        <input
          id="show_residuals"
          v-model="show_residuals"
          class="form-check-input"
          type="checkbox"
          @change="draw_plot"
        />
        <label class="form-check-label" for="show_residuals">Residuals</label>
      </div>
      <div class="col-auto form-check">
        <input id="log_y" v-model="log_y" type="checkbox" class="form-check-input" @change="draw_plot" />
        <label for="log_y" class="form-check-label">Log y</label>
      </div>
      <div class="col-auto form-check">
        <input id="log_x" v-model="log_x" type="checkbox" class="form-check-input" @change="draw_plot" />
        <label for="log_x" class="form-check-label">Log x</label>
      </div>
      <div class="col-auto form-check">
        <input
          id="show_resolution"
          v-model="show_resolution"
          type="checkbox"
          class="form-check-input"
          @change="draw_plot"
        />
        <label for="show_resolution" class="form-check-label">dQ</label>
      </div>
    </div>
    <div class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="plot_offset_control" class="col-form-label">Plot offset</label>
      </div>
      <div class="col">
        <input
          id="plot_offset_control"
          v-model.number="plot_offset"
          type="range"
          min="0"
          max="1.0"
          step="0.01"
          class="form-range"
          @input="draw_plot"
        />
      </div>
    </div>
    <div id="plot_div" ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
.plot-mode {
  width: 100%;
}
</style>
