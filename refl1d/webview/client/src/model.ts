/**
 * This file contains the types for the serialized model format used by the refl1d webapp. Note that the fields must
 * match the Python model serialization format, and must therefore be snake_case
 */

import type { NumpyArray, ParameterLike } from "bumps-webview-client/src/model";

export interface Reference {
  id: string;
  __class__: "Reference";
}

export type BoundsValue = number | "-inf" | "inf";

/* Sample models */

export interface Slab {
  name: string;
  thickness: ParameterLike;
  magnetism: Magnetism | null;
  material: SLD;
  interface: ParameterLike;
  __class__: "refl1d.sample.layers.Slab";
}

export interface Repeat {
  name: string;
  interface: ParameterLike;
  magnetism: Magnetism | null;
  repeat: ParameterLike;
  stack: Stack;
  thickness: ParameterLike;
  __class__: "refl1d.sample.layers.Repeat";
}

export interface SLD {
  name: string;
  rho: ParameterLike;
  irho: ParameterLike;
  __class__: "refl1d.sample.material.SLD";
}

export interface Magnetism {
  name: string;
  rhoM: ParameterLike;
  thetaM: ParameterLike;
  phiM: ParameterLike;
  __class__: "refl1d.sample.material.Magnetism";
}

export interface Stack {
  layers: (Slab | Repeat)[];
  __class__: "refl1d.sample.layers.Stack";
}

/* Experiment models */

export interface Experiment {
  /*
   * Why does this not include name, or the other fields?
   * Why is this a QProbe and not a Probe?
   */
  sample: Stack;
  probe: QProbe;
  __class__: "refl1d.experiment.Experiment";
}

export interface MixedExperiment {
  name: string;
  ratio: (number | ParameterLike)[];
  samples?: Stack[];
  probe: Probe | PolarizedNeutronProbe;
  coherent: boolean;
  interpolation: number;
  __class__: "refl1d.experiment.MixedExperiment";
}

/* Probe models */

export interface Probe {
  /* W... what? should go here? */
  name?: string;
}

export interface PolarizedNeutronProbe {
  /* W... what? should go here? */
  name: string;
}

export interface QProbe {
  Q: NumpyArray;
  dQ: NumpyArray;
  name?: string;
  filename?: string;
  intensity: ParameterLike;
  back_absorption: ParameterLike;
  background: ParameterLike;
  back_reflectivity: boolean;
  R?: NumpyArray;
  dR?: NumpyArray;
  resolution: "normal" | "uniform";
  __class__: "refl1d.probe.QProbe";
}
