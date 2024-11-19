export interface SerializedModel {
    references: { [key: string]: Parameter };
    object: any;
    "$schema": "bumps-draft-02";
}

export interface Variable {
    value: number;
    __class__: "bumps.parameter.Variable";
}

export interface Reference {
    id: string;
    __class__: "Reference";
}

export type BoundsValue = number | "-inf" | "inf";

export interface Parameter {
    id: string;
    name: string;
    fixed: boolean;
    slot: Variable;
    limits: (number | "-inf" | "inf")[];
    bounds: (number | "-inf" | "inf")[] | null;
    tags: string[];
    __class__: "bumps.parameter.Parameter";
}

export interface ParameterLike {
    id: string;
    __class__: "bumps.parameter.Parameter" | "Reference";
}

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

export interface Experiment {
    sample: Stack;
    probe: QProbe;
    __class__: "refl1d.probe.experiment.Experiment";
}

export interface NumpyArray {
    values: number[];
    dtype: string;
    __class__: "bumps.util.NumpyArray";
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