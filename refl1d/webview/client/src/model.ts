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

export interface Layer {
    name: string;
    thickness: ParameterLike;
    magnetism: Magnetism | null;
    material: SLD;
    interface: ParameterLike;
    __class__: "refl1d.model.Slab";
}

export interface SLD {
    name: string;
    rho: ParameterLike;
    irho: ParameterLike;
    __class__: "refl1d.material.SLD";
}

export interface Magnetism {
    name: string;
    rhoM: ParameterLike;
    thetaM: ParameterLike;
    phiM: ParameterLike;
    __class__: "refl1d.material.Magnetism";
}

export interface Stack {
    layers: (Layer | Stack)[];
    __class__: "refl1d.model.Stack";
}

export interface Experiment {
    sample: Stack;
    probe: QProbe;
    __class__: "refl1d.experiment.Experiment";
}

export interface NumpyArray {
    values: number[];
    dtype: string;
    __class__: "bumps.util.NumpyArray";
}

export interface QProbe {
    Q: NumpyArray;
    dQ: NumpyArray;
    __class__: "refl1d.probe.QProbe";
}

export function generateQProbe(qmin: number = 0, qmax: number = 0.1, qsteps: number = 250, dQ: number = 0.00001) {
    const Q_arr = Array.from({ length: qsteps }, (_, i) => qmin + i * (qmax - qmin) / qsteps);
    const dQ_arr = Array.from({ length: qsteps }, () => dQ);
    const probe: QProbe = { 
        Q: { values: Q_arr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
        dQ: { values: dQ_arr, dtype: "float64", __class__: "bumps.util.NumpyArray" },
        __class__: "refl1d.probe.QProbe"
    };
    return probe;
}

