import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
from numpy import radians

B2SLD = 2.31604654e-6
GEPORE_SRC = "gepore.f"
GEPORE_ZEEMAN_SRC = "gepore_zeeman.f"


class GeporeRunner:
    binary_name = "gepore"
    binary_zeeman_name = "gepore_zeeman"

    def __init__(self):
        self.binary_path = Path(tempfile.mkdtemp())
        self.source_folder = Path(__file__).parent
        self.gepore_src_path = self.source_folder / GEPORE_SRC
        self.gepore_zeeman_src_path = self.source_folder / GEPORE_ZEEMAN_SRC
        self.gepore_binary = self.binary_path / self.binary_name
        self.gepore_zeeman_binary = self.binary_path / self.binary_zeeman_name
        self._compile_gepore(self.gepore_binary, self.gepore_src_path)
        self._compile_gepore(self.gepore_zeeman_binary, self.gepore_zeeman_src_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.binary_path)

    def _compile_gepore(self, target: Path, source: Path):
        status = os.system(f"gfortran -o {target} {source}")
        if status != 0:
            raise RuntimeError("Could not compile %r" % source)
        if not target.exists():
            raise RuntimeError("No gepore created in %r" % target)

    def run(self, layers, QS, DQ, NQ, EPS, H, zeeman_corrections=True, output_folder: Optional[Path] = None):
        start_path = Path.cwd()  # save the current path

        if zeeman_corrections:
            layers = add_H(layers, H, EPS=EPS)
            gepore = self.gepore_zeeman_binary
        else:
            gepore = self.gepore_binary
        # layers = add_H(layers, H, EPS-270, 0)
        depth, rho, rhoB, thetaB, phiB = list(zip(*layers))

        NL = len(rho) - 2
        NC = 1
        ROSUP = rho[-1] + rhoB[-1]
        ROSUM = rho[-1] - rhoB[-1]
        ROINP = rho[0] + rhoB[0]
        ROINM = rho[0] - rhoB[0]

        path = Path(tempfile.mkdtemp())
        gepore_local = path / "gepore"
        shutil.copy(gepore, gepore_local)
        header = path / "inpt.d"
        layers = path / "tro.d"
        rm_real = path / "rrem.d"
        rm_imag = path / "rimm.d"
        rp_real = path / "rrep.d"
        rp_imag = path / "rimp.d"
        output_files = (header, layers, rm_real, rm_imag, rp_real, rp_imag)

        with open(layers, "w") as fid:
            for T, BN, PN, THE, PHI in list(zip(depth, rho, rhoB, thetaB, phiB))[1:-1]:
                fid.write(f"{T} {1e-6 * BN} {1e-6 * PN} {radians(THE)} {radians(PHI)}\n")
                # fid.write("%f %e %e %f %f\n" % (T, 1e-6 * BN, 1e-6 * PN, radians(THE), radians(PHI)))

        for IP, IM in ((0.0, 1.0), (1.0, 0.0)):
            with open(header, "w") as fid:
                fid.write(
                    f"{NL} {NC} {QS} {DQ} {NQ} {radians(EPS)} ({IP},0.0) ({IM},0.0) {1e-6 * ROINP} {1e-6 * ROINM} {1e-6 * ROSUP} {1e-6 * ROSUM}\n"
                    # "%d %d %f %f %d %f (%f,0.0) (%f,0.0) %e %e %e %e\n"
                    # % (NL, NC, QS, DQ, NQ, radians(EPS), IP, IM, 1e-6 * ROINP, 1e-6 * ROINM, 1e-6 * ROSUP, 1e-6 * ROSUM)
                )

            os.chdir(path)
            result = subprocess.run(["./gepore"], capture_output=True, check=True)

            rp = np.loadtxt(rp_real).T[1] + 1j * np.loadtxt(rp_imag).T[1]
            rm = np.loadtxt(rm_real).T[1] + 1j * np.loadtxt(rm_imag).T[1]
            if IP > 0.5:
                Rpp, Rpm = rp, rm
                if output_folder is not None:
                    plus_files = output_folder / "gepore_plus"
                    plus_files.mkdir(exist_ok=True, parents=True)
                    for f in output_files:
                        shutil.copy(f, plus_files)
            else:
                Rmp, Rmm = rp, rm
                if output_folder is not None:
                    minus_files = output_folder / "gepore_minus"
                    minus_files.mkdir(exist_ok=True, parents=True)
                    for f in output_files:
                        shutil.copy(f, minus_files)

        # clean up
        os.chdir(start_path)
        shutil.rmtree(path)
        return Rpp, Rpm, Rmp, Rmm


def add_H(layers, H=0.0, EPS=270.0):
    """Take H (vector) as input and add H to 4piM:"""
    new_layers = []
    for layer in layers:
        thickness, sld_n, sld_m, theta_m, phi_m = layer
        # we read phi_m, but it must be zero so we don't use it.
        sld_m_x = sld_m * np.cos(theta_m * np.pi / 180.0)  # phi_m = 0
        sld_m_y = sld_m * np.sin(theta_m * np.pi / 180.0)  # phi_m = 0
        sld_m_z = 0.0  # by Maxwell's equations, H_demag = mz so we'll just cancel it here
        sld_h = B2SLD * 1.0e6 * H
        # this code was completely wrong except for the case AGUIDE=270
        sld_h_x = 0  # by definition, H is along the z,lab direction and x,lab = x,sam so Hx,sam must = 0
        sld_h_y = -sld_h * np.sin(EPS * np.pi / 180.0)
        sld_h_z = sld_h * np.cos(EPS * np.pi / 180.0)
        sld_b_x = sld_h_x + sld_m_x
        sld_b_y = sld_h_y + sld_m_y
        sld_b_z = sld_h_z + sld_m_z
        sld_b = np.sqrt((sld_b_z) ** 2 + (sld_b_y) ** 2 + (sld_b_x) ** 2)
        # this was wrong:
        # theta_b = np.arctan2(sld_b_y, sld_b_x)
        theta_b = np.arccos(sld_b_x / sld_b)
        # this didn't hurt anything but is also unneeded:
        # theta_b = np.mod(theta_b, 2.0*np.pi)
        # this wasn't even close to correct:
        # phi_b = np.arcsin(sld_b_z/sld_b)
        phi_b = np.arctan2(sld_b_z, sld_b_y)
        phi_b = np.mod(phi_b, 2.0 * np.pi)
        new_layer = [thickness, sld_n, sld_b, theta_b * 180.0 / np.pi, phi_b * 180.0 / np.pi]
        new_layers.append(new_layer)
    return new_layers
