"""
Test that Repeat and flattened multilayers produce identical results.

Compares:
1. A sample using Repeat(stack, N) with shared parameters across periods
2. A sample with N independent copies of the stack (flattened)

Both should produce identical reflectivity and profiles when parameters are matched.
"""

import numpy as np
from refl1d.sample import material, layers, magnetism
from refl1d.probe.probe import PolarizedNeutronProbe, NeutronProbe
from refl1d.experiment import Experiment


def build_multilayer(N=2, flat=False):
    """Build a multilayer using Repeat structure."""
    Si = material.Material(formula="Si")
    Ni = material.Material(formula="Ni[58]")
    Si_ml = material.Material(formula="Si")
    air = material.Vacuum()


    # Single bilayer (Ni|Si)
    Ni_layer = layers.Slab(
        material=Ni,
        thickness=100,
        interface=5,
        magnetism=magnetism.Magnetism(rhoM=2.0, interface_above=5.0, interface_below=5.0, name="Ni Layer 1.0T")
    )
    Si_layer = layers.Slab(material=Si_ml, thickness=100, interface=5)

    Ni_layer.thickness.range(50, 200)
    Ni_layer.interface.range(1, 20)
    Si_layer.thickness.range(50, 200)
    Si_layer.interface.range(1, 20)

    if flat:
        multilayer = (Ni_layer, Si_layer)*N
    else:
        multilayer = (Ni_layer | Si_layer)*N

    # Substrate and repeat
    Si_sub = layers.Slab(material=Si, thickness=0, interface=5)

    return Si_sub | multilayer | air

def build_experiments(N=2):
    # Build both samples
    sample_repeat = build_multilayer(N, flat=False)
    sample_flat = build_multilayer(N, flat=True)

    # Create T and L arrays
    T = np.logspace(-3, 0, 50)
    L = 5.0

    # Create four NeutronProbe instances, one for each cross-section
    probe_pp = NeutronProbe(T=T, L=L)
    probe_pm = NeutronProbe(T=T, L=L)
    probe_mp = NeutronProbe(T=T, L=L)
    probe_mm = NeutronProbe(T=T, L=L)

    # Create the polarized probe
    probe = PolarizedNeutronProbe(pp=probe_pp, pm=probe_pm, mp=probe_mp, mm=probe_mm)

    # Create experiments
    exp_repeat = Experiment(sample=sample_repeat, probe=probe, dz=0.5, step_interfaces=True, dA=None)
    exp_flat = Experiment(sample=sample_flat, probe=probe, dz=0.5, step_interfaces=True, dA=None)
    return exp_repeat, exp_flat

def test_multilayer_equivalence(N=2):
    """Test that Repeat and flattened structures give identical results."""
    exp_repeat, exp_flat = build_experiments(N)

    # Compare reflectivity
    print(f"\n{'=' * 70}")
    print(f"Testing magnetic multilayer: N={N} periods")
    print(f"{'=' * 70}")

    R_repeat = exp_repeat.reflectivity(resolution=False)
    R_flat = exp_flat.reflectivity(resolution=False)

    # PolarizedNeutronProbe returns 4 cross-sections: (Q, R_mm), (Q, R_mp), (Q, R_pm), (Q, R_pp)
    Q_repeat, R_mm_repeat, R_mp_repeat, R_pm_repeat, R_pp_repeat = (
        R_repeat[0][0],  # Q from first cross-section
        R_repeat[0][1],  # R_mm
        R_repeat[1][1],  # R_mp
        R_repeat[2][1],  # R_pm
        R_repeat[3][1],  # R_pp
    )
    Q_flat, R_mm_flat, R_mp_flat, R_pm_flat, R_pp_flat = (
        R_flat[0][0],
        R_flat[0][1],
        R_flat[1][1],
        R_flat[2][1],
        R_flat[3][1],
    )

    # Check Q points match
    assert np.allclose(Q_repeat, Q_flat), "Q points differ between Repeat and flattened"
    print("✓ Q points match")

    # Check reflectivity values match for all cross-sections
    for label, R_r, R_f in [
        ("mm", R_mm_repeat, R_mm_flat),
        ("mp", R_mp_repeat, R_mp_flat),
        ("pm", R_pm_repeat, R_pm_flat),
        ("pp", R_pp_repeat, R_pp_flat),
    ]:
        R_diff = np.abs(R_r - R_f)
        R_rel_err = R_diff / (np.abs(R_r) + 1e-10)
        max_abs_err = np.max(R_diff)
        max_rel_err = np.max(R_rel_err)

        print(f"✓ Reflectivity {label} comparison:")
        print(f"  Max absolute difference: {max_abs_err:.3e}")
        print(f"  Max relative difference: {max_rel_err:.3e}")
        assert max_abs_err < 1e-12, f"Reflectivity {label} differs: max error {max_abs_err}"
    print("  ✓ All cross-section reflectivity values match")

    # Compare magnetic slab profiles
    w_r, sigma_r, rho_r, irho_r, rhoM_r, thetaM_r = exp_repeat.magnetic_slabs()
    w_f, sigma_f, rho_f, irho_f, rhoM_f, thetaM_f = exp_flat.magnetic_slabs()

    print(f"\n✓ Magnetic slab profile comparison:")
    print(f"  Repeat:   {len(w_r)} slabs, thickness sum = {np.sum(w_r):.1f}")
    print(f"  Flattened: {len(w_f)} slabs, thickness sum = {np.sum(w_f):.1f}")

    # Check slab counts match
    assert len(w_r) == len(w_f), f"Slab counts differ: {len(w_r)} vs {len(w_f)}"
    print("  ✓ Slab counts match")

    # Check thickness profiles match
    assert np.allclose(w_r, w_f), "Thickness arrays differ"
    assert np.allclose(sigma_r, sigma_f), "Sigma arrays differ"
    print("  ✓ Thickness and roughness arrays match")

    # Check rho profiles match
    assert np.allclose(rho_r, rho_f), "Nuclear SLD arrays differ"
    assert np.allclose(irho_r, irho_f), "Nuclear imaginary SLD arrays differ"
    print("  ✓ Nuclear SLD arrays match")

    # Check magnetic profiles match (with slight tolerance for sign/phase wrapping)
    rhoM_diff = np.abs(rhoM_r - rhoM_f)
    thetaM_diff = np.abs(thetaM_r - thetaM_f)
    # Handle 360-degree wrapping in angle
    thetaM_diff = np.minimum(thetaM_diff, 360 - thetaM_diff)

    assert np.allclose(rhoM_diff, 0, atol=1e-10), f"Magnetic SLD differs: max {np.max(rhoM_diff)}"
    assert np.allclose(thetaM_diff, 0, atol=1e-10), f"Magnetic angle differs: max {np.max(thetaM_diff)}"
    print("  ✓ Magnetic SLD and angle arrays match")

    # Print depth profile info
    print(f"\n✓ Magnetic structure (first few slabs):")
    for i in range(min(5, len(w_r))):
        depth = np.sum(w_r[: i + 1])
        print(
            f"  Slab {i}: z={depth:.1f}, w={w_r[i]:.2f}, "
            f"rho={rho_r[i]:.3f}, rhoM={rhoM_r[i]:.3f}, "
            f"theta={thetaM_r[i]:.1f}°"
        )
    if len(w_r) > 5:
        print(f"  ... ({len(w_r) - 5} more slabs)")

    # Count magnetic slabs (nonzero rhoM)
    mag_slabs_r = np.sum(np.abs(rhoM_r) > 1e-9)
    mag_slabs_f = np.sum(np.abs(rhoM_f) > 1e-9)
    print(f"\n✓ Magnetic slabs: {mag_slabs_r} (Repeat), {mag_slabs_f} (Flattened)")
    assert mag_slabs_r == mag_slabs_f, "Number of magnetic slabs differs"

    print(f"\n{'=' * 70}")
    print(f"✓ ALL TESTS PASSED for N={N}")
    print(f"{'=' * 70}\n")

    return True


if __name__ == "__main__":
    # Test for different repeat counts
    for N in [2, 3, 10]:
        try:
            test_multilayer_equivalence(N)
        except Exception as e:
            print(f"\n✗ TEST FAILED for N={N}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            exit(1)

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED FOR ALL N VALUES")
    print("=" * 70)

else:
    # Allow loading into refl1d
    from bumps.names import FitProblem

    problem = FitProblem(build_experiments(N=4))
