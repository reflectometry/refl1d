from pathlib import Path
from refl1d.probe.data_loaders.load4 import load4, PolarizedQProbe


def test_load4_polarized_qprobe():
    """Test that load4 correctly loads polarized data and creates a PolarizedQProbe."""
    # Get the path to the example data file (located in the doc/examples/four_column directory)
    test_file = Path(__file__).parent.parent.parent / "doc" / "examples" / "four_column" / "refl.txt"

    # Load the probe
    probe = load4(str(test_file))

    # Verify it's a PolarizedQProbe
    assert isinstance(probe, PolarizedQProbe), f"Expected PolarizedQProbe, got {type(probe)}"

    # Verify the specific cross sections are pp and mm
    assert probe.pp is not None, "Expected pp cross section to be present"
    assert probe.mm is not None, "Expected mm cross section to be present"
    assert probe.pm is None, "Expected pm cross section to be None"
    assert probe.mp is None, "Expected mp cross section to be None"


if __name__ == "__main__":
    test_load4_polarized_qprobe()
