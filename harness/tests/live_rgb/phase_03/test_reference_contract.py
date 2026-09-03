from pathlib import Path


def test_phase_three_spec_and_validator_exist():
    root = Path(__file__).resolve().parents[4]
    assert (root / "docs/rgb-live-reconstruction-phases/03-versioned-draft-3d.md").is_file()
    assert "validate_phase_three" in (root / "scripts/live_rgb/validate_phase.py").read_text()


def test_new_data_flow_remains_rgb_only():
    root = Path(__file__).resolve().parents[4]
    sources = "\n".join(
        path.read_text().lower()
        for path in (root / "harness/recon/live_rgb").glob("*.py")
        if path.name != "schema.py"
    )
    assert "com.google.ar" not in sources
    assert "mobile_pose" not in sources
    assert "mobile_depth" not in sources
