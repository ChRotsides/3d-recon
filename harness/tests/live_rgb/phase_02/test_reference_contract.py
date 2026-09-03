from pathlib import Path


def test_phase_two_spec_and_validator_exist():
    root = Path(__file__).resolve().parents[4]
    assert (root / "docs/rgb-live-reconstruction-phases/02-incremental-rgb-guidance.md").is_file()
    validator = (root / "scripts/live_rgb/validate_phase.py").read_text()
    assert "phase" in validator
