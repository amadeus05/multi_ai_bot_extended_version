import importlib.util
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "runners" / "run_live.py"
SPEC = importlib.util.spec_from_file_location("run_live", RUNNER_PATH)
run_live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_live)
build_live_safety = run_live.build_live_safety


def test_live_safety_requires_private_sync() -> None:
    safety = build_live_safety()

    assert safety.require_private_sync is True
