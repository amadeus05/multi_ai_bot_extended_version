import importlib.util
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "runners" / "run_bybit_testnet.py"
SPEC = importlib.util.spec_from_file_location("run_bybit_testnet", RUNNER_PATH)
run_bybit_testnet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_bybit_testnet)
build_testnet_safety_from_env = run_bybit_testnet.build_testnet_safety_from_env


def test_testnet_safety_defaults_to_dry_run_private_sync_and_symbol_allowlist(monkeypatch) -> None:
    monkeypatch.delenv("BYBIT_TESTNET_DRY_RUN", raising=False)
    monkeypatch.delenv("BYBIT_TESTNET_KILL_SWITCH", raising=False)
    monkeypatch.delenv("BYBIT_TESTNET_MAX_ORDER_NOTIONAL", raising=False)
    monkeypatch.delenv("BYBIT_TESTNET_ALLOWLIST", raising=False)

    safety = build_testnet_safety_from_env(("BTC/USDT", "ETH/USDT"))

    assert safety.dry_run is True
    assert safety.kill_switch is False
    assert safety.max_order_notional == 50.0
    assert safety.allowlist_symbols == frozenset({"BTC/USDT", "ETH/USDT"})
    assert safety.require_private_sync is True


def test_testnet_safety_allows_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BYBIT_TESTNET_DRY_RUN", "0")
    monkeypatch.setenv("BYBIT_TESTNET_KILL_SWITCH", "1")
    monkeypatch.setenv("BYBIT_TESTNET_MAX_ORDER_NOTIONAL", "125.5")
    monkeypatch.setenv("BYBIT_TESTNET_ALLOWLIST", "SOL/USDT")

    safety = build_testnet_safety_from_env(("BTC/USDT",))

    assert safety.dry_run is False
    assert safety.kill_switch is True
    assert safety.max_order_notional == 125.5
    assert safety.allowlist_symbols == frozenset({"SOL/USDT"})
    assert safety.require_private_sync is True
