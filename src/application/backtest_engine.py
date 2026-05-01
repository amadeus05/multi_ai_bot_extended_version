from __future__ import annotations

"""
Бэктест: каузальная симуляция по общим временным меткам.

Порядок времени на шаг i (current_ts=T, next_ts=T+1):
  - Эквити в точке T: cash + MTM позиций по close(T).
  - Выходы: TP/SL по OHLC следующей свечи T+1 (путь цены после входа).
  - Входы: фичи и barrier_* для сайза только из строки T; исполнение по open следующей свечи ± slippage.

В ``model.predict()`` передаются только колонки из ``*_features.json`` (не весь parquet).
Строка parquet может содержать Target/barriers для сайзинга — таргет в LGB не попадает.
Ошибочный leakage по фичам в основном из dataset/feature builder, не из этого файла.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from domain.strategy.signal_resolver import resolve_directional_signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[91m"
_ANSI_GREEN = "\033[92m"


class BacktestEngine:
    def __init__(self, config, strategy=None, exit_manager=None) -> None:
        self.config = config
        self.strategy = strategy
        self.exit_manager = exit_manager

    @staticmethod
    def _trade_log_colorize(text: str, color: str) -> str:
        return f"{color}{text}{_ANSI_RESET}"

    @classmethod
    def _format_trade_log_pnl_pct(cls, pnl_pct: float) -> str:
        color = _ANSI_GREEN if pnl_pct >= 0 else _ANSI_RED
        return cls._trade_log_colorize(f"{pnl_pct:+.2f}%", color)

    @classmethod
    def _format_trade_log_reason(cls, reason: str) -> str:
        if reason == "TP":
            return cls._trade_log_colorize("TP", _ANSI_GREEN)
        if reason == "SL":
            return cls._trade_log_colorize("SL", _ANSI_RED)
        return reason

    @staticmethod
    def _audit_inference_features(model) -> None:
        """Жёсткая проверка: в обучающие признаки модели не попали таргет / явный будущий контекст."""
        fc = getattr(model, "feature_columns", None)
        if not fc:
            return
        forbidden_exact = {"target", "label", "y", "class"}
        bad = [c for c in fc if c.lower() in forbidden_exact or c == "Target"]
        if bad:
            raise ValueError(
                "Утечка таргета в признаки модели (features.json): "
                + ", ".join(bad)
                + ". Обучение/экспорт фич неверны."
            )
        barrier_in_model = [c for c in fc if c.startswith("barrier_")]
        if barrier_in_model:
            print(
                "\u26a0\ufe0f  В модели перечислены barrier_* признаки: "
                + ", ".join(barrier_in_model)
                + ". Они задаются тем же билдом данных, что и triple-barrier-лейбл; при сомнениях исключите из обучения."
            )
        suspicious_substrings = ("_fwd", "_future", "_lead", "forward_ret", "next_open", "next_close")
        warn = [c for c in fc if any(s in c.lower() for s in suspicious_substrings)]
        if warn:
            print(
                "\u26a0\ufe0f  В именах признаков есть подозрительные паттерны (возможный look-ahead): "
                + ", ".join(warn[:15])
                + ("..." if len(warn) > 15 else "")
            )

    @staticmethod
    def _log_backtest_causal_contract(model) -> None:
        wb: int | None = None
        req_fn = getattr(model, "required_bars", None)
        if callable(req_fn):
            try:
                wb = int(req_fn())
            except (TypeError, ValueError):
                wb = None
        skip_hint = ""
        if wb is not None and wb > 0:
            skip_hint = (
                f" Рекомендация для паритета с live-warmup: BACKTEST_SKIP_INITIAL_BARS={max(0, wb - 1)} "
                f"(или оставьте 0, если в parquet уже отрезаны неполные окна индикаторов)."
            )
        print("\nПроверка look-ahead / leakage (бэктест):")
        print("   • Фичи модели: строка с timestamp T (известны после close свечи T).")
        print("   • Вход: open следующей свечи T→T+1 ± slippage (цена close T+1 в решение не входит).")
        print("   • Выход: OHLC свечи после входа (как в triple-barrier / etl).")
        print("   • predict(): только колонки из *_features.json — Target в LGBM не передаётся.")
        print("   • Барьеры для объёма: из той же строки T (данные ≤ close T).")
        if skip_hint and wb is not None:
            print(f"   • Warmup (required_bars={wb}):{skip_hint}")
        print(
            "   • Методологическая утечка обучения: prod-модель из train_pipeline учится на всех строках "
            "датасета — бэктест по полному parquet без BACKTEST_START без OOS. "
            "Для явного окна: BACKTEST_START[/END]; принудительно: BACKTEST_STRICT_OOS=1.\n"
        )

    async def run(
        self,
        *,
        symbols: list[str],
        data_by_symbol: dict[str, pd.DataFrame],
        model,
    ) -> dict:
        symbols = [s for s in symbols if s in data_by_symbol and not data_by_symbol[s].empty]
        if not symbols:
            raise ValueError("No symbols with data for backtest")

        if not self.config.allow_longs and not self.config.allow_shorts:
            raise ValueError("Both ALLOW_LONGS and ALLOW_SHORTS are disabled")

        for symbol in symbols:
            frame = data_by_symbol[symbol].copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            data_by_symbol[symbol] = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        symbols = [s for s in symbols if {"open", "high", "low", "close", "timestamp"}.issubset(data_by_symbol[s].columns)]
        if not symbols:
            raise ValueError("No symbols contain required OHLC columns")

        common_ts = self._common_timestamps(data_by_symbol, symbols)
        if len(common_ts) < 2:
            raise ValueError("Too little common data between symbols")

        self._audit_inference_features(model)
        self._log_backtest_causal_contract(model)

        loop_start = max(0, int(self.config.backtest_skip_initial_bars))
        if loop_start >= len(common_ts) - 1:
            raise ValueError(
                f"BACKTEST_SKIP_INITIAL_BARS={loop_start} съедает всю историю "
                f"(осталось баров для симуляции: {len(common_ts) - 1})."
            )

        initial_balance = float(self.config.initial_capital)
        balance = initial_balance
        positions = {s: None for s in symbols}
        used_margin = 0.0
        next_trade_number = 1
        stop_cooldown_until_idx = {s: -1 for s in symbols}
        daily_sl_count = 0
        daily_stop_announced = False
        current_day = None
        consecutive_loss_count = 0

        trades = []
        trade_events = []
        equity_curve = []
        equity_timestamps = []
        monthly = defaultdict(lambda: {"pnl_abs": 0.0, "trades": 0, "wins": 0, "losses": 0})
        by_symbol = defaultdict(lambda: {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0})

        index_by_symbol = {s: data_by_symbol[s].set_index("timestamp", drop=False).to_dict("index") for s in symbols}
        peak_equity = balance
        max_drawdown = 0.0

        print(
            f"\nBacktest on {len(common_ts)} common timestamps | "
            f"simulation starts at index {loop_start} (BACKTEST_SKIP_INITIAL_BARS)"
        )
        print("-" * 80)

        for i in range(loop_start, len(common_ts) - 1):
            current_ts = common_ts[i]
            next_ts = common_ts[i + 1]
            day = pd.Timestamp(next_ts).normalize()
            if current_day is None or day != current_day:
                current_day = day
                daily_sl_count = 0
                daily_stop_announced = False

            market = {}
            for symbol in symbols:
                cur = index_by_symbol[symbol].get(current_ts)
                nxt = index_by_symbol[symbol].get(next_ts)
                if cur is None or nxt is None:
                    continue
                market[symbol] = {
                    "cur_close": float(cur["close"]),
                    "next_open": float(nxt["open"]),
                    "next_high": float(nxt["high"]),
                    "next_low": float(nxt["low"]),
                }

            mark_prices = {s: payload["cur_close"] for s, payload in market.items()}
            equity = self._compute_portfolio_equity(balance, positions, mark_prices)
            equity_curve.append(equity)
            equity_timestamps.append(current_ts)
            peak_equity, max_drawdown = self._update_drawdown_stats(equity, peak_equity, max_drawdown)

            # exits first
            for symbol, ctx in market.items():
                position = positions[symbol]
                if position is None:
                    continue
                closed = self._try_close_position(position, ctx)
                if closed is None:
                    continue
                exit_price, reason = closed
                pnl_pct, pnl_abs, commission = self._compute_trade_outcome(position, exit_price)
                previous_loss_streak = consecutive_loss_count

                used_margin = max(0.0, used_margin - float(position["margin"]))
                balance += pnl_abs

                trade = {
                    "trade_number": position["trade_number"],
                    "symbol": symbol,
                    "direction": "LONG" if int(position["dir"]) == 1 else "SHORT",
                    "reason": reason,
                    "pnl_pct": float(pnl_pct),
                    "pnl_abs": float(pnl_abs),
                    "commission": float(commission),
                    "ts": next_ts,
                }
                trades.append(trade)
                trade_events.append(
                    {
                        "type": "CLOSE",
                        "trade_number": position["trade_number"],
                        "symbol": symbol,
                        "reason": reason,
                        "pnl_pct": float(pnl_pct),
                        "pnl_abs": float(pnl_abs),
                        "commission": float(commission),
                        "ts": next_ts,
                        "balance": float(balance),
                    }
                )
                month_key = pd.Timestamp(next_ts).strftime("%Y-%m")
                monthly[month_key]["pnl_abs"] += float(pnl_abs)
                monthly[month_key]["trades"] += 1
                by_symbol[symbol]["trades"] += 1
                by_symbol[symbol]["pnl_abs"] += float(pnl_abs)
                if reason == "TP":
                    by_symbol[symbol]["tp"] += 1
                if reason == "SL":
                    by_symbol[symbol]["sl"] += 1
                    daily_sl_count += 1
                    if int(self.config.backtest_sl_cooldown_bars) > 0:
                        stop_cooldown_until_idx[symbol] = i + int(self.config.backtest_sl_cooldown_bars)

                if pnl_pct > 0:
                    monthly[month_key]["wins"] += 1
                    by_symbol[symbol]["wins"] += 1
                    consecutive_loss_count = 0
                else:
                    monthly[month_key]["losses"] += 1
                    by_symbol[symbol]["losses"] += 1
                    consecutive_loss_count += 1
                positions[symbol] = None

                exit_icon = "\u274C" if reason == "SL" else "\u2705" if reason == "TP" else "\u2139\uFE0F"
                print(
                    f"[{next_ts}] \u2116 {position['trade_number']} {exit_icon} {symbol}: "
                    f"{self._format_trade_log_reason(reason)} | "
                    f"PnL: {self._format_trade_log_pnl_pct(float(pnl_pct) * 100.0)} | "
                    f"Com: {commission:.2f}$ | "
                    f"Bal: {balance:.2f}"
                )
                reduce_after = int(self.config.backtest_reduce_risk_after_consecutive_losses)
                reduced_risk = float(self.config.backtest_reduced_risk_per_trade)
                base_risk = float(self.config.risk_per_trade)
                if reduce_after > 0 and reduced_risk < base_risk:
                    if previous_loss_streak < reduce_after <= consecutive_loss_count:
                        print(
                            f"[{next_ts}] \u26A0\uFE0F Loss streak {consecutive_loss_count}: "
                            f"risk per trade reduced to {reduced_risk * 100:.2f}%"
                        )
                    elif float(pnl_pct) > 0 and previous_loss_streak >= reduce_after:
                        print(
                            f"[{next_ts}] \u2139\uFE0F Loss streak reset: "
                            f"risk per trade restored to {base_risk * 100:.2f}%"
                        )
                if (
                    reason == "SL"
                    and int(self.config.backtest_max_sl_per_day) > 0
                    and daily_sl_count >= int(self.config.backtest_max_sl_per_day)
                    and not daily_stop_announced
                ):
                    daily_stop_announced = True
                    print(
                        f"[{next_ts}] \u26D4 Daily SL limit reached ({daily_sl_count}), "
                        "new entries are paused until next day"
                    )

            # entries
            if (
                int(self.config.backtest_max_sl_per_day) > 0
                and daily_sl_count >= int(self.config.backtest_max_sl_per_day)
            ):
                continue

            effective_risk = float(self.config.risk_per_trade)
            if (
                int(self.config.backtest_reduce_risk_after_consecutive_losses) > 0
                and consecutive_loss_count >= int(self.config.backtest_reduce_risk_after_consecutive_losses)
                and float(self.config.backtest_reduced_risk_per_trade) > 0
            ):
                effective_risk = float(self.config.backtest_reduced_risk_per_trade)

            snapshot_balance = balance
            candidates = []
            for symbol, ctx in market.items():
                if positions[symbol] is not None:
                    continue
                if i < stop_cooldown_until_idx.get(symbol, -1):
                    continue
                row = index_by_symbol[symbol].get(current_ts)
                if row is None:
                    continue
                row_df = pd.DataFrame([row])
                stop_pct = self._float_or_none(row.get("barrier_stop_pct"))
                take_pct = self._float_or_none(row.get("barrier_take_pct"))
                if stop_pct is None or take_pct is None or stop_pct <= 0:
                    continue

                pred = model.predict(row_df)
                pred["barrier_stop_pct"] = stop_pct
                pred["barrier_take_pct"] = take_pct
                
                from core.types.domain_types import Tick
                tick = Tick(symbol=symbol, ts=current_ts, bid=ctx["cur_close"], ask=ctx["cur_close"], price=ctx["cur_close"], volume=0.0)
                order = self.strategy.on_prediction(tick, pred, portfolio=None)
                if not order:
                    continue
                    
                from core.types.enums import OrderSide
                signal = 1 if order.side == OrderSide.BUY else -1
                p_long = float(pred.get("p_long", 0.5))
                p_short = float(pred.get("p_short", 0.5))
                direction_prob = max(p_long, p_short)
                signal_gap = abs(p_long - p_short)

                entry_price = (
                    ctx["next_open"] * (1.0 + float(self.config.slippage))
                    if signal == 1
                    else ctx["next_open"] * (1.0 - float(self.config.slippage))
                )
                risk_capital = snapshot_balance * effective_risk
                position_notional = min(risk_capital / stop_pct, snapshot_balance * float(self.config.leverage))
                required_margin = position_notional / max(1e-9, float(self.config.leverage))
                if position_notional < float(self.config.min_position_notional):
                    continue
                score = self._build_entry_score(direction_prob, signal_gap)
                candidates.append(
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "entry_price": float(entry_price),
                        "position_notional": float(position_notional),
                        "required_margin": float(required_margin),
                        "stop_pct": float(stop_pct),
                        "take_pct": float(take_pct),
                        "p_long": p_long,
                        "p_short": p_short,
                        "direction_prob": float(direction_prob),
                        "score": float(score),
                    }
                )

            if not candidates:
                continue

            candidates.sort(key=lambda x: (x["score"], x["direction_prob"]), reverse=True)
            opened_this_bar = 0
            open_positions_count = sum(1 for value in positions.values() if value is not None)
            for candidate in candidates:
                if opened_this_bar >= int(self.config.backtest_max_new_positions_per_bar):
                    break
                if open_positions_count >= int(self.config.backtest_max_open_positions):
                    break
                available_balance = balance - used_margin
                if available_balance <= 0:
                    break

                required_margin = min(candidate["required_margin"], available_balance)
                position_notional = min(candidate["position_notional"], required_margin * float(self.config.leverage))
                if (
                    required_margin <= 0
                    or position_notional < float(self.config.min_position_notional)
                ):
                    continue

                trade_number = next_trade_number
                next_trade_number += 1
                used_margin += required_margin
                positions[candidate["symbol"]] = {
                    "trade_number": trade_number,
                    "dir": int(candidate["signal"]),
                    "entry": float(candidate["entry_price"]),
                    "size": float(position_notional),
                    "margin": float(required_margin),
                    "stop_pct": float(candidate["stop_pct"]),
                    "take_pct": float(candidate["take_pct"]),
                    "ts_open": next_ts,
                }
                trade_events.append(
                    {
                        "type": "OPEN",
                        "trade_number": trade_number,
                        "symbol": candidate["symbol"],
                        "side": "LONG" if candidate["signal"] == 1 else "SHORT",
                        "price": float(candidate["entry_price"]),
                        "qty": float(position_notional / max(1e-9, candidate["entry_price"])),
                        "ts": next_ts,
                        "balance": float(balance),
                        "p_long": candidate["p_long"],
                        "p_short": candidate["p_short"],
                        "score": candidate["score"],
                    }
                )
                direction_str = "LONG" if candidate["signal"] == 1 else "SHORT"
                print(
                    f"[{next_ts}] \u2116 {trade_number} \U0001F525 OPEN {direction_str}: {candidate['symbol']} "
                    f"(Long={candidate['p_long']:.2f}, Short={candidate['p_short']:.2f}, "
                    f"Score={candidate['score']:.3f}) "
                    f"at {candidate['entry_price']:.4f} | "
                    f"Size: {position_notional:.2f}$ "
                    f"Margin: {required_margin:.2f}$"
                )
                opened_this_bar += 1
                open_positions_count += 1

                # IMMEDIATELY CHECK EXIT ON ENTRY BAR (fixes Entry Bar Leakage)
                ctx_intra = market[candidate["symbol"]]
                closed_intra = self._try_close_position(positions[candidate["symbol"]], ctx_intra)
                if closed_intra is not None:
                    exit_price_intra, reason_intra = closed_intra
                    pnl_pct_intra, pnl_abs_intra, commission_intra = self._compute_trade_outcome(positions[candidate["symbol"]], exit_price_intra)
                    previous_loss_streak_intra = consecutive_loss_count

                    used_margin = max(0.0, used_margin - float(positions[candidate["symbol"]]["margin"]))
                    balance += pnl_abs_intra

                    trade_intra = {
                        "trade_number": positions[candidate["symbol"]]["trade_number"],
                        "symbol": candidate["symbol"],
                        "direction": "LONG" if int(positions[candidate["symbol"]]["dir"]) == 1 else "SHORT",
                        "reason": reason_intra,
                        "pnl_pct": float(pnl_pct_intra),
                        "pnl_abs": float(pnl_abs_intra),
                        "commission": float(commission_intra),
                        "ts": next_ts,
                    }
                    trades.append(trade_intra)
                    trade_events.append(
                        {
                            "type": "CLOSE",
                            "trade_number": positions[candidate["symbol"]]["trade_number"],
                            "symbol": candidate["symbol"],
                            "reason": reason_intra,
                            "pnl_pct": float(pnl_pct_intra),
                            "pnl_abs": float(pnl_abs_intra),
                            "commission": float(commission_intra),
                            "ts": next_ts,
                            "balance": float(balance),
                        }
                    )
                    month_key_intra = pd.Timestamp(next_ts).strftime("%Y-%m")
                    monthly[month_key_intra]["pnl_abs"] += float(pnl_abs_intra)
                    monthly[month_key_intra]["trades"] += 1
                    by_symbol[candidate["symbol"]]["trades"] += 1
                    by_symbol[candidate["symbol"]]["pnl_abs"] += float(pnl_abs_intra)
                    if reason_intra == "TP":
                        by_symbol[candidate["symbol"]]["tp"] += 1
                    if reason_intra == "SL":
                        by_symbol[candidate["symbol"]]["sl"] += 1
                        daily_sl_count += 1
                        if int(self.config.backtest_sl_cooldown_bars) > 0:
                            stop_cooldown_until_idx[candidate["symbol"]] = i + int(self.config.backtest_sl_cooldown_bars)

                    if pnl_pct_intra > 0:
                        monthly[month_key_intra]["wins"] += 1
                        by_symbol[candidate["symbol"]]["wins"] += 1
                        consecutive_loss_count = 0
                    else:
                        monthly[month_key_intra]["losses"] += 1
                        by_symbol[candidate["symbol"]]["losses"] += 1
                        consecutive_loss_count += 1
                    
                    positions[candidate["symbol"]] = None
                    open_positions_count -= 1

                    exit_icon_intra = "\u274C" if reason_intra == "SL" else "\u2705" if reason_intra == "TP" else "\u2139\uFE0F"
                    print(
                        f"[{next_ts}] \u2116 {trade_intra['trade_number']} {exit_icon_intra} {candidate['symbol']}: "
                        f"{self._format_trade_log_reason(reason_intra)} (INTRA-BAR) | "
                        f"PnL: {self._format_trade_log_pnl_pct(float(pnl_pct_intra) * 100.0)} | "
                        f"Com: {commission_intra:.2f}$ | "
                        f"Bal: {balance:.2f}"
                    )
                    reduce_after_intra = int(self.config.backtest_reduce_risk_after_consecutive_losses)
                    reduced_risk_intra = float(self.config.backtest_reduced_risk_per_trade)
                    base_risk_intra = float(self.config.risk_per_trade)
                    if reduce_after_intra > 0 and reduced_risk_intra < base_risk_intra:
                        if previous_loss_streak_intra < reduce_after_intra <= consecutive_loss_count:
                            print(
                                f"[{next_ts}] \u26A0\uFE0F Loss streak {consecutive_loss_count}: "
                                f"risk per trade reduced to {reduced_risk_intra * 100:.2f}%"
                            )
                        elif float(pnl_pct_intra) > 0 and previous_loss_streak_intra >= reduce_after_intra:
                            print(
                                f"[{next_ts}] \u2139\uFE0F Loss streak reset: "
                                f"risk per trade restored to {base_risk_intra * 100:.2f}%"
                            )
                    if (
                        reason_intra == "SL"
                        and int(self.config.backtest_max_sl_per_day) > 0
                        and daily_sl_count >= int(self.config.backtest_max_sl_per_day)
                        and not daily_stop_announced
                    ):
                        daily_stop_announced = True
                        print(
                            f"[{next_ts}] \u26D4 Daily SL limit reached ({daily_sl_count}), "
                            "new entries are paused until next day"
                        )

        # final close
        last_ts = common_ts[-1]
        last_mark = {}
        for symbol in symbols:
            row = index_by_symbol[symbol].get(last_ts)
            if row is not None:
                last_mark[symbol] = float(row["close"])
        for symbol, position in list(positions.items()):
            if position is None:
                continue
            mark_price = last_mark.get(symbol)
            if mark_price is None:
                continue
            exit_price = (
                mark_price * (1.0 - float(self.config.slippage))
                if int(position["dir"]) == 1
                else mark_price * (1.0 + float(self.config.slippage))
            )
            pnl_pct, pnl_abs, commission = self._compute_trade_outcome(position, exit_price)
            used_margin = max(0.0, used_margin - float(position["margin"]))
            balance += pnl_abs

            trade = {
                "trade_number": position["trade_number"],
                "symbol": symbol,
                "direction": "LONG" if int(position["dir"]) == 1 else "SHORT",
                "reason": "FINAL",
                "pnl_pct": float(pnl_pct),
                "pnl_abs": float(pnl_abs),
                "commission": float(commission),
                "ts": last_ts,
            }
            trades.append(trade)
            month_key = pd.Timestamp(last_ts).strftime("%Y-%m")
            monthly[month_key]["pnl_abs"] += float(pnl_abs)
            monthly[month_key]["trades"] += 1
            by_symbol[symbol]["trades"] += 1
            by_symbol[symbol]["pnl_abs"] += float(pnl_abs)
            if pnl_pct > 0:
                monthly[month_key]["wins"] += 1
                by_symbol[symbol]["wins"] += 1
            else:
                monthly[month_key]["losses"] += 1
                by_symbol[symbol]["losses"] += 1
            positions[symbol] = None
            print(
                f"[{last_ts}] \u2116 {position['trade_number']} \u23F9 CLOSE {symbol}: FINAL | "
                f"PnL: {self._format_trade_log_pnl_pct(float(pnl_pct) * 100.0)} | "
                f"Com: {commission:.2f}$ | "
                f"Bal: {balance:.2f}"
            )

        final_equity = self._compute_portfolio_equity(balance, positions, last_mark)
        equity_curve.append(final_equity)
        equity_timestamps.append(last_ts)
        peak_equity, max_drawdown = self._update_drawdown_stats(final_equity, peak_equity, max_drawdown)
        report = self._build_report(
            initial_balance=initial_balance,
            end_balance=balance,
            trades=trades,
            equity_curve=equity_curve,
            equity_timestamps=equity_timestamps,
            max_drawdown=max_drawdown,
            monthly=monthly,
            by_symbol=by_symbol,
            trade_events=trade_events,
        )
        self._print_report(report)
        chart_path = self._save_equity_chart(report)
        if chart_path is not None:
            report["equity_chart_path"] = str(chart_path)
        return report

    def _save_equity_chart(self, report: dict) -> Path | None:
        equity_curve = report.get("equity_curve") or []
        equity_timestamps = report.get("equity_timestamps") or []
        if len(equity_curve) <= 1 or len(equity_timestamps) != len(equity_curve):
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        initial_balance = float(report.get("start_balance", 0.0))
        total_trades = int(report.get("trades", 0))
        max_dd = float(report.get("max_drawdown_pct", 0.0))

        idx = pd.DatetimeIndex(pd.to_datetime(equity_timestamps, utc=False))
        ser = pd.Series(equity_curve, index=idx).sort_index()

        plt.figure(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.14)
        plt.step(ser.index, ser.values, where="post", linewidth=1.2, label="Equity (step, дискретный бэктест)")
        if len(ser) > 2:
            daily = ser.resample("1D").last().ffill()
            if len(daily) > 1:
                plt.plot(
                    daily.index,
                    daily.values,
                    color="gray",
                    alpha=0.45,
                    linewidth=1.0,
                    label="Daily last (сглаживание по дням)",
                )

        plt.axhline(y=initial_balance, linestyle="--", color="black", alpha=0.35)
        plt.title(f"Multi-Symbol Equity Curve | {total_trades} trades | DD: {max_dd:.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.gcf().autofmt_xdate()

        if os.getenv("BACKTEST_EQUITY_LOG_Y", "").strip().lower() in ("1", "true", "yes"):
            plt.yscale("log")
            plt.ylabel("Equity (log)")

        risk = float(self.config.risk_per_trade)
        lev = float(self.config.leverage)
        plt.figtext(
            0.5,
            0.02,
            f"RISK_PER_TRADE={risk:.0%}, LEVERAGE={lev:.0f}x — позиция ~ min(risk/stop, equity×lev); "
            f"рост баланса увеличивает номинал (реинвестирование). Линейный plot между точками вводил в заблуждение — используйте step. "
            f"Log-Y: BACKTEST_EQUITY_LOG_Y=1",
            ha="center",
            fontsize=8,
            color="dimgray",
        )

        out_dir = Path(self.config.backtest_charts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "equity_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved chart: {output_path.resolve()}")
        return output_path

    def _build_report(
        self,
        *,
        initial_balance: float,
        end_balance: float,
        trades: list[dict],
        equity_curve: list[float],
        equity_timestamps: list[pd.Timestamp],
        max_drawdown: float,
        monthly: dict,
        by_symbol: dict,
        trade_events: list[dict],
    ) -> dict:
        total_pnl_abs = float(sum(float(row.get("pnl_abs", 0.0)) for row in trades))
        total_fees = float(sum(float(row.get("commission", 0.0)) for row in trades))
        total_trades = len(trades)
        total_wins = sum(1 for row in trades if float(row.get("pnl_abs", 0.0)) > 0)
        total_losses = total_trades - total_wins
        winrate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        expectancy = (total_pnl_abs / total_trades) if total_trades > 0 else 0.0
        total_return_pct = ((end_balance - initial_balance) / initial_balance * 100.0) if initial_balance > 0 else 0.0
        gains = sum(float(row["pnl_abs"]) for row in trades if float(row["pnl_abs"]) > 0)
        losses_abs = abs(sum(float(row["pnl_abs"]) for row in trades if float(row["pnl_abs"]) < 0))
        profit_factor = (gains / losses_abs) if losses_abs > 0 else (float("inf") if gains > 0 else 0.0)
        sharpe, sortino, calmar, cagr = self._risk_metrics(equity_curve, equity_timestamps, max_drawdown)
        direction_stats = {
            "LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
            "SHORT": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        }
        for row in trades:
            direction = str(row.get("direction", "LONG")).upper()
            if direction not in direction_stats:
                continue
            direction_stats[direction]["total"] += 1
            direction_stats[direction]["pnl_abs"] += float(row.get("pnl_abs", 0.0))
            if float(row.get("pnl_abs", 0.0)) > 0:
                direction_stats[direction]["wins"] += 1
            else:
                direction_stats[direction]["losses"] += 1

        return {
            "trades": total_trades,
            "wins": total_wins,
            "losses": total_losses,
            "winrate_pct": winrate,
            "start_balance": float(initial_balance),
            "end_balance": float(end_balance),
            "pnl_abs": float(total_pnl_abs),
            "return_pct": float(total_return_pct),
            "fees": float(total_fees),
            "max_drawdown_pct": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "cagr": float(cagr),
            "monthly": dict(monthly),
            "by_symbol": dict(by_symbol),
            "direction_stats": direction_stats,
            "closed_trades": trades,
            "trade_events": trade_events,
            "equity_curve": list(equity_curve),
            "equity_timestamps": list(equity_timestamps),
            "equity_chart_path": None,
        }

    @staticmethod
    def _max_drawdown_pct(equity_curve: list[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    @staticmethod
    def _risk_metrics(
        equity_curve: list[float],
        equity_ts: list[pd.Timestamp],
        max_drawdown: float,
    ) -> tuple[float, float, float, float]:
        if len(equity_curve) < 3 or not equity_ts:
            return 0.0, 0.0, 0.0, 0.0
        if len(equity_curve) != len(equity_ts):
            min_len = min(len(equity_curve), len(equity_ts))
            if min_len < 3:
                return 0.0, 0.0, 0.0, 0.0
            equity_curve = equity_curve[:min_len]
            equity_ts = equity_ts[:min_len]
        series = pd.Series(equity_curve, index=equity_ts)
        daily = series.resample("D").last().ffill()
        returns = daily.pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0, 0.0, 0.0, 0.0

        mean_daily = float(returns.mean())
        std_daily = float(returns.std())
        sharpe = (mean_daily / std_daily) * np.sqrt(365) if std_daily > 0 else 0.0

        downside = returns[returns < 0]
        if len(downside) > 1 and float(downside.std()) > 0:
            sortino = (mean_daily / float(downside.std())) * np.sqrt(365)
        else:
            sortino = 0.0

        total_days = int((daily.index[-1] - daily.index[0]).days)
        if total_days > 0 and daily.iloc[0] > 0:
            cagr = float((daily.iloc[-1] / daily.iloc[0]) ** (365 / total_days) - 1)
        else:
            cagr = 0.0
        calmar = (cagr / (max_drawdown / 100.0)) if max_drawdown > 0 else 0.0
        return float(sharpe), float(sortino), float(calmar), float(cagr)

    @staticmethod
    def _fmt_money(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}${abs(value):.2f}"

    @staticmethod
    def _fmt_pct(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}{abs(value):.2f}%"

    @staticmethod
    def _print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
        right_align = right_align or set()
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(str(cell)))

        def format_row(row_values: list[str]) -> str:
            formatted: list[str] = []
            for idx, cell in enumerate(row_values):
                text = str(cell)
                formatted.append(text.rjust(widths[idx]) if idx in right_align else text.ljust(widths[idx]))
            return "│ " + " │ ".join(formatted) + " │"

        top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
        mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
        bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"
        print(top)
        print(format_row(headers))
        print(mid)
        for row in rows:
            print(format_row(row))
        print(bottom)

    def _print_report(self, report: dict) -> None:
        print("\nSimulation finished.\n")
        print("╔═══════════════════════════════════════════════════════════╗")
        print(f"║{'Backtest Summary'[:59].center(59)}║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(f"\n📊 Trades: {report['trades']} (W: {report['wins']} / L: {report['losses']})")
        print("💰 Equity:")
        print(f"   Start: ${report['start_balance']:.2f}")
        print(f"   End:   ${report['end_balance']:.2f}")
        print(f"   PnL:   {self._fmt_money(report['pnl_abs'])} ({self._fmt_pct(report['return_pct'])})")
        print(f"   Fees:  ${report['fees']:.2f}")
        print("📉 Risk:")
        print(f"   Max DD: {report['max_drawdown_pct']:.2f}%")
        print(f"   Profit Factor: {report['profit_factor']:.2f}")
        print(f"   Expectancy: {self._fmt_money(report['expectancy'])}")
        print(f"   Sharpe: {report['sharpe']:.2f}")
        print(f"   Sortino: {report.get('sortino', 0.0):.2f}")
        print(f"   Calmar: {report.get('calmar', 0.0):.2f}")
        print(f"   CAGR: {report.get('cagr', 0.0) * 100.0:.2f}%")
        if report["monthly"]:
            monthly_rows: list[list[str]] = []
            for month_key in sorted(report["monthly"].keys()):
                stats = report["monthly"][month_key]
                monthly_rows.append(
                    [
                        month_key,
                        self._fmt_money(stats["pnl_abs"]),
                        str(stats["trades"]),
                        str(stats["wins"]),
                        str(stats["losses"]),
                    ]
                )
            print("\n📅 Monthly Performance Extended:")
            self._print_table(["Month", "PnL", "Total", "Wins", "Losses"], monthly_rows, right_align={1, 2, 3, 4})
        configured_symbols = list(getattr(self.config, "symbols", []) or [])
        by_sym = report.get("by_symbol", {})
        if configured_symbols:
            coin_rows: list[list[str]] = []
            for symbol in configured_symbols:
                stats = by_sym.get(
                    symbol,
                    {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
                )
                winrate = (stats["wins"] / stats["trades"] * 100.0) if stats["trades"] > 0 else 0.0
                coin_rows.append(
                    [
                        symbol.replace("/", ""),
                        str(stats["trades"]),
                        str(stats.get("tp", 0)),
                        str(stats.get("sl", 0)),
                        f"{winrate:.1f}%",
                        self._fmt_money(stats["pnl_abs"]),
                    ]
                )
            print("\n📊 Summary by Coin (all SYMBOLS from config; 0 = no closed trades):")
            self._print_table(["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"], coin_rows, right_align={1, 2, 3, 4, 5})
        elif by_sym:
            coin_rows = []
            for symbol in sorted(by_sym.keys()):
                stats = by_sym[symbol]
                winrate = (stats["wins"] / stats["trades"] * 100.0) if stats["trades"] > 0 else 0.0
                coin_rows.append(
                    [
                        symbol.replace("/", ""),
                        str(stats["trades"]),
                        str(stats.get("tp", 0)),
                        str(stats.get("sl", 0)),
                        f"{winrate:.1f}%",
                        self._fmt_money(stats["pnl_abs"]),
                    ]
                )
            print("\n📊 Summary by Coin:")
            self._print_table(["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"], coin_rows, right_align={1, 2, 3, 4, 5})
        direction_stats = report.get("direction_stats", {})
        if direction_stats:
            direction_rows = []
            for direction in ("LONG", "SHORT"):
                stats = direction_stats.get(direction, {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0})
                direction_rows.append(
                    [
                        direction,
                        str(stats["total"]),
                        str(stats["wins"]),
                        str(stats["losses"]),
                        self._fmt_money(stats["pnl_abs"]),
                    ]
                )
            print("\n📈 Long / Short Summary:")
            self._print_table(
                ["Direction", "Total", "Wins", "Losses", "PnL"],
                direction_rows,
                right_align={1, 2, 3, 4},
            )

    @staticmethod
    def _common_timestamps(data_by_symbol: dict[str, pd.DataFrame], symbols: list[str]) -> list[pd.Timestamp]:
        ts_sets = [set(data_by_symbol[s]["timestamp"]) for s in symbols]
        if not ts_sets:
            return []
        return sorted(list(set.intersection(*ts_sets)))


    def _build_entry_score(self, direction_prob: float, signal_gap: float) -> float:
        edge = max(0.0, float(direction_prob) - float(self.config.directional_proba_threshold))
        return edge * 10.0 + float(signal_gap)

    def _compute_net_pnl_pct(self, direction: int, entry_price: float, exit_price: float) -> float:
        if direction == 1:
            raw = (exit_price - entry_price) / entry_price
        else:
            raw = (entry_price - exit_price) / entry_price
        return float(raw - (float(self.config.taker_com) + float(self.config.taker_com)))

    @staticmethod
    def _mark_to_market_return_pct(direction: int, entry_price: float, mark_price: float) -> float:
        """Без комиссий — только для нереализованного PnL в эквити (иначе «ломается» шаг при входе)."""
        if direction == 1:
            return float((mark_price - entry_price) / entry_price)
        return float((entry_price - mark_price) / entry_price)

    def _compute_trade_outcome(self, position: dict, exit_price: float) -> tuple[float, float, float]:
        pnl_pct = self._compute_net_pnl_pct(int(position["dir"]), float(position["entry"]), float(exit_price))
        position_notional = float(position["size"])
        commission = position_notional * (float(self.config.taker_com) + float(self.config.taker_com))
        pnl_abs = position_notional * pnl_pct
        return float(pnl_pct), float(pnl_abs), float(commission)

    def _compute_portfolio_equity(self, balance: float, positions: dict, mark_prices: dict[str, float]) -> float:
        equity = float(balance)
        for symbol, position in positions.items():
            if position is None:
                continue
            mark = mark_prices.get(symbol)
            if mark is None:
                continue
            mtm_pct = self._mark_to_market_return_pct(int(position["dir"]), float(position["entry"]), float(mark))
            equity += float(position["size"]) * mtm_pct
        return float(equity)

    @staticmethod
    def _update_drawdown_stats(equity: float, peak_equity: float, max_drawdown: float) -> tuple[float, float]:
        peak_equity = max(float(peak_equity), float(equity))
        if peak_equity > 0:
            current_dd = (peak_equity - float(equity)) / peak_equity * 100.0
            max_drawdown = max(float(max_drawdown), float(current_dd))
        return float(peak_equity), float(max_drawdown)

    def _try_close_position(self, position: dict, ctx: dict) -> tuple[float, str] | None:
        if not getattr(self, "exit_manager", None):
            return None
            
        from core.types.domain_types import Position
        from core.types.enums import PositionSide

        pos_obj = Position(
            symbol="dummy",
            side=PositionSide.LONG if position["dir"] == 1 else PositionSide.SHORT,
            amount=1.0,
            entry_price=float(position["entry"]),
        )
        
        exit_price, reason = self.exit_manager.check_causal_exit(
            position=pos_obj,
            next_open=float(ctx["next_open"]),
            next_high=float(ctx["next_high"]),
            next_low=float(ctx["next_low"]),
            stop_pct=float(position["stop_pct"]),
            take_pct=float(position["take_pct"]),
        )
        if exit_price is not None and reason is not None:
            return float(exit_price), str(reason)
        return None

    @staticmethod
    def _float_or_none(value) -> float | None:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value_float):
            return None
        return float(value_float)

