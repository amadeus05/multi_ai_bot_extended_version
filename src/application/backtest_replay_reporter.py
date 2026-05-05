from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[91m"
_ANSI_GREEN = "\033[92m"


class BacktestReplayReporter:
    """Backtest-style logging and reporting backed by PortfolioManager state."""

    def __init__(self, *, config, portfolio, charts_dir: str | None = None) -> None:
        self.config = config
        self.portfolio = portfolio
        self.charts_dir = charts_dir or getattr(config, "backtest_charts_dir", "backtest_charts")
        self.equity_curve: list[float] = []
        self.equity_timestamps: list[pd.Timestamp] = []
        self._seen_trade_events = 0

    @staticmethod
    def _colorize(text: str, color: str) -> str:
        return f"{color}{text}{_ANSI_RESET}"

    @classmethod
    def _format_pnl_pct(cls, pnl_pct: float) -> str:
        color = _ANSI_GREEN if pnl_pct >= 0 else _ANSI_RED
        return cls._colorize(f"{pnl_pct:+.2f}%", color)

    @classmethod
    def _format_reason(cls, reason: str) -> str:
        if reason == "TP":
            return cls._colorize("TP", _ANSI_GREEN)
        if reason == "SL":
            return cls._colorize("SL", _ANSI_RED)
        return reason

    @staticmethod
    def _fmt_money(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}${abs(value):.2f}"

    @staticmethod
    def _fmt_pct(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}{abs(value):.2f}%"

    def flush_trade_events(self) -> None:
        events = self.portfolio.trade_events[self._seen_trade_events :]
        self._seen_trade_events = len(self.portfolio.trade_events)
        for event in events:
            if event.get("type") == "OPEN":
                self._print_open(event)
            elif event.get("type") == "CLOSE":
                self._print_close(event)

    def record_equity(self, ts, mark_prices: dict[str, float]) -> float:
        equity = self._compute_equity(mark_prices)
        self.equity_curve.append(float(equity))
        self.equity_timestamps.append(pd.Timestamp(ts))
        return float(equity)

    def _compute_equity(self, mark_prices: dict[str, float]) -> float:
        equity = float(self.portfolio.cash.get("USDT", 0.0))
        for pos in self.portfolio.get_open_positions():
            mark = float(mark_prices.get(pos.symbol, pos.entry_price))
            if pos.side.value == "long":
                equity += float(pos.amount) * (mark - float(pos.entry_price))
            else:
                equity += float(pos.amount) * (float(pos.entry_price) - mark)
        return float(equity)

    def _print_open(self, event: dict) -> None:
        p_long = float(event.get("p_long", 0.5))
        p_short = float(event.get("p_short", 0.5))
        score = float(event.get("score", max(p_long, p_short)))
        notional = float(event.get("qty", 0.0)) * float(event.get("price", 0.0))
        margin = notional / max(1e-9, float(getattr(self.config, "leverage", 1.0)))
        print(
            f"[{event.get('ts')}] № {event.get('trade_number')} 🔥 OPEN {event.get('side')}: {event.get('symbol')} "
            f"(Long={p_long:.2f}, Short={p_short:.2f}, Score={score:.3f}) "
            f"at {float(event.get('price', 0.0)):.4f} | "
            f"Size: {notional:.2f}$ Margin: {margin:.2f}$"
        )

    def _print_close(self, event: dict) -> None:
        reason = str(event.get("reason", "CLOSE"))
        suffix = event.get("reason_suffix")
        label = f"{reason} ({suffix})" if suffix else reason
        icon = "❌" if reason == "SL" else "✅" if reason == "TP" else "ℹ️"
        pnl_pct = self._event_pnl_pct(event)
        print(
            f"[{event.get('ts')}] № {event.get('trade_number')} {icon} {event.get('symbol')}: "
            f"{self._format_reason(reason) if not suffix else label} | "
            f"PnL: {self._format_pnl_pct(pnl_pct * 100.0)} | "
            f"Com: {float(event.get('commission', 0.0)):.2f}$ | "
            f"Bal: {float(event.get('balance', 0.0)):.2f}"
        )

    def _event_pnl_pct(self, event: dict) -> float:
        trade_number = event.get("trade_number")
        for row in reversed(self.portfolio.closed_trade_results):
            if row.get("trade_number") == trade_number and row.get("ts") == event.get("ts"):
                notional = abs(float(row.get("qty", 0.0)) * float(row.get("entry_price", 0.0)))
                if notional > 0:
                    return float(row.get("pnl_abs", 0.0)) / notional
                return float(row.get("pnl_pct", 0.0))
        return 0.0

    def build_report(self) -> dict:
        closed = [dict(row) for row in self.portfolio.closed_trade_results]
        initial_balance = float(getattr(self.config, "initial_capital", 0.0))
        end_balance = float(self.portfolio.cash.get("USDT", 0.0))
        total_pnl_abs = float(sum(float(row.get("pnl_abs", 0.0)) for row in closed))
        total_fees = float(sum(float(row.get("commission", 0.0)) for row in closed))
        total_trades = len(closed)
        total_wins = sum(1 for row in closed if float(row.get("pnl_abs", 0.0)) > 0)
        total_losses = total_trades - total_wins
        gains = sum(float(row.get("pnl_abs", 0.0)) for row in closed if float(row.get("pnl_abs", 0.0)) > 0)
        losses_abs = abs(sum(float(row.get("pnl_abs", 0.0)) for row in closed if float(row.get("pnl_abs", 0.0)) < 0))
        profit_factor = gains / losses_abs if losses_abs > 0 else (float("inf") if gains > 0 else 0.0)
        max_dd = self._max_drawdown_pct(self.equity_curve)
        sharpe, sortino, calmar, cagr = self._risk_metrics(self.equity_curve, self.equity_timestamps, max_dd)

        monthly = defaultdict(lambda: {"pnl_abs": 0.0, "trades": 0, "wins": 0, "losses": 0})
        by_symbol = defaultdict(lambda: {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0})
        direction_stats = {
            "LONG": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
            "SHORT": {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0},
        }
        for row in closed:
            ts = pd.Timestamp(row.get("ts"))
            month_key = ts.strftime("%Y-%m")
            pnl_abs = float(row.get("pnl_abs", 0.0))
            symbol = str(row.get("symbol", ""))
            reason = str(row.get("reason", ""))
            direction = "LONG" if str(row.get("side", "")).lower() == "long" else "SHORT"
            monthly[month_key]["pnl_abs"] += pnl_abs
            monthly[month_key]["trades"] += 1
            by_symbol[symbol]["trades"] += 1
            by_symbol[symbol]["pnl_abs"] += pnl_abs
            direction_stats[direction]["total"] += 1
            direction_stats[direction]["pnl_abs"] += pnl_abs
            if reason == "TP":
                by_symbol[symbol]["tp"] += 1
            if reason == "SL":
                by_symbol[symbol]["sl"] += 1
            if pnl_abs > 0:
                monthly[month_key]["wins"] += 1
                by_symbol[symbol]["wins"] += 1
                direction_stats[direction]["wins"] += 1
            else:
                monthly[month_key]["losses"] += 1
                by_symbol[symbol]["losses"] += 1
                direction_stats[direction]["losses"] += 1

        return {
            "trades": total_trades,
            "wins": total_wins,
            "losses": total_losses,
            "winrate_pct": (total_wins / total_trades * 100.0) if total_trades else 0.0,
            "start_balance": initial_balance,
            "end_balance": end_balance,
            "pnl_abs": total_pnl_abs,
            "return_pct": ((end_balance - initial_balance) / initial_balance * 100.0) if initial_balance > 0 else 0.0,
            "fees": total_fees,
            "max_drawdown_pct": max_dd,
            "profit_factor": float(profit_factor),
            "expectancy": (total_pnl_abs / total_trades) if total_trades else 0.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "cagr": cagr,
            "monthly": dict(monthly),
            "by_symbol": dict(by_symbol),
            "direction_stats": direction_stats,
            "closed_trades": closed,
            "trade_events": list(self.portfolio.trade_events),
            "equity_curve": list(self.equity_curve),
            "equity_timestamps": list(self.equity_timestamps),
            "equity_chart_path": None,
        }

    def finish(self) -> dict:
        report = self.build_report()
        self._print_report(report)
        chart_path = self._save_equity_chart(report)
        if chart_path is not None:
            report["equity_chart_path"] = str(chart_path)
        return report

    @staticmethod
    def _max_drawdown_pct(equity_curve: list[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak * 100.0)
        return float(max_dd)

    @staticmethod
    def _risk_metrics(equity_curve: list[float], equity_ts: list[pd.Timestamp], max_drawdown: float) -> tuple[float, float, float, float]:
        if len(equity_curve) < 3 or len(equity_curve) != len(equity_ts):
            return 0.0, 0.0, 0.0, 0.0
        series = pd.Series(equity_curve, index=pd.to_datetime(equity_ts)).sort_index()
        daily = series.resample("D").last().ffill()
        returns = daily.pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean_daily = float(returns.mean())
        std_daily = float(returns.std())
        sharpe = (mean_daily / std_daily) * np.sqrt(365) if std_daily > 0 else 0.0
        downside = returns[returns < 0]
        sortino = (mean_daily / float(downside.std())) * np.sqrt(365) if len(downside) > 1 and float(downside.std()) > 0 else 0.0
        total_days = int((daily.index[-1] - daily.index[0]).days)
        cagr = float((daily.iloc[-1] / daily.iloc[0]) ** (365 / total_days) - 1) if total_days > 0 and daily.iloc[0] > 0 else 0.0
        calmar = cagr / (max_drawdown / 100.0) if max_drawdown > 0 else 0.0
        return float(sharpe), float(sortino), float(calmar), float(cagr)

    @staticmethod
    def _print_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> None:
        right_align = right_align or set()
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(str(cell)))

        def format_row(row_values: list[str]) -> str:
            formatted = []
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
        print("╔" + "═" * 59 + "╗")
        print(f"║{'Backtest Summary'[:59].center(59)}║")
        print("╚" + "═" * 59 + "╝")
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
            rows = []
            for month_key in sorted(report["monthly"]):
                stats = report["monthly"][month_key]
                rows.append([month_key, self._fmt_money(stats["pnl_abs"]), str(stats["trades"]), str(stats["wins"]), str(stats["losses"])])
            print("\n📅 Monthly Performance Extended:")
            self._print_table(["Month", "PnL", "Total", "Wins", "Losses"], rows, right_align={1, 2, 3, 4})

        symbols = list(getattr(self.config, "symbols", []) or [])
        if symbols:
            rows = []
            for symbol in symbols:
                stats = report["by_symbol"].get(symbol, {"trades": 0, "tp": 0, "sl": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0})
                winrate = (stats["wins"] / stats["trades"] * 100.0) if stats["trades"] > 0 else 0.0
                rows.append([symbol.replace("/", ""), str(stats["trades"]), str(stats.get("tp", 0)), str(stats.get("sl", 0)), f"{winrate:.1f}%", self._fmt_money(stats["pnl_abs"])])
            print("\n📊 Summary by Coin (all SYMBOLS from config; 0 = no closed trades):")
            self._print_table(["Symbol", "Trades", "TP", "SL", "Winrate", "PnL"], rows, right_align={1, 2, 3, 4, 5})

        direction_rows = []
        for direction in ("LONG", "SHORT"):
            stats = report["direction_stats"].get(direction, {"total": 0, "wins": 0, "losses": 0, "pnl_abs": 0.0})
            direction_rows.append([direction, str(stats["total"]), str(stats["wins"]), str(stats["losses"]), self._fmt_money(stats["pnl_abs"])])
        print("\n📈 Long / Short Summary:")
        self._print_table(["Direction", "Total", "Wins", "Losses", "PnL"], direction_rows, right_align={1, 2, 3, 4})

    def _save_equity_chart(self, report: dict) -> Path | None:
        if len(self.equity_curve) <= 1 or len(self.equity_curve) != len(self.equity_timestamps):
            return None
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idx = pd.DatetimeIndex(pd.to_datetime(self.equity_timestamps, utc=False))
        ser = pd.Series(self.equity_curve, index=idx).sort_index()
        plt.figure(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.14)
        plt.step(ser.index, ser.values, where="post", linewidth=1.2, label="Equity (step, replay TradingEngine)")
        if len(ser) > 2:
            daily = ser.resample("1D").last().ffill()
            if len(daily) > 1:
                plt.plot(daily.index, daily.values, color="gray", alpha=0.45, linewidth=1.0, label="Daily last")
        plt.axhline(y=float(report.get("start_balance", 0.0)), linestyle="--", color="black", alpha=0.35)
        plt.title(f"Replay Equity Curve | {int(report.get('trades', 0))} trades | DD: {float(report.get('max_drawdown_pct', 0.0)):.1f}%")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.gcf().autofmt_xdate()
        out_dir = Path(self.charts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "equity_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved chart: {output_path.resolve()}")
        return output_path
