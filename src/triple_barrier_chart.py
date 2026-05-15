from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


BarrierHit = Literal["take", "stop"]
TradeDirection = Literal["long", "short"]


@dataclass(frozen=True)
class Candle:
    time: Any
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class BarrierBreak:
    kind: BarrierHit
    candle_index: int
    candle_time: Any
    price: float


@dataclass(frozen=True)
class TripleBarrierResult:
    direction: TradeDirection
    entry_index: int
    entry_time: Any
    entry: float
    stop: float
    take: float
    break_event: BarrierBreak | None
    output_path: Path | None


def render_triple_barrier_chart(
    candles: Iterable[Any],
    *,
    entry: float,
    stop: float,
    take: float,
    entry_index: int | None = None,
    entry_time: Any | None = None,
    title: str | None = None,
    probability: float | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12.0, 6.5),
    min_visible_candles: int = 200,
) -> TripleBarrierResult:
    """Draw a TradingView-like triple-barrier chart.

    The module is intentionally standalone: copy this file into another project
    and pass candles as dicts, dataclasses, objects, or a pandas DataFrame.

    Case 1: pass only history up to the entry candle and omit entry_index/time.
    Case 2: pass candles after entry too and provide entry_index or entry_time.
    min_visible_candles keeps short test samples from stretching across the chart.
    """

    normalized = normalize_candles(candles)
    if not normalized:
        raise ValueError("candles must contain at least one candle")

    resolved_entry_index = resolve_entry_index(normalized, entry_index=entry_index, entry_time=entry_time)
    direction = infer_direction(entry=entry, stop=stop, take=take)
    break_event = find_first_barrier_break(
        normalized,
        entry_index=resolved_entry_index,
        direction=direction,
        stop=stop,
        take=take,
    )

    path = Path(output_path) if output_path is not None else None
    draw_chart(
        normalized,
        entry=entry,
        stop=stop,
        take=take,
        direction=direction,
        entry_index=resolved_entry_index,
        break_event=break_event,
        title=title,
        probability=probability,
        output_path=path,
        show=show,
        figsize=figsize,
        min_visible_candles=min_visible_candles,
    )

    return TripleBarrierResult(
        direction=direction,
        entry_index=resolved_entry_index,
        entry_time=normalized[resolved_entry_index].time,
        entry=entry,
        stop=stop,
        take=take,
        break_event=break_event,
        output_path=path,
    )


def normalize_candles(candles: Iterable[Any]) -> list[Candle]:
    if _looks_like_dataframe(candles):
        candles = candles.to_dict("records")

    normalized = []
    for i, candle in enumerate(candles):
        time_value = _get_first(candle, ("time", "timestamp", "open_time", "date", "datetime"), default=i)
        normalized.append(
            Candle(
                time=time_value,
                open=float(_get_required(candle, "open")),
                high=float(_get_required(candle, "high")),
                low=float(_get_required(candle, "low")),
                close=float(_get_required(candle, "close")),
            )
        )
    return normalized


def resolve_entry_index(candles: Sequence[Candle], *, entry_index: int | None, entry_time: Any | None) -> int:
    if entry_index is not None and entry_time is not None:
        raise ValueError("pass either entry_index or entry_time, not both")

    if entry_index is not None:
        if entry_index < 0:
            entry_index = len(candles) + entry_index
        if not 0 <= entry_index < len(candles):
            raise IndexError("entry_index is outside candles range")
        return entry_index

    if entry_time is not None:
        normalized_entry_time = _normalize_time(entry_time)
        for i, candle in enumerate(candles):
            if _normalize_time(candle.time) == normalized_entry_time:
                return i
        raise ValueError("entry_time was not found in candles")

    return len(candles) - 1


def infer_direction(*, entry: float, stop: float, take: float) -> TradeDirection:
    if stop < entry < take:
        return "long"
    if take < entry < stop:
        return "short"
    raise ValueError("cannot infer direction: for long use stop < entry < take, for short use take < entry < stop")


def find_first_barrier_break(
    candles: Sequence[Candle],
    *,
    entry_index: int,
    direction: TradeDirection,
    stop: float,
    take: float,
) -> BarrierBreak | None:
    for candle_index in range(entry_index + 1, len(candles)):
        candle = candles[candle_index]
        if direction == "long":
            if candle.low <= stop:
                return BarrierBreak("stop", candle_index, candle.time, stop)
            if candle.high >= take:
                return BarrierBreak("take", candle_index, candle.time, take)
        else:
            if candle.high >= stop:
                return BarrierBreak("stop", candle_index, candle.time, stop)
            if candle.low <= take:
                return BarrierBreak("take", candle_index, candle.time, take)
    return None


def draw_chart(
    candles: Sequence[Candle],
    *,
    entry: float,
    stop: float,
    take: float,
    direction: TradeDirection,
    entry_index: int,
    break_event: BarrierBreak | None,
    title: str | None,
    probability: float | None,
    output_path: Path | None,
    show: bool,
    figsize: tuple[float, float],
    min_visible_candles: int,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=figsize)
    candle_width = 0.58
    visible_candles = max(len(candles), int(min_visible_candles))
    bg = "#0d0d12"
    grid = "#1f1f28"
    muted = "#6f7582"
    green = "#089981"
    red = "#f23645"
    entry_color = "#f6c343"
    entry_marker_color = "#d8dde6"

    for i, candle in enumerate(candles):
        up = candle.close >= candle.open
        color = green if up else red
        body_low = min(candle.open, candle.close)
        body_height = abs(candle.close - candle.open) or _minimum_body_height(candles)

        wick = ax.vlines(
            i,
            candle.low,
            candle.high,
            color=color,
            alpha=0.55,
            linewidth=0.06,
            antialiased=False,
            zorder=2,
        )
        wick.set_snap(True)
        ax.add_patch(
            Rectangle(
                (i - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
                antialiased=False,
                snap=True,
                zorder=3,
            )
        )

    last_barrier_index = break_event.candle_index if break_event is not None else visible_candles
    right_edge = max(entry_index + 1, last_barrier_index)

    _draw_zone(ax, entry_index, right_edge, min(entry, take), max(entry, take), green, alpha=0.26)
    _draw_zone(ax, entry_index, right_edge, min(entry, stop), max(entry, stop), red, alpha=0.22)

    _draw_level(ax, entry_index, right_edge, entry, entry_color, "--")
    _draw_level(ax, entry_index, right_edge, take, green, "--")
    _draw_level(ax, entry_index, right_edge, stop, red, "--")

    ax.axvline(entry_index, color="#4d4d59", linestyle=":", linewidth=1.1, alpha=0.9)
    ax.scatter([entry_index], [entry], color=entry_marker_color, edgecolor=bg, linewidth=1.0, s=48, zorder=5)
    ax.annotate(
        "ENTRY",
        (entry_index, entry),
        xytext=(0, 10),
        textcoords="offset points",
        color="#2d313a",
        fontsize=8,
        weight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": entry_marker_color, "edgecolor": "none", "alpha": 0.95},
    )

    if break_event is not None:
        break_color = green if break_event.kind == "take" else red
        marker_label = "TP" if break_event.kind == "take" else "SL"
        hit_label = "TP HIT" if break_event.kind == "take" else "SL HIT"
        ax.scatter([break_event.candle_index], [break_event.price], color=break_color, edgecolor=bg, linewidth=1.0, s=70, zorder=6)
        ax.axvline(break_event.candle_index, color="#4d4d59", linestyle=":", linewidth=1.1, alpha=0.9)
        ax.annotate(
            marker_label,
            (break_event.candle_index, break_event.price),
            xytext=(0, 12),
            textcoords="offset points",
            color="white",
            fontsize=8,
            weight="bold",
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": break_color, "edgecolor": "none", "alpha": 0.95},
        )
    else:
        hit_label = None

    _draw_symbol_watermark(ax, title or "TRIPLE BARRIER")
    _draw_trade_info(
        ax,
        title=title or "TRIPLE BARRIER",
        direction=direction,
        entry=entry,
        take=take,
        stop=stop,
        probability=probability,
        green=green,
        red=red,
        entry_color=entry_color,
    )

    label_x = visible_candles + 1.5
    _price_badge(ax, label_x, stop, f"STOP {stop:.2f}", red)
    _price_badge(ax, label_x, entry, f"ENTRY {entry:.2f}", entry_color, color="#101014")
    if hit_label is not None:
        _price_badge(ax, label_x, break_event.price, f"{hit_label} {break_event.price:.2f}", green if break_event.kind == "take" else red)
    else:
        _price_badge(ax, label_x, take, f"TP {take:.2f}", green)

    ax.grid(True, color=grid, linewidth=1.05, alpha=0.82)
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="both", colors=muted, labelsize=8, length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color("#2a2a35")
    ax.spines["bottom"].set_color("#2a2a35")
    ax.spines["right"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    tick_positions = _build_time_ticks(len(candles), max_ticks=7)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([_format_time(candles[i].time) for i in tick_positions], rotation=0, ha="center")

    ax.set_xlim(-1, visible_candles + 15)
    ax.margins(y=0.12)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def _draw_level(ax: Any, left: int, right: int, price: float, color: str, linestyle: str) -> None:
    ax.hlines(price, left, right, colors=color, linestyles=linestyle, linewidth=1.3, alpha=0.95)


def _draw_zone(ax: Any, left: int, right: int, bottom: float, top: float, color: str, *, alpha: float) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
    )


def _draw_symbol_watermark(ax: Any, title: str) -> None:
    ax.text(
        0.5,
        0.52,
        title.upper(),
        transform=ax.transAxes,
        color="#d8dde6",
        fontsize=42,
        weight="bold",
        alpha=0.08,
        va="center",
        ha="center",
        zorder=0,
    )


def _draw_trade_info(
    ax: Any,
    *,
    title: str,
    direction: TradeDirection,
    entry: float,
    take: float,
    stop: float,
    probability: float | None,
    green: str,
    red: str,
    entry_color: str,
) -> None:
    order_type = "buy" if direction == "long" else "sell"
    expected_direction = "up" if direction == "long" else "down"
    direction_arrow = "↗" if direction == "long" else "↘"
    risk_pct = abs(entry - stop) / entry * 100 if entry else 0.0
    reward_pct = abs(entry - take) / entry * 100 if entry else 0.0
    rr = reward_pct / risk_pct if risk_pct else 0.0

    x = 0.012
    value_x = 0.125
    y = 0.92
    line_gap = 0.033
    section_gap = 0.047
    label_color = "#d8dde6"
    panel_bottom = 0.475 if probability is not None else 0.51
    _draw_info_panel(ax, x - 0.012, panel_bottom, 0.255, 0.475 if probability is not None else 0.44)

    _info_row(ax, x, value_x, y, "Монета:", title.upper(), label_color=label_color, value_color="white")
    y -= section_gap
    _info_text(ax, x, y, "Сделка:", color="white", weight="bold")
    y -= line_gap
    _info_row(ax, x, value_x, y, "Тип ордера:", order_type, label_color=label_color, value_color="white")
    y -= line_gap
    _info_row(ax, x, value_x, y, "Точка входа:", f"{entry:.5f}", label_color=label_color, value_color=entry_color)
    y -= line_gap
    _info_row(ax, x, value_x, y, "Тейк-профит:", f"{take:.5f}", label_color=label_color, value_color=green)
    y -= line_gap
    _info_row(ax, x, value_x, y, "Стоп-Лосс:", f"{stop:.5f}", label_color=label_color, value_color=red)

    y -= section_gap
    _info_text(ax, x, y, "Риск менеджмент:", color="white", weight="bold")
    y -= line_gap
    _info_row(ax, x, value_x, y, "Риск:", f"{risk_pct:.2f}%", label_color=label_color, value_color=label_color, value_weight="normal")
    y -= line_gap
    _info_row(ax, x, value_x, y, "Вознагражд.:", f"{reward_pct:.2f}%", label_color=label_color, value_color=label_color, value_weight="normal")
    y -= line_gap
    _info_row(ax, x, value_x, y, "R/R:", f"{rr:.2f}", label_color=label_color, value_color=label_color, value_weight="bold")

    y -= section_gap
    _info_row(
        ax,
        x,
        value_x,
        y,
        "Направление:",
        f"{expected_direction} {direction_arrow}",
        label_color=label_color,
        value_color=green if direction == "long" else red,
    )
    if probability is not None:
        y -= line_gap
        probability_value = probability * 100 if 0 <= probability <= 1 else probability
        quality = "высокая" if probability_value >= 70 else "средняя" if probability_value >= 55 else "низкая"
        _info_row(
            ax,
            x,
            value_x,
            y,
            "Вероятность:",
            f"{probability_value:.0f}% ({quality})",
            label_color=label_color,
            value_color=label_color,
        )


def _draw_info_panel(ax: Any, x: float, y: float, width: float, height: float) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor="#3a3d46",
            edgecolor="#595d68",
            linewidth=0.8,
            alpha=0.42,
            zorder=9,
        )
    )


def _info_row(
    ax: Any,
    label_x: float,
    value_x: float,
    y: float,
    label: str,
    value: str,
    *,
    label_color: str,
    value_color: str,
    value_weight: str = "bold",
) -> None:
    _info_text(ax, label_x, y, label, color=label_color)
    _info_text(ax, value_x, y, value, color=value_color, weight=value_weight)


def _info_text(ax: Any, x: float, y: float, text: str, *, color: str, weight: str = "normal") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        color=color,
        fontsize=7.5,
        weight=weight,
        va="top",
        ha="left",
        zorder=10,
    )


def _build_time_ticks(candle_count: int, *, max_ticks: int) -> list[int]:
    if candle_count <= 0:
        return []
    if candle_count == 1:
        return [0]

    step = max(1, round((candle_count - 1) / max(1, max_ticks - 1)))
    ticks = list(range(0, candle_count, step))
    last = candle_count - 1
    min_gap = max(2, step // 2)
    if ticks[-1] != last and last - ticks[-1] >= min_gap:
        ticks.append(last)
    return ticks


def _price_badge(ax: Any, x: float, y: float, label: str, facecolor: str, *, color: str = "white") -> None:
    ax.text(
        x,
        y,
        label,
        color=color,
        fontsize=8,
        weight="bold",
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": facecolor, "edgecolor": "none", "alpha": 0.95},
        clip_on=False,
    )


def _minimum_body_height(candles: Sequence[Candle]) -> float:
    prices = [price for candle in candles for price in (candle.high, candle.low)]
    price_range = max(prices) - min(prices)
    return price_range * 0.002 if price_range else 0.0001


def _get_required(candle: Any, key: str) -> Any:
    value = _get_first(candle, (key,), default=None)
    if value is None:
        raise ValueError(f"candle is missing required field: {key}")
    return value


def _get_first(candle: Any, keys: tuple[str, ...], *, default: Any) -> Any:
    if isinstance(candle, Mapping):
        for key in keys:
            if key in candle:
                return candle[key]
        return default

    for key in keys:
        if hasattr(candle, key):
            return getattr(candle, key)
    return default


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "to_dict") and hasattr(value, "columns")


def _normalize_time(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value
    return value


def _format_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)
